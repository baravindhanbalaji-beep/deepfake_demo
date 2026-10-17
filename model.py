import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------
swin_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

tex_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# -----------------------------------------------------------------------
# Texture feature extractors
# -----------------------------------------------------------------------

def compute_lbp(image):
    """Compute 8-neighbor Local Binary Pattern map on grayscale image."""
    image = image.astype(np.uint8)
    center = image[1:-1, 1:-1]
    neighbors = [
        image[:-2, :-2], image[:-2, 1:-1], image[:-2, 2:],
        image[1:-1, 2:], image[2:, 2:], image[2:, 1:-1],
        image[2:, :-2], image[1:-1, :-2]
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    result = np.zeros_like(image, dtype=np.uint8)
    result[1:-1, 1:-1] = lbp
    return result


def compute_dct(gray_image):
    """Compute 2D DCT with log scaling and min-max normalization."""
    im_f = gray_image.astype(np.float32) / 255.0
    coeff = cv2.dct(im_f)
    coeff = np.log1p(np.abs(coeff))
    coeff = cv2.normalize(coeff, None, 0, 255, cv2.NORM_MINMAX)
    return coeff.astype(np.uint8)


# -----------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------

class FeatureBranch(nn.Module):
    """3-layer CNN feature extractor for LBP and DCT maps."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class SwinTripleFusionNet(nn.Module):
    """Swin V2 Tiny + LBP CNN + DCT CNN Triple-Fusion Architecture."""
    def __init__(self, use_layernorm=True):
        super().__init__()
        self.swin = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2)
        self.swin.head = nn.Identity()

        self.lbp_branch = FeatureBranch()
        self.dct_branch = FeatureBranch()

        if use_layernorm:
            # Celeb-DF trained checkpoint architecture
            self.classifier = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )
        else:
            # FF++ trained checkpoint architecture
            self.classifier = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )

    def forward(self, rgb, lbp, dct):
        swin_feats = self.swin(rgb)
        if swin_feats.ndim == 4:
            swin_feats = swin_feats.mean(dim=(1, 2)) if swin_feats.shape[-1] == 768 else swin_feats.mean(dim=(2, 3))
        elif swin_feats.ndim == 3:
            swin_feats = swin_feats.mean(dim=1)

        lbp_feats = self.lbp_branch(lbp)
        dct_feats = self.dct_branch(dct)

        combined = torch.cat([swin_feats, lbp_feats, dct_feats], dim=1)
        return self.classifier(combined)


# -----------------------------------------------------------------------
# Face Detector — MediaPipe Short-Range (primary) → center-crop (fallback)
# -----------------------------------------------------------------------

def _load_mediapipe_detector():
    """
    Attempts to load MediaPipe FaceDetection.
    Returns (detector, True) on success, (None, False) on failure.

    MediaPipe's Short-Range model is optimised for faces within ~2 m of
    the camera — perfect for selfie/interview-style deepfake videos.
    """
    try:
        import mediapipe as mp
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,      # 0 = short-range (~2 m), 1 = full-range (~5 m)
            min_detection_confidence=0.5
        )
        print("Face detector: MediaPipe loaded successfully.")
        return detector, True
    except Exception as e:
        print(f"WARNING: MediaPipe unavailable ({e}). Using center-crop fallback.")
        return None, False


def _mediapipe_detect(detector, frame_bgr):
    """
    Runs MediaPipe on a BGR frame.
    Returns list of (x, y, w, h) in pixel coordinates, or [].
    """
    import mediapipe as mp
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            boxes.append((x, y, bw, bh))
    return boxes


# -----------------------------------------------------------------------
# DeepfakeDetector
# -----------------------------------------------------------------------

class DeepfakeDetector:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # Load face detector
        self._mp_detector, self.use_face_detector = _load_mediapipe_detector()

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
        use_layernorm = "classifier.4.weight" in cleaned

        self.model = SwinTripleFusionNet(use_layernorm=use_layernorm).to(self.device)
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()
        print(f"Model loaded (LayerNorm head = {use_layernorm})")

    def extract_face(self, frame_bgr):
        """
        1. Run MediaPipe face detection.
        2. Pick the largest bounding box.
        3. Expand by 20% margin (captures jawline / boundary seams).
        4. Fallback → centred square crop if no face found.
        5. Resize to 256×256 (the model's expected input resolution).
        """
        h, w = frame_bgr.shape[:2]

        faces = []
        if self.use_face_detector:
            try:
                faces = _mediapipe_detect(self._mp_detector, frame_bgr)
            except Exception as e:
                print(f"Face detection error: {e}")

        if faces:
            # Pick largest face by area
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])

            # 20% margin expansion — same as training preprocessing
            margin_x = int(fw * 0.20)
            margin_y = int(fh * 0.20)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + fw + margin_x)
            y2 = min(h, y + fh + margin_y)

            face_crop = frame_bgr[y1:y2, x1:x2]
        else:
            # Fallback: centred square crop
            min_dim = min(h, w)
            cy, cx = h // 2, w // 2
            half = min_dim // 2
            face_crop = frame_bgr[cy - half: cy + half, cx - half: cx + half]

        return cv2.resize(face_crop, (256, 256))

    @torch.no_grad()
    def predict_frame(self, face_bgr_256):
        """
        Runs triple-fusion inference on a 256×256 BGR face crop.
        Returns (real_probability, fake_probability) as floats in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model(path) first.")

        rgb  = cv2.cvtColor(face_bgr_256, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(face_bgr_256, cv2.COLOR_BGR2GRAY)

        rgb_t = swin_transform(rgb).unsqueeze(0).to(self.device)
        lbp_t = tex_transform(compute_lbp(gray)).unsqueeze(0).to(self.device)
        dct_t = tex_transform(compute_dct(gray)).unsqueeze(0).to(self.device)

        logits = self.model(rgb_t, lbp_t, dct_t)
        probs  = F.softmax(logits.float(), dim=1).squeeze(0)

        return float(probs[0].item()), float(probs[1].item())
