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
            # Architecture matching Celeb-DF trained checkpoint
            self.classifier = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )
        else:
            # Architecture matching FF++ trained checkpoint
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
# DeepfakeDetector
# -----------------------------------------------------------------------

class DeepfakeDetector:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # Initialize MTCNN face detector for focused facial crops
        try:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(
                keep_all=False,
                select_largest=True,
                post_process=False,
                device=self.device
            )
            print(f"Face detector: MTCNN initialized successfully on {self.device}")
        except Exception as e:
            self.face_detector = None
            print(f"Face detector: MTCNN initialization failed ({e}). Using center-crop fallback.")

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
        print(f"Model loaded successfully (LayerNorm head = {use_layernorm})")

    def extract_face(self, frame_bgr):
        """
        Detects the primary face using MTCNN and crops a focused region
        with proportional padding (scale=1.4x face size, y_shift=0.05)
        to capture the full face, jawline, and boundary seams.
        Resizes to (256, 256). Falls back to centered crop if no face detected.
        """
        h, w = frame_bgr.shape[:2]

        if self.face_detector is not None:
            try:
                from PIL import Image
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                boxes, probs = self.face_detector.detect(pil_img)

                if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.6:
                    x1, y1, x2, y2 = boxes[0]
                    fw = x2 - x1
                    fh = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # 1.28x proportional scaling around face center matching dataset face crops
                    crop_size = max(fw, fh) * 1.28
                    half = crop_size / 2.0
                    scy = cy - fh * 0.02

                    nx1 = int(round(cx - half))
                    ny1 = int(round(scy - half))
                    nx2 = int(round(cx + half))
                    ny2 = int(round(scy + half))

                    # Keep crop in bounds while preserving 1:1 aspect ratio
                    if nx1 < 0:
                        nx2 = min(w, nx2 - nx1)
                        nx1 = 0
                    if ny1 < 0:
                        ny2 = min(h, ny2 - ny1)
                        ny1 = 0
                    if nx2 > w:
                        nx1 = max(0, nx1 - (nx2 - w))
                        nx2 = w
                    if ny2 > h:
                        ny1 = max(0, ny1 - (ny2 - h))
                        ny2 = h

                    if (nx2 - nx1) > 20 and (ny2 - ny1) > 20:
                        crop = frame_bgr[ny1:ny2, nx1:nx2]
                        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                pass

        # Fallback: centered square crop
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        half = min_dim // 2
        crop = frame_bgr[cy - half: cy + half, cx - half: cx + half]
        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)

    @torch.no_grad()
    def predict_frame(self, face_bgr_256):
        """
        Runs triple-fusion inference on a 256x256 BGR face crop.
        Returns (real_probability, fake_probability) as floats in [0.0, 1.0].
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
