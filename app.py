import os
import uuid
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from model import DeepfakeDetector

# Use absolute paths so Flask finds templates/static from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config["SECRET_KEY"] = "deepfake-swin-v2-demo-secret"
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max video upload
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -------------------------------------------------------------
# Model Initialization
# -------------------------------------------------------------
# Check known locations for best checkpoint
DEFAULT_CHECKPOINTS = [
    os.path.join(os.path.dirname(__file__), "best_celebdf_to_ff.pth"),
    os.path.join(os.path.dirname(__file__), "best_swinv2_lbp_dct.pth"),
    "/kaggle/working/celebdf_trained_checkpoints/best_celebdf_to_ff.pth",
    "/kaggle/working/threeway_checkpoints/best_swinv2_lbp_dct.pth",
    "best_celebdf_to_ff.pth",
    "best_swinv2_lbp_dct.pth"
]

CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT")
if not CHECKPOINT_PATH or not os.path.isfile(CHECKPOINT_PATH):
    for candidate in DEFAULT_CHECKPOINTS:
        if os.path.isfile(candidate):
            CHECKPOINT_PATH = candidate
            break

detector = None
try:
    if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        detector = DeepfakeDetector(checkpoint_path=CHECKPOINT_PATH)
    else:
        print("WARNING: Checkpoint file not found. Place 'best_celebdf_to_ff.pth' in the project folder.")
        detector = DeepfakeDetector()
except Exception as e:
    print(f"Error loading model: {e}")
    detector = DeepfakeDetector()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", checkpoint_loaded=(detector.model is not None))

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    if detector.model is None:
        return jsonify({
            "error": "Model weights are not loaded. Place your .pth checkpoint in the directory or set MODEL_CHECKPOINT."
        }), 500

    start_time = time.time()
    
    # Create unique session folder for this video run
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_filename = secure_filename(file.filename)
    video_path = os.path.join(session_dir, video_filename)
    file.save(video_path)

    # -------------------------------------------------------------
    # Frame Extraction (1 FPS, matching training preprocessing)
    # -------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file."}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # safe default

    FPS_TO_EXTRACT = 1.0
    frame_interval = max(1, round(fps / FPS_TO_EXTRACT))
    MAX_FRAMES_TO_ANALYZE = 30  # Cap to 30 frames for fast, responsive live demo

    frame_idx = 0
    saved_count = 0
    frame_results = []
    fake_probs = []

    while True:
        success, frame = cap.read()
        if not success or saved_count >= MAX_FRAMES_TO_ANALYZE:
            break

        if frame_idx % frame_interval == 0:
            # 1. Detect & Crop Face
            face_crop = detector.extract_face(frame)

            # 2. Save thumbnail for frontend timeline
            thumb_filename = f"frame_{saved_count:04d}.jpg"
            thumb_path = os.path.join(session_dir, thumb_filename)
            cv2.imwrite(thumb_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # 3. Model Inference (Swin + LBP + DCT)
            real_p, fake_p = detector.predict_frame(face_crop)
            fake_probs.append(fake_p)

            # Timestamp calculation
            sec = int(frame_idx / fps)
            timestamp_str = f"{sec // 60:02d}:{sec % 60:02d}"

            frame_results.append({
                "frame_num": saved_count + 1,
                "timestamp": timestamp_str,
                "image_url": f"/static/uploads/{session_id}/{thumb_filename}",
                "fake_prob": round(fake_p * 100, 2),
                "real_prob": round(real_p * 100, 2),
                "is_fake": fake_p >= 0.5
            })

            saved_count += 1

        frame_idx += 1

    cap.release()

    if len(fake_probs) == 0:
        return jsonify({"error": "No frames could be extracted from this video."}), 400

    # -------------------------------------------------------------
    # Video-Level Temporal Mean-Pooling
    # -------------------------------------------------------------
    video_fake_prob = float(np.mean(fake_probs))
    video_real_prob = 1.0 - video_fake_prob

    is_fake_verdict = video_fake_prob >= 0.50
    verdict = "DEEPFAKE / MANIPULATED" if is_fake_verdict else "AUTHENTIC / REAL"
    confidence = (video_fake_prob if is_fake_verdict else video_real_prob) * 100

    fake_frames_count = sum(1 for p in fake_probs if p >= 0.5)
    real_frames_count = len(fake_probs) - fake_frames_count
    elapsed_time = round(time.time() - start_time, 2)

    return jsonify({
        "verdict": verdict,
        "is_fake": is_fake_verdict,
        "confidence": f"{confidence:.1f}%",
        "video_fake_prob": round(video_fake_prob * 100, 2),
        "video_real_prob": round(video_real_prob * 100, 2),
        "total_frames": len(fake_probs),
        "fake_frames": fake_frames_count,
        "real_frames": real_frames_count,
        "elapsed_time": f"{elapsed_time}s",
        "frames": frame_results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
