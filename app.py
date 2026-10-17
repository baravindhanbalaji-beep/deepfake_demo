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
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS

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

def is_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def is_image_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", checkpoint_loaded=(detector.model is not None))

@app.route("/predict", methods=["POST"])
def predict():
    uploaded_files = []
    for key in ("video", "frames", "images", "file"):
        if key in request.files:
            uploaded_files.extend(request.files.getlist(key))

    uploaded_files = [f for f in uploaded_files if f and f.filename and f.filename.strip()]

    if not uploaded_files:
        return jsonify({"error": "No video or image frame files provided."}), 400

    if detector.model is None:
        return jsonify({
            "error": "Model weights are not loaded. Place your .pth checkpoint in the directory or set MODEL_CHECKPOINT."
        }), 500

    invalid_files = [f.filename for f in uploaded_files if not allowed_file(f.filename)]
    if invalid_files:
        return jsonify({
            "error": f"Invalid format for: {', '.join(invalid_files[:3])}. Allowed videos: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}, Allowed frames: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        }), 400

    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_files = [f for f in uploaded_files if is_video_file(f.filename)]
    image_files = [f for f in uploaded_files if is_image_file(f.filename)]

    frame_results = []
    fake_probs = []
    is_video_mode = len(video_files) > 0

    if is_video_mode:
        file = video_files[0]
        video_filename = secure_filename(file.filename)
        video_path = os.path.join(session_dir, video_filename)
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file."}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        FPS_TO_EXTRACT = 1.0
        frame_interval = max(1, round(fps / FPS_TO_EXTRACT))
        MAX_FRAMES_TO_ANALYZE = 30

        frame_idx = 0
        saved_count = 0

        while True:
            success, frame = cap.read()
            if not success or saved_count >= MAX_FRAMES_TO_ANALYZE:
                break

            if frame_idx % frame_interval == 0:
                face_crop = detector.extract_face(frame)

                thumb_filename = f"frame_{saved_count:04d}.jpg"
                thumb_path = os.path.join(session_dir, thumb_filename)
                cv2.imwrite(thumb_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

                real_p, fake_p = detector.predict_frame(face_crop)
                fake_probs.append(fake_p)

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

    else:
        import re
        def natural_sort_key(f):
            name = getattr(f, "filename", "") or ""
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]

        image_files = sorted(image_files, key=natural_sort_key)

        MAX_FRAMES_TO_ANALYZE = 50
        if len(image_files) > MAX_FRAMES_TO_ANALYZE:
            step = len(image_files) / float(MAX_FRAMES_TO_ANALYZE)
            sampled_indices = [int(i * step) for i in range(MAX_FRAMES_TO_ANALYZE)]
            frames_to_process = [image_files[i] for i in sampled_indices]
        else:
            frames_to_process = image_files

        for idx, file in enumerate(frames_to_process):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            face_crop = detector.extract_face(frame)

            thumb_filename = f"frame_{idx:04d}.jpg"
            thumb_path = os.path.join(session_dir, thumb_filename)
            cv2.imwrite(thumb_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

            real_p, fake_p = detector.predict_frame(face_crop)
            fake_probs.append(fake_p)

            clean_name = os.path.basename(file.filename.replace("\\", "/"))
            if not clean_name:
                clean_name = f"Frame #{idx + 1}"

            frame_results.append({
                "frame_num": idx + 1,
                "timestamp": clean_name,
                "image_url": f"/static/uploads/{session_id}/{thumb_filename}",
                "fake_prob": round(fake_p * 100, 2),
                "real_prob": round(real_p * 100, 2),
                "is_fake": fake_p >= 0.5
            })

        if len(fake_probs) == 0:
            return jsonify({"error": "No valid image frames could be read."}), 400

    video_fake_prob = float(np.mean(fake_probs))
    video_real_prob = 1.0 - video_fake_prob
    is_fake_verdict = video_fake_prob >= 0.50

    if not is_video_mode and len(fake_probs) == 1:
        verdict = "DEEPFAKE / MANIPULATED FRAME" if is_fake_verdict else "AUTHENTIC / REAL FRAME"
    else:
        verdict = "DEEPFAKE / MANIPULATED" if is_fake_verdict else "AUTHENTIC / REAL"

    confidence = (video_fake_prob if is_fake_verdict else video_real_prob) * 100
    fake_frames_count = sum(1 for p in fake_probs if p >= 0.5)
    real_frames_count = len(fake_probs) - fake_frames_count
    elapsed_time = round(time.time() - start_time, 2)

    if is_fake_verdict:
        if is_video_mode or len(fake_probs) > 1:
            verdict_title = "High Likelihood of AI Manipulation"
            human_summary = f"Detected synthetic anomalies in {fake_frames_count} of {len(fake_probs)} analyzed frames. Facial boundary seams and pore-smoothing patterns indicate generative AI or face replacement."
        else:
            verdict_title = "Likely AI-Generated / Manipulated"
            human_summary = "This portrait exhibits unnatural skin texture smoothing and digital generator frequency anomalies."
    else:
        if is_video_mode or len(fake_probs) > 1:
            verdict_title = "Looks Natural & Authentic"
            human_summary = f"All {len(fake_probs)} frames maintain realistic micro-textures, organic pore structure, and coherent lighting without face-swap seams."
        else:
            verdict_title = "Looks Natural & Authentic"
            human_summary = "Natural skin micro-texture, organic lighting transitions, and coherent boundaries were observed."

    return jsonify({
        "verdict": verdict,
        "verdict_title": verdict_title,
        "human_summary": human_summary,
        "is_fake": is_fake_verdict,
        "confidence": f"{confidence:.1f}%",
        "video_fake_prob": round(video_fake_prob * 100, 2),
        "video_real_prob": round(video_real_prob * 100, 2),
        "total_frames": len(fake_probs),
        "fake_frames": fake_frames_count,
        "real_frames": real_frames_count,
        "elapsed_time": f"{elapsed_time}s",
        "frames": frame_results,
        "mode": "video" if is_video_mode else ("single_frame" if len(fake_probs) == 1 else "multi_frames")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
