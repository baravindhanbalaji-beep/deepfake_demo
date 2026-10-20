import os
import uuid
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from model import DeepfakeDetector

# Use absolute paths so Flask locates templates and static files reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config["SECRET_KEY"] = "deepfake-swin-v2-demo-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload limit

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# -------------------------------------------------------------
# Model Initialization
# -------------------------------------------------------------
DEFAULT_CHECKPOINTS = [
    os.path.join(BASE_DIR, "best_celebdf_to_ff.pth"),
    os.path.join(BASE_DIR, "best_swinv2_lbp_dct.pth"),
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


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", checkpoint_loaded=(detector.model is not None))


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video selected."}), 400

    if not allowed_video(file.filename):
        return jsonify({
            "error": f"Invalid format. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        }), 400

    if detector.model is None:
        return jsonify({
            "error": "Model weights are not loaded. Place your .pth checkpoint in the directory or set MODEL_CHECKPOINT."
        }), 503

    start_time = time.time()

    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_filename = secure_filename(file.filename)
    video_path = os.path.join(session_dir, video_filename)
    file.save(video_path)

    # Frame Extraction (approx 1 FPS, max 30 frames for responsive demo)
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
    frame_results = []
    fake_probs = []

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
                "image_url": url_for("static", filename=f"uploads/{session_id}/{thumb_filename}"),
                "fake_prob": round(fake_p * 100, 2),
                "real_prob": round(real_p * 100, 2),
                "is_fake": fake_p >= 0.5
            })

            saved_count += 1

        frame_idx += 1

    cap.release()

    if not fake_probs:
        return jsonify({"error": "No frames could be extracted from this video."}), 400

    # Robust Temporal Aggregation:
    # A video is manipulated if a significant cluster of frames shows strong forgery artifacts.
    # We use Top-K Suspicious Frame Pooling (top 25% highest fake probabilities) combined
    # with threshold verification, preventing diluted arithmetic averages on transient deepfakes.
    sorted_fake_probs = sorted(fake_probs, reverse=True)
    top_k_count = max(1, int(np.ceil(len(fake_probs) * 0.25)))
    top_k_fake_prob = float(np.mean(sorted_fake_probs[:top_k_count]))
    overall_mean_fake = float(np.mean(fake_probs))

    fake_frames_count = sum(1 for p in fake_probs if p >= 0.5)
    real_frames_count = len(fake_probs) - fake_frames_count

    is_fake_verdict = (top_k_fake_prob >= 0.50 and fake_frames_count >= 2) or (overall_mean_fake >= 0.50)

    if is_fake_verdict:
        final_fake_prob = top_k_fake_prob
        confidence = top_k_fake_prob * 100
    else:
        final_fake_prob = overall_mean_fake
        confidence = (1.0 - overall_mean_fake) * 100

    final_real_prob = 1.0 - final_fake_prob
    verdict = "DEEPFAKE / MANIPULATED" if is_fake_verdict else "AUTHENTIC / REAL"
    elapsed_time = round(time.time() - start_time, 2)

    return jsonify({
        "verdict": verdict,
        "is_fake": is_fake_verdict,
        "confidence": round(confidence, 2),
        "video_fake_prob": round(final_fake_prob * 100, 2),
        "video_real_prob": round(final_real_prob * 100, 2),
        "total_frames": len(fake_probs),
        "fake_frames": fake_frames_count,
        "real_frames": real_frames_count,
        "elapsed_time": f"{elapsed_time}s",
        "frames": frame_results
    })


@app.route("/predict-folder", methods=["POST"])
def predict_folder():
    start_time = time.time()

    if detector.model is None:
        return jsonify({
            "error": "The model checkpoint is not loaded. Ensure the .pth file is available."
        }), 503

    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        return jsonify({
            "error": "Please select a folder containing images."
        }), 400

    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)

    frame_results = []
    fake_probs = []
    MAX_FOLDER_FRAMES = 60  # Cap to prevent timeouts or memory exhaustion

    for uploaded_file in uploaded_files:
        if len(frame_results) >= MAX_FOLDER_FRAMES:
            break

        original_name = (
            uploaded_file.filename or ""
        ).replace("\\", "/").split("/")[-1]

        if not original_name or not allowed_image(original_name):
            continue

        image_bytes = uploaded_file.read()
        if not image_bytes:
            continue

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            continue

        # Resize directly to 256x256 (preprocessed face crops)
        face_crop = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

        try:
            real_probability, fake_probability = detector.predict_frame(face_crop)
        except Exception:
            continue

        real_p = float(real_probability)
        fake_p = float(fake_probability)
        fake_probs.append(fake_p)

        frame_number = len(frame_results) + 1
        preview_name = f"frame_{frame_number:04d}.jpg"
        preview_path = os.path.join(session_dir, preview_name)
        cv2.imwrite(preview_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        frame_results.append({
            "frame_num": frame_number,
            "timestamp": secure_filename(original_name) or f"img_{frame_number}",
            "image_url": url_for("static", filename=f"uploads/{session_id}/{preview_name}"),
            "is_fake": fake_p >= 0.5,
            "fake_prob": round(fake_p * 100, 2),
            "real_prob": round(real_p * 100, 2)
        })

    if not frame_results:
        return jsonify({
            "error": "No valid images could be processed. Check that the folder contains JPG, PNG, WEBP, or BMP images."
        }), 400

    # Aggregate frame predictions into folder-level prediction
    sorted_fake_probs = sorted(fake_probs, reverse=True)
    top_k_count = max(1, int(np.ceil(len(fake_probs) * 0.25)))
    top_k_fake_prob = float(np.mean(sorted_fake_probs[:top_k_count]))
    overall_mean_fake = float(np.mean(fake_probs))

    fake_frames_count = sum(1 for p in fake_probs if p >= 0.5)
    real_frames_count = len(fake_probs) - fake_frames_count

    folder_is_fake = (top_k_fake_prob >= 0.50 and fake_frames_count >= 2) or (overall_mean_fake >= 0.50)

    if folder_is_fake:
        final_fake_prob = top_k_fake_prob
        confidence = top_k_fake_prob * 100
    else:
        final_fake_prob = overall_mean_fake
        confidence = (1.0 - overall_mean_fake) * 100

    final_real_prob = 1.0 - final_fake_prob
    elapsed_seconds = round(time.time() - start_time, 2)
    verdict = "DEEPFAKE / MANIPULATED" if folder_is_fake else "AUTHENTIC / REAL"

    return jsonify({
        "verdict": verdict,
        "is_fake": folder_is_fake,
        "confidence": round(confidence, 2),
        "video_fake_prob": round(final_fake_prob * 100, 2),
        "video_real_prob": round(final_real_prob * 100, 2),
        "total_frames": len(frame_results),
        "fake_frames": fake_frames_count,
        "real_frames": real_frames_count,
        "elapsed_time": f"{elapsed_seconds}s",
        "frames": frame_results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
