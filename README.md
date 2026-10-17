# Deepfake Video Detection Web Demo (Flask)

Live demonstration system for **Swin V2 + LBP + DCT Triple-Fusion** Deepfake Detection with **Temporal Mean-Pooling Video-Level Prediction**.

---

## 🚀 Quickstart Guide

### 1. Set up your Virtual Environment

```bash
# Navigate to the demo directory
cd deepfake_demo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(If you have a GPU with CUDA, install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` or `cu121`)*.

---

### 3. Add Your Trained Checkpoint

Place your best trained checkpoint inside the `deepfake_demo/` folder and name it either:
* `best_celebdf_to_ff.pth` (The model that achieved **99.2% Video ROC-AUC**), or
* `best_swinv2_lbp_dct.pth`

*(Alternatively, you can set an environment variable: `set MODEL_CHECKPOINT="C:\path\to\checkpoint.pth"`)*.

---

### 4. Run the Flask Server

```bash
python app.py
```

Open your browser at:
```
http://127.0.0.1:5000
```

---

## 🌟 Demo Features for Review / Viva

1. **Video Upload & De-framing at 1 FPS:** Automatically opens the video using OpenCV and samples frames at 1 frame-per-second, replicating the project's exact preprocessing pipeline.
2. **Face Extraction with Margin Expansion:** Automatically locates the face in each frame and expands the bounding box by 20% to capture jawline and blending seams.
3. **Triple-Fusion Inference:** Evaluates each facial crop through Swin V2 Tiny (768-d), LBP CNN (128-d), and DCT CNN (128-d).
4. **Temporal Mean-Pooling:** Combines all frame probabilities into a single, highly robust video-level score ($P_{\text{video}} = \frac{1}{N} \sum p_i$).
5. **Interactive Gallery:** Displays a visual timeline of all analyzed frames with their individual manipulation probabilities, allowing evaluators to see exactly which frames triggered manipulation alarms.
