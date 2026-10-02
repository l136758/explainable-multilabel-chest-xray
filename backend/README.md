# Team Core — Backend (ConvNeXt-Tiny Chest X-Ray API)

FastAPI backend serving the final ConvNeXt-Tiny 320×320 BCE model for multi-label chest X-ray classification, with Grad-CAM explainability.

## Requirements
- Python 3.x
- Virtual environment with dependencies installed (see `requirements.txt` if present, or install manually: `fastapi`, `uvicorn`, `torch`, `torchvision`, `pillow`, `numpy`, `matplotlib`, `python-multipart`)

## Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows

pip install -r ../requirements.txt
pip install fastapi uvicorn python-multipart   # not yet in requirements.txt — needed to run the API

uvicorn main:app --reload
```

> **Note:** `requirements.txt` currently only lists training dependencies (numpy, pandas, scikit-learn, Pillow, torch, torchvision, matplotlib). `fastapi`, `uvicorn`, and `python-multipart` are required to run the backend but are missing — install them manually until `requirements.txt` is updated.
Server runs at `http://127.0.0.1:8000`. On startup you should see:
```
✓ الموديل جاهز على: cpu
Application startup complete.
```

Model checkpoint and thresholds are expected at:
```
model/convnext_tiny_320_best.pth
model/convnext_tiny_320_bce_thresholds.json
```
`convnext_tiny_320_bce_thresholds.json` is committed to GitHub. **`convnext_tiny_320_best.pth` is NOT** (too large / excluded per team storage rules) — after cloning, copy it manually into `backend/model/` from the shared location (Kaggle/team storage) before starting the server.

## Endpoints

### `GET /`
Health check. Returns `{"status": "Backend is running"}`.

### `POST /predict`
Multipart form, field `file` (JPEG/PNG). Returns 14 pathology predictions with probability, threshold, and positive flag, plus `inference_time_ms` and `model_version`.

### `POST /gradcam`
Multipart form, fields `file` (image) and `pathology` (one of the 14 label names). Returns a PNG heatmap image (colored overlay) directly as the response body.

## Error handling
- Unsupported file type or corrupt image → `400` with a clear `detail` message.
- Grad-CAM generation failure → `500` with a clear `detail` message.

## Notes
- CORS is enabled (`allow_origins=["*"]`) for local frontend integration (e.g. `http://127.0.0.1:5500`).
- Label order is fixed and must match the training notebook — do not reorder `LABELS` in `model_utils.py`.
