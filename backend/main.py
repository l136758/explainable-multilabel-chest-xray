from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import io
import time
import base64

from model_utils import predict, generate_gradcam_base64, LABELS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/")
def home():
    return {"status": "Backend is running"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: JPEG, PNG"
        )

    image_bytes = await file.read()

    try:
        image_check = Image.open(io.BytesIO(image_bytes))
        image_check.verify()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is corrupt or not a valid image"
        )

    start_time = time.time()
    results = predict(image)
    inference_time_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "model_version": "convnext_tiny_320_bce_v1",
        "inference_time_ms": inference_time_ms,
        "predictions": results
    }


@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...), pathology: str = Form(...)):
    if pathology not in LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pathology: {pathology}"
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is corrupt or not a valid image"
        )

    target_class = LABELS.index(pathology)

    try:
        heatmap_b64 = generate_gradcam_base64(image, target_class)
        heatmap_bytes = base64.b64decode(heatmap_b64)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not generate heatmap for this image"
        )

    return Response(content=heatmap_bytes, media_type="image/png")