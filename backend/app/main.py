import base64
import os
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .gradcam import make_gradcam, overlay
from .model_loader import model_state
from .preprocess import preprocess, crop_resize


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dr_backend")

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_LONGEST_SIDE = 2500


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state.load()
    yield


app = FastAPI(title="Diabetic Retinopathy Detector", lifespan=lifespan)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
allow_wildcard = "*" in allowed_origins
if allow_wildcard:
    allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if allow_wildcard else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_state.model is not None}


def _decode_image(upload: UploadFile, contents: bytes):
    if upload.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted",
        )
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The image exceeds the 10 MB size limit",
        )
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Could not read this file as an image",
        )

    data = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read this file as an image",
        )

    height, width = image.shape[:2]
    if height < 64 or width < 64:
        raise HTTPException(
            status_code=400,
            detail="Image must be at least 64 pixels on each side",
        )

    longest_side = max(height, width)
    if longest_side > MAX_LONGEST_SIDE:
        scale = MAX_LONGEST_SIDE / float(longest_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
    return image


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    filename = file.filename or "<unnamed>"

    if model_state.model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable. Please try again later.",
        )

    contents = file.file.read(MAX_FILE_BYTES + 1)
    image = _decode_image(file, contents)
    input_tensor = preprocess(image, model_state.config["img_size"])

    with model_state.lock:
        try:
            probabilities = model_state.model.predict(
                input_tensor,
                verbose=0,
            )[0]
            heatmap = make_gradcam(model_state.model, input_tensor)
        except Exception:
            logger.exception("Model prediction failed")
            raise HTTPException(
                status_code=500,
                detail="The model could not process this image",
            )

    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    grade = int(np.argmax(probabilities))
    confidence = float(probabilities[grade])
    threshold = float(model_state.config["threshold"])
    classes = model_state.config["classes"]
    label = str(classes[grade])

    resized = crop_resize(image, model_state.config["img_size"])
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_overlay = overlay(resized_rgb, heatmap)
    rgb_overlay = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)
    success, png = cv2.imencode(".png", rgb_overlay)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode the Grad-CAM image",
        )
    gradcam_base64 = base64.b64encode(png.tobytes()).decode("ascii")

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request filename=%s size=%d grade=%d confidence=%.6f latency_ms=%.2f",
        filename,
        len(contents),
        grade,
        confidence,
        latency_ms,
    )

    return {
        "grade": grade,
        "label": label,
        "confidence": confidence,
        "probabilities": [float(value) for value in probabilities],
        "low_confidence": confidence < threshold,
        "threshold": threshold,
        "gradcam_base64": gradcam_base64,
    }
