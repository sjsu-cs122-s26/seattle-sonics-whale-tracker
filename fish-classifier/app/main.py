from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import load_model, predict as run_predict

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fish_model.pt"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", "models/class_names.json"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH.resolve()}")
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(f"class_names.json not found: {CLASS_NAMES_PATH.resolve()}")
    load_model(MODEL_PATH, CLASS_NAMES_PATH)
    yield


app = FastAPI(title="Fish Classifier API", version="1.0.0", lifespan=lifespan)

_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int | None = Query(default=None, ge=2, le=20, description="Return top-k predictions"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return run_predict(image, top_k=top_k)
