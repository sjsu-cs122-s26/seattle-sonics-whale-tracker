from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL.Image import Image
from torchvision import models

from .preprocess import preprocess

_model: nn.Module | None = None
_class_names: list[str] = []
_device: torch.device = torch.device("cpu")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    return m


def load_model(model_path: Path, class_names_path: Path) -> None:
    global _model, _class_names, _device
    _class_names = json.loads(class_names_path.read_text())
    _device = _pick_device()
    _model = _build_model(len(_class_names))
    state = torch.load(model_path, map_location=_device, weights_only=True)
    _model.load_state_dict(state)
    _model.to(_device).eval()
    print(f"Model loaded: {len(_class_names)} classes on {_device}")


def predict(image: Image, top_k: int | None = None) -> dict:
    assert _model is not None, "Model not loaded — call load_model first"
    tensor = preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
    probs = torch.softmax(logits[0], dim=0)
    best_idx = int(probs.argmax().item())
    result: dict = {
        "label": _class_names[best_idx],
        "confidence": round(float(probs[best_idx].item()), 6),
    }
    if top_k is not None and top_k > 1:
        k = min(top_k, len(_class_names))
        topk_probs, topk_idxs = probs.topk(k)
        result["top_k"] = [
            {"label": _class_names[int(i)], "confidence": round(float(p), 6)}
            for p, i in zip(topk_probs, topk_idxs)
        ]
    return result
