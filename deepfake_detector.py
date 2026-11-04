"""deepfake_detector.py - pretrained ViT deepfake image classifier.

Model: prithivMLmods/Deep-Fake-Detector-v2-Model (Apache-2.0, Hugging Face).
This is NOT a face-similarity check. It only scores signs of manipulation
learned from its own training data, and it can be wrong."""
import logging
import os

os.environ.setdefault("USE_TF", "0")   # stop transformers importing TensorFlow

import numpy as np
from PIL import Image

from image_processor import detect_faces

logger = logging.getLogger("identity_guard")

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "deepfake_vit")
FACE_MARGIN = 0.30   # extra context around the face crop (a design choice)

LIMITATIONS = (
    "This classifier was trained on a specific dataset and may not generalise "
    "to other manipulation methods. Low resolution or heavy compression can reduce "
    "reliability. A prediction is a statistical signal, not proof."
)

_cache = {"model": None, "processor": None}


def _has_weights() -> bool:
    """True if a weights file is present in the model folder."""
    if not os.path.isdir(MODEL_DIR):
        return False
    return any(f.endswith((".safetensors", ".bin")) for f in os.listdir(MODEL_DIR))


def download_model() -> str:
    """Download the model into models/deepfake_vit.

    First tries only the files needed to run (config + weights, a few hundred MB).
    If no weights arrive, falls back to downloading the whole repository."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR,
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
    if not (model_available() and _has_weights()):
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
    return MODEL_DIR


def model_available() -> bool:
    """True if the model files have been downloaded to models/deepfake_vit."""
    return os.path.exists(os.path.join(MODEL_DIR, "config.json"))


def _load_model():
    """Load once and reuse."""
    if _cache["model"] is not None:
        return
    if not model_available():
        raise FileNotFoundError("model files not found")
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        model.eval()
    except Exception as exc:
        logger.error("Deepfake model load failed: %s", type(exc).__name__)
        raise RuntimeError("model could not be loaded") from exc
    _cache["model"], _cache["processor"] = model, processor


def _crop_largest_face(img_rgb: np.ndarray):
    """Crop the largest detected face with a margin. Falls back to full image."""
    faces = detect_faces(img_rgb)
    if not faces:
        return img_rgb, "full image (no face found to crop)"
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(FACE_MARGIN * max(w, h))
    H, W = img_rgb.shape[:2]
    crop = img_rgb[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
    return crop, "largest face crop"


def detect_deepfake(img_rgb: np.ndarray) -> dict:
    """Return a dict with status 'ok', 'unavailable' or 'error'."""
    result = {"status": "ok", "model": MODEL_ID, "limitations": LIMITATIONS}

    # 1. Make sure the model exists and loads
    try:
        _load_model()
    except FileNotFoundError:
        result.update(status="unavailable", message=(
            "Deepfake detector unavailable: model files not found. "
            "Run  python download_model.py  once, then restart the app."))
        return result
    except RuntimeError:
        result.update(status="unavailable", message=(
            "Deepfake detector unavailable: the model failed to load "
            "(check the torch/transformers installation)."))
        return result

    # 2. Run the model
    try:
        import torch
        crop, input_used = _crop_largest_face(img_rgb)
        inputs = _cache["processor"](images=Image.fromarray(crop), return_tensors="pt")
        with torch.no_grad():
            logits = _cache["model"](**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].tolist()

        id2label = _cache["model"].config.id2label
        scores = {str(id2label[i]): round(p, 4) for i, p in enumerate(probs)}
        fake_idx = next((i for i, l in id2label.items() if "fake" in str(l).lower()), None)
        if fake_idx is None:
            raise ValueError("no 'fake' label in model config")

        best = int(np.argmax(probs))
        result.update(
            label=str(id2label[best]),
            confidence=round(probs[best], 4),
            fake_probability=round(probs[fake_idx], 4),
            scores=scores,
            input_used=input_used,
        )
        return result
    except Exception as exc:
        logger.error("Deepfake inference failed: %s", type(exc).__name__)
        result.update(status="error", message="The deepfake model failed while analysing the image.")
        return result
