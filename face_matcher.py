"""face_matcher.py - face embeddings (FaceNet via DeepFace) + cosine similarity."""
import logging

import cv2
import numpy as np

logger = logging.getLogger("identity_guard")

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"   # same family of detector as image_processor


class FaceMatchError(Exception):
    """Raised with a user-friendly message when face matching fails."""


def _largest_face_embedding(img_rgb: np.ndarray):
    """Return (embedding, number_of_faces_found) for the largest face."""
    try:
        from deepface import DeepFace   # imported here so the app can still start
    except Exception:                   # if DeepFace/TensorFlow is broken
        logger.error("DeepFace import failed")
        raise FaceMatchError("Face recognition library (DeepFace) could not be loaded.")

    # DeepFace expects numpy images in BGR order (OpenCV convention)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        results = DeepFace.represent(
            img_path=bgr,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError:
        raise FaceMatchError("No face could be detected in this image.")
    except Exception as exc:
        # Log only the error class, never image data or filenames
        logger.error("Embedding failed: %s", type(exc).__name__)
        raise FaceMatchError(
            "The face model failed to run. On first use it must download its "
            "weights, so check your internet connection."
        )

    # Choose the face with the largest bounding-box area
    best = max(results, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
    return np.array(best["embedding"], dtype=np.float64), len(results)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(theta) = (a . b) / (|a| * |b|). 1.0 = same direction, 0 = unrelated."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise FaceMatchError("Invalid embedding (zero vector).")
    return float(np.dot(a, b) / denom)


def compare_faces(reference_rgb: np.ndarray, suspected_rgb: np.ndarray) -> dict:
    """Compare two images and return the similarity result as a dictionary."""
    emb_ref, n_ref = _largest_face_embedding(reference_rgb)
    emb_sus, n_sus = _largest_face_embedding(suspected_rgb)
    sim = cosine_similarity(emb_ref, emb_sus)
    return {
        "similarity": round(sim, 4),
        "cosine_distance": round(1.0 - sim, 4),
        "faces_reference": n_ref,
        "faces_suspected": n_sus,
        "model": f"{MODEL_NAME} (via DeepFace)",
        "note": "Visual face similarity is not proof of identity.",
    }
