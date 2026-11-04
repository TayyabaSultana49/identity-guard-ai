"""image_processor.py - validation, loading and face detection for uploads.

Uploads are handled in memory only; nothing is written to disk and filenames
are never logged."""
import io
import logging
import os

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("identity_guard")

# ---- Configuration -------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}    # real format, checked from file contents
MAX_FILE_SIZE_MB = 10
MAX_SIDE_PX = 1600                   # large images are shrunk for speed
Image.MAX_IMAGE_PIXELS = 50_000_000  # protects against decompression bombs


class ImageValidationError(Exception):
    """Raised with a user-friendly message when an upload is not usable."""


def load_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Validate an upload and return it as an RGB numpy array."""
    # 1. Extension check (filename is used ONLY for this, never logged)
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Unsupported file type. Please upload a JPG, JPEG or PNG.")

    # 2. Size check
    if len(file_bytes) == 0:
        raise ImageValidationError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ImageValidationError(f"File is larger than {MAX_FILE_SIZE_MB} MB.")

    # 3. Content check: is it really a JPEG/PNG image?
    try:
        with Image.open(io.BytesIO(file_bytes)) as probe:
            real_format = probe.format
            probe.verify()  # detects many kinds of corruption
        if real_format not in ALLOWED_FORMATS:
            raise ImageValidationError("File contents are not a valid JPG or PNG image.")

        # verify() invalidates the object, so open again for real use
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)   # fix phone-photo rotation
        img = img.convert("RGB")             # drop alpha channel etc.
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
        return np.array(img)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        logger.warning("Image rejected: unreadable or corrupted")
        raise ImageValidationError("The image is corrupted or could not be read.")


# ---- Face detection (OpenCV Haar cascade) ----------------------------------
_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError("Could not load OpenCV face detector.")
        _cascade = cascade
    return _cascade


def detect_faces(img_rgb: np.ndarray) -> list:
    """Return a list of (x, y, w, h) face boxes."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [tuple(int(v) for v in f) for f in faces]


def face_status(faces: list) -> str:
    """Plain-language face detection status for the UI and report."""
    if len(faces) == 0:
        return "No face detected"
    if len(faces) == 1:
        return "1 face detected"
    return f"{len(faces)} faces detected (largest face will be used)"


def draw_face_boxes(img_rgb: np.ndarray, faces: list) -> np.ndarray:
    """Return a copy of the image with cyan boxes drawn around faces."""
    out = img_rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 3)
    return out


def crop_largest_face(img_rgb: np.ndarray, faces: list, margin: float = 0.25):
    """Return the largest face (with a margin) as a new array, or None."""
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(margin * max(w, h))
    H, W = img_rgb.shape[:2]
    return img_rgb[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)].copy()
