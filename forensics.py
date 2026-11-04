"""forensics.py - supporting forensic signals and the evidence record.

IMPORTANT: Error Level Analysis (ELA) and metadata checks are WEAK, supporting
hints. They do not prove an image was edited or generated, and they are NOT used
to change the risk category. Everything runs locally on in-memory bytes."""
import hashlib
import io
import platform
from importlib import metadata as importlib_metadata

import cv2
import numpy as np
from PIL import Image, ImageOps

ELA_QUALITY = 90          # JPEG quality used for the re-save comparison
ELA_MAX_SIDE = 2400       # larger images are shrunk (this can blur ELA signals)

# Words in the EXIF "Software" tag that suggest editing/generation software
EDITOR_KEYWORDS = ("photoshop", "gimp", "lightroom", "snapseed", "canva", "facetune",
                   "picsart", "stable diffusion", "midjourney", "dall", "firefly",
                   "comfyui", "automatic1111")

PACKAGES = ["streamlit", "deepface", "torch", "transformers",
            "opencv-python", "numpy", "pillow", "reportlab"]

ELA_CAVEAT = ("ELA highlights regions whose JPEG compression differs from the rest of the "
              "image. Edits can cause this, but so can sharp edges, text, resizing and "
              "re-uploads. It is a weak hint only and never proves manipulation.")


def sha256_hex(data: bytes) -> str:
    """SHA-256 fingerprint of the exact file bytes (identifies this exact file)."""
    return hashlib.sha256(data).hexdigest()


def library_versions() -> dict:
    """Versions of the tools used, for the evidence record."""
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run_ela(file_bytes: bytes) -> dict:
    """Error Level Analysis. Returns an overlay image plus simple statistics."""
    with Image.open(io.BytesIO(file_bytes)) as raw:
        source_format = raw.format
        img = ImageOps.exif_transpose(raw).convert("RGB")
    downscaled = max(img.size) > ELA_MAX_SIDE
    img.thumbnail((ELA_MAX_SIDE, ELA_MAX_SIDE))

    # Re-save as JPEG in memory and measure how much each pixel changes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    a = np.asarray(img, dtype=np.int16)
    b = np.asarray(resaved, dtype=np.int16)
    diff = np.abs(a - b).max(axis=2).astype(np.float32)

    max_err = float(diff.max())
    scale = 255.0 / max_err if max_err > 0 else 1.0
    norm = np.clip(diff * scale, 0, 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.asarray(img), 0.45, heat, 0.55, 0)

    notes = [ELA_CAVEAT, "The colours are relative to this image only (red = largest "
                         "difference), so a heatmap is always produced."]
    if source_format == "PNG":
        notes.append("This is a PNG (lossless). ELA is mainly meaningful for JPEG photos.")
    if downscaled:
        notes.append("The image was shrunk before analysis, which can hide compression traces.")
    return {
        "overlay": overlay,
        "mean_error": round(float(diff.mean()), 3),
        "p99_error": round(float(np.percentile(diff, 99)), 3),
        "max_error": round(max_err, 3),
        "source_format": source_format,
        "notes": notes,
    }


def _clean(value, limit=80):
    """Turn an EXIF value into short, safe text."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().replace("\x00", "")[:limit]


def read_metadata(file_bytes: bytes) -> dict:
    """Read basic EXIF info. GPS coordinates are NEVER shown, only whether they exist."""
    with Image.open(io.BytesIO(file_bytes)) as img:
        fmt = img.format
        exif = img.getexif()
        make, model = exif.get(0x010F), exif.get(0x0110)
        software, taken = exif.get(0x0131), exif.get(0x0132)
        try:
            taken = exif.get_ifd(0x8769).get(0x9003) or taken
            has_gps = bool(exif.get_ifd(0x8825))
        except Exception:
            has_gps = False

    camera = " ".join(_clean(x) for x in (make, model) if x) or None
    has_exif = bool(len(exif))
    notes = []
    if not has_exif:
        notes.append("No EXIF metadata found. This is common for screenshots and images "
                     "re-uploaded to social media, and does not indicate manipulation.")
    if software:
        low = _clean(software).lower()
        hits = [k for k in EDITOR_KEYWORDS if k in low]
        if hits:
            notes.append(f"The Software tag mentions editing or generation software ({_clean(software)}).")
        else:
            notes.append(f"Software tag: {_clean(software)}.")
    if has_exif and not camera:
        notes.append("No camera make/model recorded.")
    if has_gps:
        notes.append("The file contains location (GPS) data. Remove it before sharing the photo.")
    return {
        "format": fmt,
        "has_exif": has_exif,
        "camera": camera,
        "software": _clean(software) if software else None,
        "date_time": _clean(taken) if taken else None,
        "has_gps": has_gps,
        "notes": notes,
    }
