"""Automated tests that need NO AI models. Run from the project root:  pytest -q
(Face similarity and the real deepfake model are tested manually; see README.)"""
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfake_detector import detect_deepfake
from face_matcher import cosine_similarity
from image_processor import ImageValidationError, detect_faces, face_status, load_image
from report_generator import generate_report
from risk_assessment import assess_risk


def make_image(fmt="PNG", size=(200, 200)):
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


# ---- image validation ----
def test_valid_png_and_jpeg():
    assert load_image(make_image("PNG"), "a.png").shape == (200, 200, 3)
    assert load_image(make_image("JPEG"), "a.jpg").ndim == 3


@pytest.mark.parametrize("name", ["a.gif", "a.txt", "a.pdf", "noext"])
def test_unsupported_extension(name):
    with pytest.raises(ImageValidationError):
        load_image(make_image(), name)


def test_empty_and_oversized():
    with pytest.raises(ImageValidationError):
        load_image(b"", "a.png")
    with pytest.raises(ImageValidationError):
        load_image(b"0" * (11 * 1024 * 1024), "a.png")


def test_corrupted_and_fake_image():
    with pytest.raises(ImageValidationError):
        load_image(b"this is not an image", "a.jpg")
    with pytest.raises(ImageValidationError):
        load_image(make_image("PNG")[:60], "a.png")          # truncated file
    with pytest.raises(ImageValidationError):
        load_image(make_image("GIF"), "a.png")               # GIF renamed to .png


def test_no_face_in_noise():
    faces = detect_faces(load_image(make_image(), "a.png"))
    assert faces == [] and face_status(faces) == "No face detected"


# ---- face similarity maths ----
def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)


# ---- deepfake model missing -> explicit unavailable, never a fake prediction ----
def test_missing_model_is_unavailable(monkeypatch, tmp_path):
    # Pretend the model folder does not exist, even if it is downloaded on this machine
    import deepfake_detector
    monkeypatch.setattr(deepfake_detector, "MODEL_DIR", str(tmp_path / "no_model_here"))
    monkeypatch.setitem(deepfake_detector._cache, "model", None)
    monkeypatch.setitem(deepfake_detector._cache, "processor", None)
    res = deepfake_detector.detect_deepfake(np.zeros((100, 100, 3), dtype=np.uint8))
    assert res["status"] == "unavailable" and "label" not in res


# ---- risk logic ----
FACE = lambda s: {"similarity": s, "faces_reference": 1, "faces_suspected": 1}
FAKE = lambda p: {"status": "ok", "fake_probability": p}


@pytest.mark.parametrize("sim,fake,expected", [
    (0.9, 0.95, "High Risk"), (0.9, 0.5, "Moderate Risk"), (0.9, 0.1, "Low Risk"),
    (0.1, 0.95, "Low Risk"), (0.55, 0.95, "Moderate Risk"), (0.55, 0.5, "Inconclusive"),
])
def test_risk_table(sim, fake, expected):
    assert assess_risk(FACE(sim), FAKE(fake))["category"] == expected


def test_risk_inconclusive_when_inputs_missing():
    assert assess_risk(None, FAKE(0.9))["category"] == "Inconclusive"
    assert assess_risk(FACE(0.9), {"status": "unavailable", "message": "x"})["category"] == "Inconclusive"


# ---- PDF report ----
def _results(with_model=True):
    face, fake = FACE(0.8) | {"model": "Facenet"}, FAKE(0.9) | {
        "label": "Deepfake", "confidence": 0.9, "model": "m", "input_used": "crop", "limitations": "lim"}
    if not with_model:
        fake = {"status": "unavailable", "message": "Deepfake detector unavailable"}
    img = np.zeros((120, 100, 3), dtype=np.uint8)
    return {"timestamp": "2026-01-01 10:00:00", "ref_status": "1 face detected",
            "sus_status": "1 face detected", "face": face, "face_error": None,
            "deepfake": fake, "risk": assess_risk(face, fake), "ref_img": img, "sus_img": img}


@pytest.mark.parametrize("with_model", [True, False])
@pytest.mark.parametrize("include", [True, False])
def test_report_is_valid_pdf(with_model, include):
    pdf = generate_report(_results(with_model), include_images=include)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


def test_report_without_images_is_smaller():
    r = _results()
    assert len(generate_report(r, False)) < len(generate_report(r, True))
