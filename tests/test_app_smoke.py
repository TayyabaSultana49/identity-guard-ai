"""Smoke test: every page of the Streamlit app renders without exceptions."""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("page", ["Home", "Identity Analysis", "Results Dashboard", "Next Steps", "About the Project"])
def test_pages_render(page):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception


def test_results_page_with_data():
    import numpy as np
    from risk_assessment import assess_risk
    face = {"similarity": 0.82, "faces_reference": 1, "faces_suspected": 1, "model": "Facenet"}
    fake = {"status": "ok", "label": "Deepfake", "confidence": 0.9, "fake_probability": 0.9,
            "model": "m", "input_used": "crop", "limitations": "lim"}
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.session_state["page"] = "Results Dashboard"
    at.session_state["results"] = {
        "timestamp": "2026-01-01 10:00:00", "ref_status": "1 face detected",
        "sus_status": "1 face detected", "face": face, "face_error": None,
        "deepfake": fake, "risk": assess_risk(face, fake), "ref_img": img, "sus_img": img}
    at.run()
    assert not at.exception, at.exception


def test_results_page_with_forensics_and_evidence():
    import io
    import numpy as np
    from PIL import Image
    from forensics import library_versions, read_metadata, run_ela, sha256_hex
    from risk_assessment import assess_risk
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (120, 150, 3), dtype=np.uint8)).save(buf, format="JPEG")
    raw = buf.getvalue()
    face = {"similarity": 0.78, "faces_reference": 1, "faces_suspected": 1, "model": "Facenet"}
    fake = {"status": "ok", "label": "Deepfake", "confidence": 0.78, "fake_probability": 0.78,
            "model": "m", "input_used": "crop", "limitations": "lim"}
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.session_state["page"] = "Results Dashboard"
    at.session_state["results"] = {
        "timestamp": "2026-01-01 10:00:00", "ref_status": "1 face detected",
        "sus_status": "1 face detected", "face": face, "face_error": None,
        "deepfake": fake, "risk": assess_risk(face, fake), "ref_img": img, "sus_img": img,
        "ref_crop": img, "sus_crop": img,
        "forensics": {"ela": run_ela(raw), "metadata": read_metadata(raw)},
        "evidence": {"sha256": sha256_hex(raw), "versions": library_versions(),
                     "models": {"face": "Facenet", "deepfake": "m"}}}
    at.run()
    assert not at.exception, at.exception


def test_results_page_when_deepfake_unavailable():
    import numpy as np
    from risk_assessment import assess_risk
    face = {"similarity": 0.6, "faces_reference": 1, "faces_suspected": 1, "model": "Facenet"}
    fake = {"status": "unavailable", "message": "Deepfake detector unavailable"}
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.session_state["page"] = "Results Dashboard"
    at.session_state["results"] = {
        "timestamp": "t", "ref_status": "1 face detected", "sus_status": "1 face detected",
        "face": face, "face_error": None, "deepfake": fake, "risk": assess_risk(face, fake),
        "ref_img": img, "sus_img": img}
    at.run()
    assert not at.exception, at.exception
