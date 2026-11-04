"""Tests for navigation buttons, the pipeline, batch/protect pages and protect.py."""
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline
from forensics import read_metadata
from protect import add_watermark, clean_copy, protect_photo


def app(page, **state):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def results_state():
    from risk_assessment import assess_risk
    face = {"similarity": 0.6, "faces_reference": 1, "faces_suspected": 1, "model": "Facenet"}
    fake = {"status": "ok", "label": "Deepfake", "confidence": 0.7, "fake_probability": 0.7,
            "model": "m", "input_used": "crop", "limitations": "lim"}
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    return {"timestamp": "t", "ref_status": "1 face detected", "sus_status": "1 face detected",
            "face": face, "face_error": None, "deepfake": fake, "risk": assess_risk(face, fake),
            "ref_img": img, "sus_img": img}


# ---- every page renders ----
@pytest.mark.parametrize("page", ["Batch Scan", "Protect My Photo"])
def test_new_pages_render(page):
    assert not app(page).exception


# ---- in-page buttons behave like the sidebar ----
@pytest.mark.parametrize("page,key,target", [
    ("Home", "home_start", "Identity Analysis"),
    ("Home", "home_batch", "Batch Scan"),
    ("Home", "home_protect", "Protect My Photo"),
    ("Results Dashboard", "res_go_analysis", "Identity Analysis"),
    ("Next Steps", "ns_new", "Identity Analysis"),
    ("Next Steps", "ns_results", "Results Dashboard"),
])
def test_buttons_navigate(page, key, target):
    at = app(page)
    at.button(key=key).click().run()
    assert not at.exception
    assert at.session_state["page"] == target


def test_results_buttons_navigate():
    at = app("Results Dashboard", results=results_state())
    at.button(key="res_next").click().run()
    assert at.session_state["page"] == "Next Steps"


def test_analysis_page_offers_latest_results():
    at = app("Identity Analysis", results=results_state())
    at.button(key="view_latest").click().run()
    assert at.session_state["page"] == "Results Dashboard"


def test_batch_results_table_renders():
    small = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
    batch = {"timestamp": "t", "thumbs": [("Image 1", "High Risk", small)],
             "rows": [{"Image": "Image 1", "Faces": "1 face detected", "Similarity": "0.90",
                       "Manipulation score": "0.95", "Risk": "High Risk", "_rank": 0},
                      {"Image": "Image 2", "Faces": "No face detected", "Similarity": "n/a",
                       "Manipulation score": "n/a", "Risk": "Skipped: no face detected", "_rank": 4}]}
    assert not app("Batch Scan", batch=batch).exception


# ---- pipeline (models replaced by stand-ins) ----
def test_pipeline_combines_results(monkeypatch):
    monkeypatch.setattr(pipeline, "compare_faces", lambda a, b: {"similarity": 0.9, "model": "x",
                        "faces_reference": 1, "faces_suspected": 1})
    monkeypatch.setattr(pipeline, "detect_deepfake", lambda img: {"status": "ok", "fake_probability": 0.95})
    out = pipeline.analyse_pair(None, None)
    assert out["risk"]["category"] == "High Risk" and out["face_error"] is None


def test_pipeline_face_failure_is_inconclusive(monkeypatch):
    def boom(a, b):
        raise pipeline.FaceMatchError("No face could be detected in this image.")
    monkeypatch.setattr(pipeline, "compare_faces", boom)
    monkeypatch.setattr(pipeline, "detect_deepfake", lambda img: {"status": "ok", "fake_probability": 0.95})
    out = pipeline.analyse_pair(None, None)
    assert out["face"] is None and out["risk"]["category"] == "Inconclusive"


# ---- protect.py ----
def jpeg_with_metadata():
    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0131] = "Canon", "EOS", "Photoshop"
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (17.0, 23.0, 0.0), "E", (78.0, 29.0, 0.0)
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)).save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_protect_removes_metadata_and_gps():
    raw = jpeg_with_metadata()
    before = read_metadata(raw)
    assert before["has_exif"] and before["has_gps"]
    _, out, fmt = protect_photo(raw, "Hello")
    after = read_metadata(out)
    assert fmt == "JPEG" and not after["has_exif"] and not after["has_gps"] and after["camera"] is None


def test_watermark_changes_pixels_only_when_text_given():
    img, _ = clean_copy(jpeg_with_metadata())
    assert np.array_equal(np.asarray(add_watermark(img, "   ")), np.asarray(img))
    assert not np.array_equal(np.asarray(add_watermark(img, "Private", 0.5)), np.asarray(img))
    assert add_watermark(img, "Private").size == img.size


def test_protect_png_stays_png():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(buf, "PNG")
    _, out, fmt = protect_photo(buf.getvalue(), "")
    assert fmt == "PNG" and out.startswith(b"\x89PNG")
