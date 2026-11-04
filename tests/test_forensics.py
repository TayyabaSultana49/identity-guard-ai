"""Tests for forensics.py, guidance.py and the new report sections (no AI models needed)."""
import hashlib
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forensics import library_versions, read_metadata, run_ela, sha256_hex
from guidance import GUIDANCE
from image_processor import crop_largest_face
from report_generator import generate_report
from risk_assessment import assess_risk, manipulation_signal, similarity_signal


def jpeg_bytes(size=(160, 220), software=None, camera=None):
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    exif = Image.Exif()
    if software:
        exif[0x0131] = software
    if camera:
        exif[0x010F], exif[0x0110] = camera
    buf = io.BytesIO()
    kwargs = {"exif": exif} if (software or camera) else {}
    Image.fromarray(arr).save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def test_sha256_matches_hashlib():
    assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert sha256_hex(b"abc") != sha256_hex(b"abd")


def test_library_versions_has_python_and_packages():
    v = library_versions()
    assert "python" in v and "numpy" in v and "streamlit" in v


def test_ela_output_shape_and_stats():
    res = run_ela(jpeg_bytes())
    assert res["overlay"].shape == (160, 220, 3)
    assert 0 <= res["mean_error"] <= res["max_error"]
    assert res["source_format"] == "JPEG" and res["notes"]


def test_ela_png_gets_note():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((90, 90, 3), dtype=np.uint8)).save(buf, format="PNG")
    res = run_ela(buf.getvalue())
    assert any("PNG" in n for n in res["notes"])


def test_metadata_without_exif():
    meta = read_metadata(jpeg_bytes())
    assert meta["has_exif"] is False and meta["camera"] is None
    assert any("does not indicate manipulation" in n for n in meta["notes"])


def test_metadata_detects_editor_and_camera():
    meta = read_metadata(jpeg_bytes(software="Adobe Photoshop 25.0", camera=("Canon", "EOS 90D")))
    assert meta["has_exif"] and meta["camera"] == "Canon EOS 90D"
    assert any("editing or generation software" in n for n in meta["notes"])


def test_crop_largest_face():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_largest_face(img, [(50, 50, 40, 40), (10, 10, 20, 20)])
    assert crop.shape[0] >= 40 and crop.shape[1] >= 40
    assert crop_largest_face(img, []) is None


def test_signal_text():
    assert "weak" in manipulation_signal(0.1)
    assert "not conclusive" in manipulation_signal(0.5)
    assert "strong" in manipulation_signal(0.95)
    assert "do not look similar" in similarity_signal(0.1)


def test_guidance_content():
    assert GUIDANCE["steps"] and GUIDANCE["resources"]
    urls = [r["url"] for r in GUIDANCE["resources"] if r["url"]]
    assert all(u.startswith("https://") for u in urls)


def test_report_with_forensics_and_evidence():
    raw = jpeg_bytes()
    face = {"similarity": 0.8, "faces_reference": 1, "faces_suspected": 1, "model": "Facenet"}
    fake = {"status": "ok", "label": "Deepfake", "confidence": 0.9, "fake_probability": 0.9,
            "model": "m", "input_used": "crop", "limitations": "lim"}
    img = np.zeros((90, 80, 3), dtype=np.uint8)
    res = {"timestamp": "2026-01-01 10:00:00", "ref_status": "1 face detected",
           "sus_status": "1 face detected", "face": face, "face_error": None, "deepfake": fake,
           "risk": assess_risk(face, fake), "ref_img": img, "sus_img": img,
           "forensics": {"ela": run_ela(raw), "metadata": read_metadata(raw)},
           "evidence": {"sha256": sha256_hex(raw), "versions": library_versions()}}
    for include in (True, False):
        pdf = generate_report(res, include_images=include)
        assert pdf.startswith(b"%PDF") and len(pdf) > 3000
    assert len(generate_report(res, False)) < len(generate_report(res, True))
