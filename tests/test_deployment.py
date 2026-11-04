"""Tests for hosted-demo mode and the model download helper (no network needed)."""
import os
import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deepfake_detector


def fake_hub(calls, write_weights_on_call):
    """A stand-in for huggingface_hub that records calls and writes fake files."""
    def snapshot_download(repo_id, local_dir, allow_patterns=None):
        calls.append(allow_patterns)
        os.makedirs(local_dir, exist_ok=True)
        open(os.path.join(local_dir, "config.json"), "w").write("{}")
        if write_weights_on_call(len(calls)):
            open(os.path.join(local_dir, "model.safetensors"), "wb").write(b"x")
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = snapshot_download
    return mod


def test_download_uses_small_patterns_first(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls, lambda n: True))
    monkeypatch.setattr(deepfake_detector, "MODEL_DIR", str(tmp_path / "m"))
    deepfake_detector.download_model()
    assert len(calls) == 1 and calls[0] is not None
    assert deepfake_detector.model_available() and deepfake_detector._has_weights()


def test_download_falls_back_to_full_repo(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls, lambda n: n == 2))
    monkeypatch.setattr(deepfake_detector, "MODEL_DIR", str(tmp_path / "m"))
    deepfake_detector.download_model()
    assert len(calls) == 2 and calls[1] is None
    assert deepfake_detector._has_weights()


def run_app(page, monkeypatch, hosted):
    import streamlit as st
    st.cache_resource.clear()          # each test starts with a fresh server-side cache
    if hosted:
        monkeypatch.setenv("IDENTITY_GUARD_HOSTED", "1")
    else:
        monkeypatch.delenv("IDENTITY_GUARD_HOSTED", raising=False)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.session_state["page"] = page
    return at.run()


@pytest.mark.parametrize("page", ["Identity Analysis", "Batch Scan", "Protect My Photo"])
def test_hosted_notice_shown_on_upload_pages(monkeypatch, page):
    monkeypatch.setattr(deepfake_detector, "download_model", lambda: None)
    at = run_app(page, monkeypatch, hosted=True)
    assert not at.exception
    assert any("hosted demo" in w.value.lower() for w in at.warning)


def test_no_hosted_notice_when_running_locally(monkeypatch):
    at = run_app("Identity Analysis", monkeypatch, hosted=False)
    assert not at.exception
    assert not any("hosted demo" in w.value.lower() for w in at.warning)


def test_hosted_download_failure_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(deepfake_detector, "MODEL_DIR", str(tmp_path / "none"))

    def boom():
        raise OSError("offline")
    monkeypatch.setattr(deepfake_detector, "download_model", boom)
    at = run_app("Identity Analysis", monkeypatch, hosted=True)
    assert not at.exception
    assert any("could not be downloaded" in e.value for e in at.error)
