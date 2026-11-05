from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from adapters.ocr_runtime_catalog import managed_model_requirements  # noqa: E402
from ui.common import model_download  # noqa: E402
from utils.model_manager import ModelError, ModelStatus  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(list())


def _wait(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _fake_spec(model_id):
    return type("Spec", (), {"display_name": model_id})()


class _FakeManager:
    def __init__(self, statuses=None, error=None):
        self.statuses = statuses or {}
        self.error = error
        self.ensured = list()

    def status(self, model_id, *, target_dir=None):
        return self.statuses.get(model_id, ModelStatus.READY)

    def ensure(self, model_id, *, target_dir=None, progress_callback=None, cancel_check=None):
        if self.error is not None:
            raise self.error
        if progress_callback is not None:
            progress_callback(512 * 1024, 1024 * 1024)
            progress_callback(1024 * 1024, 1024 * 1024)
        self.ensured.append(model_id)
        return target_dir


def test_managed_requirements_only_for_48px():
    assert not managed_model_requirements({"ndlocr_lite", "paddle"})
    ids = list(model_id for model_id, _target in managed_model_requirements({"manga_48px"}))
    assert len(ids) == 2
    assert all(model_id.startswith("manga_48px") for model_id in ids)


def test_pending_models_filters_ready(monkeypatch):
    fake = _FakeManager({"a": ModelStatus.MISSING, "b": ModelStatus.READY, "c": ModelStatus.CORRUPT})
    monkeypatch.setattr(model_download, "default_manager", lambda: fake)
    pending = model_download.pending_models((("a", None), ("b", None), ("c", None)))
    assert list(model_id for model_id, _target in pending) == list("ac")


def test_empty_requirements_resume_immediately(app):
    calls = list()
    model_download.ensure_models_in_background(QWidget(), (), on_ready=lambda: calls.append(True))
    assert calls.count(True) == 1


def test_download_runs_then_resumes(app, monkeypatch):
    fake = _FakeManager()
    monkeypatch.setattr(model_download, "default_manager", lambda: fake)
    monkeypatch.setattr(model_download, "get_spec", _fake_spec)
    ready, logs = list(), list()
    model_download.ensure_models_in_background(
        QWidget(),
        (("m1", None), ("m2", None)),
        on_ready=lambda: ready.append(True),
        log=logs.append,
    )
    assert _wait(app, lambda: bool(ready))
    assert "-".join(fake.ensured) == "m1-m2"
    assert logs
    assert not model_download._ACTIVE


def test_model_error_shown_not_raised(app, monkeypatch):
    fake = _FakeManager(error=ModelError("boom: place file manually"))
    monkeypatch.setattr(model_download, "default_manager", lambda: fake)
    monkeypatch.setattr(model_download, "get_spec", _fake_spec)
    shown, ready = list(), list()
    monkeypatch.setattr(
        model_download.LocalizedMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    model_download.ensure_models_in_background(
        QWidget(), (("m1", None),), on_ready=lambda: ready.append(True)
    )
    assert _wait(app, lambda: bool(shown))
    _parent, _title, message = shown.pop()
    assert "boom" in message
    assert not ready
    assert not model_download._ACTIVE
