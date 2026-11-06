from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget  # noqa: E402

from adapters.ocr_runtime_catalog import managed_model_requirements  # noqa: E402
from ui.common import model_download  # noqa: E402
from utils.model_manager import ModelCancelledError, ModelError, ModelStatus  # noqa: E402


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


@pytest.mark.parametrize("outcome", ("ready", "error", "cancel"))
def test_background_progress_and_completion_keep_ui_responsive(app, monkeypatch, outcome):
    main_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    worker_threads, callback_threads = [], []
    progress, ready, shown, logs, ticks = [], [], [], [], []
    detail = "Model m1 request timed out (timeout 0.25 seconds); install manually."

    class ControlledManager:
        def ensure(self, model_id, *, target_dir=None, progress_callback=None, cancel_check=None):
            worker_threads.append(threading.get_ident())
            try:
                progress_callback(512 * 1024, 1024 * 1024)
                entered.set()
                if not release.wait(5):
                    raise AssertionError("UI did not release the background worker")
                if outcome == "cancel":
                    assert cancel_check()
                    raise ModelCancelledError("cancelled")
                assert not cancel_check()
                if outcome == "error":
                    raise ModelError(detail)
                progress_callback(1024 * 1024, 1024 * 1024)
                return target_dir
            finally:
                exited.set()

    original_format = model_download._format_progress

    def record_progress(label, done, total):
        callback_threads.append(threading.get_ident())
        progress.append((done, total))
        return original_format(label, done, total)

    def on_ready():
        callback_threads.append(threading.get_ident())
        ready.append(True)

    def on_log(message):
        callback_threads.append(threading.get_ident())
        logs.append(message)

    def on_error(*args, **kwargs):
        callback_threads.append(threading.get_ident())
        shown.append(args)

    monkeypatch.setattr(model_download, "default_manager", ControlledManager)
    monkeypatch.setattr(model_download, "get_spec", _fake_spec)
    monkeypatch.setattr(model_download, "_format_progress", record_progress)
    monkeypatch.setattr(model_download.LocalizedMessageBox, "critical", staticmethod(on_error))
    parent = QWidget()
    try:
        model_download.ensure_models_in_background(
            parent, (("m1", None),), on_ready=on_ready, log=on_log,
        )
        assert _wait(app, lambda: entered.is_set() and (512 * 1024, 1024 * 1024) in progress)
        assert not exited.is_set()
        QTimer.singleShot(0, lambda: ticks.append(threading.get_ident()))
        assert _wait(app, lambda: bool(ticks))
        assert ticks == [main_thread]
        assert not ready and not shown
        dialog = parent.findChild(QProgressDialog)
        assert dialog is not None
        assert dialog.value() == 500
        if outcome == "cancel":
            dialog.canceled.emit()
        release.set()
        assert _wait(app, lambda: exited.is_set() and not model_download._ACTIVE)
        assert worker_threads and all(ident != main_thread for ident in worker_threads)
        assert callback_threads and set(callback_threads) == {main_thread}
        assert logs
        if outcome == "ready":
            assert ready == [True]
            assert not shown
            assert (1024 * 1024, 1024 * 1024) in progress
        elif outcome == "error":
            assert not ready
            assert len(shown) == 1
            ((_parent, _title, message),) = shown
            assert detail in message
            assert any(detail in message for message in logs)
        else:
            assert not ready and not shown
    finally:
        release.set()
        _wait(app, lambda: exited.is_set() and not model_download._ACTIVE)
        parent.close()
        parent.deleteLater()


@pytest.mark.parametrize("language", ("zh", "en", "ja"))
def test_error_dialog_localizes_summary_and_preserves_details(app, monkeypatch, language):
    from ui.localization import translate_text
    from ui.message_catalog import format_message

    detail = (
        "Cannot download model 模型/完成-v2: request timed out (timeout 0.25 seconds). "
        "Download it manually from https://models.invalid/weights?name=完成 "
        "and place it in /tmp/模型/{weights}."
    )
    fake = _FakeManager(error=ModelError(detail))
    monkeypatch.setattr(model_download, "default_manager", lambda: fake)
    monkeypatch.setattr(model_download, "get_spec", _fake_spec)
    shown, ready, logs = [], [], []
    monkeypatch.setattr(
        model_download.LocalizedMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    previous_language = app.property("nfLanguage")
    parent = QWidget()
    app.setProperty("nfLanguage", language)
    try:
        model_download.ensure_models_in_background(
            parent, (("m1", None),), on_ready=lambda: ready.append(True), log=logs.append,
        )
        assert _wait(app, lambda: bool(shown))
        ((_parent, title, message),) = shown
        expected = format_message("model_download.failed.body", language, details=detail)
        assert title == translate_text("模型下载失败", language)
        assert message == expected
        assert message.endswith("\n\n" + detail)
        assert expected != detail
        assert any(item.endswith(expected) for item in logs)
        assert not ready and not model_download._ACTIVE
    finally:
        app.setProperty("nfLanguage", previous_language)
        parent.close()
        parent.deleteLater()


@pytest.mark.parametrize("outcome", ("ready", "cancel", "policy", "hash", "limit", "curl_hash"))
def test_actual_gui_preflight_uses_protected_fallback(app, tmp_path, monkeypatch, outcome):
    import dataclasses
    import hashlib
    import io
    import sys
    import types
    import urllib.error

    import gui_pyside6 as gui
    from adapters import manga_48px_runtime as runtime
    from adapters import ocr_runtime_catalog as catalog
    from utils import model_manager as mm

    data = b"tiny verified weights"
    ids = (runtime.MODEL_ID_CHECKPOINT, runtime.MODEL_ID_DICTIONARY)
    registry = {
        model_id: dataclasses.replace(
            mm.get_spec(model_id),
            urls=("https://models.invalid/" + mm.get_spec(model_id).filename,),
            size=len(data), min_size=1, sha256=hashlib.sha256(data).hexdigest(),
            legacy_dirs=(),
        )
        for model_id in ids
    }
    directory = tmp_path / "models"
    entered, release = threading.Event(), threading.Event()
    requests, commands, processes = [], [], []
    ready, shown, logs = [], [], []

    class Response(io.BytesIO):
        status = 200
        headers = {}

    def primary(request, timeout):
        requests.append(request.full_url)
        entered.set()
        if outcome == "cancel":
            assert release.wait(5), "test did not release primary transport"
        if outcome == "policy":
            raise mm._UnsafeDownloadURL("injected unsafe redirect")
        if outcome == "hash":
            return Response(b"x" * len(data))
        if outcome == "limit":
            return Response(data + b"oversized")
        raise urllib.error.URLError("injected Python transport failure")

    manager = mm.ModelManager(
        tmp_path / "cache", registry=registry, opener=primary,
        retries=0, backoff=0, auto_download=True, timeout=2,
        project_root=tmp_path / "project", chunk_size=4,
    )
    real_popen = mm.subprocess.Popen

    def launch(command, **kwargs):
        commands.append(command)
        payload = b"x" * len(data) if outcome == "curl_hash" else data
        process = real_popen(
            [sys.executable, "-u", "-c", f"import os; os.write(1, {payload!r})"],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(mm, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(mm, "default_manager", lambda: manager)
    monkeypatch.setattr(model_download, "default_manager", lambda: manager)
    monkeypatch.setattr(runtime, "default_cache_dir", lambda: directory)
    monkeypatch.setattr(catalog, "missing_components", lambda ids: [])
    monkeypatch.setattr(mm.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(mm.subprocess, "Popen", launch)
    monkeypatch.setattr(
        model_download.LocalizedMessageBox, "critical",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )

    class PreflightWidget(QWidget):
        _begin_runtime_installation_preflight = gui.OCRTab._begin_runtime_installation_preflight

        def _runtime_installation_component_ids(self, engine_ids, *, include_ruby):
            return ["manga_48px"]

        def _flush_ocr_log_buffer(self, **kwargs):
            pass

    parent = PreflightWidget()
    parent._ocr_preflight_generation = gui.GenerationGuard()
    parent._ocr_preflight_signal_refs = {}
    parent._ocr_preflight_active = False
    parent._ocr_run_active = False
    parent._run_btn = QWidget(parent)
    parent._ocr_log_buffer = types.SimpleNamespace(push=logs.append)
    try:
        parent._begin_runtime_installation_preflight(
            ["manga_48px"], include_ruby=False,
            resume_callback=lambda: ready.append(True),
        )
        assert _wait(app, entered.is_set)
        if outcome == "cancel":
            dialog = parent.findChild(QProgressDialog)
            assert dialog is not None
            dialog.canceled.emit()
            release.set()
        assert _wait(app, lambda: bool(ready or shown or logs) and not model_download._ACTIVE)
        app.processEvents()
        assert not parent._ocr_preflight_active
        assert not parent._ocr_preflight_signal_refs
        if outcome == "ready":
            assert ready == [True]
            assert not shown
            assert len(requests) == len(ids)
            assert len(commands) == len(ids)
            for model_id in ids:
                installed = directory / registry[model_id].filename
                assert installed.read_bytes() == data
                assert manager.verify(
                    model_id, force=True, target_dir=directory,
                ) is mm.ModelStatus.READY
        else:
            assert not ready
            assert len(requests) == 1
            assert len(commands) == (1 if outcome == "curl_hash" else 0)
            assert bool(shown) is (outcome != "cancel")
            for model_id in ids:
                installed = directory / registry[model_id].filename
                assert not installed.exists()
                assert not manager._marker_path(installed).exists()
        for command in commands:
            assert "-q" in command and "--proto" in command
            assert "--output" in command and "-" in command
            assert all(not str(item).endswith(".part") for item in command)
        assert all(process.poll() is not None for process in processes)
        assert all(process.stdout.closed and process.stderr.closed for process in processes)
    finally:
        release.set()
        _wait(app, lambda: not model_download._ACTIVE and not parent._ocr_preflight_active)
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        parent.close()
        parent.deleteLater()
