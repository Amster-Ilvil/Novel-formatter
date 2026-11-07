"""Regression coverage for transactional local model imports."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from utils import model_manager as mm


def make_case(tmp_path):
    data = b"verified model bytes"
    spec = mm.ModelSpec(
        model_id="test.import", engine="test",
        urls=("https://models.invalid/model.bin",),
        filename="model.bin", sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )
    manager = mm.ModelManager(
        tmp_path / "cache", registry={spec.model_id: spec},
        auto_download=False, project_root=tmp_path / "project",
    )
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    final = directory / spec.filename
    final.write_bytes(data)
    assert manager.verify(spec.model_id) is mm.ModelStatus.READY
    marker = manager._marker_path(final)
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    return manager, spec, source, final, marker, data


def test_import_rejects_source_changed_before_copy(tmp_path, monkeypatch):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    original_marker = marker.read_bytes()
    foreign = final.parent / ".foreign.importing"
    foreign.write_bytes(b"another operation")
    copy = mm.shutil.copy2
    copied = []

    def replace_source_then_copy(src, dst, *args, **kwargs):
        assert Path(src) == source
        source.write_bytes(b"x" * len(data))
        copied.append(Path(dst))
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(mm.shutil, "copy2", replace_source_then_copy)
    with pytest.raises(mm.ModelIntegrityError):
        manager.import_file(spec, source)
    assert copied
    assert source.read_bytes() == b"x" * len(data)
    assert final.read_bytes() == data
    assert marker.read_bytes() == original_marker
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY
    assert foreign.read_bytes() == b"another operation"
    assert all(not path.exists() for path in copied)


@pytest.mark.parametrize("move", (False, True))
def test_import_replace_failure_preserves_source_and_install(tmp_path, monkeypatch, move):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    original_marker = marker.read_bytes()
    replace = mm.os.replace
    attempted = []

    def fail_install(src, dst):
        if Path(dst) == final:
            attempted.append(Path(src))
            raise OSError("injected final replace failure")
        return replace(src, dst)

    monkeypatch.setattr(mm.os, "replace", fail_install)
    with pytest.raises(OSError, match="injected final replace failure"):
        manager.import_file(spec, source, move=move)
    assert attempted
    assert source.read_bytes() == data
    assert final.read_bytes() == data
    assert marker.read_bytes() == original_marker
    assert all(not path.exists() for path in attempted)


@pytest.mark.parametrize("move", (False, True))
def test_import_bad_digest_preserves_source_and_install(tmp_path, move):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    original_marker = marker.read_bytes()
    source.write_bytes(b"x" * len(data))
    with pytest.raises(mm.ModelIntegrityError):
        manager.import_file(spec, source, move=move)
    assert source.read_bytes() == b"x" * len(data)
    assert final.read_bytes() == data
    assert marker.read_bytes() == original_marker


def test_move_success(tmp_path):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    assert manager.import_file(spec, source, move=True) == final
    assert not source.exists()
    assert final.read_bytes() == data
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY


@pytest.mark.parametrize("alias", ("same", "hardlink", "symlink", "directory_symlink"))
def test_move_alias_preserves_installed_bytes(tmp_path, alias):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    if alias == "same":
        source = final
    elif alias == "hardlink":
        source.unlink()
        source.hardlink_to(final)
    elif alias == "symlink":
        source.unlink()
        source.symlink_to(final)
    else:
        directory_alias = tmp_path / "directory-alias"
        directory_alias.symlink_to(final.parent, target_is_directory=True)
        source = directory_alias / final.name
    assert manager.import_file(spec, source, move=True) == final
    assert final.read_bytes() == data
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY


def test_move_unlink_failure_reports_installed_model(tmp_path, monkeypatch):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    unlink = Path.unlink

    def deny_source_unlink(path, *args, **kwargs):
        if path == source:
            raise PermissionError("source removal denied")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_source_unlink)
    with pytest.raises(mm.ModelError, match="installed.*source"):
        manager.import_file(spec, source, move=True)
    assert source.read_bytes() == data
    assert final.read_bytes() == data
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY


@pytest.mark.parametrize("change", ("replace", "rewrite", "symlink_to_final"))
def test_move_does_not_delete_changed_source(tmp_path, monkeypatch, change):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    copy = mm.shutil.copy2
    changed = b"new source contents"

    def copy_then_change(src, dst, *args, **kwargs):
        result = copy(src, dst, *args, **kwargs)
        if change == "replace":
            replacement = tmp_path / "replacement.bin"
            replacement.write_bytes(changed)
            mm.os.replace(replacement, source)
        elif change == "rewrite":
            source.write_bytes(changed)
        else:
            source.unlink()
            source.symlink_to(final)
        return result

    monkeypatch.setattr(mm.shutil, "copy2", copy_then_change)
    with pytest.raises(mm.ModelError, match="installed.*source"):
        manager.import_file(spec, source, move=True)
    assert final.read_bytes() == data
    assert source.read_bytes() == (data if change == "symlink_to_final" else changed)
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY


@pytest.mark.parametrize(
    "first_kind,second_kind",
    (("import", "download"), ("download", "import"), ("import", "import")),
)
def test_import_concurrency_uses_shared_destination_lock(
    tmp_path, monkeypatch, first_kind, second_kind,
):
    import io
    import threading

    manager, spec, source, final, marker, data = make_case(tmp_path)
    final.unlink()
    marker.unlink()
    manager._auto_download = True
    entered = threading.Event()
    release = threading.Event()
    decided = threading.Event()
    blocked = threading.Event()
    overlap = threading.Event()
    errors, results = [], []
    original_lock_for = mm._thread_lock_for
    copy = mm.shutil.copy2

    class ObservedLock:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                blocked.set()
                decided.set()
                assert self.lock.acquire(timeout=5), "destination lock was not released"
            return self

        def __exit__(self, *args):
            self.lock.release()

    def critical_section():
        if entered.is_set():
            if not release.is_set():
                overlap.set()
                decided.set()
        else:
            entered.set()
            assert release.wait(5), "first operation was not released"

    def controlled_copy(src, dst, *args, **kwargs):
        critical_section()
        return copy(src, dst, *args, **kwargs)

    class Response(io.BytesIO):
        status = 200
        headers = {}

    def controlled_open(*args, **kwargs):
        critical_section()
        return Response(data)

    monkeypatch.setattr(
        mm, "_thread_lock_for", lambda key: ObservedLock(original_lock_for(key)),
    )
    monkeypatch.setattr(mm.shutil, "copy2", controlled_copy)
    manager._open = controlled_open

    def run(kind):
        try:
            if kind == "import":
                results.append(manager.import_file(spec, source))
            else:
                results.append(manager.ensure(spec.model_id))
        except BaseException as exc:
            errors.append(exc)
            decided.set()

    first = threading.Thread(target=run, args=(first_kind,), daemon=True)
    second = threading.Thread(target=run, args=(second_kind,), daemon=True)
    try:
        first.start()
        assert entered.wait(5), "first operation did not enter"
        second.start()
        assert decided.wait(5), "second operation neither waited nor entered"
        assert blocked.is_set(), "import bypassed the shared destination lock"
        assert not overlap.is_set()
    finally:
        release.set()
        first.join(6)
        if second.ident is not None:
            second.join(6)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert results == [final, final]
    assert final.read_bytes() == data
    assert source.read_bytes() == data
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY


def _import_in_process(root, spec, source, decided, blocked, results):
    import fcntl

    manager = mm.ModelManager(
        root, registry={spec.model_id: spec}, auto_download=False, lock_timeout=5,
    )
    flock = fcntl.flock

    def observe_flock(fd, operation):
        try:
            return flock(fd, operation)
        except BlockingIOError:
            blocked.set()
            decided.set()
            raise

    fcntl.flock = observe_flock
    try:
        installed = manager.import_file(spec, source)
        results.put(("ok", installed.read_bytes(), manager.verify(spec.model_id).value))
    except BaseException as exc:
        results.put(("error", repr(exc)))
    finally:
        fcntl.flock = flock
        decided.set()


@pytest.mark.skipif(mm.os.name != "posix", reason="observes POSIX flock contention")
def test_import_concurrency_respects_process_lock(tmp_path):
    import multiprocessing

    manager, spec, source, final, marker, data = make_case(tmp_path)
    context = multiprocessing.get_context("spawn")
    decided, blocked = context.Event(), context.Event()
    results = context.Queue()
    process = context.Process(
        target=_import_in_process,
        args=(manager.models_dir, spec, source, decided, blocked, results),
    )
    try:
        with manager._process_lock(spec, final.parent, None):
            process.start()
            assert decided.wait(10), "child neither waited for the OS lock nor finished"
            assert blocked.is_set(), "import bypassed the inter-process lock"
            assert final.read_bytes() == data
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert results.get(timeout=2) == ("ok", data, "ready")
        assert source.read_bytes() == data
        assert final.read_bytes() == data
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        results.close()
        results.join_thread()


@pytest.mark.parametrize("move", (False, True))
def test_import_changed_source_with_no_previous_install(tmp_path, monkeypatch, move):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    final.unlink()
    marker.unlink()
    copy = mm.shutil.copy2

    def corrupt_before_copy(src, dst, *args, **kwargs):
        source.write_bytes(b"x" * len(data))
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(mm.shutil, "copy2", corrupt_before_copy)
    with pytest.raises(mm.ModelIntegrityError):
        manager.import_file(spec, source, move=move)
    assert source.read_bytes() == b"x" * len(data)
    assert not final.exists() and not marker.exists()
    assert not list(final.parent.glob("*.importing"))


@pytest.mark.parametrize(
    "model_id,override,expected",
    (
        ("manga_48px.checkpoint", None, True),
        ("manga_48px.dictionary", None, True),
        ("openvino_handwriting.xml", None, False),
        ("manga_48px.checkpoint", False, False),
    ),
)
def test_ensure_fallback_follows_artifact_policy(tmp_path, monkeypatch, model_id, override, expected):
    import dataclasses

    spec = dataclasses.replace(mm.get_spec(model_id), legacy_dirs=())
    manager = mm.ModelManager(
        tmp_path / "cache", registry={model_id: spec}, auto_download=True,
    )
    calls = []

    def download(spec, directory, progress, cancel, heartbeat, *, curl_fallback):
        calls.append(curl_fallback)
        return directory / spec.filename

    monkeypatch.setattr(manager, "_download_and_install", download)
    if override is None:
        manager.ensure(model_id)
    else:
        manager.ensure_spec(spec, curl_fallback=override)
    assert calls == [expected]


def test_import_marker_failure_cleans_only_owned_temporary_file(tmp_path, monkeypatch):
    manager, spec, source, final, marker, data = make_case(tmp_path)
    marker.unlink()
    foreign = final.parent / ".foreign-marker.tmp"
    foreign.write_bytes(b"another operation")
    replace = mm.os.replace

    def deny_marker_replace(src, dst):
        if Path(dst) == marker:
            raise PermissionError("marker installation denied")
        return replace(src, dst)

    monkeypatch.setattr(mm.os, "replace", deny_marker_replace)
    assert manager.import_file(spec, source) == final
    assert final.read_bytes() == data
    assert source.read_bytes() == data
    assert not marker.exists()
    assert foreign.read_bytes() == b"another operation"
    assert set(final.parent.glob("*.tmp")) == {foreign}
    assert manager.verify(spec.model_id, force=True) is mm.ModelStatus.READY
    assert set(final.parent.glob("*.tmp")) == {foreign}
