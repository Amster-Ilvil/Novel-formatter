"""OpenVINO must verify local artifacts before passing them to Core."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import types

import pytest

from adapters import openvino_handwriting_engine as ov
from utils import model_manager as mm


@pytest.fixture
def local_openvino(tmp_path, monkeypatch):
    payloads = (b"model xml", b"model bin", b"ab")
    registry = {
        model_id: dataclasses.replace(
            mm.get_spec(model_id),
            size=len(data), sha256=hashlib.sha256(data).hexdigest(),
            sha384=None, legacy_dirs=(),
        )
        for model_id, data in zip(ov._MODEL_IDS, payloads)
    }
    network, reads, checks = [], [], []

    def forbidden(*args, **kwargs):
        network.append((args, kwargs))
        raise AssertionError("local validation attempted a download")

    manager = mm.ModelManager(
        tmp_path / "cache", registry=registry, opener=forbidden,
        project_root=tmp_path / "project",
    )
    monkeypatch.setattr(mm, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(ov, "default_manager", lambda: manager)
    monkeypatch.setattr(manager, "ensure", forbidden)
    monkeypatch.setattr(manager, "ensure_spec", forbidden)
    monkeypatch.setattr(mm, "_CurlResponse", forbidden)
    directory = manager.target_dir(next(iter(ov._MODEL_IDS)))
    directory.mkdir(parents=True)
    paths = []
    for model_id, data in zip(ov._MODEL_IDS, payloads):
        path = directory / registry[model_id].filename
        path.write_bytes(data)
        assert manager.verify(model_id) is mm.ModelStatus.READY
        paths.append(path)

    verify = manager.verify

    def record_verify(model_id, **kwargs):
        checks.append((model_id, kwargs))
        return verify(model_id, **kwargs)

    monkeypatch.setattr(manager, "verify", record_verify)

    class Core:
        def read_model(self, path):
            reads.append(path)
            return types.SimpleNamespace(
                inputs=[types.SimpleNamespace(shape=(1, 1, 32, 128))],
                outputs=[types.SimpleNamespace(shape=(1, 1, 3))],
            )

        def compile_model(self, model, device):
            return object()

    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    return manager, paths, payloads, network, reads, checks


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("preserve_marker", (False, True))
def test_openvino_corruption_rejected_before_read_model(local_openvino, index, preserve_marker):
    manager, paths, payloads, network, reads, checks = local_openvino
    path = paths[index]
    stat = path.stat()
    path.write_bytes(b"x" * len(payloads[index]))
    if preserve_marker:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    else:
        manager._marker_path(path).unlink()
    available, detail = ov.model_available()
    assert available, detail
    with pytest.raises(RuntimeError):
        ov.OpenVINOJapaneseHandwritingRecognizer()
    assert reads == []
    assert network == []


@pytest.mark.parametrize("index", (0, 1, 2))
def test_openvino_missing_file_is_clear_and_offline(local_openvino, index):
    manager, paths, payloads, network, reads, checks = local_openvino
    paths[index].unlink()
    with pytest.raises(RuntimeError) as error:
        ov.OpenVINOJapaneseHandwritingRecognizer()
    assert paths[index].name in str(error.value)
    assert reads == []
    assert network == []


def test_openvino_valid_files_are_fully_verified_before_use(local_openvino):
    manager, paths, payloads, network, reads, checks = local_openvino
    recognizer = ov.OpenVINOJapaneseHandwritingRecognizer()
    assert recognizer._characters == ["[blank]", "a", "b"]
    xml_path = next(iter(paths))
    assert reads == [str(xml_path)]
    assert network == []
    assert {model_id for model_id, kwargs in checks} == set(ov._MODEL_IDS)
    assert all(kwargs.get("force") is True for model_id, kwargs in checks)
    assert all(kwargs.get("target_dir") == xml_path.parent for model_id, kwargs in checks)
