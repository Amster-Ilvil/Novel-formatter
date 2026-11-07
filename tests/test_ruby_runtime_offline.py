"""Ruby preparation policy tests; no real downloads or package installation."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from adapters import findtext_centernet_ruby as ruby
from utils import model_manager


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = tmp_path / "runtime" / "src"
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    for name in (
        "run_ocr.py", "process_ocr_base.py", "models/detector.py",
        "models/transformer.py", "textline_detect/Makefile",
        "textline_detect/linedetect", "model.pt",
    ):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(ruby, "DEFAULT_SOURCE_DIR", source)
    monkeypatch.setattr(ruby, "RUNTIME_ROOT", source.parent)
    monkeypatch.setattr(ruby, "DEFAULT_VENV_DIR", python.parent.parent)
    monkeypatch.setattr(ruby, "_source_dir", lambda: source)
    monkeypatch.setattr(ruby, "_runtime_python", lambda source: python)
    monkeypatch.setattr(
        ruby, "_ready_backend",
        lambda source: "torch" if (source / "model.pt").exists() else "",
    )
    monkeypatch.setattr(model_manager, "_PREFERENCES", {
        "models_dir": None, "auto_download": False,
    })
    monkeypatch.setenv(model_manager.ENV_AUTO_DOWNLOAD, "0")
    monkeypatch.setenv(model_manager.ENV_MODELS_DIR, str(tmp_path / "models"))
    monkeypatch.setenv("NOVEL_FORMATTER_FINDTEXT_CACHE_DIR", str(tmp_path / "ruby-cache"))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("offline Ruby attempted installation or download")

    for name in (
        "_install_source", "_install_source_with_git", "_ensure_venv",
        "ensure_venv", "_download", "_hf_xet_download",
    ):
        monkeypatch.setattr(ruby, name, forbidden)
    monkeypatch.setattr(ruby.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(ruby.subprocess, "Popen", forbidden)
    probes = []

    def local_probe(command, **kwargs):
        assert "-c" in command, "offline Ruby launched an installation subprocess"
        assert str(python) == next(iter(command))
        probes.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ruby.subprocess, "run", local_probe)
    return SimpleNamespace(
        source=source, python=python, calls=calls, probes=probes,
        forbidden=forbidden, tmp_path=tmp_path,
    )


@pytest.mark.parametrize("component", ("source", "weights", "python", "linedetect"))
def test_missing_runtime_fails_before_installation(runtime, component):
    missing = {
        "source": runtime.source / "run_ocr.py",
        "weights": runtime.source / "model.pt",
        "python": runtime.python,
        "linedetect": ruby._linedetect_path(runtime.source),
    }
    missing.get(component).unlink()
    with pytest.raises(RuntimeError) as caught:
        ruby.prepare_runtime()
    message = str(caught.value)
    assert "--no-download" in message
    assert component in message.lower()
    assert "prepare_runtime" in message
    assert runtime.calls == []


def test_ready_local_runtime_is_used_without_installation(runtime):
    assert ruby.prepare_runtime() == (runtime.source, runtime.python)
    assert runtime.calls == []
    assert runtime.probes, "offline runtime dependencies were not checked"


def test_missing_dependencies_fail_without_pip(runtime, monkeypatch):
    monkeypatch.setattr(
        ruby.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="ModuleNotFoundError: torch",
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        ruby.prepare_runtime()
    message = str(caught.value)
    assert "dependencies" in message.lower()
    assert "--no-download" in message
    assert "prepare_runtime" in message
    assert runtime.calls == []


def test_normal_mode_keeps_preparation(runtime, monkeypatch):
    model_manager._PREFERENCES.update(auto_download=True)
    steps = []
    monkeypatch.setattr(ruby, "_install_source", lambda *a, **k: steps.append("source"))
    monkeypatch.setattr(
        ruby, "_ensure_venv",
        lambda *a, **k: steps.append("venv") or runtime.python,
    )
    monkeypatch.setattr(ruby, "_compile_linedetect", lambda *a, **k: steps.append("compile"))
    from adapters import ocr_runtime_catalog
    monkeypatch.setattr(ocr_runtime_catalog, "mark_runtime_ready", lambda *a, **k: None)
    assert ruby.prepare_runtime() == (runtime.source, runtime.python)
    assert steps == ["source", "venv", "compile"]


def test_cli_no_download_preserve_ruby_reaches_offline_guard(runtime, monkeypatch, capsys):
    import run
    from adapters import apple_vision_adapter
    from builder import epub_builder
    from engine import formatter
    from models.document import UnifiedDocument

    image = runtime.tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image)
    document = UnifiedDocument()
    monkeypatch.setattr(apple_vision_adapter, "run", lambda **kwargs: document)
    monkeypatch.setattr(formatter, "run_pipeline", lambda doc, **kwargs: doc)
    monkeypatch.setattr(epub_builder, "build_epub", lambda *args, **kwargs: None)
    monkeypatch.setattr(ruby, "_text_page_images", lambda doc: [(1, str(image))])
    monkeypatch.setattr(ruby.RubyResultCache, "get", lambda *args, **kwargs: None)
    (runtime.source / "run_ocr.py").unlink()
    assert run.main([
        str(image), str(runtime.tmp_path / "book.epub"),
        "--no-download", "--preserve-ruby", "--ruby-scan-mode", "full_page",
        "--no-preview",
    ]) == 0
    output = capsys.readouterr().out
    assert "--no-download" in output
    assert "source" in output.lower()
    assert runtime.calls == []
