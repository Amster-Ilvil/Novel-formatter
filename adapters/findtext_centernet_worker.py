#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent, upstream-native findtextCenterNet worker.

Novel Formatter deliberately does *not* reimplement findtextCenterNet here.
The worker changes cwd/sys.path to the pinned upstream source tree, imports the
upstream ``run_ocr.py`` exactly as shipped, and reuses its already-created
``processer`` for all Smart-ROI images in this process.

Protocol (JSONL over stdin/stdout):
  request: {"request_id": 1, "images": ["/tmp/a.png", ...], "resize": 1.0}
  result:  {"request_id": 1, "items": [{"path":..., "ok":true,"payload":...}]}

All noisy upstream prints are redirected to stderr so stdout remains a clean
machine protocol.  The upstream sidecar ``<image>.json`` is read verbatim and
returned; its schema and Ruby semantics therefore remain upstream-owned.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _load_upstream(source_root: Path):
    source_root = source_root.resolve()
    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    # run_ocr.py selects CoreML -> ONNX -> Torch exactly as upstream defines and
    # instantiates OCR_Processer at module import.  Keep its banner/noise off the
    # JSONL protocol channel.
    trace = _trace_coreml_model_loading(source_root)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import run_ocr as upstream  # type: ignore
    finally:
        if trace is not None:
            model_class, original_init = trace
            model_class.__init__ = original_init
    models = list(getattr(upstream, "models", []) or [])
    if not models or not hasattr(upstream, "processer"):
        raise RuntimeError("上游 run_ocr.py 未能选择可用模型后端")
    backend = str(models[0])
    if backend == "coreml":
        model_attributes = (
            "mlmodel_detector",
            "mlmodel_transformer_encoder",
            "mlmodel_transformer_decoder",
        )
        unavailable = []
        for attribute in model_attributes:
            model = getattr(upstream.processer, attribute, None)
            if model is None or (hasattr(model, "__proxy__") and model.__proxy__ is None):
                unavailable.append(attribute)
        if unavailable:
            raise RuntimeError(
                "CoreML 模型未能编译为可执行对象：" + ", ".join(unavailable)
                + "；worker 不会将不可预测的模型报告为 ready"
            )
    return upstream, backend


def _process_one(upstream, image_path: str, resize: float) -> dict:
    path = Path(image_path).resolve()
    sidecar = Path(str(path) + ".json")
    sidecar.unlink(missing_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        upstream.processer.call_OCR(str(path), resize)
    if not sidecar.is_file():
        raise RuntimeError(f"上游未生成 JSON：{sidecar.name}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    finally:
        # Parent owns persistent caching.  Do not leave transient ROI sidecars
        # behind in Novel Formatter's temporary directory.
        sidecar.unlink(missing_ok=True)
    if not isinstance(payload, dict):
        raise RuntimeError("上游 JSON 顶层不是对象")
    return payload


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _write_progress(**event) -> None:
    # Upstream construction redirects sys.stdout to stderr; keep protocol events
    # on the original pipe so the parent can consume them as structured progress.
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    stream.write(json.dumps({"type": "progress", **event}, ensure_ascii=False) + "\n")
    stream.flush()




def _coreml_package_signature(package: Path) -> str:
    """Cheap invalidation key for a local ``.mlpackage`` tree.

    The official model archives are already pinned/validated by the parent.  A
    metadata fingerprint is sufficient here to invalidate compiled CoreML output
    when those extracted package files are replaced.
    """
    digest = hashlib.sha256()
    for child in sorted(package.rglob("*"), key=lambda item: item.as_posix()):
        if not child.is_file():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        digest.update(str(child.relative_to(package)).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b":")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:20]


def _compiled_coreml_model(ct, source_root: Path, raw_model) -> Path | None:
    """Return a persistent ``.mlmodelc`` for a package, fail-open on any error."""
    if not isinstance(raw_model, (str, os.PathLike)):
        return None
    package = Path(os.fspath(raw_model))
    if not package.is_absolute():
        package = (source_root / package).resolve()
    if package.suffix != ".mlpackage" or not package.is_dir():
        return None
    compile_model = getattr(getattr(ct, "utils", None), "compile_model", None)
    if not callable(compile_model):
        return None
    configured = str(os.environ.get("NOVEL_FORMATTER_FINDTEXT_COREML_CACHE", "") or "").strip()
    cache_root = (
        Path(configured).expanduser().resolve()
        if configured else (source_root.parent / "coreml-compiled-v1").resolve()
    )
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        signature = _coreml_package_signature(package)
        target = cache_root / f"{package.stem}-{signature}.mlmodelc"
        if target.is_dir():
            if (target / "Manifest.json").is_file():
                _write_progress(stage="coreml_compiled_cache_hit", model=package.name)
                return target
            # CoreML can return a compiled asset directory that lacks the
            # package manifest required by MLModel(path). Do not persist or
            # reuse that partial artifact; let MLModel load the source package.
            shutil.rmtree(target, ignore_errors=True)
        _write_progress(stage="coreml_compile_started", model=package.name)
        started = time.monotonic()
        compiled = Path(os.fspath(compile_model(str(package)))).resolve()
        if not compiled.exists():
            raise RuntimeError("coremltools.compile_model 未返回有效 .mlmodelc")
        if not compiled.is_dir() or not (compiled / "Manifest.json").is_file():
            if compiled.is_dir():
                shutil.rmtree(compiled, ignore_errors=True)
            raise RuntimeError("CoreML 编译结果缺少 Manifest.json；改由 MLModel 加载原始模型包")
        with tempfile.TemporaryDirectory(prefix="nf-findtext-coreml-", dir=str(cache_root)) as td:
            stage = Path(td) / target.name
            if compiled.is_dir():
                shutil.copytree(compiled, stage)
            else:
                raise RuntimeError("CoreML 编译结果不是目录")
            try:
                os.replace(stage, target)
            except OSError:
                if not target.is_dir():
                    raise
        # Keep only the current compiled generation for this package.
        for old in cache_root.glob(f"{package.stem}-*.mlmodelc"):
            if old != target:
                shutil.rmtree(old, ignore_errors=True)
        _write_progress(
            stage="coreml_compile_finished", model=package.name,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        return target
    except Exception as exc:
        _write_progress(
            stage="coreml_compile_failed", model=package.name, error=type(exc).__name__
        )
        return None

def _trace_coreml_model_loading(source_root: Path):
    package_names = (
        "TextDetector.mlpackage",
        "TransformerEncoder.mlpackage",
        "TransformerDecoder.mlpackage",
    )
    if not all((source_root / name).is_dir() for name in package_names):
        return None
    try:
        # Observe CoreML model construction without importing any private
        # findtextCenterNet implementation module.  The only upstream entry
        # point owned by this bridge remains run_ocr.py.
        import coremltools as ct

        model_class = ct.models.MLModel
        original_init = model_class.__init__

        def traced_init(self, model, *args, **kwargs):
            model_name = (
                Path(os.fspath(model)).name
                if isinstance(model, (str, os.PathLike)) else "in-memory model"
            )
            started = time.monotonic()
            _write_progress(stage="coreml_model_started", model=model_name)
            compiled = _compiled_coreml_model(ct, source_root, model)
            load_target = str(compiled) if compiled is not None else model
            try:
                result = original_init(self, load_target, *args, **kwargs)
            except BaseException:
                _write_progress(
                    stage="coreml_model_failed", model=model_name,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
                raise
            _write_progress(
                stage="coreml_model_finished", model=model_name,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            return result

        model_class.__init__ = traced_init
        return model_class, original_init
    except Exception:
        # Instrumentation is observational only and must not disable a backend.
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    print(json.dumps({
        "type": "progress",
        "stage": "upstream_import",
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "coremltools": _package_version("coremltools"),
    }, ensure_ascii=False), flush=True)
    try:
        upstream, backend = _load_upstream(source_root)
    except Exception as exc:
        print(json.dumps({"type": "ready", "ready": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        traceback.print_exc(file=sys.stderr)
        return 2

    print(json.dumps({
        "type": "ready",
        "ready": True,
        "backend": backend,
        "upstream": "lithium0003/findtextCenterNet",
    }, ensure_ascii=False), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except Exception as exc:
            print(json.dumps({"type": "protocol_error", "error": str(exc)}, ensure_ascii=False), flush=True)
            continue
        if request.get("command") == "close":
            print(json.dumps({"type": "closed"}, ensure_ascii=False), flush=True)
            return 0
        request_id = int(request.get("request_id", 0) or 0)
        images = [str(item) for item in (request.get("images") or [])]
        try:
            resize = float(request.get("resize", 1.0) or 1.0)
        except Exception:
            resize = 1.0
        items = []
        total = len(images)
        for index, image_path in enumerate(images, start=1):
            started = time.monotonic()
            print(json.dumps({
                "type": "progress", "stage": "roi_started",
                "request_id": request_id, "index": index, "total": total,
            }), flush=True)
            try:
                payload = _process_one(upstream, image_path, resize)
                items.append({"path": image_path, "ok": True, "payload": payload})
                ok = True
            except Exception as exc:
                items.append({"path": image_path, "ok": False, "error": str(exc)})
                ok = False
                error_type = type(exc).__name__
                traceback.print_exc(file=sys.stderr)
            else:
                error_type = ""
            print(json.dumps({
                "type": "progress", "stage": "roi_finished",
                "request_id": request_id, "index": index, "total": total,
                "ok": ok, "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": error_type,
            }), flush=True)
        print(json.dumps({
            "request_id": request_id,
            "items": items,
            "request_done": True,
        }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
