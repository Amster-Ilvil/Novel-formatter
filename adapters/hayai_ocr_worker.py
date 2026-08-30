#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent Hayai OCR v2.1 worker for Novel Formatter.

The worker accepts already-isolated text crops over a JSONL stream.  It never
performs page layout detection; geometry and reading order stay in the parent
process.  Batch requests are forwarded to HayaiOcr in one model call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from PIL import Image



def _torch_device_available(torch_module, device: str) -> bool:
    key = str(device or "").strip().lower()
    if key == "cuda":
        try:
            return bool(torch_module.cuda.is_available())
        except Exception:
            return False
    if key == "mps":
        try:
            return bool(hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available())
        except Exception:
            return False
    return key == "cpu"


def _resolve_torch_device(torch_module, requested: str) -> tuple[str | None, str, str]:
    """Return (device_arg, effective_device, warning) without hiding user errors."""
    raw = str(requested or "auto").strip().lower()
    if raw not in {"auto", "cpu", "cuda", "mps"}:
        raw = "auto"
    if raw == "auto":
        if _torch_device_available(torch_module, "cuda"):
            return None, "cuda", ""
        if _torch_device_available(torch_module, "mps"):
            return None, "mps", ""
        return None, "cpu", ""
    if _torch_device_available(torch_module, raw):
        return raw, raw, ""
    return "cpu", "cpu", f"请求的 {raw.upper()} 当前不可用，已自动回退到 CPU；识别功能保持可用。"


def _release_accelerator_cache() -> None:
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

def _configure_runtime(recognizer) -> tuple[object, str]:
    if bool(getattr(recognizer, "is_litert", False)):
        return nullcontext, str(getattr(recognizer, "device", "cpu"))
    try:
        import torch
    except Exception:
        return nullcontext, str(getattr(recognizer, "device", "cpu"))

    try:
        requested_threads = int(os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_THREADS", "0") or 0)
    except ValueError:
        requested_threads = 0
    if requested_threads <= 0:
        requested_threads = min(8, max(1, os.cpu_count() or 4))
    try:
        torch.set_num_threads(requested_threads)
    except Exception:
        pass
    return torch.inference_mode, str(getattr(recognizer, "device", "cpu"))


def _recognize_batch(recognizer, inference_context, paths: list[str], max_new_tokens: int) -> list[dict]:
    ordered = [str(path) for path in paths]
    if not ordered:
        return []
    try:
        sizes: list[list[int]] = []
        for path in ordered:
            with Image.open(path) as probe:
                sizes.append(list(probe.size))
        with inference_context():
            values = recognizer(ordered, max_new_tokens=max_new_tokens, repetition_penalty=1.0)
        texts = [values] if isinstance(values, str) else list(values)
        if len(texts) != len(ordered):
            raise RuntimeError(f"Hayai OCR 批量返回数量异常：输入 {len(ordered)}，返回 {len(texts)}")
        output: list[dict] = []
        for path, input_size, text in zip(ordered, sizes, texts):
            value = str(text or "").strip()
            blocks = [{
                "text": value,
                "confidence": 0.0,
                "confidence_kind": "uncalibrated",
                "box": None,
                "input_size": input_size,
                "orientation": "vertical-preserved",
            }] if value else []
            output.append({
                "ok": True,
                "path": path,
                "blocks": blocks,
                "input_size": input_size,
                "orientation": "vertical-preserved",
            })
        return output
    except Exception:
        # Isolate one corrupt/oversized crop rather than losing the whole batch.
        # Release allocator pressure first; a failed large batch may otherwise make
        # every single-crop retry fail for reasons unrelated to that crop.
        _release_accelerator_cache()
        output = []
        for path in ordered:
            try:
                with Image.open(path) as probe:
                    input_size = list(probe.size)
                with inference_context():
                    value = str(recognizer(path, max_new_tokens=max_new_tokens, repetition_penalty=1.0) or "").strip()
                blocks = [{
                    "text": value,
                    "confidence": 0.0,
                    "confidence_kind": "uncalibrated",
                    "box": None,
                    "input_size": input_size,
                    "orientation": "vertical-preserved",
                }] if value else []
                output.append({"ok": True, "path": path, "blocks": blocks, "input_size": input_size})
            except Exception as exc:
                output.append({"ok": False, "path": path, "error": str(exc)})
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--backend", choices=("torch", "litert"), default="torch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantize", choices=("none", "int8", "int4"), default="none")
    parser.add_argument("--litert-quant", default="wi4")
    parser.add_argument("--litert-threads", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.stream:
        parser.error("Hayai worker currently requires --stream")

    try:
        backend = str(args.backend or "torch").lower()
        model_name = os.environ.get(
            "NOVEL_FORMATTER_HAYAI_OCR_MODEL", "JustANormalTinkerer/hayai-ocr-v2"
        )
        kwargs = {"backend": backend}
        device_warning = ""
        if backend == "torch":
            import torch
            device_arg, effective_device_hint, device_warning = _resolve_torch_device(torch, args.device)
            # Hayai v2.1 currently calls torch.compile unconditionally. Dynamo on
            # CPU/MPS adds startup cost and has had backend-specific failures; the
            # recognizer remains identical when Dynamo is disabled. Keep compile
            # available on CUDA, where it is most useful.
            if effective_device_hint in {"cpu", "mps"}:
                os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
            kwargs["pretrained_model_name_or_path"] = model_name
            if device_arg is not None:
                kwargs["device"] = device_arg
            if args.quantize and args.quantize != "none":
                kwargs["quantize"] = args.quantize
        else:
            kwargs["litert_quant"] = args.litert_quant or "wi4"
            if args.litert_threads > 0:
                kwargs["litert_threads"] = args.litert_threads
            litert_repo = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_LITERT_REPO", "").strip()
            if litert_repo:
                kwargs["litert_repo"] = litert_repo
            litert_model_path = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_LITERT_MODEL_PATH", "").strip()
            if litert_model_path:
                kwargs["litert_model_path"] = litert_model_path

        from hayai_ocr import HayaiOcr

        requested_quantize = args.quantize if backend == "torch" else args.litert_quant
        effective_quantize = requested_quantize
        startup_warning = device_warning
        try:
            recognizer = HayaiOcr(**kwargs)
        except Exception as first_exc:
            # torchao support differs by PyTorch/device builds. Quantization is an
            # optimisation, not a correctness requirement, so retry unquantized
            # instead of making an otherwise supported OCR backend unusable.
            if backend != "torch" or args.quantize == "none" or "quantize" not in kwargs:
                raise
            kwargs.pop("quantize", None)
            try:
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                recognizer = HayaiOcr(**kwargs)
            except Exception:
                raise first_exc
            effective_quantize = "none"
            quant_warning = (
                f"请求的 {args.quantize.upper()} 量化在当前运行时不可用，已自动回退到非量化模式；"
                "识别功能保持可用。"
            )
            startup_warning = " ".join(item for item in (startup_warning, quant_warning) if item)
        inference_context, device_label = _configure_runtime(recognizer)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "path": "",
            "error": f"Hayai OCR 模型初始化失败: {exc}",
        }, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    print(json.dumps({
        "ok": True,
        "ready": True,
        "device": device_label,
        "backend": backend,
        "quantize": effective_quantize,
        "requested_quantize": requested_quantize,
        "effective_quantize": effective_quantize,
        "warning": startup_warning,
        "model": model_name if backend == "torch" else "JustANormalTinkerer/hayai-ocr-v2-tflite",
        "input_contract": "isolated-crop-batch",
    }, ensure_ascii=False), flush=True)

    max_token_cap = 64 if backend == "litert" else 192
    max_new_tokens = max(32, min(max_token_cap, int(args.max_new_tokens or 128)))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"请求 JSON 无效: {exc}"}, ensure_ascii=False), flush=True)
            continue
        if request.get("command") == "close":
            print(json.dumps({"ok": True, "closed": True}, ensure_ascii=False), flush=True)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(0)

        raw_paths = request.get("paths")
        if not isinstance(raw_paths, list):
            raw_path = str(request.get("path") or "")
            raw_paths = [raw_path] if raw_path else []
        paths = [str(path) for path in raw_paths if str(path)]
        response = {
            "ok": bool(paths),
            "items": _recognize_batch(recognizer, inference_context, paths, max_new_tokens) if paths else [],
        }
        if not paths:
            response["error"] = "缺少 paths"
        if "request_id" in request:
            response["request_id"] = request["request_id"]
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
