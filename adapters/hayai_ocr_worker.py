#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent Hayai OCR worker for Novel Formatter.

The worker accepts already-isolated text crops over a JSONL stream.  It never
performs page layout detection; geometry and reading order stay in the parent
process.  Batch requests are forwarded to HayaiOcr in one model call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

# Direct worker execution sets sys.path[0] to ``adapters``; add the project
# root explicitly so local ``utils`` and ``adapters`` packages cannot disappear.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from utils.apple_silicon_runtime import (
    is_m6, recommended_cpu_threads, recommended_hayai_batch,
    configure_torch_memory_safety, probe_torch_mps,
)


class _StatusEmitter:
    def __init__(self) -> None:
        self.enabled = os.environ.get("NOVEL_FORMATTER_HAYAI_STATUS") == "1"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._phase = ""
        self._started = 0.0
        self._thread: threading.Thread | None = None

    def send(self, payload: dict) -> None:
        with self._lock:
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def start(self, phase: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._phase = phase
            self._started = time.monotonic()
        # Emit every phase transition immediately, then keep the existing
        # 15-second heartbeat for genuinely long imports/model loads.
        self.send({"status": phase, "elapsed_seconds": 0})
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="hayai-status")
            self._thread.start()

    def phase(self, phase: str) -> None:
        self.start(phase)

    def pause(self) -> None:
        with self._lock:
            self._phase = ""

    def _run(self) -> None:
        while not self._stop.wait(15.0):
            with self._lock:
                phase = self._phase
                started = self._started
            if not phase:
                continue
            self.send({
                "status": phase,
                "elapsed_seconds": int(time.monotonic() - started),
            })

    def close(self) -> None:
        self._stop.set()



def _torch_device_available(torch_module, device: str) -> bool:
    key = str(device or "").strip().lower()
    if key == "cuda":
        try:
            return bool(torch_module.cuda.is_available())
        except Exception:
            return False
    if key == "mps":
        return probe_torch_mps(torch_module)[0]
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
        try:
            mps_built = bool(torch_module.backends.mps.is_built())
        except Exception:
            mps_built = False
        warning = ""
        if mps_built:
            _ok, detail = probe_torch_mps(torch_module)
            warning = (
                "PyTorch 已编译 MPS，但真实 tensor probe 未通过；本次自动使用 CPU："
                + detail + "。Apple Silicon 上请检查是否误用了 x86_64/Rosetta Python。"
            )
        return None, "cpu", warning
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
        mps_available = bool(
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if mps_available and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

def _configure_runtime(recognizer) -> tuple[object, str]:
    try:
        import torch
    except Exception:
        return nullcontext, str(getattr(recognizer, "device", "cpu"))

    try:
        requested_threads = int(os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_THREADS", "0") or 0)
    except ValueError:
        requested_threads = 0
    if requested_threads <= 0:
        requested_threads = recommended_cpu_threads()
    try:
        torch.set_num_threads(requested_threads)
    except Exception:
        pass
    configure_torch_memory_safety(torch)
    if is_m6() and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return torch.inference_mode, str(getattr(recognizer, "device", "cpu"))


_LAST_EFFECTIVE_BATCH = 0


def _recognize_direct(recognizer, inference_context, ordered: list[str], max_new_tokens: int) -> list[dict]:
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
            "ok": True, "path": path, "blocks": blocks,
            "input_size": input_size, "orientation": "vertical-preserved",
        })
    return output


def _recognize_batch(recognizer, inference_context, paths: list[str], max_new_tokens: int) -> tuple[list[dict], int, bool]:
    """Run a batch and halve on pressure before falling back to per-item work.

    A transient MPS/CUDA allocation failure should not immediately demote the
    entire OCR worker to CPU. Splitting preserves accelerator use and usually
    recovers most throughput on unified-memory machines.
    """
    global _LAST_EFFECTIVE_BATCH
    ordered = [str(path) for path in paths]
    if not ordered:
        return [], 0, False
    try:
        result = _recognize_direct(recognizer, inference_context, ordered, max_new_tokens)
        _LAST_EFFECTIVE_BATCH = max(_LAST_EFFECTIVE_BATCH, len(ordered))
        return result, len(ordered), False
    except Exception as first_error:
        _release_accelerator_cache()
        if len(ordered) > 1:
            middle = max(1, len(ordered) // 2)
            left, left_size, left_backoff = _recognize_batch(recognizer, inference_context, ordered[:middle], max_new_tokens)
            right, right_size, right_backoff = _recognize_batch(recognizer, inference_context, ordered[middle:], max_new_tokens)
            return left + right, max(left_size, right_size), True or left_backoff or right_backoff
        path = ordered[0]
        try:
            result = _recognize_direct(recognizer, inference_context, [path], max_new_tokens)
            _LAST_EFFECTIVE_BATCH = max(_LAST_EFFECTIVE_BATCH, 1)
            return result, 1, True
        except Exception as exc:
            return [{"ok": False, "path": path, "error": str(exc or first_error)}], 1, True


class _SupernovaRecognizer:
    """Official v2.7 Transformers 5 inference."""


    def __init__(self, device: str, *, max_num_patches: int, requested_quantize: str):
        from transformers import AutoModel, AutoProcessor
        from adapters.hayai_ocr_adapter import (
            SUPERNOVA_MODEL, SUPERNOVA_REVISION, _supernova_model_location,
        )
        import torch
        source = _supernova_model_location()
        local = Path(source).expanduser().is_dir()
        opts = {"trust_remote_code": True}
        if not local:
            if source != SUPERNOVA_MODEL:
                raise RuntimeError("拒绝载入未经批准的远端 Supernova 模型")
            opts["revision"] = SUPERNOVA_REVISION
            opts["code_revision"] = SUPERNOVA_REVISION
        self.device = device
        self.max_num_patches = max_num_patches
        self.requested_quantize = requested_quantize
        # Using a local model means remote code is loaded from that local path.
        # When online, both code and weights are fixed to the same revision.
        # MPS supports native half precision. Keep an explicit fp32 escape hatch
        # for diagnosing numerical differences on future model revisions.
        precision = os.environ.get("NOVEL_FORMATTER_HAYAI_MPS_DTYPE", "float16").strip().lower()
        if device == "mps" and precision not in {"float16", "float32"}:
            raise ValueError("NOVEL_FORMATTER_HAYAI_MPS_DTYPE must be float16 or float32")
        if device == "mps":
            opts["dtype"] = torch.float16 if precision == "float16" else torch.float32
        self.model = AutoModel.from_pretrained(source, **opts).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=True,
            **({"revision": SUPERNOVA_REVISION, "code_revision": SUPERNOVA_REVISION} if not local else {}))
        if device != "cuda":
            os.environ.setdefault("HAYAI_DISABLE_COMPILE", "1")

    def __call__(self, images, *, max_new_tokens=128, repetition_penalty=1.0):
        if repetition_penalty != 1.0:
            raise ValueError("Supernova OCR requires repetition_penalty=1.0")
        single = isinstance(images, (str, Path, Image.Image))
        batch = [images] if single else list(images)
        pil_images = []
        for item in batch:
            if isinstance(item, Image.Image):
                pil_images.append(item.convert("RGB"))
            else:
                with Image.open(item) as src:
                    pil_images.append(src.convert("RGB"))
        # CUDA Graphs are upstream's optimized path; CPU/MPS use eager mode.
        result = self.model.ocr(
            pil_images if not single else pil_images[0], self.processor,
            max_patches=self.max_num_patches,
            max_new_tokens=max_new_tokens,
            use_cuda_graph=self.device == "cuda",
        )
        return result


def main() -> None:
    from adapters.hayai_ocr_adapter import SUPERNOVA_REVISION, _supernova_model_location

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--backend", choices=("supernova",), default="supernova")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-num-patches", type=int, default=256)
    args = parser.parse_args()
    if not args.stream:
        parser.error("Hayai worker currently requires --stream")

    status = _StatusEmitter()
    status.start("startup")
    try:
        status.phase("import_torch")
        import torch
        status.phase("resolve_device")
        device_arg, effective_device_hint, device_warning = _resolve_torch_device(torch, args.device)
        if effective_device_hint in {"cpu", "mps"}:
            os.environ.setdefault("HAYAI_DISABLE_COMPILE", "1")
        try:
            mps_built = bool(torch.backends.mps.is_built())
        except Exception:
            mps_built = False
        mps_available, mps_probe_detail = probe_torch_mps(torch)
        if status.enabled:
            import platform
            status.send({
                "status": "device_resolved",
                "elapsed_seconds": 0,
                "requested_device": str(args.device or "auto"),
                "effective_device": effective_device_hint,
                "torch_version": str(getattr(torch, "__version__", "")),
                "machine": platform.machine(),
                "mps_built": mps_built,
                "mps_available": mps_available,
                "mps_probe_detail": mps_probe_detail,
                "offline_mode": os.environ.get("NOVEL_FORMATTER_HAYAI_OFFLINE_ACTIVE") == "1",
            })
        configure_torch_memory_safety(torch)
        status.phase("import_hayai")
        status.phase("load_model")
        recognizer = _SupernovaRecognizer(
            effective_device_hint,
            max_num_patches=max(1, int(args.max_num_patches or 256)),
            requested_quantize="none",
        )
        status.phase("configure_runtime")
        inference_context, device_label = _configure_runtime(recognizer)
    except Exception as exc:
        status.pause()
        status.send({"ok": False, "path": "", "error": f"Hayai OCR 模型初始化失败: {exc}"})
        raise SystemExit(1)

    status.pause()
    status.send({
        "ok": True,
        "ready": True,
        "device": device_label,
        "backend": "supernova",
        "quantize": "none",
        "requested_quantize": "none",
        "effective_quantize": "none",
        "warning": device_warning,
        "model": _supernova_model_location(),
        "model_revision": SUPERNOVA_REVISION,
        "input_contract": "isolated-crop-batch",
        "max_num_patches": int(getattr(recognizer, "max_num_patches", args.max_num_patches)),
        "offline_mode": os.environ.get("NOVEL_FORMATTER_HAYAI_OFFLINE_ACTIVE") == "1",
    })

    max_new_tokens = max(32, min(192, int(args.max_new_tokens or 128)))
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
            status.close()
            status.send({"ok": True, "closed": True})
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
        if paths:
            status.start("recognition")
            items, effective_batch, backoff = _recognize_batch(recognizer, inference_context, paths, max_new_tokens)
            status.pause()
        else:
            items, effective_batch, backoff = [], 0, False
        if paths and backoff:
            try:
                from utils.hardware_runtime import current_available_memory_gb
                from utils.ocr_runtime_calibration import record
                record(
                    "hayai", batch_size=len(paths), elapsed_seconds=0.001,
                    items=len(paths), available_memory_gb=current_available_memory_gb(),
                    success=False, resource_error=True,
                )
            except Exception:
                pass
        response = {
            "ok": bool(paths),
            "items": items,
            "effective_batch": effective_batch,
            "backoff": backoff,
        }
        if not paths:
            response["error"] = "缺少 paths"
        if "request_id" in request:
            response["request_id"] = request["request_id"]
        status.send(response)


if __name__ == "__main__":
    main()
