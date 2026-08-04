#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent Manga-OCR recognizer for already prepared short text regions.

The parent adapter owns page layout, vertical-column detection and long-column
chunking.  This worker intentionally performs no rotation, no page masking and
no fallback crop: Manga OCR natively supports vertical Japanese and must see the
short vertical pixels exactly as printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

from PIL import Image, ImageOps


def _configure_runtime(recognizer) -> tuple[object, str]:
    """Apply safe inference-only optimizations and return a context factory."""
    try:
        import torch
    except Exception:
        return nullcontext, "default"

    try:
        requested_threads = int(os.environ.get("NOVEL_FORMATTER_MANGA_OCR_THREADS", "0") or 0)
    except ValueError:
        requested_threads = 0
    if requested_threads <= 0:
        requested_threads = min(8, max(1, os.cpu_count() or 4))
    try:
        torch.set_num_threads(requested_threads)
    except Exception:
        pass

    device_label = str(getattr(recognizer, "device", "cpu"))
    use_mps = os.environ.get("NOVEL_FORMATTER_MANGA_OCR_MPS", "1") != "0"
    if use_mps:
        try:
            mps_ready = bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
            model = getattr(recognizer, "model", None)
            if mps_ready and model is not None and "mps" not in device_label.lower():
                device = torch.device("mps")
                model.to(device)
                recognizer.device = device
                device_label = "mps"
        except Exception:
            pass

    return torch.inference_mode, device_label


class InternalMangaOcr:
    """Small, version-stable wrapper around the official model components.

    The PyPI package can lag behind the repository's tokenizer initialization.
    Using the official current tokenizer type explicitly prevents Transformers
    from selecting an incompatible fast tokenizer while keeping the exact
    ``kha-white/manga-ocr-base`` weights and post-processing contract.
    """

    def __init__(self, pretrained_model_name_or_path: str = "kha-white/manga-ocr-base"):
        import torch
        from transformers import (
            AutoTokenizer,
            GenerationMixin,
            ViTImageProcessor,
            VisionEncoderDecoderModel,
        )

        class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
            pass

        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            tokenizer_type="bert-japanese",
            use_fast=False,
        )
        self.model = MangaOcrModel.from_pretrained(pretrained_model_name_or_path)
        self.model.eval()
        self.device = torch.device("cpu")

    @staticmethod
    def _normalize_text(text: str) -> str:
        import jaconv

        value = "".join(str(text or "").split())
        value = value.replace("…", "...")
        value = re.sub(r"[・.]{2,}", lambda match: (match.end() - match.start()) * ".", value)
        return jaconv.h2z(value, ascii=True, digit=True)

    def batch(self, paths: list[str]) -> list[str]:
        """Run one real tensor batch while preserving official preprocessing."""
        images: list[Image.Image] = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(ImageOps.exif_transpose(source).convert("L").convert("RGB"))
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        finally:
            for image in images:
                image.close()
        generated = self.model.generate(
            pixel_values.to(self.model.device),
            max_length=300,
        ).cpu()
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [self._normalize_text(text) for text in decoded]

    def __call__(self, img_or_path) -> str:
        if isinstance(img_or_path, (str, Path)):
            return self.batch([str(img_or_path)])[0]
        if isinstance(img_or_path, Image.Image):
            # Keep the old object input contract for compatibility/debugging.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png") as fh:
                img_or_path.convert("RGB").save(fh.name)
                return self.batch([fh.name])[0]
        raise ValueError(f"不支持的 Manga OCR 输入类型: {type(img_or_path)!r}")



def _recognize(recognizer, inference_context, path: str) -> dict:
    try:
        with Image.open(path) as probe:
            input_size = list(probe.size)
        with inference_context():
            text = str(recognizer(path) or "").strip()
        blocks = [{
            "text": text,
            "confidence": 0.0,
            "confidence_kind": "uncalibrated",
            "box": None,
            "input_size": input_size,
            "orientation": "vertical-preserved",
        }] if text else []
        return {
            "ok": True,
            "path": path,
            "blocks": blocks,
            "input_size": input_size,
            "orientation": "vertical-preserved",
        }
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

def _recognize_batch(recognizer, inference_context, paths: list[str]) -> list[dict]:
    """Recognize a bounded path group; isolate failures without losing order."""
    ordered = [str(path) for path in paths]
    if not ordered:
        return []
    try:
        sizes = []
        for path in ordered:
            with Image.open(path) as probe:
                sizes.append(list(probe.size))
        with inference_context():
            texts = recognizer.batch(ordered)
        if len(texts) != len(ordered):
            raise RuntimeError(
                f"Manga OCR 批量返回数量异常：输入 {len(ordered)}，返回 {len(texts)}"
            )
        items = []
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
            items.append({
                "ok": True,
                "path": path,
                "blocks": blocks,
                "input_size": input_size,
                "orientation": "vertical-preserved",
            })
        return items
    except Exception:
        # A corrupt image or backend batch edge case must not invalidate the
        # other physical columns. Fall back to the proven single-image path.
        return [_recognize(recognizer, inference_context, path) for path in ordered]



def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest")
    mode.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    try:
        model_name = os.environ.get("NOVEL_FORMATTER_MANGA_OCR_MODEL", "kha-white/manga-ocr-base")
        recognizer = InternalMangaOcr(model_name)
        inference_context, device_label = _configure_runtime(recognizer)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "path": "",
            "error": f"Manga OCR 模型初始化失败: {exc}",
        }, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    if args.stream:
        print(json.dumps({
            "ok": True,
            "ready": True,
            "device": device_label,
            "model": model_name,
            "input_contract": "short-vertical-unrotated",
        }, ensure_ascii=False), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except Exception as exc:
                print(json.dumps({
                    "ok": False,
                    "path": "",
                    "error": f"请求 JSON 无效: {exc}",
                }, ensure_ascii=False), flush=True)
                continue
            if request.get("command") == "close":
                print(json.dumps({"ok": True, "closed": True}, ensure_ascii=False), flush=True)
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os._exit(0)
            raw_paths = request.get("paths")
            if isinstance(raw_paths, list):
                paths = [str(path) for path in raw_paths if str(path)]
                if not paths:
                    response = {"ok": False, "items": [], "error": "缺少 paths"}
                else:
                    response = {
                        "ok": True,
                        "items": _recognize_batch(recognizer, inference_context, paths),
                    }
                if "request_id" in request:
                    response["request_id"] = request["request_id"]
                print(json.dumps(response, ensure_ascii=False), flush=True)
                continue

            path = str(request.get("path", ""))
            if not path:
                print(json.dumps({"ok": False, "path": "", "error": "缺少 path"}, ensure_ascii=False), flush=True)
                continue
            response = _recognize(recognizer, inference_context, path)
            if "request_id" in request:
                response["request_id"] = request["request_id"]
            print(json.dumps(response, ensure_ascii=False), flush=True)
        return

    try:
        with open(args.manifest, "r", encoding="utf-8") as fh:
            paths = json.load(fh)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "path": "",
            "error": f"读取 manifest 失败: {exc}",
        }, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    for path in paths:
        print(json.dumps(_recognize(recognizer, inference_context, str(path)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
