#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline kha-white/manga-ocr-base worker used only for disagreement review.

The worker never downloads model files.  The 424 MB checkpoint is supplied by
NOVEL_FORMATTER_MANGA_OCR_MODEL (or the project/persistent model cache), while
small official config/vocabulary files are bundled with Novel Formatter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

MANGA_OCR_SHA256 = "c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb"
SPECIAL_IDS = {0, 1, 2, 3, 4}


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_device(torch, requested: str):
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        requested = "auto"
    if requested in {"auto", "mps"} and getattr(torch.backends, "mps", None) is not None:
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        if requested == "mps":
            raise RuntimeError("Manga OCR 请求 MPS，但当前 PyTorch/Mac 环境不可用")
    if requested in {"auto", "cuda"}:
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:
            pass
        if requested == "cuda":
            raise RuntimeError("Manga OCR 请求 CUDA，但当前运行时不可用")
    return torch.device("cpu")


def _clean_decoded(text: str) -> str:
    # OCR evidence must remain model-faithful.  Only strip transport whitespace
    # and normalize canonical Unicode; do not silently rewrite punctuation.
    return unicodedata.normalize("NFC", "".join(str(text or "").split()))


class Recognizer:
    def __init__(self, weights: Path, metadata_dir: Path, device: str = "auto"):
        import torch
        from PIL import Image
        from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTImageProcessor

        self.torch = torch
        self.Image = Image
        threads = int(os.environ.get("NOVEL_FORMATTER_MANGA_OCR_THREADS", "0") or 0)
        if threads > 0:
            try:
                torch.set_num_threads(max(1, min(32, threads)))
                torch.set_num_interop_threads(max(1, min(8, threads // 2 or 1)))
            except Exception:
                pass
        self.device = _select_device(torch, device)
        self.weights = Path(weights)
        self.metadata_dir = Path(metadata_dir)
        digest = _sha256(self.weights)
        if digest != MANGA_OCR_SHA256:
            raise RuntimeError(
                "Manga OCR 权重 SHA256 不匹配；需要 kha-white/manga-ocr-base 原始 pytorch_model.bin，"
                f"当前={digest}"
            )

        config = VisionEncoderDecoderConfig.from_pretrained(
            self.metadata_dir, local_files_only=True
        )
        self.model = VisionEncoderDecoderModel(config)
        state = torch.load(self.weights, map_location="cpu", weights_only=True)
        self.checkpoint_key_count = len(state)
        self.dropped_compat_keys: list[str] = []
        for key in ("decoder.bert.embeddings.position_ids",):
            if key in state and key not in self.model.state_dict():
                state.pop(key)
                self.dropped_compat_keys.append(key)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.processor = ViTImageProcessor.from_pretrained(
            self.metadata_dir, local_files_only=True
        )
        self.vocab = (self.metadata_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        if len(self.vocab) != 6144:
            raise RuntimeError(f"Manga OCR vocab 应为 6144 项，实际 {len(self.vocab)}")

    def _decode(self, ids) -> str:
        parts: list[str] = []
        for raw in ids:
            idx = int(raw)
            if idx in SPECIAL_IDS:
                continue
            if 0 <= idx < len(self.vocab):
                parts.append(self.vocab[idx])
        return _clean_decoded("".join(parts))

    def recognize_one(self, path: str, *, max_length: int = 48, num_beams: int = 4) -> dict:
        image = self.Image.open(path)
        try:
            image = image.convert("L").convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            with self.torch.inference_mode():
                ids = self.model.generate(
                    pixel_values,
                    max_length=max(16, min(128, int(max_length))),
                    num_beams=max(1, min(4, int(num_beams))),
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                )[0]
            text = self._decode(ids.tolist())
            # The upstream model does not expose a calibrated OCR probability.
            # Keep this explicitly heuristic; local adjudication counts text as
            # independent evidence, not as a probability score.
            return {
                "path": str(path),
                "text": text,
                "confidence": 0.88 if text else 0.0,
                "confidence_kind": "heuristic",
            }
        finally:
            try:
                image.close()
            except Exception:
                pass

    def recognize(self, paths: list[str], *, max_length: int = 48, num_beams: int = 4) -> list[dict]:
        return [
            self.recognize_one(path, max_length=max_length, num_beams=num_beams)
            for path in paths
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--device", default=os.environ.get("NOVEL_FORMATTER_MANGA_OCR_DEVICE", "auto"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    try:
        recognizer = Recognizer(Path(args.weights), Path(args.metadata_dir), args.device)
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 2
    _emit({
        "ready": True,
        "device": str(recognizer.device),
        "load_seconds": time.perf_counter() - t0,
        "checkpoint_key_count": recognizer.checkpoint_key_count,
        "dropped_compat_keys": recognizer.dropped_compat_keys,
        "model_sha256": MANGA_OCR_SHA256,
    })

    if not args.stream:
        return 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "error": f"invalid-json: {exc}"})
            continue
        if request.get("command") == "close":
            return 0
        request_id = int(request.get("request_id", 0) or 0)
        paths = [str(item) for item in request.get("paths", [])]
        max_length = int(request.get("max_length", 48) or 48)
        num_beams = max(1, min(4, int(request.get("num_beams", 4) or 4)))
        try:
            items = recognizer.recognize(paths, max_length=max_length, num_beams=num_beams)
            _emit({"request_id": request_id, "ok": True, "items": items})
        except Exception as exc:
            _emit({
                "request_id": request_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
