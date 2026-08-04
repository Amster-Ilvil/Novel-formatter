#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent worker for Manga Image Translator 48px autoregressive OCR."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    from adapters.manga_48px_runtime import load_ocr_class
except ModuleNotFoundError:  # direct worker execution adds adapters/ to sys.path
    from manga_48px_runtime import load_ocr_class


def _device(torch):
    requested = os.environ.get(
        "NOVEL_FORMATTER_MANGA_48PX_DEVICE", "auto"
    ).strip().lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        requested = "auto"
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if os.environ.get("NOVEL_FORMATTER_MANGA_48PX_MPS", "1") != "0":
            try:
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps":
        try:
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                return torch.device("cpu")
        except Exception:
            return torch.device("cpu")
    return torch.device(requested)


def _torch_load_supports_weights_only(torch) -> bool:
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except Exception:
        return True


def _load_official_state_dict(torch, model_path: Path) -> dict:
    """Load the verified official checkpoint across PyTorch 2.3-2.x.

    PyTorch 2.6 defaults to ``weights_only=True`` and old checkpoints may be
    rejected. The checkpoint has already passed the official SHA-256 check, so
    an explicit legacy retry is safe. Older Torch builds that do not expose the
    ``weights_only`` argument are also supported.
    """
    errors: list[str] = []
    state = None
    if _torch_load_supports_weights_only(torch):
        for weights_only in (True, False):
            try:
                state = torch.load(
                    model_path,
                    map_location="cpu",
                    weights_only=weights_only,
                )
                break
            except Exception as exc:
                errors.append(f"weights_only={weights_only}: {exc}")
    else:
        try:
            state = torch.load(model_path, map_location="cpu")
        except Exception as exc:
            errors.append(f"legacy torch.load: {exc}")

    if state is None:
        raise RuntimeError(
            "官方 48px checkpoint 无法反序列化。" + " | ".join(errors)
        )

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict) and nested:
                state = nested
                break
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"官方 48px checkpoint 格式异常：{type(state).__name__}")

    normalized = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("model.", "module.", "_orig_mod.", "net."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    return normalized


def _prepare_image(path: str) -> tuple[np.ndarray, int, str]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    orientation = "horizontal"
    try:
        # The upstream model consumes horizontal strips. Rotate a physical
        # Japanese vertical column counter-clockwise so top-to-bottom becomes
        # left-to-right without reversing glyph order.
        if image.height > image.width * 1.20:
            image = image.transpose(Image.Transpose.ROTATE_90)
            orientation = "vertical-rotated-ccw"
        ratio = image.width / max(1.0, float(image.height))
        width = max(4, int(round(ratio * 48)))
        resized = image.resize((width, 48), Image.Resampling.LANCZOS)
        try:
            return np.asarray(resized, dtype=np.uint8), width, orientation
        finally:
            resized.close()
    finally:
        image.close()


def _decode(dictionary: list[str], item) -> tuple[str, float, list[int], list[int]]:
    indices, probability, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = item
    chars: list[str] = []
    for index in indices:
        token_index = int(index)
        if token_index < 0 or token_index >= len(dictionary):
            raise RuntimeError(f"48px OCR 返回越界字符索引：{token_index}")
        token = dictionary[token_index]
        if token == "<S>":
            continue
        if token == "</S>":
            break
        chars.append(" " if token == "<SP>" else token)

    def colour(pred, indicator):
        try:
            present = bool(indicator[1] > indicator[0])
            values = pred.detach().float().cpu().tolist()
            if not present:
                return []
            return [
                max(0, min(255, int(round(value * 255))))
                for value in values[:3]
            ]
        except Exception:
            return []

    return (
        "".join(chars),
        float(probability or 0.0),
        colour(fg_pred, fg_ind_pred),
        colour(bg_pred, bg_ind_pred),
    )


class Recognizer:
    def __init__(self, cache_dir: Path):
        import torch
        import einops  # noqa: F401 - dependency checked before model load

        try:
            threads = int(
                os.environ.get("NOVEL_FORMATTER_MANGA_48PX_THREADS", "0") or 0
            )
        except ValueError:
            threads = 0
        torch.set_num_threads(threads or min(8, max(1, os.cpu_count() or 4)))

        OCR, model_path, dict_path = load_ocr_class(cache_dir)
        with dict_path.open("r", encoding="utf-8-sig") as fh:
            self.dictionary = [line.rstrip("\r\n") for line in fh]
        if len(self.dictionary) < 1000:
            raise RuntimeError(
                f"48px 字符表异常，仅有 {len(self.dictionary)} 个条目"
            )
        # Beam decoding uses fixed official indices: pad=0, start=1, end=2.
        # Do not require a particular spelling for the padding token itself.
        if self.dictionary[1:3] != ["<S>", "</S>"]:
            raise RuntimeError(
                "48px 字符表起始/结束 token 与官方模型不匹配："
                + repr(self.dictionary[:3])
            )

        self.model = OCR(self.dictionary, 768)
        state = _load_official_state_dict(torch, model_path)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            model_keys = set(self.model.state_dict())
            state_keys = set(state)
            missing = sorted(model_keys - state_keys)[:12]
            unexpected = sorted(state_keys - model_keys)[:12]
            raise RuntimeError(
                "48px checkpoint 与固定版本网络结构不匹配。"
                f" missing={missing}; unexpected={unexpected}; detail={exc}"
            ) from exc

        self.model.eval()
        self.device = _device(torch)
        self.device_fallback_reason = ""
        try:
            self.model.to(self.device)
        except Exception as exc:
            if self.device.type == "cpu":
                raise
            self.device_fallback_reason = (
                f"{self.device} 模型加载失败，已回退 CPU：{exc}"
            )
            self.device = torch.device("cpu")
            self.model.to(self.device)
        self.torch = torch

    def _fallback_to_cpu(self, reason: Exception) -> None:
        if self.device.type == "cpu":
            raise reason
        self.device_fallback_reason = (
            f"{self.device} 推理失败，已回退 CPU：{reason}"
        )
        self.device = self.torch.device("cpu")
        self.model.to(self.device)
        try:
            if hasattr(self.torch, "mps") and hasattr(self.torch.mps, "empty_cache"):
                self.torch.mps.empty_cache()
        except Exception:
            pass

    def recognize(
        self,
        paths: list[str],
        beams_k: int = 5,
        max_seq_length: int = 255,
    ) -> list[dict]:
        prepared = []
        for path in paths:
            array, width, orientation = _prepare_image(path)
            prepared.append((path, array, width, orientation))
        output: list[dict] = []
        for offset in range(0, len(prepared), 16):
            batch = prepared[offset:offset + 16]
            widths = [item[2] for item in batch]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((len(batch), 48, max_width, 3), dtype=np.uint8)
            for index, (_, array, width, _) in enumerate(batch):
                region[index, :, :width, :] = array
            tensor = (self.torch.from_numpy(region).float() - 127.5) / 127.5
            tensor = tensor.permute(0, 3, 1, 2).to(self.device)
            kwargs = {
                "beams_k": max(1, min(8, int(beams_k))),
                "max_seq_length": max(8, min(384, int(max_seq_length))),
            }
            try:
                with self.torch.inference_mode():
                    recognized = self.model.infer_beam_batch_tensor(
                        tensor, widths, **kwargs
                    )
            except RuntimeError as exc:
                if self.device.type == "cpu":
                    raise
                self._fallback_to_cpu(exc)
                tensor = tensor.to(self.device)
                with self.torch.inference_mode():
                    recognized = self.model.infer_beam_batch_tensor(
                        tensor, widths, **kwargs
                    )
            if len(recognized) != len(batch):
                raise RuntimeError(
                    f"48px OCR 批量返回数量异常：输入 {len(batch)}，返回 {len(recognized)}"
                )
            for (path, _, width, orientation), item in zip(batch, recognized):
                text, probability, fg, bg = _decode(self.dictionary, item)
                output.append(
                    {
                        "path": path,
                        "text": text,
                        "confidence": probability,
                        "foreground": fg,
                        "background": bg,
                        "input_width": width,
                        "orientation": orientation,
                    }
                )
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    try:
        recognizer = Recognizer(Path(args.cache_dir))
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(
            json.dumps(
                {
                    "ok": False,
                    "ready": False,
                    "error": f"48px AR OCR 初始化失败: {detail}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": True,
                "ready": True,
                "device": str(recognizer.device),
                "model": "Manga Image Translator 48px AR",
                "input_contract": "48px-horizontal-strip; vertical crops auto-rotated",
                "fallback": recognizer.device_fallback_reason,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = {}
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                print(
                    json.dumps({"ok": True, "closed": True}, ensure_ascii=False),
                    flush=True,
                )
                return
            paths = [str(path) for path in request.get("paths", []) if str(path)]
            if not paths and request.get("path"):
                paths = [str(request["path"])]
            if not paths:
                raise ValueError("缺少 path/paths")
            items = recognizer.recognize(
                paths,
                beams_k=int(request.get("beams_k", 5) or 5),
                max_seq_length=int(request.get("max_seq_length", 255) or 255),
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "request_id": request.get("request_id"),
                        "items": items,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": request.get("request_id")
                        if isinstance(request, dict)
                        else None,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
