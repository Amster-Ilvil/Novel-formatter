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

# Direct worker execution uses adapters/ as sys.path[0]; pin the project root
# first so the local utils package cannot be shadowed by another installed package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageOps

try:
    from adapters.manga_48px_runtime import load_ocr_class
except ModuleNotFoundError:  # direct worker execution adds adapters/ to sys.path
    from manga_48px_runtime import load_ocr_class

from utils.apple_silicon_runtime import recommended_manga_batch, configure_torch_memory_safety, probe_torch_mps
from utils.hardware_runtime import current_available_memory_gb, memory_budget_gb


def _device(torch, *, mps_probe: tuple[bool, str] | None = None):
    requested = os.environ.get(
        "NOVEL_FORMATTER_MANGA_48PX_DEVICE", "auto"
    ).strip().lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        requested = "auto"
    mps_ok, _mps_detail = mps_probe if mps_probe is not None else probe_torch_mps(torch)
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if os.environ.get("NOVEL_FORMATTER_MANGA_48PX_MPS", "1") != "0" and mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not mps_ok:
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
        configure_torch_memory_safety(torch)

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
        self._learned_batch_cap = 0
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
        requested_device = os.environ.get("NOVEL_FORMATTER_MANGA_48PX_DEVICE", "auto").strip().lower()
        self.mps_probe_ok, self.mps_probe_detail = probe_torch_mps(torch)
        self.device = _device(torch, mps_probe=(self.mps_probe_ok, self.mps_probe_detail))
        self.device_fallback_reason = ""
        if self.device.type == "cpu" and requested_device in {"auto", "mps"}:
            try:
                if torch.backends.mps.is_built() and not self.mps_probe_ok:
                    self.device_fallback_reason = (
                        "MPS 实测不可用，已使用 CPU：" + self.mps_probe_detail
                        + "。请在设置中运行设备检测；Apple Silicon 上旧 x86_64/Rosetta venv 会自动重建。"
                    )
            except Exception:
                pass
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

    def _batch_groups(self, prepared: list[tuple[str, np.ndarray, int, str]]) -> list[list[tuple[str, np.ndarray, int, str]]]:
        """Bucket 48px strips by width so padding waste does not dominate M6 GPU time."""
        if not prepared:
            return []
        target = max(1, recommended_manga_batch()) if self.device.type != "cpu" else min(8, max(2, recommended_manga_batch()))
        learned_cap = int(getattr(self, "_learned_batch_cap", 0) or 0)
        if learned_cap:
            target = min(target, learned_cap)
        budget = max(1.5, float(memory_budget_gb()))
        # Pixel budget follows live OCR memory headroom.  The coefficient is a
        # conservative tensor-size envelope, not a fixed RAM tier.
        pixel_budget = int(max(260_000, min(1_200_000, budget * 180_000)))
        available = current_available_memory_gb()
        if available and available < 2.5:
            target = min(target, 2)
            pixel_budget = min(pixel_budget, 320_000)
        # Very wide rotated vertical columns create large padded tensors. Keep the
        # batch count high for ordinary novel columns, but shrink it automatically
        # before a single extreme column forces a large unified-memory allocation.
        ordered = sorted(prepared, key=lambda item: (int(item[2]), str(item[0])))
        groups: list[list[tuple[str, np.ndarray, int, str]]] = []
        current: list[tuple[str, np.ndarray, int, str]] = []
        current_max_width = 0
        for item in ordered:
            width = max(4, int(item[2]))
            next_max = max(current_max_width, width)
            next_count = len(current) + 1
            estimated_pixels = next_count * 48 * next_max
            if current and (next_count > target or estimated_pixels > pixel_budget):
                groups.append(current)
                current = []
                current_max_width = 0
            current.append(item)
            current_max_width = max(current_max_width, width)
        if current:
            groups.append(current)
        return groups

    def _recognize_prepared_part(self, batch, kwargs):
        widths = [item[2] for item in batch]
        max_width = 4 * (max(widths) + 7) // 4
        region = np.zeros((len(batch), 48, max_width, 3), dtype=np.uint8)
        for index, (_, array, width, _) in enumerate(batch):
            region[index, :, :width, :] = array
        tensor = (self.torch.from_numpy(region).float() - 127.5) / 127.5
        tensor = tensor.permute(0, 3, 1, 2).to(self.device)
        try:
            with self.torch.inference_mode():
                recognized = self.model.infer_beam_batch_tensor(tensor, widths, **kwargs)
            self._learned_batch_cap = max(self._learned_batch_cap, len(batch))
            return recognized
        except RuntimeError as exc:
            self._release_memory()
            if len(batch) > 1:
                self._learned_batch_cap = max(1, len(batch) // 2)
                try:
                    from utils.ocr_runtime_calibration import record
                    record("48px", batch_size=len(batch), elapsed_seconds=0.001, items=len(batch), available_memory_gb=current_available_memory_gb(), success=False, resource_error=True)
                except Exception:
                    pass
                mid = max(1, len(batch) // 2)
                result = []
                result.extend(self._recognize_prepared_part(batch[:mid], kwargs))
                result.extend(self._recognize_prepared_part(batch[mid:], kwargs))
                return result
            if self.device.type != "cpu":
                self._fallback_to_cpu(exc)
                return self._recognize_prepared_part(batch, kwargs)
            raise

    def _release_memory(self):
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            if self.device.type == "mps" and hasattr(self.torch, "mps") and hasattr(self.torch.mps, "empty_cache"):
                self.torch.mps.empty_cache()
            elif self.device.type == "cuda" and hasattr(self.torch, "cuda"):
                self.torch.cuda.empty_cache()
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
        recognized_by_path: dict[str, dict] = {}
        kwargs = {
            "beams_k": max(1, min(8, int(beams_k))),
            "max_seq_length": max(8, min(384, int(max_seq_length))),
        }
        for batch in self._batch_groups(prepared):
            widths = [item[2] for item in batch]
            def _run_current(batch_items, batch_widths):
                region_local = np.zeros((len(batch_items), 48, 4 * ((max(batch_widths) + 7) // 4), 3), dtype=np.uint8)
                for local_index, (_, local_array, local_width, _) in enumerate(batch_items):
                    region_local[local_index, :, :local_width, :] = local_array
                tensor_local = (self.torch.from_numpy(region_local).float() - 127.5) / 127.5
                tensor_local = tensor_local.permute(0, 3, 1, 2).to(self.device)
                with self.torch.inference_mode():
                    result_local = self.model.infer_beam_batch_tensor(tensor_local, batch_widths, **kwargs)
                return result_local

            try:
                recognized = _run_current(batch, widths)
                self._learned_batch_cap = max(self._learned_batch_cap, len(batch))
            except RuntimeError as exc:
                if len(batch) > 1:
                    self._learned_batch_cap = max(1, len(batch) // 2)
                    self._release_memory()
                    mid = max(1, len(batch) // 2)
                    split_recognized = []
                    for part in (batch[:mid], batch[mid:]):
                        if not part:
                            continue
                        split_recognized.extend(self._recognize_prepared_part(part, kwargs))
                    recognized = split_recognized
                else:
                    if self.device.type == "cpu":
                        raise
                    self._fallback_to_cpu(exc)
                    self._release_memory()
                    recognized = _run_current(batch, widths)
            if len(recognized) != len(batch):
                raise RuntimeError(
                    f"48px OCR 批量返回数量异常：输入 {len(batch)}，返回 {len(recognized)}"
                )
            for (path, _, width, orientation), item in zip(batch, recognized):
                text, probability, fg, bg = _decode(self.dictionary, item)
                recognized_by_path[str(path)] = {
                    "path": path,
                    "text": text,
                    "confidence": probability,
                    "foreground": fg,
                    "background": bg,
                    "input_width": width,
                    "orientation": orientation,
                }
        # Restore physical-column input order exactly; bucketing is invisible to
        # the caller and therefore cannot reorder Japanese reading flow.
        return [recognized_by_path[str(path)] for path in paths if str(path) in recognized_by_path]


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
                "torch_version": str(getattr(recognizer.torch, "__version__", "")),
                "mps_probe_ok": bool(getattr(recognizer, "mps_probe_ok", False)),
                "mps_probe_detail": str(getattr(recognizer, "mps_probe_detail", "")),
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
