"""Stable whole-run progress and ETA estimation for OCR pipelines.

The OCR UI has several nested progress domains (models, page splitting, page
recognition, sentence re-OCR and final multi-model alignment).  This module
maps them onto one monotonic 0..1 timeline without changing any OCR execution
order or recognition result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import wraps
import math
import threading
import time
from typing import Callable, Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class OCRProgressSnapshot:
    fraction: float
    percent: float
    elapsed_seconds: float
    eta_seconds: Optional[float]
    label: str
    current: int
    total: int
    unit: str


def _synchronized(method):
    """Serialize estimator mutations from parallel OCR model callbacks."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class OCRProgressEstimator:
    """Aggregate nested OCR phases into a monotonic whole-run progress value."""

    _PHASE_ORDER = ("init", "split", "recognition", "sentence_reocr")

    def __init__(
        self,
        *,
        model_count: int,
        use_column_mask: bool,
        sentence_context_reocr: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_count = max(1, int(model_count or 1))
        self.use_column_mask = bool(use_column_mask)
        self.sentence_context_reocr = bool(sentence_context_reocr and use_column_mask)
        self._clock = clock
        self._lock = threading.RLock()
        self._started_at = float(clock())
        self._last_fraction = 0.0
        self._eta_ema: Optional[float] = None
        self._samples: Deque[Tuple[float, float]] = deque([(self._started_at, 0.0)], maxlen=256)

        # Keep a small final-stage reserve so the bar never reaches 100% before
        # alignment/post-processing really finishes.
        self._final_weight = 0.04 if self.model_count > 1 else 0.02
        self._model_region_weight = 1.0 - self._final_weight
        self._final_fraction = 0.0

        if not self.use_column_mask:
            self._phase_weights = {
                "init": 0.06,
                "recognition": 0.94,
            }
        elif self.sentence_context_reocr:
            self._phase_weights = {
                "init": 0.06,
                "split": 0.17,
                "recognition": 0.55,
                "sentence_reocr": 0.22,
            }
        else:
            self._phase_weights = {
                "init": 0.06,
                "split": 0.24,
                "recognition": 0.70,
            }

        self._models: list[Dict[str, float]] = [
            {phase: 0.0 for phase in self._phase_weights}
            for _ in range(self.model_count)
        ]

    @staticmethod
    def format_duration(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
            return "计算中"
        whole = max(0, int(round(float(seconds))))
        if whole < 60:
            return f"{whole}秒"
        minutes, sec = divmod(whole, 60)
        if minutes < 60:
            return f"{minutes}分{sec:02d}秒"
        hours, minute = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}小时{minute:02d}分"
        days, hour = divmod(hours, 24)
        return f"{days}天{hour:02d}小时"

    @_synchronized
    def start_model(self, model_index: int, *, label: str = "模型加载") -> OCRProgressSnapshot:
        index = self._clamp_model_index(model_index)
        if "init" in self._models[index]:
            # A tiny non-zero marker shows that work started while still leaving
            # almost all initialization weight for the first real callback.
            self._models[index]["init"] = max(self._models[index]["init"], 0.08)
        return self._snapshot(label=label, current=0, total=1, unit="阶段")

    @_synchronized
    def update_phase(
        self,
        model_index: int,
        phase: str,
        current: int,
        total: int,
        *,
        label: str,
        unit: str,
    ) -> OCRProgressSnapshot:
        index = self._clamp_model_index(model_index)
        phase_key = str(phase or "recognition")
        if phase_key not in self._models[index]:
            # Some engines only report generic page progress.  Treat it as the
            # recognition phase rather than creating an unweighted phase.
            phase_key = "recognition"
        total_i = max(1, int(total or 1))
        current_i = max(0, min(int(current or 0), total_i))
        ratio = current_i / total_i

        phases = self._models[index]
        if "init" in phases:
            phases["init"] = 1.0

        # Once a later phase starts, earlier phases are known to be complete even
        # if an adapter omitted its final callback.
        try:
            order_index = self._PHASE_ORDER.index(phase_key)
        except ValueError:
            order_index = -1
        if order_index >= 0:
            for earlier in self._PHASE_ORDER[:order_index]:
                if earlier in phases:
                    phases[earlier] = 1.0

        phases[phase_key] = max(phases.get(phase_key, 0.0), ratio)
        return self._snapshot(
            label=label,
            current=current_i,
            total=total_i,
            unit=unit,
        )

    @_synchronized
    def complete_model(self, model_index: int, *, label: str) -> OCRProgressSnapshot:
        index = self._clamp_model_index(model_index)
        for phase in self._models[index]:
            self._models[index][phase] = 1.0
        return self._snapshot(label=label, current=1, total=1, unit="阶段")

    @_synchronized
    def update_final_stage(
        self,
        current: int,
        total: int,
        *,
        label: str,
        unit: str = "阶段",
    ) -> OCRProgressSnapshot:
        total_i = max(1, int(total or 1))
        current_i = max(0, min(int(current or 0), total_i))
        self._final_fraction = max(self._final_fraction, current_i / total_i)
        return self._snapshot(
            label=label,
            current=current_i,
            total=total_i,
            unit=unit,
        )

    @_synchronized
    def complete(self, *, label: str = "OCR 完成") -> OCRProgressSnapshot:
        for index in range(self.model_count):
            for phase in self._models[index]:
                self._models[index][phase] = 1.0
        self._final_fraction = 1.0
        self._last_fraction = 1.0
        now = float(self._clock())
        self._append_sample(now, 1.0)
        return OCRProgressSnapshot(
            fraction=1.0,
            percent=100.0,
            elapsed_seconds=max(0.0, now - self._started_at),
            eta_seconds=0.0,
            label=label,
            current=1,
            total=1,
            unit="阶段",
        )

    def _clamp_model_index(self, model_index: int) -> int:
        return max(0, min(int(model_index or 0), self.model_count - 1))

    def _model_fraction(self, model_index: int) -> float:
        phases = self._models[model_index]
        value = 0.0
        for phase, weight in self._phase_weights.items():
            value += float(weight) * max(0.0, min(1.0, phases.get(phase, 0.0)))
        return max(0.0, min(1.0, value))

    def _raw_fraction(self) -> float:
        model_average = sum(self._model_fraction(i) for i in range(self.model_count)) / self.model_count
        return (
            self._model_region_weight * model_average
            + self._final_weight * max(0.0, min(1.0, self._final_fraction))
        )

    def _append_sample(self, now: float, fraction: float) -> None:
        if self._samples and abs(self._samples[-1][1] - fraction) < 1e-9:
            return
        self._samples.append((now, fraction))
        cutoff = now - 90.0
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.popleft()

    def _estimate_eta(self, now: float, fraction: float) -> Optional[float]:
        elapsed = max(0.0, now - self._started_at)
        if fraction >= 0.999999:
            self._eta_ema = 0.0
            return 0.0
        if elapsed < 5.0 or fraction < 0.01:
            return None

        overall_rate = fraction / max(elapsed, 1e-6)
        recent_rate = overall_rate
        if len(self._samples) >= 2:
            first_t, first_f = self._samples[0]
            last_t, last_f = self._samples[-1]
            dt = last_t - first_t
            df = last_f - first_f
            if dt >= 3.0 and df > 0:
                recent_rate = df / dt

        # Blend whole-run and recent rates.  Clamp the recent rate so a tiny fast
        # stage cannot make the displayed ETA collapse unrealistically.
        recent_rate = max(overall_rate * 0.35, min(recent_rate, overall_rate * 2.5))
        rate = overall_rate * 0.45 + recent_rate * 0.55
        if rate <= 1e-9:
            return None
        raw_eta = max(0.0, (1.0 - fraction) / rate)
        if self._eta_ema is None:
            self._eta_ema = raw_eta
        else:
            self._eta_ema = self._eta_ema * 0.75 + raw_eta * 0.25
        return self._eta_ema

    def _snapshot(self, *, label: str, current: int, total: int, unit: str) -> OCRProgressSnapshot:
        now = float(self._clock())
        raw = self._raw_fraction()
        fraction = max(self._last_fraction, min(1.0, raw))
        self._last_fraction = fraction
        self._append_sample(now, fraction)
        eta = self._estimate_eta(now, fraction)
        return OCRProgressSnapshot(
            fraction=fraction,
            percent=fraction * 100.0,
            elapsed_seconds=max(0.0, now - self._started_at),
            eta_seconds=eta,
            label=str(label or "OCR 处理中"),
            current=max(0, int(current or 0)),
            total=max(1, int(total or 1)),
            unit=str(unit or "项"),
        )
