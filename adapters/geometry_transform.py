# -*- coding: utf-8 -*-
"""Small dependency-free 2D affine transform contract for OCR geometry.

Matrices map homogeneous column vectors ``[x, y, 1]`` in pixel coordinates.
``A.compose(B)`` means "apply B first, then A".  Keeping this contract outside
individual OCR adapters prevents crop/scale/rotate paths from inventing subtly
incompatible inverse-coordinate formulae.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class AffineMatrix:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 1.0
    f: float = 0.0

    @classmethod
    def identity(cls) -> "AffineMatrix":
        return cls()

    @classmethod
    def translation(cls, tx: float, ty: float) -> "AffineMatrix":
        return cls(1.0, 0.0, float(tx), 0.0, 1.0, float(ty))

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> "AffineMatrix":
        sy = sx if sy is None else sy
        return cls(float(sx), 0.0, 0.0, 0.0, float(sy), 0.0)

    @classmethod
    def rotation(cls, degrees: float) -> "AffineMatrix":
        radians = math.radians(float(degrees))
        cosine = math.cos(radians)
        sine = math.sin(radians)
        return cls(cosine, -sine, 0.0, sine, cosine, 0.0)

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.a * float(x) + self.b * float(y) + self.c,
            self.d * float(x) + self.e * float(y) + self.f,
        )

    def map_points(self, points: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
        return [self.map_point(float(point[0]), float(point[1])) for point in points]

    def map_bbox(self, bbox: Sequence[float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (float(value) for value in bbox[:4])
        points = self.map_points(((x0, y0), (x1, y0), (x0, y1), (x1, y1)))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def compose(self, other: "AffineMatrix") -> "AffineMatrix":
        """Return ``self ∘ other`` (other is applied first)."""
        return AffineMatrix(
            self.a * other.a + self.b * other.d,
            self.a * other.b + self.b * other.e,
            self.a * other.c + self.b * other.f + self.c,
            self.d * other.a + self.e * other.d,
            self.d * other.b + self.e * other.e,
            self.d * other.c + self.e * other.f + self.f,
        )

    def inverse(self) -> "AffineMatrix":
        determinant = self.a * self.e - self.b * self.d
        if abs(determinant) < 1e-12:
            raise ValueError("仿射矩阵不可逆")
        inv = 1.0 / determinant
        a = self.e * inv
        b = -self.b * inv
        d = -self.d * inv
        e = self.a * inv
        c = -(a * self.c + b * self.f)
        f = -(d * self.c + e * self.f)
        return AffineMatrix(a, b, c, d, e, f)

    def to_list(self) -> list[float]:
        return [self.a, self.b, self.c, self.d, self.e, self.f]

    @classmethod
    def from_value(cls, value) -> "AffineMatrix":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            if "matrix" in value:
                value = value["matrix"]
            else:
                return cls(*(float(value.get(key, default)) for key, default in (
                    ("a", 1), ("b", 0), ("c", 0), ("d", 0), ("e", 1), ("f", 0)
                )))
        if isinstance(value, (list, tuple)) and len(value) >= 6:
            return cls(*(float(item) for item in value[:6]))
        raise ValueError("无效的仿射矩阵")
