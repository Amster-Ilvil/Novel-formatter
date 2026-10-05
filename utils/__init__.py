"""Novel Formatter local utility package.

This explicit package marker prevents an unrelated third-party top-level
``utils`` package from shadowing the application's own utility modules when the
GUI is launched from a Finder/app shell or an environment with extra site-
packages entries.
"""
from __future__ import annotations

__all__ = []
