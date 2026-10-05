"""Backward-compatible source updater import.

Production UI imports :mod:`core.source_update` directly.  This shim remains so
older scripts using ``utils.source_update`` keep working when the local package
is selected.
"""
from __future__ import annotations

from core.source_update import *  # noqa: F401,F403
from core.source_update import _version_tuple  # compatibility for older tests/tools
