"""Phase 22 reference-UI design system.

The design layer owns visual metrics and reusable widgets only.  Business
widgets keep their existing signals, state and controllers.
"""
from .metrics import *  # noqa: F401,F403
from .components import DesignCard, DesignNavButton, DesignSwitch, SectionLabel

__all__ = ["DesignCard", "DesignNavButton", "DesignSwitch", "SectionLabel"]
