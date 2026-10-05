# -*- coding: utf-8 -*-
"""Pure QSS color conversion helpers used by the optional dark theme."""
from __future__ import annotations

import colorsys
import re

THEME_LIGHT = "light"
THEME_DARK = "dark"
SUPPORTED_THEMES = (THEME_LIGHT, THEME_DARK)


def normalize_theme(value: str | None) -> str:
    value = str(value or "").strip().lower()
    return value if value in SUPPORTED_THEMES else THEME_LIGHT


_DARK_MAP = {
    "#FFFFFF": "#1B1F24", "#FBFCFE": "#171B21", "#F8FAFD": "#171A20",
    "#F8FAFC": "#171A20", "#F7F9FC": "#111318", "#F5F7FA": "#20252C",
    "#F4F6F8": "#242930", "#F3F5F8": "#20252C", "#F7F8FA": "#192535",
    "#F1F5FB": "#222A34", "#F0F5FC": "#242D38", "#EEF4FF": "#223146",
    "#EAF3FF": "#1C2D45", "#E7EEF8": "#293645", "#E3EDFF": "#263A55",
    "#E2EEFF": "#253A56", "#EFEFF3": "#292D34", "#E3E3E8": "#363B43",
    "#202733": "#F2F4F7", "#1D2939": "#F5F7FA", "#344054": "#D6DBE1",
    "#475467": "#C0C7D1", "#667085": "#AAB4C0", "#98A2B3": "#8F9AA8",
    "#A0A8B4": "#778392", "#AAB2BF": "#707C8B", "#666666": "#AAB4C0",
    "#666": "#AAB4C0", "#CFE2F8": "#344252", "#C7DCF4": "#3A4859",
    "#D7E7F8": "#333F4D", "#D5E6FF": "#3C4B60", "#B8D4F2": "#45576C",
    "#AFCBEA": "#4C6078", "#A9CBFA": "#4E6381", "#9EBEDE": "#52677D",
    "#91B9F5": "#5B77A0", "#7FB0F4": "#5E7FAE", "#BFD5F7": "#475B74",
    "#C8DEFF": "#455B78", "#DCEAFF": "#26476D", "#C5CCD6": "#596472",
    "#AAB3BF": "#687483", "#8E99A7": "#7B8795", "#B4B4BC": "#68727E",
    "#90909A": "#7B8795", "#1677FF": "#4C9AFF", "#0F6CE8": "#3E8BE6",
    "#0A5CC7": "#3479C8", "#FFF3F2": "#3B2528", "#FFE9E7": "#4A292D",
    "#F6C8C9": "#6B3A40", "#EEAAAD": "#805058", "#E5484D": "#FF6B70",
    "#ECFDF3": "#17372A", "#B7E4CC": "#315E49", "#16794E": "#5ED39A",
    "#EAF8F0": "#183529", "#B9E2CA": "#315A47", "#147A4A": "#62D69A",
}
from ui.theme.tokens import DARK_PAIRS as _NF_DARK_PAIRS  # noqa: E402
_DARK_MAP.update({k.upper(): v for k, v in _NF_DARK_PAIRS.items()})
_HEX_RE = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
_NAMED_COLOR_RE = re.compile(r"\b(?:white|black|gray|grey)\b", re.IGNORECASE)
_RGB_RE = re.compile(
    r"\brgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([\d.]+))?\s*\)",
    re.IGNORECASE,
)
_DECLARATION_RE = re.compile(r"(?P<prop>[A-Za-z-]+)(?P<sep>\s*:\s*)(?P<value>[^;}]+)")
_FOREGROUND_PROPERTIES = {
    "color", "selection-color", "placeholder-text-color",
}
_BACKGROUND_PROPERTIES = {
    "background", "background-color", "alternate-background-color",
    "selection-background-color",
}


def _parse_hex(token: str):
    raw = token.lstrip("#")
    alpha = ""
    if len(raw) == 8:
        raw, alpha = raw[:6], raw[6:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return None
    try:
        return tuple(int(raw[i:i+2], 16) for i in (0, 2, 4)), alpha
    except ValueError:
        return None


def _rgb_luminance(r: int, g: int, b: int) -> float:
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _fallback_dark_hex(token: str) -> str:
    """Legacy/general conversion used for borders and uncategorised values."""
    raw = token.upper()
    if raw in _DARK_MAP:
        return _DARK_MAP[raw]
    parsed = _parse_hex(raw)
    if parsed is None:
        return token
    (r, g, b), alpha = parsed
    mx, mn = max(r, g, b), min(r, g, b)
    saturation = 0 if mx == 0 else (mx - mn) / mx
    luminance = _rgb_luminance(r, g, b)
    if saturation > 0.35 and luminance < 0.75:
        factor = 1.18 if luminance < 0.45 else 1.08
        rr, gg, bb = (min(255, round(v * factor)) for v in (r, g, b))
        result = f"#{rr:02X}{gg:02X}{bb:02X}"
    elif luminance >= 0.90:
        result = "#1B1F24"
    elif luminance >= 0.72:
        result = "#354150"
    elif luminance <= 0.28:
        result = "#F2F4F7"
    else:
        result = "#AAB4C0"
    return result + alpha


def _foreground_dark_hex(token: str) -> str:
    """Convert a foreground colour without turning existing white text dark.

    The original converter inverted every hex token. That made declarations
    such as ``color:#FFFFFF; background:rgba(20,36,58,145)`` unreadable in dark
    mode. Foregrounds and surfaces need different semantics.
    """
    parsed = _parse_hex(token)
    if parsed is None:
        return token
    (r, g, b), alpha = parsed
    luminance = _rgb_luminance(r, g, b)
    if luminance >= 0.72:
        # Already suitable for a dark surface; keep hue and near-white intent.
        if max(r, g, b) - min(r, g, b) < 18 and luminance > 0.9:
            return "#F2F4F7" + alpha
        return f"#{r:02X}{g:02X}{b:02X}" + alpha
    return _fallback_dark_hex(token)


def _background_dark_hex(token: str) -> str:
    """Convert surfaces while keeping already-dark surfaces dark."""
    raw = token.upper()
    if raw in _DARK_MAP and _rgb_luminance(*_parse_hex(raw)[0]) >= 0.55:
        return _DARK_MAP[raw]
    parsed = _parse_hex(raw)
    if parsed is None:
        return token
    (r, g, b), alpha = parsed
    luminance = _rgb_luminance(r, g, b)
    if luminance <= 0.34:
        # Do not invert an intentionally dark overlay/surface into light.
        return f"#{r:02X}{g:02X}{b:02X}" + alpha
    if raw in _DARK_MAP:
        return _DARK_MAP[raw]
    if luminance >= 0.90:
        return "#1B1F24" + alpha
    if luminance >= 0.72:
        return "#2B333D" + alpha
    return "#303A46" + alpha


def _convert_hex_for_property(token: str, prop: str) -> str:
    prop = prop.lower()
    if prop in _FOREGROUND_PROPERTIES or prop.endswith("-color") and "background" not in prop and "border" not in prop:
        return _foreground_dark_hex(token)
    if prop in _BACKGROUND_PROPERTIES or "background" in prop:
        return _background_dark_hex(token)
    return _fallback_dark_hex(token)


def _rgb_to_hex(match: re.Match[str]) -> tuple[str, str | None]:
    values = [max(0, min(255, int(match.group(i)))) for i in (1, 2, 3)]
    return f"#{values[0]:02X}{values[1]:02X}{values[2]:02X}", match.group(4)


def _convert_named_for_property(token: str, prop: str) -> str:
    """Convert common named colours with the same foreground/surface semantics.

    Named ``white`` is widespread in legacy local QSS.  Leaving it untouched
    kept text readable but also left input fields and review panels glaring
    white in dark mode.  Handle only the unambiguous CSS colour keywords here;
    ``transparent`` and semantic names such as ``red`` remain untouched.
    """
    name = token.strip().lower()
    prop = prop.lower()
    foreground = prop in _FOREGROUND_PROPERTIES or (
        prop.endswith("-color") and "background" not in prop and "border" not in prop
    )
    background = prop in _BACKGROUND_PROPERTIES or "background" in prop
    if foreground:
        if name in {"white", "black"}:
            return "#F2F4F7"
        if name in {"gray", "grey"}:
            return "#AAB4C0"
    if background:
        if name == "white":
            return "#1B1F24"
        if name == "black":
            return "#000000"
        if name in {"gray", "grey"}:
            return "#303A46"
    if name == "white":
        return "#687483"
    if name == "black":
        return "#F2F4F7"
    if name in {"gray", "grey"}:
        return "#7B8795"
    return token


def _convert_value(value: str, prop: str) -> str:
    value = _HEX_RE.sub(lambda m: _convert_hex_for_property(m.group(0), prop), value)

    def repl_rgb(match: re.Match[str]) -> str:
        token, alpha = _rgb_to_hex(match)
        converted = _convert_hex_for_property(token, prop)
        parsed = _parse_hex(converted)
        if parsed is None:
            return match.group(0)
        (r, g, b), _ = parsed
        if alpha is None:
            return f"rgb({r},{g},{b})"
        return f"rgba({r},{g},{b},{alpha})"

    value = _RGB_RE.sub(repl_rgb, value)
    return _NAMED_COLOR_RE.sub(lambda m: _convert_named_for_property(m.group(0), prop), value)


def darken_stylesheet(style: str) -> str:
    """Convert QSS colours by declaration semantics.

    Layout, dimensions, selectors and non-colour values are preserved. Common
    named colours (white/black/gray/grey) are converted with the same
    foreground-versus-surface semantics as hexadecimal and rgb()/rgba() values.
    """
    if not style:
        return style

    def repl_decl(match: re.Match[str]) -> str:
        prop = match.group("prop")
        return f"{prop}{match.group('sep')}{_convert_value(match.group('value'), prop)}"

    # Handle normal declarations contextually. Any hex token outside a
    # declaration (rare in QSS) falls back to the legacy safe conversion.
    converted = _DECLARATION_RE.sub(repl_decl, style)
    return converted
