# scikitplot/_brand/_kit/_color.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Colours for brand assets: parsing, tonal scales and contrast.

Everything here is pure arithmetic on sRGB hex colours, with the standard
library only, so that every result is the same on every machine.

See Also
--------
scikitplot._brand._kit._svg : writes the colours into SVG documents.

Notes
-----
**User notes:** a colour is written ``#RRGGBB`` (or ``#RGB``); the result of
every function is the upper-case ``#RRGGBB`` form. :func:`readable` returns
the colour itself when it already meets a contrast target, so brand colours
are kept wherever they work.

**Developer notes:** contrast follows the WCAG 2.x definitions (relative
luminance of linearised sRGB, ratio ``(L1 + 0.05) / (L2 + 0.05)``). Scales mix
in sRGB toward white and black with fixed amounts, as design-token scales
usually do; step ``500`` is the colour itself.

References
----------
.. [1] W3C, "Web Content Accessibility Guidelines (WCAG) 2.2", relative
   luminance and contrast ratio, https://www.w3.org/TR/WCAG22/#dfn-contrast-ratio
"""

from __future__ import annotations

import re
from typing import Mapping

__all__ = [
    "SCALE_STEPS",
    "ContrastError",
    "contrast_ratio",
    "mix",
    "normalize_hex",
    "readable",
    "relative_luminance",
    "tonal_scale",
]

#: Steps of a tonal scale, lightest first. ``500`` is the base colour.
SCALE_STEPS: tuple[int, ...] = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)

# Share of white mixed in for the light steps, and of black for the dark ones.
_TINTS: Mapping[int, float] = {50: 0.94, 100: 0.86, 200: 0.72, 300: 0.54, 400: 0.30}
_SHADES: Mapping[int, float] = {600: 0.12, 700: 0.24, 800: 0.38, 900: 0.54, 950: 0.68}

_HEX = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")


class ContrastError(ValueError):
    """
    No colour of a scale reaches the contrast that was asked for.

    Notes
    -----
    **User notes:** lower the target only if the element is decorative, or
    choose another background; the message names the best ratio reached.
    """


def normalize_hex(color: str) -> str:
    """
    Return a colour as upper-case ``#RRGGBB``.

    Parameters
    ----------
    color : str
        ``#RGB`` or ``#RRGGBB``.

    Returns
    -------
    str
        The same colour as ``#RRGGBB``, upper case.

    Raises
    ------
    TypeError
        If ``color`` is not a string.
    ValueError
        If ``color`` is not a hex colour of that form. Named colours and
        alpha are refused so that the result is exact.

    Examples
    --------
    >>> normalize_hex("#1e90ff")
    '#1E90FF'
    >>> normalize_hex("#fff")
    '#FFFFFF'
    """
    if not isinstance(color, str):
        raise TypeError(f"colour must be a string, got {type(color).__name__}")
    if not _HEX.match(color):
        raise ValueError(f"expected a colour '#RRGGBB' or '#RGB', got {color!r}")
    digits = color[1:]
    if len(digits) == 3:  # noqa: PLR2004
        digits = "".join(ch * 2 for ch in digits)
    return "#" + digits.upper()


def _rgb(color: str) -> tuple[int, int, int]:
    digits = normalize_hex(color)[1:]
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def _hex(rgb: tuple[float, float, float]) -> str:
    # round-half-even of Python's round() is exact and the same everywhere.
    return "#" + "".join(f"{min(255, max(0, round(v))):02X}" for v in rgb)


def mix(color: str, other: str, amount: float) -> str:
    """
    Mix two colours in sRGB.

    Parameters
    ----------
    color, other : str
        Hex colours.
    amount : float
        Share of ``other``, from 0 (``color``) to 1 (``other``).

    Returns
    -------
    str
        The mixed colour, ``#RRGGBB``.

    Raises
    ------
    ValueError
        If ``amount`` is outside [0, 1] or a colour is not hex.

    Examples
    --------
    >>> mix("#000000", "#FFFFFF", 0.5)
    '#808080'
    """
    amount = float(amount)
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}")
    a, b = _rgb(color), _rgb(other)
    return _hex(tuple(x + (y - x) * amount for x, y in zip(a, b)))


def tonal_scale(color: str) -> dict[int, str]:
    """
    Return the 50 to 950 tonal scale of a colour.

    Parameters
    ----------
    color : str
        The base colour; it is step ``500`` exactly.

    Returns
    -------
    dict of int to str
        Step to ``#RRGGBB``, lightest first (:data:`SCALE_STEPS`).

    Notes
    -----
    **User notes:** lower steps mix toward white, higher steps toward black.
    The scale is a design-token scale: it says nothing about contrast; use
    :func:`readable` for that.

    Examples
    --------
    >>> tonal_scale("#1E90FF")[500]
    '#1E90FF'
    """
    base = normalize_hex(color)
    scale: dict[int, str] = {}
    for step in SCALE_STEPS:
        if step in _TINTS:
            scale[step] = mix(base, "#FFFFFF", _TINTS[step])
        elif step in _SHADES:
            scale[step] = mix(base, "#000000", _SHADES[step])
        else:
            scale[step] = base
    return scale


# sRGB transfer function: the knee below which the curve is linear.
_SRGB_KNEE = 0.04045
_BASE_STEP = 500


def _linear(channel: float) -> float:
    """Linearise one sRGB channel in [0, 1]."""
    if channel <= _SRGB_KNEE:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    """
    Return the WCAG relative luminance of a colour, from 0 to 1.

    Parameters
    ----------
    color : str
        Hex colour.

    Returns
    -------
    float
        Relative luminance.

    References
    ----------
    .. [1] https://www.w3.org/TR/WCAG22/#dfn-relative-luminance
    """
    channels = [_linear(value / 255.0) for value in _rgb(color)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(color: str, background: str) -> float:
    """
    Return the WCAG contrast ratio of two colours, from 1 to 21.

    Parameters
    ----------
    color, background : str
        Hex colours; the order does not matter.

    Returns
    -------
    float
        ``(L_lighter + 0.05) / (L_darker + 0.05)``.

    Examples
    --------
    >>> round(contrast_ratio("#000000", "#FFFFFF"), 1)
    21.0
    """
    high, low = sorted(
        (relative_luminance(color), relative_luminance(background)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def readable(color: str, background: str, minimum: float) -> str:
    """
    Return the colour, or the nearest step of its scale, that reaches a contrast.

    Parameters
    ----------
    color : str
        The wanted colour (step ``500`` of its scale).
    background : str
        The colour it is drawn on.
    minimum : float
        The contrast ratio to reach, from 1 to 21: 3 for graphics and large
        text, 4.5 for small text (WCAG 2.x, level AA).

    Returns
    -------
    str
        ``color`` itself if it reaches ``minimum``; otherwise the first step
        of its scale that does, walking away from the background (toward 950
        on a background lighter than the colour, toward 50 otherwise).

    Raises
    ------
    ContrastError
        If no step reaches ``minimum``. The message names the best step.
    ValueError
        If ``minimum`` is outside [1, 21].

    Notes
    -----
    **User notes:** the rule changes a brand colour only where it does not
    work, and then by the smallest step that does, so the result stays in the
    colour's family.

    **Developer notes:** the direction is decided by luminance alone, so the
    same inputs always give the same step.

    Examples
    --------
    >>> readable("#202729", "#F7F6F3", 3.0)
    '#202729'
    >>> readable("#B58B58", "#F7F6F3", 3.0)
    '#9F7A4D'
    """
    minimum = float(minimum)
    if not 1.0 <= minimum <= 21.0:  # noqa: PLR2004
        raise ValueError(f"minimum contrast must be in [1, 21], got {minimum}")
    base = normalize_hex(color)
    if contrast_ratio(base, background) >= minimum:
        return base
    scale = tonal_scale(base)
    darker = relative_luminance(background) > relative_luminance(base)
    steps = [s for s in SCALE_STEPS if (s > _BASE_STEP if darker else s < _BASE_STEP)]
    if not darker:
        steps.reverse()
    best = (contrast_ratio(base, background), _BASE_STEP)
    for step in steps:
        ratio = contrast_ratio(scale[step], background)
        if ratio >= minimum:
            return scale[step]
        best = max(best, (ratio, step))
    raise ContrastError(
        f"no step of the scale of {base} reaches {minimum:g}:1 on {normalize_hex(background)}; "
        f"the best is step {best[1]} ({scale[best[1]]}) at {best[0]:.2f}:1"
    )
