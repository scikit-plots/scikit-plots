# scikitplot/_brand/_eukrina/_render.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Draw the Eukrina Mind AI logos as SVG.

See Also
--------
scikitplot._brand._eukrina._spec : the brand as data.
scikitplot._brand._eukrina._build : writes whole asset sets.

Notes
-----
**User notes:** :func:`render` returns the SVG text of one asset; every
argument has a default, so ``render()`` is the horizontal, warm-bronze logo
for light backgrounds.

**Developer notes:** every position is derived from the mark's bounds and the
type metrics by the rules in :data:`RULES`; no coordinate is placed by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

from .._kit._color import readable, relative_luminance
from .._kit._svg import Element, IdSpace, document, fmt, to_string, url_ref
from .._kit._text import find_font, outline
from ._spec import EDITIONS, LAYOUTS, MARK, THEMES, TYPE, Edition, Mark, mark_bounds

__all__ = ["RULES", "Palette", "palette", "render"]

#: Layout rules, as fractions. ``M`` is the drawn height of the mark.
RULES: Mapping[str, float] = {
    "lockup_mark_height": 168.0,  # M in the horizontal lockup, user units
    "stacked_mark_height": 190.0,  # M in the stacked lockup
    "pad": 0.25,  # clear space around a lockup, in M
    "title_size": 0.635,  # title em over M (horizontal)
    "stacked_title_size": 0.48,  # title em over M (stacked)
    "mark_to_text": 0.30,  # mark ink to title ink, in M (horizontal)
    "stacked_mark_to_text": 0.34,  # mark ink to title cap top, in M (stacked)
    "app_icon_fill": 0.70,  # mark's larger side over the tile side
    "app_icon_radius": 0.22,  # corner radius over the tile side
    "round_icon_fill": 0.80,  # mark's farthest ink over the radius
    "symbol_pad": 0.04,  # clear space around the mark in the symbol, in M
}

#: Contrast targets (WCAG 2.x, AA): graphics and large text, small text.
GRAPHIC_CONTRAST = 3.0
TEXT_CONTRAST = 4.5

PathLike = Union[str, Path]
_ICON_SIDE = 256.0
_DEFAULT = object()


@dataclass(frozen=True)
class Palette:
    """
    The colours one asset is drawn with.

    Attributes
    ----------
    background : str
        Colour the asset is meant for; filled only by icons.
    u : tuple of two str
        ``U`` gradient, top-left to bottom-right (equal for a solid colour).
    accent : tuple of two str
        Diagonal gradient, top to bottom.
    title, subtitle : str
        Text colours.
    """

    background: str
    u: tuple[str, str]
    accent: tuple[str, str]
    title: str
    subtitle: str


def _edition(edition: str | Edition) -> Edition:
    if isinstance(edition, Edition):
        return edition
    if edition not in EDITIONS:
        raise ValueError(
            f"unknown edition {edition!r}; expected one of {sorted(EDITIONS)} or an Edition"
        )
    return EDITIONS[edition]


def _choice(name: str, value: str, allowed) -> str:
    if value not in allowed:
        raise ValueError(f"unknown {name} {value!r}; expected one of {list(allowed)}")
    return value


def palette(
    edition: str | Edition = "warm-bronze",
    theme: str = "light",
    *,
    color: str = "full",
    contrast: bool = True,
) -> Palette:
    """
    Return the colours of the logo for an edition, theme and colour mode.

    Parameters
    ----------
    edition : str or Edition, default "warm-bronze"
        ``"warm-bronze"``, ``"blue-green"`` or a custom :class:`Edition`.
    theme : {"light", "dark"}, default "light"
        Background the asset is drawn for.
    color : {"full", "mono"}, default "full"
        Full colour, or one colour for everything.
    contrast : bool, default True
        Move each colour to the nearest step of its scale that reaches the
        contrast target on the background (3:1 for the mark and title, 4.5:1
        for the subtitle). False keeps the edition's colours as given.

    Returns
    -------
    Palette
        The resolved colours.

    Raises
    ------
    ValueError
        For an unknown edition, theme or colour mode.
    scikitplot._brand._kit.ContrastError
        If a colour cannot reach its target on the background.

    Notes
    -----
    **User notes:** on dark backgrounds the bronze and blue accents already
    pass, so the dark logos keep the exact brand colours; on light ones the
    accents and subtitle are darkened by the smallest step that passes.
    """
    ed = _edition(edition)
    _choice("theme", theme, THEMES)
    _choice("colour mode", color, ("full", "mono"))
    background = ed.surface if theme == "light" else ed.dark

    def fit(value: str, minimum: float) -> str:
        return readable(value, background, minimum) if contrast else value

    if color == "mono":
        ink = fit(ed.dark if theme == "light" else ed.on_dark, GRAPHIC_CONTRAST)
        return Palette(background, (ink, ink), (ink, ink), ink, ink)
    if theme == "light":
        u = (fit(ed.dark_highlight, GRAPHIC_CONTRAST), fit(ed.dark, GRAPHIC_CONTRAST))
        title = fit(ed.dark, GRAPHIC_CONTRAST)
        subtitle = fit(ed.support, TEXT_CONTRAST)
    else:
        u = (fit(ed.on_dark, GRAPHIC_CONTRAST),) * 2
        title = fit(ed.on_dark, GRAPHIC_CONTRAST)
        subtitle = fit(ed.muted, TEXT_CONTRAST)
    accent = _keep_order(
        (ed.accent_light, ed.accent),
        (fit(ed.accent_light, GRAPHIC_CONTRAST), fit(ed.accent, GRAPHIC_CONTRAST)),
    )
    if theme == "light":
        u = _keep_order((ed.dark_highlight, ed.dark), u)
    return Palette(background, u, accent, title, subtitle)


def _keep_order(given: tuple[str, str], fitted: tuple[str, str]) -> tuple[str, str]:
    """
    Keep a gradient's light-to-dark direction after its stops were fitted.

    Two stops fitted separately can swap their lightness (the lighter colour
    needs more darkening to pass); the fitted colours are then exchanged so
    that the gradient still runs the way the edition draws it.
    """
    before = relative_luminance(given[0]) - relative_luminance(given[1])
    after = relative_luminance(fitted[0]) - relative_luminance(fitted[1])
    if before * after < 0:
        return fitted[1], fitted[0]
    return fitted


# ---------------------------------------------------------------------------
# The mark
# ---------------------------------------------------------------------------


def _mark_radius(mark: Mark, center: tuple[float, float]) -> float:
    """Distance from ``center`` to the farthest ink of the mark."""
    points = list(mark.u_extremes) + list(mark.accent)
    return (
        max(math.hypot(x - center[0], y - center[1]) for x, y in points)
        + mark.stroke / 2.0
    )


def _paint(
    ids: IdSpace,
    name: str,
    colors: tuple[str, str],
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    defs: list[Element],
) -> str:
    """Return a solid colour, or a gradient along ``start`` -> ``end`` added to defs."""
    if colors[0] == colors[1]:
        return colors[0]
    gid = ids.new(name)
    defs.append(
        Element(
            "linearGradient",
            {
                "id": gid,
                "gradientUnits": "userSpaceOnUse",
                "x1": start[0],
                "y1": start[1],
                "x2": end[0],
                "y2": end[1],
            },
            [
                Element("stop", {"offset": 0, "stop-color": colors[0]}),
                Element("stop", {"offset": 1, "stop-color": colors[1]}),
            ],
        )
    )
    return url_ref(gid)


def _mark(
    ids: IdSpace,
    pal: Palette,
    *,
    mark: Mark,
    gap: float,
    transform: tuple[float, float, float],
    defs: list[Element],
) -> Element:
    """Return the mark, moved by ``translate(tx ty) scale(s)``."""
    tx, ty, scale = transform
    (ax, ay), (bx, by) = mark.accent
    u_paint = _paint(ids, "u", pal.u, start=(55.0, 46.0), end=(201.0, 210.0), defs=defs)
    accent_paint = _paint(
        ids, "accent", pal.accent, start=(ax, ay), end=(bx, by), defs=defs
    )
    accent_d = f"M{fmt(ax)} {fmt(ay)} L{fmt(bx)} {fmt(by)}"
    u_attrs: dict[str, object] = {
        "d": mark.u_path,
        "stroke": u_paint,
        "stroke-width": mark.stroke,
    }
    if gap > 0:
        x0, y0, x1, y1 = mark_bounds(mark)
        mid = ids.new("gap")
        defs.append(
            Element(
                "mask",
                {
                    "id": mid,
                    "maskUnits": "userSpaceOnUse",
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                },
                [
                    Element(
                        "rect",
                        {
                            "x": x0,
                            "y": y0,
                            "width": x1 - x0,
                            "height": y1 - y0,
                            "fill": "#FFFFFF",
                        },
                    ),
                    Element(
                        "path",
                        {
                            "d": accent_d,
                            "stroke": "#000000",
                            "stroke-width": mark.stroke + 2.0 * gap,
                            "stroke-linecap": "round",
                            "fill": "none",
                        },
                    ),
                ],
            )
        )
        u_attrs["mask"] = url_ref(mid)
    return Element(
        "g",
        {
            "transform": f"translate({fmt(tx)} {fmt(ty)}) scale({fmt(scale)})",
            "fill": "none",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        },
        [
            Element("path", u_attrs),
            Element(
                "path",
                {"d": accent_d, "stroke": accent_paint, "stroke-width": mark.stroke},
            ),
        ],
    )


def _quantize(scale: float) -> float:
    """
    Round a scale down to the 3 decimals it is written with.

    The file then holds exactly the geometry that was checked, and rounding
    can only make the mark smaller, never push it past its clear space.
    """
    return math.floor(scale * 1000.0) / 1000.0


def _fit_mark(
    mark: Mark, height: float, left: float, top: float
) -> tuple[float, float, float]:
    """Transform that draws the mark ``height`` tall with its ink at (left, top)."""
    x0, y0, _, y1 = mark_bounds(mark)
    scale = _quantize(height / (y1 - y0))
    return left - x0 * scale, top - y0 * scale, scale


def _center_mark(
    mark: Mark, scale: float, cx: float, cy: float
) -> tuple[float, float, float]:
    """Transform that draws the mark at ``scale`` with its bounds centred on (cx, cy)."""
    x0, y0, x1, y1 = mark_bounds(mark)
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    scale = _quantize(scale)
    return cx - mx * scale, cy - my * scale, scale


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    tracking: float
    advance: float | None  # em, at zero tracking; None: measure from the font

    def width(self, font: str | None) -> float:
        """Advance width in user units, last character's tracking excluded."""
        if self.advance is not None and font is None:
            return (self.advance + self.tracking * (len(self.text) - 1)) * self.size
        from .._kit._text import measure  # noqa: PLC0415

        return measure(self.text, font, tracking=self.tracking) * self.size


def _text_element(
    line: _Line, pen: float, baseline: float, fill: str, font: str | None
) -> Element:
    if font is None:
        return Element(
            "text",
            {
                "x": pen,
                "y": baseline,
                "font-family": TYPE.family,
                "font-size": line.size,
                "font-weight": TYPE.weight,
                "letter-spacing": line.tracking * line.size,
                "fill": fill,
            },
            text=line.text,
        )
    d, _ = outline(
        line.text,
        font,
        x=pen,
        baseline=baseline,
        size=line.size,
        tracking=line.tracking,
    )
    return Element("path", {"d": d, "fill": fill}, [])


def _lines(title, subtitle, title_size: float) -> list[_Line]:
    lines: list[_Line] = []
    style = TYPE.title
    text = style.text if title is _DEFAULT else title
    if text is not None:
        lines.append(
            _Line(
                text,
                title_size,
                style.tracking,
                style.advance if text == style.text else None,
            )
        )
    style = TYPE.subtitle
    text = style.text if subtitle is _DEFAULT else subtitle
    if text is not None:
        lines.append(
            _Line(
                text,
                title_size * TYPE.subtitle_ratio,
                style.tracking,
                style.advance if text == style.text else None,
            )
        )
    return lines


def _text_block(lines: list[_Line], top: float) -> list[float]:
    """Baselines of the lines, the first cap top at ``top``."""
    baselines: list[float] = []
    y = top
    for index, line in enumerate(lines):
        if index:
            y += TYPE.line_gap * lines[0].size
        y += TYPE.cap_height * line.size
        baselines.append(y)
    return baselines


def _block_height(lines: list[_Line]) -> float:
    if not lines:
        return 0.0
    return sum(TYPE.cap_height * line.size for line in lines) + TYPE.line_gap * lines[
        0
    ].size * (len(lines) - 1)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def render(  # noqa: PLR0912, PLR0913, PLR0915
    layout: str = "horizontal",
    edition: str | Edition = "warm-bronze",
    theme: str = "light",
    *,
    color: str = "full",
    text: str = "live",
    font: PathLike | None = None,
    title=_DEFAULT,
    subtitle=_DEFAULT,
    gap: float | None = None,
    height: float | None = None,
    contrast: bool = True,
    id_prefix: str | None = None,
    mark: Mark = MARK,
) -> str:
    """
    Return one Eukrina Mind AI logo as SVG text.

    Parameters
    ----------
    layout : {"symbol", "horizontal", "stacked", "wordmark", "app-icon", "round-icon"}, default "horizontal"
        The mark alone, the two lockups (mark beside or above the text), the
        text alone, or the mark on a rounded-square or round tile.
    edition : str or Edition, default "warm-bronze"
        ``"warm-bronze"``, ``"blue-green"`` or a custom :class:`Edition`.
    theme : {"light", "dark"}, default "light"
        Background the asset is for. Icons fill their tile with it.
    color : {"full", "mono"}, default "full"
        Full colour or one colour.
    text : {"live", "outline"}, default "live"
        ``"live"`` writes ``<text>`` with the Inter font stack (small,
        editable; drawn with the viewer's fonts). ``"outline"`` draws the
        glyphs as paths from a font file, identical in every viewer.
    font : str or pathlib.Path, optional
        Font file for outlines, or for measuring custom text in live mode.
        Default for outlines: the installed Inter SemiBold.
    title, subtitle : str or None, optional
        Replace "Eukrina" or "MIND AI"; ``None`` leaves the line out. Custom
        text in live mode needs ``font`` to be measured.
    gap : float, optional
        Clear space cut out of the ``U`` around the diagonal, in design units.
        Default: the mark's gap for ``color="mono"``, none for full colour.
    height : float, optional
        Rendered height; the width follows. Default: the viewBox height.
    contrast : bool, default True
        Apply the contrast rule of :func:`palette`.
    id_prefix : str, optional
        Prefix of every id in the document; default
        ``"eukrina-<layout>-<edition>-<theme>"``.
    mark : Mark, default MARK
        The mark's geometry.

    Returns
    -------
    str
        A standalone SVG document.

    Raises
    ------
    ValueError
        For an unknown choice, a negative gap or a non-positive height, custom
        live text without ``font``, or a layout left without content.
    LookupError
        If ``text="outline"`` and no font is given and Inter is not installed.
    scikitplot._brand._kit.ContrastError
        If a colour cannot reach its contrast target.

    See Also
    --------
    palette : the colours used.
    scikitplot._brand._eukrina.build : write many assets at once.

    Notes
    -----
    **User notes:** live text is laid out with Inter's metrics; a viewer
    without Inter uses the next font of the stack, which can be slightly
    wider or narrower. Use ``text="outline"`` for files that must look the
    same everywhere (favicons, print, social images).

    **Developer notes:** the canvas is tight around the content plus the
    clear space of :data:`RULES`; the ``U`` and the diagonal use
    ``userSpaceOnUse`` gradients, which work on any stroke direction.

    Examples
    --------
    >>> svg = render("symbol", "blue-green", "dark")
    >>> svg.startswith("<svg") and "Eukrina" in svg
    True
    """
    _choice("layout", layout, LAYOUTS)
    _choice("text mode", text, ("live", "outline"))
    ed = _edition(edition)
    pal = palette(ed, theme, color=color, contrast=contrast)
    if gap is None:
        gap = mark.gap if color == "mono" else 0.0
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    if height is not None and not height > 0:
        raise ValueError(f"height must be > 0, got {height}")

    font_file: str | None = None
    if text == "outline":
        font_file = (
            str(font)
            if font is not None
            else find_font(TYPE.outline_family, TYPE.weight)
        )
    custom = (title is not _DEFAULT and title != TYPE.title.text) or (
        subtitle is not _DEFAULT and subtitle != TYPE.subtitle.text
    )
    measure_font = (
        font_file
        if font_file is not None
        else (str(font) if font is not None else None)
    )
    if (
        text == "live"
        and custom
        and measure_font is None
        and layout in ("horizontal", "stacked", "wordmark")
    ):
        lines_check = _lines(title, subtitle, 1.0)
        if any(line.advance is None for line in lines_check):
            raise ValueError(
                "custom text in live mode needs font= (a font file) to be measured"
            )

    ids = IdSpace(id_prefix or f"eukrina-{layout}-{ed.key}-{theme}")
    defs: list[Element] = []
    body: list[Element] = []
    x0, y0, x1, y1 = mark_bounds(mark)
    mark_w, mark_h = x1 - x0, y1 - y0

    def draw_lines(
        lines: list[_Line], left: float, top: float, center: float | None = None
    ) -> float:
        """Draw lines with their ink starting at ``left`` (or centred); return right edge."""
        right = left
        for line, baseline, fill in zip(
            lines, _text_block(lines, top), (pal.title, pal.subtitle)
        ):
            width = line.width(measure_font)
            bearing = TYPE.side_bearing * line.size
            pen = left - bearing if center is None else center - (width + bearing) / 2.0
            body.append(_text_element(line, pen, baseline, fill, font_file))
            right = max(right, pen + width)
        return right

    if layout in ("symbol", "app-icon", "round-icon"):
        side = _ICON_SIDE
        if layout == "symbol":
            pad = RULES["symbol_pad"] * mark_h
            side_w, side_h = mark_w + 2 * pad, mark_h + 2 * pad
            transform = (pad - x0, pad - y0, 1.0)
            view = (0.0, 0.0, side_w, side_h)
        elif layout == "app-icon":
            scale = RULES["app_icon_fill"] * side / max(mark_w, mark_h)
            transform = _center_mark(mark, scale, side / 2.0, side / 2.0)
            body.append(
                Element(
                    "rect",
                    {
                        "width": side,
                        "height": side,
                        "rx": RULES["app_icon_radius"] * side,
                        "fill": pal.background,
                    },
                )
            )
            view = (0.0, 0.0, side, side)
        else:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            scale = (
                RULES["round_icon_fill"] * (side / 2.0) / _mark_radius(mark, (cx, cy))
            )
            transform = _center_mark(mark, scale, side / 2.0, side / 2.0)
            body.append(
                Element(
                    "circle",
                    {
                        "cx": side / 2.0,
                        "cy": side / 2.0,
                        "r": side / 2.0,
                        "fill": pal.background,
                    },
                )
            )
            view = (0.0, 0.0, side, side)
        body.append(_mark(ids, pal, mark=mark, gap=gap, transform=transform, defs=defs))
    elif layout == "horizontal":
        m = RULES["lockup_mark_height"]
        pad = RULES["pad"] * m
        lines = _lines(title, subtitle, RULES["title_size"] * m)
        transform = _fit_mark(mark, m, pad, pad)
        drawn = mark_h * transform[2]  # m, or a hair less after rounding
        body.append(_mark(ids, pal, mark=mark, gap=gap, transform=transform, defs=defs))
        right = pad + mark_w * transform[2]
        if lines:
            block = _block_height(lines)
            right = draw_lines(
                lines, right + RULES["mark_to_text"] * m, pad + (drawn - block) / 2.0
            )
        view = (0.0, 0.0, right + pad, drawn + 2 * pad)
    elif layout == "stacked":
        m = RULES["stacked_mark_height"]
        pad = RULES["pad"] * m
        lines = _lines(title, subtitle, RULES["stacked_title_size"] * m)
        widths = [
            line.width(measure_font) + TYPE.side_bearing * line.size for line in lines
        ]
        content_w = max([mark_w * _quantize(m / mark_h), *widths])
        width_box = content_w + 2 * pad
        center = width_box / 2.0
        scale = m / mark_h
        transform = _center_mark(mark, scale, center, pad + m / 2.0)
        body.append(_mark(ids, pal, mark=mark, gap=gap, transform=transform, defs=defs))
        bottom = pad + m
        if lines:
            top = bottom + RULES["stacked_mark_to_text"] * m
            draw_lines(lines, 0.0, top, center=center)
            bottom = top + _block_height(lines)
        view = (0.0, 0.0, width_box, bottom + pad)
    else:  # wordmark
        lines = _lines(
            title, subtitle, RULES["title_size"] * RULES["lockup_mark_height"]
        )
        if not lines:
            raise ValueError("the wordmark needs a title or a subtitle")
        pad = RULES["pad"] * lines[0].size
        right = draw_lines(lines, pad, pad)
        view = (0.0, 0.0, right + pad, _block_height(lines) + 2 * pad)

    if defs:
        body.insert(0, Element("defs", {}, defs))
    out_h = view[3] if height is None else float(height)
    out_w = view[2] * out_h / view[3]
    root = document(
        view,
        body,
        title=" ".join(filter(None, ["Eukrina Mind AI", layout.replace("-", " ")])),
        width=out_w,
        height=out_h,
        id_space=ids,
    )
    return to_string(root)
