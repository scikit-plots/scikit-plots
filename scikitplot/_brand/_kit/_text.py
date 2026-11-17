# scikitplot/_brand/_kit/_text.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Text for brand assets: font lookup, advance widths and glyph outlines.

A logo's text can be written two ways. As *live* text (``<text>``) it is small
and editable, but each viewer draws it with whatever font it has. As
*outlines* (``<path>``) it looks the same everywhere, drawn from one named font
file. This module provides the outlines and the measurements both need.

See Also
--------
scikitplot._brand._kit._svg : writes the paths.

Notes
-----
**User notes:** outlines need a font file. :func:`find_font` looks one up by
family and weight and stops with a clear message if it is not installed;
nothing falls back silently to another font.

**Developer notes:** Matplotlib (a dependency of ``scikitplot._brand``) reads
the font through FreeType and is imported only when a function here is called.
Glyphs are placed one by one: advance from the font's ``hmtx`` table, plus the
tracking, plus the ``kern`` table where the font has one. OpenType ``GPOS``
kerning is not applied, the same as for the measured widths that live text is
laid out with, so both modes agree.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from ._svg import fmt

__all__ = [
    "FontInfo",
    "find_font",
    "font_info",
    "measure",
    "outline",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class FontInfo:
    """
    What identifies a font file, for the records of a build.

    Attributes
    ----------
    path : str
        File the glyphs were read from.
    family, style : str
        Names stored in the font.
    sha256 : str
        Hash of the file, so a build can say exactly which font it used.
    units_per_em : int
        Design units per em.
    ascender, descender : float
        In em; ``descender`` is negative.
    """

    path: str
    family: str
    style: str
    sha256: str
    units_per_em: int
    ascender: float
    descender: float


def _ft2font(path: PathLike):
    from matplotlib.ft2font import FT2Font  # noqa: PLC0415

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"font file not found: {file}")
    return FT2Font(str(file))


def find_font(family: str, weight: int | str = "normal") -> str:
    """
    Return the path of an installed font, without falling back to another.

    Parameters
    ----------
    family : str
        Font family, for example ``"Inter"``.
    weight : int or str, default "normal"
        CSS weight, for example ``600``.

    Returns
    -------
    str
        Path of the font file.

    Raises
    ------
    LookupError
        If no installed font has that family.

    Notes
    -----
    **User notes:** Matplotlib's font cache is used; after installing a font,
    a new Python process may be needed for it to be seen.
    """
    from matplotlib.font_manager import FontProperties, findfont  # noqa: PLC0415

    try:
        return findfont(
            FontProperties(family=family, weight=weight), fallback_to_default=False
        )
    except ValueError as exc:
        raise LookupError(
            f"the font {family!r} (weight {weight}) is not installed; install it, or pass the path "
            "of a font file. Text outlines never fall back to another font."
        ) from exc


def font_info(path: PathLike) -> FontInfo:
    """
    Describe a font file.

    Parameters
    ----------
    path : str or pathlib.Path
        Font file (TrueType or OpenType).

    Returns
    -------
    FontInfo
        Names, metrics and hash.
    """
    font = _ft2font(path)
    upem = int(font.units_per_EM)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return FontInfo(
        path=str(path),
        family=str(font.family_name),
        style=str(font.style_name),
        sha256=digest,
        units_per_em=upem,
        ascender=font.ascender / upem,
        descender=font.descender / upem,
    )


def _freetype_flags():
    """
    FreeType's "no scaling" load flag and "unscaled" kerning mode.

    Matplotlib 3.10 made them enums (``LoadFlags``, ``Kerning``); earlier
    releases, back to the 3.4 this package supports, have the constants.
    """
    from matplotlib import ft2font  # noqa: PLC0415

    if hasattr(ft2font, "LoadFlags"):
        return ft2font.LoadFlags.NO_SCALE, ft2font.Kerning.UNSCALED
    return ft2font.LOAD_NO_SCALE, ft2font.KERNING_UNSCALED


def _advances(text: str, path: PathLike) -> list[float]:
    """Advance of each character in em, kern table included."""
    no_scale, unscaled = _freetype_flags()
    font = _ft2font(path)
    upem = float(font.units_per_EM)
    advances: list[float] = []
    previous: int | None = None
    for ch in text:
        glyph = font.get_char_index(ord(ch))
        if glyph == 0 and not ch.isspace():
            raise ValueError(f"the font {Path(path).name} has no glyph for {ch!r}")
        advance = font.load_char(ord(ch), flags=no_scale).horiAdvance / upem
        if previous is not None and advances:
            advances[-1] += font.get_kerning(previous, glyph, unscaled) / upem
        advances.append(advance)
        previous = glyph
    return advances


def measure(text: str, path: PathLike, *, tracking: float = 0.0) -> float:
    """
    Return the advance width of a line of text, in em.

    Parameters
    ----------
    text : str
        One line, not empty.
    path : str or pathlib.Path
        Font file.
    tracking : float, default 0.0
        Space added between characters, in em (negative tightens).

    Returns
    -------
    float
        Sum of the advances plus ``tracking`` between characters (not after
        the last one).

    Raises
    ------
    ValueError
        If ``text`` is empty or has a line break, or the font lacks a glyph.
    """
    _check_line(text)
    return sum(_advances(text, path)) + tracking * (len(text) - 1)


def _check_line(text: str) -> None:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if "\n" in text or "\r" in text:
        raise ValueError("text must be one line")


def _path_data(vertices, codes, dx: float, dy: float, size: float) -> list[str]:
    """SVG path commands of one glyph, y flipped (SVG grows downward)."""
    from matplotlib.path import Path as MplPath  # noqa: PLC0415

    parts: list[str] = []
    index = 0
    count = len(codes)

    def point(i: int) -> str:
        x, y = vertices[i]
        return f"{fmt(dx + x * size)} {fmt(dy - y * size)}"

    while index < count:
        code = codes[index]
        if code == MplPath.MOVETO:
            parts.append("M" + point(index))
            index += 1
        elif code == MplPath.LINETO:
            parts.append("L" + point(index))
            index += 1
        elif code == MplPath.CURVE3:
            parts.append("Q" + point(index) + " " + point(index + 1))
            index += 2
        elif code == MplPath.CURVE4:
            parts.append(
                "C" + point(index) + " " + point(index + 1) + " " + point(index + 2)
            )
            index += 3
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
            index += 1
        else:  # STOP
            index += 1
    return parts


def outline(
    text: str,
    path: PathLike,
    *,
    x: float,
    baseline: float,
    size: float,
    tracking: float = 0.0,
    anchor: str = "start",
) -> tuple[str, float]:
    """
    Return the outline of a line of text as SVG path data.

    Parameters
    ----------
    text : str
        One line, not empty.
    path : str or pathlib.Path
        Font file the glyphs are read from.
    x, baseline : float
        Where the line starts (or is centred, see ``anchor``) and its baseline,
        in SVG user units.
    size : float
        Font size (em) in user units, > 0.
    tracking : float, default 0.0
        Space between characters, in em.
    anchor : {"start", "middle"}, default "start"
        Whether ``x`` is the start or the centre of the line.

    Returns
    -------
    d : str
        Path data, absolute commands, y downward.
    width : float
        Advance width of the line in user units.

    Raises
    ------
    ValueError
        For empty or multi-line text, a non-positive size, an unknown anchor,
        or a glyph the font lacks.
    """
    from matplotlib.font_manager import FontProperties  # noqa: PLC0415
    from matplotlib.textpath import TextToPath  # noqa: PLC0415

    _check_line(text)
    if not size > 0:
        raise ValueError(f"size must be > 0, got {size}")
    if anchor not in ("start", "middle"):
        raise ValueError(f"anchor must be 'start' or 'middle', got {anchor!r}")
    advances = _advances(text, path)
    width = (sum(advances) + tracking * (len(text) - 1)) * size
    pen = x - width / 2.0 if anchor == "middle" else x
    prop = FontProperties(fname=str(path))
    converter = TextToPath()
    scale = 1.0 / converter.FONT_SCALE
    parts: list[str] = []
    for ch, advance in zip(text, advances):
        if not ch.isspace():
            vertices, codes = converter.get_text_path(prop, ch)
            if len(codes):
                parts.extend(
                    _path_data(
                        [(vx * scale, vy * scale) for vx, vy in vertices],
                        codes,
                        pen,
                        baseline,
                        size,
                    )
                )
        pen += (advance + tracking) * size
    return " ".join(parts), width
