# scikitplot/_brand/_eukrina/_spec.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The Eukrina Mind AI brand, as data: the mark, the two editions, the type.

Nothing here draws. :mod:`._render` turns this description into SVG.

Notes
-----
**User notes:** to change a colour, edit :data:`EDITIONS`; to add an edition,
add an :class:`Edition` with the same roles. Every layout and asset follows.

**Developer notes:**

- The mark is two round-capped strokes in a 256-unit box: a ``U`` and a
  diagonal accent that crosses its right arm. Its bounding box is computed
  from the listed extreme points and the stroke width (:func:`mark_bounds`),
  so layouts centre and fit it by rule.
- Type metrics are those of Inter SemiBold (OpenType, ``fontversion`` 262144,
  units per em 2048), measured with FreeType: they place live text and size
  the canvas. Outlined text is measured from the font file it is drawn with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from .._kit._color import normalize_hex

__all__ = [
    "EDITIONS",
    "LAYOUTS",
    "MARK",
    "THEMES",
    "TYPE",
    "Edition",
    "Mark",
    "TextStyle",
    "Typography",
    "mark_bounds",
]

#: Background themes an asset is drawn for.
THEMES: tuple[str, ...] = ("light", "dark")

#: Layouts, from the mark alone to the full lockups and icons.
LAYOUTS: tuple[str, ...] = (
    "symbol",
    "horizontal",
    "stacked",
    "wordmark",
    "app-icon",
    "round-icon",
)

Point = Tuple[float, float]


@dataclass(frozen=True)
class Mark:
    """
    Geometry of the Eukrina mark, in a 256 x 256 design box.

    Attributes
    ----------
    u_path : str
        SVG path data of the ``U`` stroke.
    u_extremes : tuple of points
        Points that bound the ``U`` centre line (ends and the lowest point).
    accent : tuple of two points
        Start (top) and end (bottom) of the diagonal accent.
    stroke : float
        Stroke width of both strokes; caps and joins are round.
    gap : float
        Clear space cut out of the ``U`` around the accent in one-colour
        versions, so that the two strokes stay apart.
    """

    u_path: str
    u_extremes: tuple[Point, ...]
    accent: tuple[Point, Point]
    stroke: float
    gap: float


MARK = Mark(
    u_path="M55 46 V137 C55 181 85 210 128 210 C171 210 201 181 201 137 V46",
    u_extremes=(
        (55.0, 46.0),
        (201.0, 46.0),
        (128.0, 210.0),
        (55.0, 137.0),
        (201.0, 137.0),
    ),
    accent=((166.0, 45.0), (215.0, 210.0)),
    stroke=25.0,
    gap=7.0,
)


def mark_bounds(mark: Mark = MARK) -> tuple[float, float, float, float]:
    """
    Return the bounding box of the drawn mark: ``(x0, y0, x1, y1)``.

    Parameters
    ----------
    mark : Mark, default MARK
        The geometry.

    Returns
    -------
    tuple of 4 float
        Extremes of the centre lines widened by half the stroke (round caps).

    Notes
    -----
    **Developer notes:** the ``U``'s curves stay inside the hull of their
    end points, whose extremes are listed in ``u_extremes``; the result is
    exact for this mark. A test renders the mark and checks the ink against it.
    """
    points = list(mark.u_extremes) + list(mark.accent)
    half = mark.stroke / 2.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - half, min(ys) - half, max(xs) + half, max(ys) + half)


@dataclass(frozen=True)
class Edition:
    """
    One colour edition of the brand.

    Attributes
    ----------
    key : str
        Identifier used in file names, for example ``"warm-bronze"``.
    name : str
        Human-readable name.
    dark : str
        Foundation colour: the mark and text on light, the icon background.
    dark_highlight : str
        Lighter companion of ``dark``; the ``U`` fades from it to ``dark``.
    accent, accent_light : str
        The diagonal fades from ``accent_light`` (top) to ``accent`` (bottom).
    support : str
        Subtitle on light backgrounds.
    muted : str
        Subtitle on dark backgrounds.
    surface : str, default "#F7F6F3"
        The light background colours are checked against.
    on_dark : str, default "#FFFFFF"
        Mark and title on dark backgrounds.
    """

    key: str
    name: str
    dark: str
    dark_highlight: str
    accent: str
    accent_light: str
    support: str
    muted: str
    surface: str = "#F7F6F3"
    on_dark: str = "#FFFFFF"

    def __post_init__(self) -> None:
        for role in (
            "dark",
            "dark_highlight",
            "accent",
            "accent_light",
            "support",
            "muted",
            "surface",
            "on_dark",
        ):
            object.__setattr__(self, role, normalize_hex(getattr(self, role)))

    def colors(self) -> dict[str, str]:
        """
        Return the role names and colours.

        Returns
        -------
        dict of str to str
            Role to ``#RRGGBB``, in declaration order.
        """
        return {
            role: getattr(self, role)
            for role in (
                "dark",
                "dark_highlight",
                "accent",
                "accent_light",
                "support",
                "muted",
                "surface",
                "on_dark",
            )
        }


EDITIONS: Mapping[str, Edition] = {
    "warm-bronze": Edition(
        key="warm-bronze",
        name="Charcoal / Bronze",
        dark="#202729",
        dark_highlight="#303735",
        accent="#B58B58",
        accent_light="#C6A477",
        support="#96836D",
        muted="#D7C7B2",
    ),
    "blue-green": Edition(
        key="blue-green",
        name="Deep Navy / Blue / Forest Green",
        dark="#071B2D",
        dark_highlight="#0D304A",
        accent="#1E90FF",
        accent_light="#00BFFF",
        support="#228B22",
        muted="#A8E5F5",
    ),
}


@dataclass(frozen=True)
class TextStyle:
    """
    One line of the logo's text.

    Attributes
    ----------
    text : str
        The words.
    advance : float
        Advance width in em at zero tracking, in the reference font.
    tracking : float
        Space between characters, in em.
    """

    text: str
    advance: float
    tracking: float


@dataclass(frozen=True)
class Typography:
    """
    The logo's type: font, metrics of the reference font, and the two lines.

    Attributes
    ----------
    family : str
        CSS font stack of live text.
    outline_family : str
        Family looked up for outlined text when no font file is given.
    weight : int
        Font weight of both lines.
    cap_height : float
        Cap height of the reference font, in em.
    side_bearing : float
        Left side bearing of the first letter of each line, in em ("E", "M").
    title, subtitle : TextStyle
        "Eukrina" and "MIND AI".
    subtitle_ratio : float
        Subtitle size over title size.
    line_gap : float
        Space between the title's baseline and the subtitle's cap top, in
        title em.
    """

    family: str
    outline_family: str
    weight: int
    cap_height: float
    side_bearing: float
    title: TextStyle
    subtitle: TextStyle
    subtitle_ratio: float
    line_gap: float = field(default=0.34)


TYPE = Typography(
    family="Inter, 'Helvetica Neue', Arial, sans-serif",
    outline_family="Inter",
    weight=600,
    cap_height=1490 / 2048,
    side_bearing=150 / 2048,
    title=TextStyle(text="Eukrina", advance=7438 / 2048, tracking=-0.047),
    subtitle=TextStyle(text="MIND AI", advance=8063 / 2048, tracking=0.48),
    subtitle_ratio=0.255,
)
