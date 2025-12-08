"""
Brand logo utilities for scikit-plots.

This module provides a small, self-contained brand system built entirely
with Matplotlib primitives. It is designed for:

- Deterministic, code-only logo generation (no image reading).
- Multiple variants for different contexts (docs, favicon, submodules).
- Simple public API aligned with Matplotlib's save semantics.
- Optional wordmark lockups using the same visual language.

The default primary mark is deterministic via a fixed dot layout.

See Also
--------
matplotlib.figure.Figure.savefig
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge

Variant = Literal["primary", "small", "metrics", "knn"]
Theme = Literal["light", "dark"]
DotsMode = Literal["fixed", "random", "none"]
Preset = Literal["favicon", "avatar", "docs-hero"]


# -----------------------------
# Public constants
# -----------------------------

_VARIANTS: tuple[Variant, ...] = ("primary", "small", "metrics", "knn")


_SIZE_PRESETS: dict[Preset, dict[str, object]] = {
    # Designed for tiny app icons / browser favicons.
    # Uses the simplified variant for clarity at small sizes.
    "favicon": {
        "variant": "small",
        "size": 1.2,  # inches
        "dpi": 256,  # ensures crisp PNG
        "dots": "none",
    },
    # Balanced for GitHub org avatar / social icons.
    "avatar": {
        "variant": "primary",
        "size": 3.2,
        "dpi": 220,
        "dots": "fixed",
    },
    # Great default for docs landing pages or banners
    # (icon-only hero; for a text lockup use wordmark).
    "docs-hero": {
        "variant": "primary",
        "size": 5.0,
        "dpi": 200,
        "dots": "fixed",
    },
}


# -----------------------------
# Fixed dot layout (brand lock)
# -----------------------------
# (x, y, r, color_key)
_FIXED_DOTS: Sequence[tuple[float, float, float, str]] = [
    (-0.10, 0.55, 0.035, "NAVY"),
    (0.12, 0.52, 0.030, "BLUE_LIGHT"),
    (0.28, 0.50, 0.032, "NAVY"),
    (0.45, 0.47, 0.030, "BLUE_LIGHT"),
    (-0.30, 0.48, 0.028, "NAVY"),
    (-0.45, 0.60, 0.030, "BLUE_LIGHT"),
    (0.60, 0.10, 0.030, "NAVY"),
    (0.70, 0.32, 0.028, "BLUE_LIGHT"),
    (0.20, 0.72, 0.026, "NAVY"),
    (-0.05, 0.78, 0.024, "BLUE_LIGHT"),
]


# -----------------------------
# Color + geometry helpers
# -----------------------------


def _lerp_color(c1: str, c2: str, t: float):
    c1 = np.array(mcolors.to_rgb(c1))
    c2 = np.array(mcolors.to_rgb(c2))
    return (1 - t) * c1 + t * c2


def _palette(theme: Theme = "light", mono: bool = False):
    """Return the brand palette for a theme."""
    if mono:
        if theme == "dark":
            return {
                "NAVY": "#e6f2f7",
                "BLUE": "#e6f2f7",
                "BLUE_LIGHT": "#e6f2f7",
                "ORANGE": "#e6f2f7",
                "ORANGE_LIGHT": "#e6f2f7",
                "BG": "#0b141a",
            }
        return {
            "NAVY": "#002030",
            "BLUE": "#002030",
            "BLUE_LIGHT": "#002030",
            "ORANGE": "#002030",
            "ORANGE_LIGHT": "#002030",
            "BG": "white",
        }

    if theme == "dark":
        return {
            "NAVY": "#e6f2f7",
            "BLUE": "#7fc1da",
            "BLUE_LIGHT": "#a7dced",
            "ORANGE": "#ff8b5a",
            "ORANGE_LIGHT": "#ffb08c",
            "BG": "#0b141a",
        }

    return {
        "NAVY": "#002030",
        "BLUE": "#2f7fa3",
        "BLUE_LIGHT": "#6fb7d2",
        "ORANGE": "#c85028",
        "ORANGE_LIGHT": "#e07b3a",
        "BG": "white",
    }


def list_variants() -> tuple[Variant, ...]:
    """
    List supported logo variants.

    Returns
    -------
    tuple
        ("primary", "small", "metrics", "knn")

    Examples
    --------
    >>> import scikitplot as sp
    >>> sp._logo.list_variants()
    ('primary', 'small', 'metrics', 'knn')
    """
    return _VARIANTS


def list_size_presets() -> tuple[Preset, ...]:
    """
    List supported size presets.

    Returns
    -------
    tuple
        ("favicon", "avatar", "docs-hero")

    Examples
    --------
    >>> import scikitplot as sp
    >>> sp._logo.list_size_presets()
    ('favicon', 'avatar', 'docs-hero')
    """
    return tuple(_SIZE_PRESETS.keys())


def _apply_preset(
    *,
    preset: Preset | None,
    variant: Variant,
    dots: DotsMode,
    size: float,
    dpi: int,
) -> tuple[Variant, DotsMode, float, int]:
    """Resolve preset defaults without overriding explicit parameters."""
    if preset is None:
        return variant, dots, size, dpi

    cfg = _SIZE_PRESETS[preset]
    variant = (
        variant if variant != "primary" or "variant" not in cfg else cfg["variant"]
    )  # type: ignore[]
    dots = dots if dots != "fixed" or "dots" not in cfg else cfg["dots"]  # type: ignore[]

    # Only override size/dpi if user kept defaults
    size = size if size != 4 else cfg.get("size", size)  # type: ignore[]  # noqa: PLR2004
    dpi = dpi if dpi != 200 else cfg.get("dpi", dpi)  # type: ignore[]  # noqa: PLR2004
    return variant, dots, size, dpi


# -----------------------------
# Icon primitives
# -----------------------------


def _add_confusion_icon(ax, clip, center=(0.02, 0.65), size=0.22, color="#002030"):
    x0, y0 = center[0] - size / 2, center[1] - size / 2
    cell = size / 3
    for i in range(3):
        for j in range(3):
            r = Rectangle(
                (x0 + i * cell, y0 + j * cell),
                cell * 0.9,
                cell * 0.9,
                facecolor="none",
                edgecolor=color,
                linewidth=2.5,
                zorder=4,
            )
            r.set_clip_path(clip)
            ax.add_patch(r)


def _add_spark(ax, clip, center, arms, color, hole_face="white"):
    for k in range(arms):
        ang = k * np.pi / arms
        dx = 0.12 * np.cos(ang)
        dy = 0.12 * np.sin(ang)

        l = Line2D(
            [center[0] - dx, center[0] + dx],
            [center[1] - dy, center[1] + dy],
            color=color,
            linewidth=6,
            solid_capstyle="round",
            zorder=4,
        )
        l.set_clip_path(clip)
        ax.add_line(l)

    hole = Circle(center, 0.04, facecolor=hole_face, edgecolor="none", zorder=5)
    hole.set_clip_path(clip)
    ax.add_patch(hole)


# -----------------------------
# Core drawing
# -----------------------------


def draw(  # noqa: PLR0912
    ax,
    *,
    variant: Variant = "primary",
    theme: Theme = "light",
    mono: bool = False,
    dots: DotsMode = "fixed",
    seed: int = 2,
):
    """
    Draw a scikit-plots logo variant onto an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Destination axes.
    variant : {"primary", "small", "metrics", "knn"}, default="primary"
        Which icon complexity to render.
    theme : {"light", "dark"}, default="light"
        Palette theme.
    mono : bool, default=False
        Render in monochrome.
    dots : {"fixed", "random", "none"}, default="fixed"
        Decorative dot mode (only used in "primary").
        Use "fixed" for brand-stable output.
    seed : int, default=2
        Seed for "random" dot placement.

    Notes
    -----
    This function is deterministic for all variants when `dots="fixed"`.

    See Also
    --------
    create
    save
    show

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import scikitplot as sp
    >>> fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    >>> sp._logo.draw(ax, variant="metrics")
    """
    P = _palette(theme=theme, mono=mono)  # noqa: N806
    NAVY, BLUE, BLUE_LIGHT = P["NAVY"], P["BLUE"], P["BLUE_LIGHT"]  # noqa: N806
    ORANGE, ORANGE_LIGHT, BG = P["ORANGE"], P["ORANGE_LIGHT"], P["BG"]  # noqa: N806

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_facecolor(BG)

    inner_r = 0.93
    ax.add_patch(Circle((0, 0), inner_r, facecolor=BG, edgecolor="none", zorder=0))
    clip = Circle((0, 0), inner_r, transform=ax.transData)

    # Bars
    n_bars = 12 if variant in ("primary", "metrics", "knn") else 8
    xs = np.linspace(-0.8, 0.8, n_bars)
    width = 0.095 if n_bars >= 10 else 0.12  # noqa: PLR2004
    baseline = -0.82
    heights = np.linspace(0.22, 0.75, n_bars)

    for i, (x, h) in enumerate(zip(xs, heights)):
        t = i / max(1, (n_bars - 1))
        col = _lerp_color(BLUE, BLUE_LIGHT, 0.6 * t) if not mono else NAVY

        bar = FancyBboxPatch(
            (x - width / 2, baseline),
            width,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.045",
            linewidth=0,
            facecolor=col,
            zorder=2,
        )
        bar.set_clip_path(clip)
        ax.add_patch(bar)

    # Orange signal dots
    if variant != "small":
        orange_x = np.linspace(-0.15, 0.75, 7)
        orange_y = np.linspace(-0.15, 0.35, 7)
        for i, (x, y) in enumerate(zip(orange_x, orange_y)):
            c = ORANGE if i < 5 else ORANGE_LIGHT  # noqa: PLR2004
            r = 0.055 if i < 5 else 0.045  # noqa: PLR2004
            dot = Circle((x, y), r, facecolor=c, edgecolor="none", zorder=3)
            dot.set_clip_path(clip)
            ax.add_patch(dot)

    # Pie chart
    pie_center = (-0.55, 0.25)
    pie_r = 0.22 if variant != "small" else 0.18
    wedges = (
        Wedge(pie_center, pie_r, 220, 40, facecolor=ORANGE, edgecolor="none", zorder=4),
        Wedge(pie_center, pie_r, 40, 140, facecolor=BLUE, edgecolor="none", zorder=4),
        Wedge(pie_center, pie_r, 140, 220, facecolor=BG, edgecolor="none", zorder=4),
    )
    for w in wedges:
        w.set_clip_path(clip)
        ax.add_patch(w)

    # Structured dots
    if variant in ("primary", "metrics", "knn"):
        grid_x = np.linspace(-0.75, -0.15, 5)
        grid_y = np.linspace(-0.15, -0.55, 4)
        for gx in grid_x:
            for gy in grid_y:
                col = NAVY if (int((gx + 1) * 10 + (gy + 1) * 10) % 3) else BLUE_LIGHT
                d = Circle((gx, gy), 0.045, facecolor=col, edgecolor="none", zorder=2.5)
                d.set_clip_path(clip)
                ax.add_patch(d)

    # Decorative dots (only primary)
    if variant == "primary":
        if dots == "fixed":
            for x, y, r, key in _FIXED_DOTS:
                col = NAVY if key == "NAVY" else BLUE_LIGHT
                d = Circle(
                    (x, y), r, facecolor=col, edgecolor="none", alpha=0.98, zorder=2
                )
                d.set_clip_path(clip)
                ax.add_patch(d)
        elif dots == "random":
            rng = np.random.default_rng(seed)
            pts = rng.uniform(-0.8, 0.8, size=(18, 2))
            for x, y in pts:
                if x * x + y * y > inner_r * inner_r:
                    continue
                if y < -0.2 and x > -0.2:  # noqa: PLR2004
                    continue
                col = NAVY if rng.random() < 0.6 else BLUE_LIGHT  # noqa: PLR2004
                rr = rng.uniform(0.025, 0.045)
                d = Circle(
                    (x, y), rr, facecolor=col, edgecolor="none", alpha=0.95, zorder=2
                )
                d.set_clip_path(clip)
                ax.add_patch(d)

    # Trend line
    line_pts = np.array([[-0.05, -0.05], [0.25, 0.10], [0.55, 0.22]])
    line = Line2D(
        line_pts[:, 0],
        line_pts[:, 1],
        color=NAVY,
        linewidth=10,
        solid_capstyle="round",
        zorder=4,
    )
    line.set_clip_path(clip)
    ax.add_line(line)

    for x, y in line_pts[1:]:
        outer_node = Circle((x, y), 0.07, facecolor=NAVY, edgecolor="none", zorder=5)
        inner_node = Circle((x, y), 0.035, facecolor=BG, edgecolor="none", zorder=6)
        for n in (outer_node, inner_node):
            n.set_clip_path(clip)
            ax.add_patch(n)

    # Top icon
    icon_center = (0.02, 0.65)
    if variant == "metrics":
        _add_confusion_icon(ax, clip, center=icon_center, size=0.22, color=NAVY)
    else:
        _add_spark(ax, clip, center=icon_center, arms=6, color=NAVY, hole_face=BG)

    # KNN mini motif
    if variant == "knn":
        nodes = [(-0.05, 0.52), (0.10, 0.45), (0.25, 0.55)]
        for x, y in nodes:
            c = Circle((x, y), 0.035, facecolor=BLUE_LIGHT, edgecolor="none", zorder=4)
            c.set_clip_path(clip)
            ax.add_patch(c)
        links = [(-0.05, 0.52, 0.10, 0.45), (0.10, 0.45, 0.25, 0.55)]
        for x1, y1, x2, y2 in links:
            l = Line2D([x1, x2], [y1, y2], color=BLUE, linewidth=4, zorder=3.8)
            l.set_clip_path(clip)
            ax.add_line(l)

    # Accent dots
    if variant != "small":
        accents = [
            (0.75, 0.25, ORANGE),
            (0.68, 0.45, ORANGE_LIGHT),
            (0.50, 0.50, BLUE_LIGHT),
            (0.35, 0.60, NAVY),
        ]
        for x, y, c in accents:
            d = Circle((x, y), 0.04, facecolor=c, edgecolor="none", zorder=4)
            d.set_clip_path(clip)
            ax.add_patch(d)

    # Outer ring
    outer = Circle(
        (0, 0),
        1.0,
        facecolor="none",
        edgecolor=NAVY,
        linewidth=18 if variant != "small" else 16,
        zorder=10,
    )
    ax.add_patch(outer)


# -----------------------------
# Figure factories
# -----------------------------


def create(
    *,
    variant: Variant = "primary",
    theme: Theme = "light",
    mono: bool = False,
    dots: DotsMode = "fixed",
    preset: Preset | None = None,
    size: float = 4,
    dpi: int = 200,
    seed: int = 2,
):
    """
    Create a logo figure.

    Parameters
    ----------
    variant : {"primary", "small", "metrics", "knn"}, default="primary"
        Logo variant to render.
    theme : {"light", "dark"}, default="light"
        Palette theme.
    mono : bool, default=False
        Monochrome rendering.
    dots : {"fixed", "random", "none"}, default="fixed"
        Decorative dot mode (only used in "primary").
    preset : {"favicon", "avatar", "docs-hero"} or None, default=None
        Convenience preset for common output contexts.
    size : float, default=4
        Figure size in inches (square).
    dpi : int, default=200
        Figure DPI.
    seed : int, default=2
        Random seed used only when `dots="random"`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    draw
    save
    show
    wordmark.create

    Examples
    --------
    >>> import scikitplot as sp
    >>> fig, ax = sp._logo.create(variant="small", preset="favicon")
    >>> fig.savefig("favicon.png", transparent=True)
    """
    variant, dots, size, dpi = _apply_preset(
        preset=preset, variant=variant, dots=dots, size=size, dpi=dpi
    )
    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    draw(ax, variant=variant, theme=theme, mono=mono, dots=dots, seed=seed)
    fig.tight_layout(pad=0)
    return fig, ax


def show(
    *,
    variant: Variant = "primary",
    theme: Theme = "light",
    mono: bool = False,
    dots: DotsMode = "fixed",
    preset: Preset | None = None,
    size: float = 4,
    dpi: int = 200,
    seed: int = 2,
    block: bool | None = None,
):
    """
    Display the logo.

    Parameters mirror :func:`create`.

    Returns
    -------
    fig : matplotlib.figure.Figure

    See Also
    --------
    create
    save
    wordmark.show

    Examples
    --------
    >>> import scikitplot as sp
    >>> sp._logo.show()
    >>> sp._logo.show(variant="metrics", theme="dark")
    """
    fig, _ = create(
        variant=variant,
        theme=theme,
        mono=mono,
        dots=dots,
        preset=preset,
        size=size,
        dpi=dpi,
        seed=seed,
    )
    plt.show(block=block)
    return fig


# -----------------------------
# Save helpers (Matplotlib-like)
# -----------------------------


def _infer_format_from_filename(
    filename: str | Path, ext: str | None, format: str | None
):
    p = Path(filename)

    # Resolve explicit format
    chosen = ext or format
    if chosen:
        e = str(chosen).lower().lstrip(".")
        return e, p.with_suffix("." + e)

    # Infer from suffix
    if p.suffix:
        e = p.suffix.lower().lstrip(".")
        return e, p

    # Default to svg when no suffix
    return "svg", p.with_suffix(".svg")


def _resolve_output_names(
    filename: str | Path,
    variants: Sequence[Variant],
    ext: str | None,
    format: str | None,
):
    fmt, base = _infer_format_from_filename(filename, ext, format)
    base_str = str(base)

    out: list[tuple[Variant, str, str]] = []

    # Template mode: "scikit-plots-{variant}.svg"
    if "{variant}" in base_str:
        for v in variants:
            out.append((v, base_str.format(variant=v), fmt))
        return out

    # Multiple variants: append "-{variant}"
    if len(variants) > 1:
        stem = str(base.with_suffix(""))
        suffix = base.suffix or f".{fmt}"
        for v in variants:
            out.append((v, f"{stem}-{v}{suffix}", fmt))
        return out

    return [(variants[0], base_str, fmt)]


def save(  # noqa: D417
    filename: str | Path,
    *,
    ext: str | None = None,
    format: str | None = None,
    variants: Variant | Sequence[Variant] | None = None,
    variant: Variant | None = None,
    theme: Theme = "light",
    mono: bool = False,
    dots: DotsMode = "fixed",
    preset: Preset | None = None,
    size: float = 4,
    dpi: int = 200,
    seed: int = 2,
    transparent: bool | None = True,
    bbox_inches=None,
    pad_inches: float = 0.0,
    **kwargs,
):
    """
    Save one or multiple logo variants to disk.

    This function follows Matplotlib conventions:

    - Format is inferred from the filename suffix.
    - You can override inference with ``ext=`` or ``format=``.
    - Any extra ``**kwargs`` are forwarded to ``Figure.savefig``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output filename. If multiple variants are requested:
        - If the name contains ``{variant}``, it will be expanded.
        - Otherwise, ``-<variant>`` will be appended before the extension.
    ext : str, optional
        Format override (e.g., "svg", "png"). Preferred over ``format``.
    format : str, optional
        Alias for Matplotlib-like parameter naming.
    variants : {"primary","small","metrics","knn"} or sequence, optional
        Batch export. If provided, takes precedence over ``variant``.
    variant : {"primary","small","metrics","knn"}, optional
        Single-variant convenience when ``variants`` is not used.
    theme : {"light","dark"}, default="light"
    mono : bool, default=False
    dots : {"fixed","random","none"}, default="fixed"
    preset : {"favicon","avatar","docs-hero"} or None, default=None
        Applies contextual defaults to variant/dots/size/dpi.
    size : float, default=4
        Figure size (square inches).
    dpi : int, default=200
        Figure DPI.
    seed : int, default=2
        Random seed for ``dots="random"``.
    transparent : bool or None, default=True
        Passed to ``savefig``.
    bbox_inches : optional
        Passed to ``savefig``.
    pad_inches : float, default=0.0
        Passed to ``savefig``.
    **kwargs
        Forwarded to ``Figure.savefig``.

    Returns
    -------
    list of str
        Paths written.

    See Also
    --------
    matplotlib.figure.Figure.savefig
    create
    show
    wordmark.save

    Examples
    --------
    >>> import scikitplot as sp
    >>> sp._logo.save("scikit-plots.svg")
    ['scikit-plots.svg']

    >>> sp._logo.save("scikit-plots.png", dpi=300)
    ['scikit-plots.png']

    >>> sp._logo.save(
    ...     "assets/scikit-plots-{variant}.svg",
    ...     variants=["primary", "small", "metrics", "knn"],
    ... )
    ['assets/scikit-plots-primary.svg', ...]
    """
    if variants is not None:
        if isinstance(variants, str):
            variants_list: Sequence[Variant] = (variants,)  # type: ignore[]
        else:
            variants_list = tuple(variants)
    else:
        variants_list = (variant or "primary",)

    # Apply preset defaults (only if user kept global defaults)
    v0 = variants_list[0]
    v_resolved, dots_resolved, size_resolved, dpi_resolved = _apply_preset(
        preset=preset,
        variant=v0,
        dots=dots,
        size=size,
        dpi=dpi,
    )

    # If preset changed the single variant and user didn't explicitly pass variants,
    # update variants_list to reflect preset.
    if variants is None:
        variants_list = (v_resolved,)
    else:
        # keep the explicit variants as-is
        dots_resolved, size_resolved, dpi_resolved = dots, size, dpi

    outputs = _resolve_output_names(filename, variants_list, ext, format)
    written: list[str] = []

    for v, out_name, fmt in outputs:
        fig, _ = create(
            variant=v,
            theme=theme,
            mono=mono,
            dots=(
                dots_resolved
                if v == "primary"
                else ("none" if v == "small" else dots_resolved)
            ),
            preset=None,  # avoid double-applying preset
            size=size_resolved,
            dpi=dpi_resolved,
            seed=seed,
        )

        fig.savefig(
            out_name,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **kwargs,
        )
        plt.close(fig)
        written.append(out_name)

    return written


# -----------------------------
# Wordmark API
# -----------------------------


def _draw_wordmark(
    ax_icon,
    ax_text,
    *,
    text: str,
    icon_variant: Variant,
    theme: Theme,
    mono: bool,
    dots: DotsMode,
    seed: int,
    text_size: int,
    weight: str,
    letter_spacing: float,
):
    # Icon axes
    draw(ax_icon, variant=icon_variant, theme=theme, mono=mono, dots=dots, seed=seed)

    # Text axes
    P = _palette(theme=theme, mono=mono)  # noqa: N806
    NAVY, BG = P["NAVY"], P["BG"]  # noqa: N806
    ax_text.axis("off")
    ax_text.set_facecolor(BG)

    # Simple letter spacing emulation by tiny offset per char
    # (kept conservative for Matplotlib portability)
    if letter_spacing <= 0:
        ax_text.text(
            0.0,
            0.5,
            text,
            ha="left",
            va="center",
            fontsize=text_size,
            color=NAVY,
            fontweight=weight,
            transform=ax_text.transAxes,
        )
        return

    x = 0.0
    for ch in text:
        ax_text.text(
            x,
            0.5,
            ch,
            ha="left",
            va="center",
            fontsize=text_size,
            color=NAVY,
            fontweight=weight,
            transform=ax_text.transAxes,
        )
        # heuristic advance
        x += 0.03 + letter_spacing


@dataclass
class _WordmarkAPI:
    """
    Wordmark helper API.

    This object is exposed as ``scikitplot._logo.wordmark``.
    """

    def create(  # noqa: D417
        self,
        *,
        text: str = "scikit-plots",
        icon_variant: Variant = "primary",
        theme: Theme = "light",
        mono: bool = False,
        dots: DotsMode = "fixed",
        preset: Preset | None = None,
        size: float = 6.0,
        dpi: int = 200,
        seed: int = 2,
        text_size: int = 40,
        weight: str = "semibold",
        letter_spacing: float = 0.0,
        icon_ratio: float = 0.28,
    ):
        """
        Create a wordmark (icon + text) figure.

        Parameters
        ----------
        text : str, default="scikit-plots"
            Wordmark text.
        icon_variant : {"primary","small","metrics","knn"}, default="primary"
            Variant used for the icon.
        theme : {"light","dark"}, default="light"
        mono : bool, default=False
        dots : {"fixed","random","none"}, default="fixed"
        preset : {"favicon","avatar","docs-hero"} or None, default=None
            Applies contextual defaults to size/dpi and may recommend icon variant.
        size : float, default=6.0
            Figure width in inches (height is derived for a clean lockup).
        dpi : int, default=200
        seed : int, default=2
        text_size : int, default=40
        weight : str, default="semibold"
        letter_spacing : float, default=0.0
            Small positive values add subtle spacing.
        icon_ratio : float, default=0.28
            Horizontal fraction of the figure reserved for the icon.

        Returns
        -------
        fig, (ax_icon, ax_text)

        See Also
        --------
        scikitplot._logo.create
        scikitplot._logo.save

        Examples
        --------
        >>> import scikitplot as sp
        >>> fig, (ax_i, ax_t) = sp._logo.wordmark.create()
        >>> fig.savefig("lockup.svg", transparent=True)
        """
        # Preset application (light touch)
        if preset == "docs-hero":
            icon_variant = "primary"
            text_size = max(text_size, 44)

        # Height tuned for a modern horizontal lockup
        height = size * 0.28
        fig = plt.figure(figsize=(size, height), dpi=dpi)

        # Two axes: icon + text
        ax_icon = fig.add_axes([0.0, 0.0, icon_ratio, 1.0])
        ax_text = fig.add_axes([icon_ratio + 0.02, 0.0, 1.0 - icon_ratio - 0.02, 1.0])

        _draw_wordmark(
            ax_icon,
            ax_text,
            text=text,
            icon_variant=icon_variant,
            theme=theme,
            mono=mono,
            dots=dots,
            seed=seed,
            text_size=text_size,
            weight=weight,
            letter_spacing=letter_spacing,
        )

        return fig, (ax_icon, ax_text)

    def show(self, **kwargs):
        """
        Display a wordmark.

        Parameters are forwarded to :meth:`create`.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Examples
        --------
        >>> import scikitplot as sp
        >>> sp._logo.wordmark.show()
        """
        fig, _ = self.create(**kwargs)
        plt.show()
        return fig

    def save(
        self,
        filename: str | Path,
        *,
        ext: str | None = None,
        format: str | None = None,
        preset: Preset | None = None,
        transparent: bool | None = True,
        bbox_inches=None,
        pad_inches: float = 0.0,
        **kwargs,
    ):
        """
        Save a wordmark to disk.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output filename. Format inferred from suffix unless overridden.
        ext, format : str, optional
            Format override (e.g., "svg", "png").
        preset : {"favicon","avatar","docs-hero"} or None, default=None
            Applies contextual defaults.
        transparent, bbox_inches, pad_inches :
            Passed to ``savefig``.
        **kwargs :
            Passed to :meth:`create` and then to ``savefig``.

        Returns
        -------
        str
            Path written.

        See Also
        --------
        scikitplot._logo.save
        matplotlib.figure.Figure.savefig

        Examples
        --------
        >>> import scikitplot as sp
        >>> sp._logo.wordmark.save("scikit-plots-lockup.svg")
        """
        fmt, out_path = _infer_format_from_filename(filename, ext, format)

        # Allow preset to influence size/dpi if caller didn't override
        size = kwargs.pop("size", 6.0)
        dpi = kwargs.pop("dpi", 200)
        if preset == "docs-hero":
            size = 8.0 if size == 6.0 else size  # noqa: PLR2004
            dpi = 200 if dpi == 200 else dpi  # noqa: PLR2004

        fig, _ = self.create(preset=preset, size=size, dpi=dpi, **kwargs)
        fig.savefig(
            str(out_path),
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        plt.close(fig)
        return str(out_path)


# Public wordmark API object
wordmark = _WordmarkAPI()


# -----------------------------
# CLI entry point
# -----------------------------


def _cli_build_variants(arg_variants: list[str] | None, all_flag: bool):
    if all_flag:
        return list(_VARIANTS)
    if not arg_variants:
        return ["primary"]
    return arg_variants


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command line interface for logo generation.

    Examples
    --------
    Generate all variants into a directory:

    >>> python -m scikitplot.logo --all --out assets

    Generate a favicon-sized small icon:

    >>> python -m scikitplot.logo --preset favicon --out assets

    Save a wordmark lockup:

    >>> python -m scikitplot.logo --wordmark --out assets --format svg
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m scikitplot.logo")
    parser.add_argument("--out", type=str, default=".", help="Output directory.")
    parser.add_argument(
        "--name", type=str, default="scikit-plots", help="Base filename stem."
    )
    parser.add_argument(
        "--format", type=str, default="svg", help="Output format (svg/png/pdf)."
    )
    parser.add_argument("--theme", type=str, default="light", choices=["light", "dark"])
    parser.add_argument("--mono", action="store_true", help="Monochrome rendering.")
    parser.add_argument(
        "--dots", type=str, default="fixed", choices=["fixed", "random", "none"]
    )
    parser.add_argument(
        "--preset", type=str, default=None, choices=[*list_size_presets()]
    )
    parser.add_argument(
        "--variant", action="append", help="Variant to export (repeatable)."
    )
    parser.add_argument("--all", action="store_true", help="Export all variants.")
    parser.add_argument("--size", type=float, default=4.0, help="Icon size in inches.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for raster outputs.")
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument(
        "--wordmark", action="store_true", help="Generate wordmark instead of icon."
    )
    parser.add_argument(
        "--text", type=str, default="scikit-plots", help="Wordmark text."
    )

    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wordmark:
        fname = out_dir / f"{args.name}-lockup.{args.format}"
        wordmark.save(
            fname,
            format=args.format,
            preset=args.preset,  # type: ignore[]
            text=args.text,
            theme=args.theme,  # type: ignore[]
            mono=args.mono,
            dots=args.dots,  # type: ignore[]
            size=max(args.size, 6.0),
            dpi=args.dpi,
            seed=args.seed,
        )
        return 0

    variants = _cli_build_variants(args.variant, args.all)
    template = out_dir / f"{args.name}-{{variant}}.{args.format}"

    save(
        template,
        format=args.format,
        variants=variants,  # type: ignore[]
        theme=args.theme,  # type: ignore[]
        mono=args.mono,
        dots=args.dots,  # type: ignore[]
        preset=args.preset,  # type: ignore[]
        size=args.size,
        dpi=args.dpi,
        seed=args.seed,
        transparent=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
