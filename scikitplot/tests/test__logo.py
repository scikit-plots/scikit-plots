"""
Tests for scikitplot._logo

Goals
-----
- Validate public helper lists.
- Ensure save() works and infers formats.
- Guarantee brand determinism at the artist/geometry level.
- Avoid flaky byte-for-byte SVG comparisons unless Matplotlib supports
  stable SVG hashing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scikitplot as sp

from matplotlib.patches import Circle, Wedge, FancyBboxPatch, Rectangle


# -----------------------------------------------------------------------------
# Helpers: stable artist signatures
# -----------------------------------------------------------------------------

def _normalize_svg_bytes(data: bytes) -> bytes:
    # NOTE: This intentionally collapses ALL ids to a single token for comparison.
    # # It will produce invalid SVG if used for output — tests only.
    s = data.decode("utf-8")

    # Remove metadata block entirely (can embed tool/version/date)
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.DOTALL)

    # Normalize ANY id="..."
    s = re.sub(r'\bid="[^"]+"', 'id="FIXED"', s)

    # Normalize url(#...) references (clip-path, masks, filters, css)
    s = re.sub(r'url\(#[-A-Za-z0-9_:\.]+\)', 'url(#FIXED)', s)

    # Normalize href/xlink:href="#..."
    s = re.sub(r'\b(xlink:href|href)="#[^"]+"', r'\1="#FIXED"', s)

    # Normalize direct attributes that point to ids
    s = re.sub(
        r'\b(clip-path|mask|filter)="url\(#[-A-Za-z0-9_:\.]+\)"',
        lambda m: f'{m.group(1)}="url(#FIXED)"',
        s,
    )

    # Normalize whitespace between tags
    s = re.sub(r">\s+<", "><", s)
    s = s.strip()

    return s.encode("utf-8")

def _round_tuple(x, n=6):
    if x is None:
        return None
    return tuple(round(float(v), n) for v in x)


def _patch_signature(p):
    """Return a stable, renderer-independent signature for a patch."""
    name = p.__class__.__name__

    # Circle
    if isinstance(p, Circle):
        c = p.center
        r = p.radius
        return (
            "patch", name,
            _round_tuple(c),
            round(float(r), 6),
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    # Wedge (pie slices)
    if isinstance(p, Wedge):
        # Wedge exposes center, r, theta1, theta2, width
        return (
            "patch", name,
            _round_tuple(p.center),
            round(float(p.r), 6),
            round(float(p.theta1), 6),
            round(float(p.theta2), 6),
            None if p.width is None else round(float(p.width), 6),
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    # FancyBboxPatch (bars)
    if isinstance(p, FancyBboxPatch):
        x, y = p.get_x(), p.get_y()
        w, h = p.get_width(), p.get_height()
        # boxstyle contains rounding details; stringify type for stability
        bs = p.get_boxstyle()
        bs_name = bs.__class__.__name__ if bs is not None else None
        return (
            "patch", name,
            round(float(x), 6),
            round(float(y), 6),
            round(float(w), 6),
            round(float(h), 6),
            bs_name,
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    # Rectangle
    if isinstance(p, Rectangle):
        x, y = p.get_x(), p.get_y()
        w, h = p.get_width(), p.get_height()
        return (
            "patch", name,
            round(float(x), 6),
            round(float(y), 6),
            round(float(w), 6),
            round(float(h), 6),
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    # Fallback: use extents but only after draw
    # (kept as a last resort)
    try:
        bbox = p.get_extents().bounds
        return ("patch", name, _round_tuple(bbox), _round_tuple(p.get_facecolor()))
    except Exception:
        return ("patch", name, "unknown")


def _line_signature(l):
    """Return a stable signature for a Line2D."""
    x, y = l.get_data()
    color = mpl.colors.to_rgba(l.get_color())
    return (
        "line",
        tuple(np.round(np.asarray(x, dtype=float), 6)),
        tuple(np.round(np.asarray(y, dtype=float), 6)),
        _round_tuple(color),
        round(float(l.get_linewidth()), 6),
        l.get_solid_capstyle(),
    )


def _ax_signature(ax):
    """Create a stable signature of all drawn artists."""
    sig = []

    for p in ax.patches:
        sig.append(_patch_signature(p))

    for l in ax.lines:
        sig.append(_line_signature(l))

    return sorted(sig, key=repr)


# -----------------------------------------------------------------------------
# Required tests
# -----------------------------------------------------------------------------

def test_list_variants():
    assert sp._logo.list_variants() == ("primary", "small", "metrics", "knn")


def test_list_size_presets():
    assert "favicon" in sp._logo.list_size_presets()


def test_save_single(tmp_path):
    out = tmp_path / "logo.svg"
    paths = sp._logo.save(out)
    assert len(paths) == 1
    assert out.exists()
    assert out.stat().st_size > 100


def test_deterministic_primary_artists():
    """
    Main determinism guarantee.

    We validate that two independently created figures produce identical
    artist geometry + style when using fixed dots.
    """
    fig1, ax1 = sp._logo.create(variant="primary", dots="fixed")
    fig2, ax2 = sp._logo.create(variant="primary", dots="fixed")

    try:
        # Force a draw to finalize transforms (safe + consistent)
        fig1.canvas.draw()
        fig2.canvas.draw()

        assert _ax_signature(ax1) == _ax_signature(ax2)
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_save_primary_svg_smoke(tmp_path):
    out = tmp_path / "primary.svg"
    sp._logo.save(out, variant="primary", dots="fixed")
    assert out.exists()
    assert out.stat().st_size > 100


def test_wordmark_save(tmp_path):
    out = tmp_path / "lockup.svg"
    p = sp._logo.wordmark.save(out)
    assert out.exists()
    assert str(p).endswith(".svg")


# -----------------------------------------------------------------------------
# Optional strict determinism tests (safe guards)
# -----------------------------------------------------------------------------

@pytest.mark.skipif("svg.hashsalt" not in mpl.rcParams, reason="svg.hashsalt not supported")
def test_deterministic_primary_svg_bytes_with_hashsalt(tmp_path):
    """
    SVG backend stability check.

    Matplotlib 3.10.x may still differ byte-for-byte even with hashsalt.
    We therefore normalize IDs/metadata and compare the stable structure.
    """
    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"

    with mpl.rc_context({"svg.hashsalt": "scikit-plots-test"}):
        # Passing metadata={} can reduce timestamp-like noise if backend uses it
        sp._logo.save(a, variant="primary", dots="fixed", metadata={})
        sp._logo.save(b, variant="primary", dots="fixed", metadata={})

    assert _normalize_svg_bytes(a.read_bytes()) == _normalize_svg_bytes(b.read_bytes())


def test_deterministic_primary_png_bytes(tmp_path):
    """
    Raster output is typically stable for byte comparisons.
    """
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"

    sp._logo.save(a, variant="primary", dots="fixed", format="png", dpi=300)
    sp._logo.save(b, variant="primary", dots="fixed", format="png", dpi=300)

    assert a.read_bytes() == b.read_bytes()


# -----------------------------------------------------------------------------
# "More" useful coverage (recommended extras)
# -----------------------------------------------------------------------------

def test_save_multiple_variants_auto_suffix(tmp_path):
    """
    When saving multiple variants without a template,
    -{variant} should be appended.
    """
    base = tmp_path / "scikit-plots.svg"
    paths = sp._logo.save(base, variants=sp._logo.list_variants())

    assert len(paths) == 4
    for v in sp._logo.list_variants():
        assert any(str(p).endswith(f"-{v}.svg") for p in paths)


def test_save_multiple_variants_template(tmp_path):
    """
    Template format should expand {variant}.
    """
    template = tmp_path / "scikit-plots-{variant}.svg"
    paths = sp._logo.save(template, variants=["primary", "small"])

    assert len(paths) == 2
    assert (tmp_path / "scikit-plots-primary.svg").exists()
    assert (tmp_path / "scikit-plots-small.svg").exists()


def test_preset_favicon_smoke(tmp_path):
    """
    Preset should produce a small clean icon without error.
    """
    out = tmp_path / "favicon.png"
    sp._logo.save(out, preset="favicon", format="png")
    assert out.exists()
    assert out.stat().st_size > 100


def test_wordmark_create_smoke():
    fig, _ = sp._logo.wordmark.create()
    try:
        fig.canvas.draw()
    finally:
        plt.close(fig)
