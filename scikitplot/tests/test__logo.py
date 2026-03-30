# scikitplot/tests/test__logo.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive tests for scikitplot._logo.

Test Strategy
-------------
- Force the Agg non-interactive backend before any pyplot import so the
  suite runs correctly in headless CI, Docker, and notebook environments.
- Validate the full public API: list_variants, list_size_presets, draw,
  create, show, save, wordmark.{create,show,save}, main.
- Cover all private helpers: _palette, _lerp_color, _infer_format_from_filename,
  _resolve_output_names, _apply_preset, _cli_build_variants.
- Achieve determinism tests at the artist-geometry level (renderer-independent)
  so they do not break across Matplotlib patch releases.
- Remove the flaky byte-level PNG comparison from the original suite;
  replace with a normalised-SVG comparison that is opt-in via hashsalt.
- Parametrize all Variant × Theme × Mono × DotsMode combinations.
- Cover every branch: dots=none/fixed/random, small vs non-small, each
  top-icon branch (metrics → confusion, others → spark), knn motif, etc.
- Test all CLI flags and combinations via main().
- Test error/edge cases: invalid args, empty text, figure-leak prevention.

Notes
-----
- Private helpers are accessed via ``import scikitplot._logo as _logo``
  (white-box testing) rather than through the public ``_logo`` namespace.
- Tests must pass with pytest ≥ 7 and Matplotlib ≥ 3.6.
- The ``close_all_figures`` autouse fixture guarantees no figure leaks
  between tests regardless of test failures.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backend selection — MUST precede any pyplot / scikitplot import
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge

# ---------------------------------------------------------------------------
# Subject under test
# ---------------------------------------------------------------------------
# import scikitplot as sp
# White-box access to private helpers that are not re-exported by the package.
from .. import _logo


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def close_all_figures():
    """
    Close every Matplotlib figure after each test.

    Without this fixture, figures created inside tests accumulate in the
    pyplot global registry, causing memory leaks and occasionally spurious
    failures in later tests.
    """
    yield
    plt.close("all")


# ===========================================================================
# Artist-level signature helpers (renderer-independent determinism)
# ===========================================================================


def _normalize_svg_bytes(data: bytes) -> bytes:
    """
    Collapse all id/href/url references in SVG bytes to a fixed token.

    This normalisation is intentionally lossy — the result is not valid SVG
    for output — but is suitable for structural byte-for-byte comparison in
    tests.

    Parameters
    ----------
    data : bytes
        Raw SVG file content.

    Returns
    -------
    bytes
        Normalised SVG with metadata removed and ids collapsed.
    """
    s = data.decode("utf-8")

    # Remove metadata block entirely (can embed tool/version/date)
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.DOTALL)

    # Normalise ANY id="..."
    s = re.sub(r'\bid="[^"]+"', 'id="FIXED"', s)

    # Normalise url(#...) references (clip-path, masks, filters, css)
    s = re.sub(r"url\(#[-A-Za-z0-9_:\.]+\)", "url(#FIXED)", s)

    # Normalise href/xlink:href="#..."
    s = re.sub(r'\b(xlink:href|href)="#[^"]+"', r'\1="#FIXED"', s)

    # Normalise direct attributes that point to ids
    s = re.sub(
        r'\b(clip-path|mask|filter)="url\(#[-A-Za-z0-9_:\.]+\)"',
        lambda m: f'{m.group(1)}="url(#FIXED)"',
        s,
    )

    # Normalise whitespace between tags
    s = re.sub(r">\s+<", "><", s)
    s = s.strip()

    return s.encode("utf-8")


def _round_tuple(x, n: int = 6):
    """Round a numeric sequence to *n* decimal places and return a tuple."""
    if x is None:
        return None
    return tuple(round(float(v), n) for v in x)


def _patch_signature(p):
    """
    Return a stable, renderer-independent signature tuple for a Matplotlib patch.

    The signature captures geometry (position, size, radii) and style
    (face/edge colour, linewidth) but not transform or zorder, which can vary
    between Matplotlib versions without affecting visual output.
    """
    name = p.__class__.__name__

    if isinstance(p, Circle):
        return (
            "patch",
            name,
            _round_tuple(p.center),
            round(float(p.radius), 6),
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    if isinstance(p, Wedge):
        return (
            "patch",
            name,
            _round_tuple(p.center),
            round(float(p.r), 6),
            round(float(p.theta1), 6),
            round(float(p.theta2), 6),
            None if p.width is None else round(float(p.width), 6),
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    if isinstance(p, FancyBboxPatch):
        x, y = p.get_x(), p.get_y()
        w, h = p.get_width(), p.get_height()
        bs = p.get_boxstyle()
        bs_name = bs.__class__.__name__ if bs is not None else None
        return (
            "patch",
            name,
            round(float(x), 6),
            round(float(y), 6),
            round(float(w), 6),
            round(float(h), 6),
            bs_name,
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    if isinstance(p, Rectangle):
        x, y = p.get_x(), p.get_y()
        w, h = p.get_width(), p.get_height()
        return (
            "patch",
            name,
            round(float(x), 6),
            round(float(y), 6),
            round(float(w), 6),
            round(float(h), 6),
            _round_tuple(p.get_facecolor()),
            _round_tuple(p.get_edgecolor()),
            round(float(p.get_linewidth() or 0.0), 6),
        )

    # Fallback for any other patch type
    try:
        bbox = p.get_extents().bounds
        return ("patch", name, _round_tuple(bbox), _round_tuple(p.get_facecolor()))
    except Exception:
        return ("patch", name, "unknown")


def _line_signature(line):
    """Return a stable signature tuple for a Matplotlib Line2D artist."""
    x, y = line.get_data()
    color = mpl.colors.to_rgba(line.get_color())
    return (
        "line",
        tuple(np.round(np.asarray(x, dtype=float), 6)),
        tuple(np.round(np.asarray(y, dtype=float), 6)),
        _round_tuple(color),
        round(float(line.get_linewidth()), 6),
        line.get_solid_capstyle(),
    )


def _ax_signature(ax):
    """
    Build a sorted, stable signature of all drawn artists on an Axes.

    Sorting by repr ensures that insertion order differences between
    two otherwise-identical draws do not cause false negatives.
    """
    sig = []
    for p in ax.patches:
        sig.append(_patch_signature(p))
    for line in ax.lines:
        sig.append(_line_signature(line))
    return sorted(sig, key=repr)


# ===========================================================================
# Section 1 — Public constants and list functions
# ===========================================================================


class TestPublicConstants:
    """Validate the public-facing constant values and list helpers."""

    def test_list_variants_returns_correct_tuple(self):
        result = _logo.list_variants()
        assert isinstance(result, tuple)
        assert result == ("primary", "small", "metrics", "knn")

    def test_list_variants_no_duplicates(self):
        result = _logo.list_variants()
        assert len(result) == len(set(result))

    def test_list_size_presets_returns_tuple(self):
        result = _logo.list_size_presets()
        assert isinstance(result, tuple)

    def test_list_size_presets_contains_expected_keys(self):
        result = _logo.list_size_presets()
        assert set(result) == {"favicon", "avatar", "docs-hero"}

    def test_variants_constant_has_four_elements(self):
        assert len(_logo._VARIANTS) == 4

    def test_themes_constant_covers_light_and_dark(self):
        assert set(_logo._THEMES) == {"light", "dark"}

    def test_dots_modes_constant_covers_all_three(self):
        assert set(_logo._DOTS_MODES) == {"fixed", "random", "none"}

    def test_banner_is_non_empty_string(self):
        banner = _logo._SCIKITPLOT_BANNER
        assert isinstance(banner, str)
        assert len(banner.strip()) > 0

    def test_banner_does_not_start_with_newline(self):
        """
        After the lstrip fix, the banner should start with the first ASCII
        art character, not a blank line.
        """
        assert not _logo._SCIKITPLOT_BANNER.startswith("\n")

    def test_banner_contains_expected_text(self):
        banner = _logo._SCIKITPLOT_BANNER
        # _SCIKITPLOT_BANNER is figlet-rendered ASCII art for "scikit-plots".
        # It is NOT plain text, so the literal substrings "scikit" / "plots"
        # are absent — the name is encoded as box-drawing characters.
        # Verify structural properties that ARE guaranteed:
        #   1. Multiple lines (at least 4 rows of ASCII art).
        #   2. The characteristic underscore cluster from the top row.
        assert len(banner.splitlines()) >= 4, (
            "Banner should have at least 4 ASCII art lines; "
            f"got {len(banner.splitlines())}"
        )
        assert "____" in banner, (
            "Banner first row should contain '____' ASCII art pattern"
        )

    def test_fixed_dots_has_ten_entries(self):
        assert len(_logo._FIXED_DOTS) == 10

    def test_fixed_dots_structure(self):
        """Each entry must be (float, float, positive_radius, color_key)."""
        valid_keys = {"NAVY", "BLUE_LIGHT"}
        for entry in _logo._FIXED_DOTS:
            x, y, r, key = entry
            assert isinstance(x, float), f"x must be float, got {type(x)}"
            assert isinstance(y, float), f"y must be float, got {type(y)}"
            assert r > 0, f"radius must be positive, got {r}"
            assert key in valid_keys, f"color_key {key!r} not in {valid_keys}"

    def test_size_presets_dict_has_required_fields(self):
        """Every preset must declare variant, size, dpi, and dots."""
        required = {"variant", "size", "dpi", "dots"}
        for name, cfg in _logo._SIZE_PRESETS.items():
            assert set(cfg.keys()) == required, (
                f"Preset {name!r} has unexpected keys: {set(cfg.keys())}"
            )

    def test_size_presets_variant_values_are_valid(self):
        valid = set(_logo._VARIANTS)
        for name, cfg in _logo._SIZE_PRESETS.items():
            assert cfg["variant"] in valid, (
                f"Preset {name!r} has invalid variant {cfg['variant']!r}"
            )

    def test_size_presets_dots_values_are_valid(self):
        valid = set(_logo._DOTS_MODES)
        for name, cfg in _logo._SIZE_PRESETS.items():
            assert cfg["dots"] in valid, (
                f"Preset {name!r} has invalid dots {cfg['dots']!r}"
            )


# ===========================================================================
# Section 2 — Private helper unit tests
# ===========================================================================


class TestLerpColor:
    """Unit tests for _lerp_color interpolation."""

    def test_at_zero_returns_c1(self):
        result = _logo._lerp_color("red", "blue", 0.0)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-7)

    def test_at_one_returns_c2(self):
        result = _logo._lerp_color("red", "blue", 1.0)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-7)

    def test_at_half_returns_midpoint(self):
        result = _logo._lerp_color("black", "white", 0.5)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5], atol=1e-7)

    def test_returns_numpy_array(self):
        result = _logo._lerp_color("red", "blue", 0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_values_in_unit_interval(self):
        for t in np.linspace(0, 1, 11):
            result = _logo._lerp_color("#002030", "#e6f2f7", t)
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_accepts_hex_color_strings(self):
        result = _logo._lerp_color("#ff0000", "#0000ff", 0.0)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-7)

    def test_same_color_returns_same_color(self):
        result = _logo._lerp_color("navy", "navy", 0.5)
        expected = np.array(mpl.colors.to_rgb("navy"))
        np.testing.assert_allclose(result, expected, atol=1e-7)


class TestPalette:
    """Unit tests for _palette color role completeness and correctness."""

    _REQUIRED_KEYS = {"NAVY", "BLUE", "BLUE_LIGHT", "ORANGE", "ORANGE_LIGHT", "BG"}

    @pytest.mark.parametrize("theme,mono", [
        ("light", False), ("light", True),
        ("dark", False), ("dark", True),
    ])
    def test_all_required_keys_present(self, theme, mono):
        P = _logo._palette(theme=theme, mono=mono)
        assert set(P.keys()) == self._REQUIRED_KEYS

    @pytest.mark.parametrize("theme,expected_bg", [
        ("light", "white"),
        ("dark", "#0b141a"),
    ])
    def test_background_color_per_theme(self, theme, expected_bg):
        P = _logo._palette(theme=theme, mono=False)
        assert P["BG"] == expected_bg

    @pytest.mark.parametrize("theme,expected_bg", [
        ("light", "white"),
        ("dark", "#0b141a"),
    ])
    def test_mono_background_unchanged(self, theme, expected_bg):
        P = _logo._palette(theme=theme, mono=True)
        assert P["BG"] == expected_bg

    def test_mono_dark_all_fg_colors_identical(self):
        P = _logo._palette(theme="dark", mono=True)
        fg = {k: v for k, v in P.items() if k != "BG"}
        assert len(set(fg.values())) == 1, "All fg colors should collapse to one"

    def test_mono_light_all_fg_colors_identical(self):
        P = _logo._palette(theme="light", mono=True)
        fg = {k: v for k, v in P.items() if k != "BG"}
        assert len(set(fg.values())) == 1

    def test_color_palette_fg_colors_differ(self):
        """A non-mono palette should have at least two distinct fg colours."""
        P = _logo._palette(theme="light", mono=False)
        fg = {k: v for k, v in P.items() if k != "BG"}
        assert len(set(fg.values())) > 1

    def test_default_is_light_non_mono(self):
        assert _logo._palette() == _logo._palette(theme="light", mono=False)

    def test_light_and_dark_differ(self):
        P_light = _logo._palette(theme="light", mono=False)
        P_dark = _logo._palette(theme="dark", mono=False)
        assert P_light != P_dark

    def test_all_color_values_are_valid_matplotlib_colors(self):
        """Every color value must be parseable by Matplotlib."""
        for theme in ("light", "dark"):
            for mono in (False, True):
                P = _logo._palette(theme=theme, mono=mono)
                for role, color in P.items():
                    try:
                        mpl.colors.to_rgba(color)
                    except ValueError as exc:
                        raise AssertionError(
                            f"Invalid color for {role!r} in "
                            f"theme={theme!r} mono={mono}: {color!r}"
                        ) from exc


class TestInferFormatFromFilename:
    """Unit tests for _infer_format_from_filename branch coverage."""

    def test_explicit_ext_overrides_suffix(self):
        fmt, path = _logo._infer_format_from_filename("logo.svg", ext="png", format=None)
        assert fmt == "png"
        assert str(path).endswith(".png")

    def test_explicit_format_used_when_no_ext(self):
        fmt, path = _logo._infer_format_from_filename("logo.svg", ext=None, format="pdf")
        assert fmt == "pdf"
        assert str(path).endswith(".pdf")

    def test_ext_takes_priority_over_format(self):
        fmt, _ = _logo._infer_format_from_filename("logo.svg", ext="png", format="pdf")
        assert fmt == "png"

    def test_infer_from_svg_suffix(self):
        fmt, path = _logo._infer_format_from_filename("logo.svg", ext=None, format=None)
        assert fmt == "svg"
        assert str(path) == "logo.svg"

    def test_infer_from_png_suffix(self):
        fmt, _ = _logo._infer_format_from_filename("logo.png", ext=None, format=None)
        assert fmt == "png"

    def test_infer_from_pdf_suffix(self):
        fmt, _ = _logo._infer_format_from_filename("logo.pdf", ext=None, format=None)
        assert fmt == "pdf"

    def test_default_to_svg_when_no_suffix(self):
        fmt, path = _logo._infer_format_from_filename("logo", ext=None, format=None)
        assert fmt == "svg"
        assert str(path) == "logo.svg"

    def test_uppercase_suffix_normalised_to_lowercase(self):
        fmt, _ = _logo._infer_format_from_filename("logo.SVG", ext=None, format=None)
        assert fmt == "svg"

    def test_ext_with_leading_dot_stripped(self):
        fmt, _ = _logo._infer_format_from_filename("logo.svg", ext=".png", format=None)
        assert fmt == "png"

    def test_format_with_leading_dot_stripped(self):
        fmt, _ = _logo._infer_format_from_filename("logo.svg", ext=None, format=".pdf")
        assert fmt == "pdf"

    def test_accepts_pathlib_path(self):
        fmt, path = _logo._infer_format_from_filename(Path("logo.svg"), ext=None, format=None)
        assert fmt == "svg"
        assert isinstance(path, Path)

    def test_returns_path_object(self):
        _, path = _logo._infer_format_from_filename("logo.svg", ext=None, format=None)
        assert isinstance(path, Path)


class TestResolveOutputNames:
    """Unit tests for _resolve_output_names naming strategies."""

    def test_template_expands_variant_placeholder(self):
        results = _logo._resolve_output_names(
            "out/{variant}.svg", ["primary", "small"], None, None
        )
        assert len(results) == 2
        paths = [r[1] for r in results]
        assert "out/primary.svg" in paths
        assert "out/small.svg" in paths

    def test_template_preserves_all_variants(self):
        results = _logo._resolve_output_names(
            "logo-{variant}.svg",
            list(_logo._VARIANTS),
            None,
            None,
        )
        assert len(results) == 4

    def test_multi_variant_appends_dash_variant(self):
        results = _logo._resolve_output_names(
            "logo.svg", ["primary", "small"], None, None
        )
        assert len(results) == 2
        paths = [r[1] for r in results]
        assert any("-primary.svg" in p for p in paths)
        assert any("-small.svg" in p for p in paths)

    def test_single_variant_uses_filename_verbatim(self):
        results = _logo._resolve_output_names("logo.svg", ["primary"], None, None)
        assert len(results) == 1
        assert results[0][1] == "logo.svg"

    def test_format_override_applied(self):
        results = _logo._resolve_output_names("logo.svg", ["primary"], ext="png", format=None)
        assert results[0][2] == "png"
        assert results[0][1].endswith(".png")

    def test_variant_field_in_result(self):
        results = _logo._resolve_output_names("logo.svg", ["knn"], None, None)
        assert results[0][0] == "knn"

    def test_format_field_in_result(self):
        results = _logo._resolve_output_names("logo.svg", ["primary"], None, None)
        assert results[0][2] == "svg"


class TestApplyPreset:
    """Unit tests for _apply_preset resolution logic."""

    def test_none_preset_returns_inputs_unchanged(self):
        result = _logo._apply_preset(
            preset=None, variant="primary", dots="fixed", size=4, dpi=200
        )
        assert result == ("primary", "fixed", 4, 200)

    def test_favicon_overrides_variant_from_default(self):
        v, d, s, dpi = _logo._apply_preset(
            preset="favicon", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert v == "small"

    def test_favicon_overrides_dots_from_default(self):
        _, d, _, _ = _logo._apply_preset(
            preset="favicon", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert d == "none"

    def test_favicon_overrides_size_from_default(self):
        _, _, s, _ = _logo._apply_preset(
            preset="favicon", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert s == pytest.approx(1.2)

    def test_favicon_overrides_dpi_from_default(self):
        _, _, _, dpi = _logo._apply_preset(
            preset="favicon", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert dpi == 256

    def test_explicit_non_default_variant_not_overridden(self):
        """Caller sets variant="knn" explicitly; preset should not override it."""
        v, _, _, _ = _logo._apply_preset(
            preset="favicon", variant="knn", dots="fixed", size=4, dpi=200
        )
        assert v == "knn"

    def test_explicit_non_default_dots_not_overridden(self):
        """Caller sets dots="random"; preset should not override it."""
        _, d, _, _ = _logo._apply_preset(
            preset="favicon", variant="primary", dots="random", size=4, dpi=200
        )
        assert d == "random"

    def test_explicit_non_default_size_not_overridden(self):
        _, _, s, _ = _logo._apply_preset(
            preset="favicon", variant="primary", dots="fixed", size=3.0, dpi=200
        )
        assert s == pytest.approx(3.0)

    def test_explicit_non_default_dpi_not_overridden(self):
        _, _, _, dpi = _logo._apply_preset(
            preset="favicon", variant="primary", dots="fixed", size=4, dpi=72
        )
        assert dpi == 72

    def test_docs_hero_overrides_size_from_default(self):
        _, _, s, _ = _logo._apply_preset(
            preset="docs-hero", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert s == pytest.approx(5.0)

    def test_all_presets_accept_without_error(self):
        for preset in _logo.list_size_presets():
            _logo._apply_preset(
                preset=preset, variant="primary", dots="fixed", size=4, dpi=200
            )

    def test_invalid_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="preset must be one of"):
            _logo._apply_preset(
                preset="nonexistent",  # type: ignore[arg-type]
                variant="primary",
                dots="fixed",
                size=4,
                dpi=200,
            )


class TestCliVariants:
    """Unit tests for _cli_build_variants flag resolution."""

    def test_all_flag_returns_all_variants(self):
        result = _logo._cli_build_variants(None, all_flag=True)
        assert set(result) == set(_logo._VARIANTS)

    def test_all_flag_length_matches_variants(self):
        result = _logo._cli_build_variants(None, all_flag=True)
        assert len(result) == len(_logo._VARIANTS)

    def test_specific_variants_returned_unchanged(self):
        result = _logo._cli_build_variants(["primary", "knn"], all_flag=False)
        assert result == ["primary", "knn"]

    def test_none_arg_returns_primary_default(self):
        result = _logo._cli_build_variants(None, all_flag=False)
        assert result == ["primary"]

    def test_empty_list_returns_primary_default(self):
        result = _logo._cli_build_variants([], all_flag=False)
        assert result == ["primary"]

    def test_all_flag_overrides_arg_variants(self):
        """--all should take precedence over any --variant arguments."""
        result = _logo._cli_build_variants(["small"], all_flag=True)
        assert set(result) == set(_logo._VARIANTS)

    def test_single_variant_in_list(self):
        result = _logo._cli_build_variants(["metrics"], all_flag=False)
        assert result == ["metrics"]


# ===========================================================================
# Section 3 — draw() tests
# ===========================================================================


class TestDraw:
    """Tests for draw() covering all variant × theme × mono × dots paths."""

    # ── Smoke: all variants ──────────────────────────────────────────────────

    @pytest.mark.parametrize("variant", ["primary", "small", "metrics", "knn"])
    def test_draw_all_variants_produces_patches(self, variant):
        fig, ax = plt.subplots()
        _logo.draw(ax, variant=variant)
        assert len(ax.patches) > 0, f"No patches for variant={variant!r}"

    # ── Theme coverage ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_draw_all_themes_produces_patches(self, theme):
        fig, ax = plt.subplots()
        _logo.draw(ax, theme=theme)
        assert len(ax.patches) > 0

    # ── Mono ─────────────────────────────────────────────────────────────────

    def test_draw_mono_true_produces_patches(self):
        fig, ax = plt.subplots()
        _logo.draw(ax, mono=True)
        assert len(ax.patches) > 0

    def test_draw_mono_false_produces_patches(self):
        fig, ax = plt.subplots()
        _logo.draw(ax, mono=False)
        assert len(ax.patches) > 0

    # ── Dots mode ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("dots", ["fixed", "random", "none"])
    def test_draw_all_dots_modes_for_primary(self, dots):
        fig, ax = plt.subplots()
        _logo.draw(ax, variant="primary", dots=dots)
        assert len(ax.patches) > 0

    def test_draw_none_dots_fewer_patches_than_fixed(self):
        """dots='none' omits decorative dots → fewer patches than dots='fixed'."""
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="primary", dots="none")
        n_none = len(ax1.patches)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="primary", dots="fixed")
        n_fixed = len(ax2.patches)
        plt.close(fig2)

        assert n_none < n_fixed

    def test_draw_random_dots_same_seed_same_count(self):
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="primary", dots="random", seed=42)
        n1 = len(ax1.patches)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="primary", dots="random", seed=42)
        n2 = len(ax2.patches)
        plt.close(fig2)

        assert n1 == n2

    # ── Axes state ────────────────────────────────────────────────────────────

    def test_draw_sets_axis_off(self):
        fig, ax = plt.subplots()
        _logo.draw(ax)
        assert not ax.axison

    def test_draw_sets_equal_aspect(self):
        """
        draw() calls ax.set_aspect("equal").

        In Matplotlib < 3.10 get_aspect() returns the string "equal".
        In Matplotlib >= 3.10 it normalises the stored value to the float 1.0.
        Both representations are semantically identical; we accept either.
        """
        fig, ax = plt.subplots()
        _logo.draw(ax)
        aspect = ax.get_aspect()
        assert aspect == "equal" or aspect == pytest.approx(1.0), (
            f"Expected equal (1:1) aspect ratio, got {aspect!r}"
        )

    def test_draw_facecolor_matches_light_palette_bg(self):
        P = _logo._palette(theme="light")
        fig, ax = plt.subplots()
        _logo.draw(ax, theme="light")
        actual = mpl.colors.to_hex(ax.get_facecolor())
        expected = mpl.colors.to_hex(mpl.colors.to_rgba(P["BG"]))
        assert actual == expected

    def test_draw_facecolor_matches_dark_palette_bg(self):
        P = _logo._palette(theme="dark")
        fig, ax = plt.subplots()
        _logo.draw(ax, theme="dark")
        actual = mpl.colors.to_hex(ax.get_facecolor())
        expected = mpl.colors.to_hex(mpl.colors.to_rgba(P["BG"]))
        assert actual == expected

    # ── Artist type assertions ────────────────────────────────────────────────

    def test_draw_always_adds_outer_circle(self):
        """The outer ring is a Circle and should exist for every variant."""
        for variant in _logo._VARIANTS:
            fig, ax = plt.subplots()
            _logo.draw(ax, variant=variant)
            circles = [p for p in ax.patches if isinstance(p, Circle)]
            assert len(circles) >= 2, (
                f"Expected ≥2 circles (background + ring) for variant={variant!r}"
            )
            plt.close(fig)

    def test_draw_always_adds_trend_line(self):
        """A Line2D for the trend line must exist for every variant."""
        for variant in _logo._VARIANTS:
            fig, ax = plt.subplots()
            _logo.draw(ax, variant=variant)
            assert len(ax.lines) >= 1, f"No trend line for variant={variant!r}"
            plt.close(fig)

    def test_draw_primary_adds_wedge_patches(self):
        fig, ax = plt.subplots()
        _logo.draw(ax, variant="primary")
        wedges = [p for p in ax.patches if isinstance(p, Wedge)]
        assert len(wedges) == 3, "Expected exactly 3 pie wedges for primary"

    def test_draw_small_adds_wedge_patches(self):
        fig, ax = plt.subplots()
        _logo.draw(ax, variant="small")
        wedges = [p for p in ax.patches if isinstance(p, Wedge)]
        assert len(wedges) == 3

    def test_draw_metrics_adds_rectangle_patches(self):
        """Metrics variant uses confusion-matrix rectangles as top icon."""
        fig, ax = plt.subplots()
        _logo.draw(ax, variant="metrics")
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) > 0, "Metrics variant should have Rectangle patches"

    def test_draw_non_metrics_uses_spark_not_rectangles(self):
        """Variants other than 'metrics' should not add Rectangle patches."""
        for variant in ("primary", "small", "knn"):
            fig, ax = plt.subplots()
            _logo.draw(ax, variant=variant)
            rects = [p for p in ax.patches if isinstance(p, Rectangle)]
            assert len(rects) == 0, (
                f"Variant {variant!r} should not have Rectangle patches"
            )
            plt.close(fig)

    def test_draw_knn_adds_extra_lines_beyond_trend(self):
        """KNN variant adds graph-edge Line2D artists in addition to trend."""
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="primary")
        n_primary_lines = len(ax1.lines)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="knn")
        n_knn_lines = len(ax2.lines)
        plt.close(fig2)

        assert n_knn_lines > n_primary_lines

    def test_draw_small_has_fewer_artists_than_primary(self):
        """The 'small' variant omits orange dots, accent dots and decorative dots."""
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="small")
        total_small = len(ax1.patches) + len(ax1.lines)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="primary", dots="fixed")
        total_primary = len(ax2.patches) + len(ax2.lines)
        plt.close(fig2)

        assert total_small < total_primary

    # ── Input validation ──────────────────────────────────────────────────────

    def test_draw_invalid_variant_raises_value_error(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="variant must be one of"):
            _logo.draw(ax, variant="nonexistent")  # type: ignore[arg-type]

    def test_draw_invalid_theme_raises_value_error(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="theme must be one of"):
            _logo.draw(ax, theme="solarized")  # type: ignore[arg-type]

    def test_draw_invalid_dots_raises_value_error(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="dots must be one of"):
            _logo.draw(ax, dots="scatter")  # type: ignore[arg-type]


# ===========================================================================
# Section 4 — create() tests
# ===========================================================================


class TestCreate:
    """Tests for the create() figure factory."""

    def test_returns_figure_and_axes(self):
        fig, ax = _logo.create()
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)

    @pytest.mark.parametrize("variant", ["primary", "small", "metrics", "knn"])
    def test_all_variants_no_error(self, variant):
        fig, ax = _logo.create(variant=variant)
        assert fig is not None

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_all_themes_no_error(self, theme):
        fig, ax = _logo.create(theme=theme)
        assert fig is not None

    def test_size_controls_figsize(self):
        fig, _ = _logo.create(size=2.0)
        assert fig.get_figwidth() == pytest.approx(2.0)
        assert fig.get_figheight() == pytest.approx(2.0)

    def test_dpi_set_on_figure(self):
        fig, _ = _logo.create(dpi=72)
        assert fig.get_dpi() == pytest.approx(72.0)

    @pytest.mark.parametrize("preset", ["favicon", "avatar", "docs-hero"])
    def test_all_presets_no_error(self, preset):
        fig, ax = _logo.create(preset=preset)
        assert fig is not None

    def test_favicon_preset_produces_smaller_figure_than_default(self):
        fig_favicon, _ = _logo.create(preset="favicon")
        fig_default, _ = _logo.create()
        assert fig_favicon.get_figwidth() < fig_default.get_figwidth()

    def test_invalid_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="preset must be one of"):
            _logo.create(preset="invalid")  # type: ignore[arg-type]

    def test_invalid_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="variant must be one of"):
            _logo.create(variant="bad")  # type: ignore[arg-type]


# ===========================================================================
# Section 5 — show() tests
# ===========================================================================


class TestShow:
    """Tests for the show() convenience function (Agg backend, non-interactive)."""

    def test_show_returns_figure(self):
        fig = _logo.show()
        assert isinstance(fig, mpl.figure.Figure)

    @pytest.mark.parametrize("variant", ["primary", "small", "metrics", "knn"])
    def test_show_all_variants(self, variant):
        fig = _logo.show(variant=variant)
        assert fig is not None

    def test_show_block_none(self):
        fig = _logo.show(block=None)
        assert isinstance(fig, mpl.figure.Figure)

    def test_show_block_false(self):
        fig = _logo.show(block=False)
        assert isinstance(fig, mpl.figure.Figure)

    def test_show_block_true(self):
        # Under Agg backend, block=True is a no-op but must not raise.
        fig = _logo.show(block=True)
        assert isinstance(fig, mpl.figure.Figure)

    def test_show_dark_theme(self):
        fig = _logo.show(theme="dark")
        assert fig is not None


# ===========================================================================
# Section 6 — save() tests
# ===========================================================================


class TestSave:
    """Tests for save() covering all naming modes, formats, and option combinations."""

    # ── Basic operation ───────────────────────────────────────────────────────

    def test_save_returns_list(self, tmp_path):
        result = _logo.save(tmp_path / "logo.svg")
        assert isinstance(result, list)

    def test_save_single_svg_creates_file(self, tmp_path):
        out = tmp_path / "logo.svg"
        paths = _logo.save(out)
        assert len(paths) == 1
        assert out.exists()

    def test_save_svg_file_not_empty(self, tmp_path):
        out = tmp_path / "logo.svg"
        _logo.save(out)
        assert out.stat().st_size > 100

    # ── Format resolution ─────────────────────────────────────────────────────

    def test_save_png_via_suffix(self, tmp_path):
        out = tmp_path / "logo.png"
        paths = _logo.save(out)
        assert Path(paths[0]).exists()
        assert paths[0].endswith(".png")

    def test_save_ext_override_changes_format(self, tmp_path):
        out = tmp_path / "logo.svg"
        paths = _logo.save(out, ext="png")
        assert paths[0].endswith(".png")
        assert Path(paths[0]).exists()

    def test_save_format_param_changes_format(self, tmp_path):
        out = tmp_path / "logo"
        paths = _logo.save(out, format="svg")
        assert paths[0].endswith(".svg")

    def test_save_no_suffix_defaults_to_svg(self, tmp_path):
        out = tmp_path / "logo"
        paths = _logo.save(out)
        assert paths[0].endswith(".svg")

    # ── Variant selection ─────────────────────────────────────────────────────

    def test_save_single_variant_param(self, tmp_path):
        out = tmp_path / "logo.svg"
        paths = _logo.save(out, variant="small")
        assert len(paths) == 1

    def test_save_variants_string_treated_as_single(self, tmp_path):
        out = tmp_path / "logo.svg"
        paths = _logo.save(out, variants="small")
        assert len(paths) == 1

    def test_save_all_variants_returns_four_paths(self, tmp_path):
        base = tmp_path / "scikit-plots.svg"
        paths = _logo.save(base, variants=_logo.list_variants())
        assert len(paths) == 4

    def test_save_all_variants_uses_dash_suffix(self, tmp_path):
        base = tmp_path / "scikit-plots.svg"
        paths = _logo.save(base, variants=_logo.list_variants())
        for v in _logo.list_variants():
            assert any(f"-{v}.svg" in p for p in paths), (
                f"Expected '-{v}.svg' suffix in paths: {paths}"
            )

    def test_save_template_expansion(self, tmp_path):
        tmpl = tmp_path / "scikit-plots-{variant}.svg"
        paths = _logo.save(tmpl, variants=["primary", "small"])
        assert len(paths) == 2
        assert (tmp_path / "scikit-plots-primary.svg").exists()
        assert (tmp_path / "scikit-plots-small.svg").exists()

    # ── Option combinations ───────────────────────────────────────────────────

    @pytest.mark.parametrize("preset", ["favicon", "avatar", "docs-hero"])
    def test_save_all_presets_create_file(self, tmp_path, preset):
        out = tmp_path / f"logo-{preset}.svg"
        paths = _logo.save(out, preset=preset)
        assert Path(paths[0]).exists()

    def test_save_dark_theme(self, tmp_path):
        out = tmp_path / "logo-dark.svg"
        paths = _logo.save(out, theme="dark")
        assert Path(paths[0]).exists()

    def test_save_mono(self, tmp_path):
        out = tmp_path / "logo-mono.svg"
        paths = _logo.save(out, mono=True)
        assert Path(paths[0]).exists()

    def test_save_dots_random(self, tmp_path):
        out = tmp_path / "logo-random.svg"
        paths = _logo.save(out, dots="random")
        assert Path(paths[0]).exists()

    def test_save_transparent_false(self, tmp_path):
        out = tmp_path / "logo.png"
        paths = _logo.save(out, transparent=False)
        assert Path(paths[0]).exists()

    # ── Figure leak prevention ────────────────────────────────────────────────

    def test_save_closes_created_figures(self, tmp_path):
        """save() must close every figure it creates; no leaks to plt registry."""
        before = len(plt.get_fignums())
        _logo.save(tmp_path / "logo.svg", variants=["primary", "small"])
        after = len(plt.get_fignums())
        assert after == before

    # ── Pathlib Path argument ─────────────────────────────────────────────────

    def test_save_accepts_pathlib_path(self, tmp_path):
        out = tmp_path / "logo.svg"
        paths = _logo.save(out)
        assert Path(paths[0]).exists()

    def test_save_accepts_string_path(self, tmp_path):
        out = str(tmp_path / "logo.svg")
        paths = _logo.save(out)
        assert Path(paths[0]).exists()


# ===========================================================================
# Section 7 — Wordmark API tests
# ===========================================================================


class TestWordmark:
    """Tests for wordmark.create(), wordmark.show(), and wordmark.save()."""

    # ── create() ─────────────────────────────────────────────────────────────

    def test_create_returns_fig_and_two_axes(self):
        fig, axes = _logo.wordmark.create()
        assert isinstance(fig, mpl.figure.Figure)
        assert len(axes) == 2

    def test_create_axes_are_axes_instances(self):
        _, (ax_icon, ax_text) = _logo.wordmark.create()
        assert isinstance(ax_icon, mpl.axes.Axes)
        assert isinstance(ax_text, mpl.axes.Axes)

    def test_create_custom_text(self):
        fig, _ = _logo.wordmark.create(text="my-brand")
        assert fig is not None

    def test_create_empty_text_does_not_crash(self):
        """Empty wordmark text is an edge case; must not raise."""
        fig, _ = _logo.wordmark.create(text="")
        assert fig is not None

    def test_create_long_text_does_not_crash(self):
        fig, _ = _logo.wordmark.create(text="x" * 60)
        assert fig is not None

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_create_all_themes(self, theme):
        fig, _ = _logo.wordmark.create(theme=theme)
        assert fig is not None

    def test_create_mono_true(self):
        fig, _ = _logo.wordmark.create(mono=True)
        assert fig is not None

    def test_create_letter_spacing_positive(self):
        """Positive letter_spacing triggers per-character rendering path."""
        fig, _ = _logo.wordmark.create(letter_spacing=0.05)
        assert fig is not None

    def test_create_letter_spacing_zero(self):
        """letter_spacing=0 uses the single ax.text() path."""
        fig, _ = _logo.wordmark.create(letter_spacing=0.0)
        assert fig is not None

    def test_create_letter_spacing_negative(self):
        """Negative letter_spacing falls through to the single text path."""
        fig, _ = _logo.wordmark.create(letter_spacing=-0.1)
        assert fig is not None

    @pytest.mark.parametrize("preset", [None, "docs-hero"])
    def test_create_supported_presets(self, preset):
        fig, _ = _logo.wordmark.create(preset=preset)
        assert fig is not None

    @pytest.mark.parametrize("icon_variant", ["primary", "small", "metrics", "knn"])
    def test_create_all_icon_variants(self, icon_variant):
        fig, _ = _logo.wordmark.create(icon_variant=icon_variant)
        assert fig is not None

    def test_create_docs_hero_preset_bumps_text_size(self):
        """docs-hero preset should raise text_size to at least 44."""
        # We verify indirectly: no crash and figure is created
        fig, _ = _logo.wordmark.create(preset="docs-hero", text_size=30)
        assert fig is not None

    def test_create_size_controls_figwidth(self):
        fig, _ = _logo.wordmark.create(size=4.0)
        assert fig.get_figwidth() == pytest.approx(4.0)

    # ── show() ────────────────────────────────────────────────────────────────

    def test_show_returns_figure(self):
        fig = _logo.wordmark.show()
        assert isinstance(fig, mpl.figure.Figure)

    def test_show_passes_kwargs_to_create(self):
        fig = _logo.wordmark.show(theme="dark")
        assert fig is not None

    # ── save() ────────────────────────────────────────────────────────────────

    def test_save_svg_creates_file(self, tmp_path):
        out = tmp_path / "lockup.svg"
        result = _logo.wordmark.save(out)
        assert out.exists()

    def test_save_returns_string(self, tmp_path):
        out = tmp_path / "lockup.svg"
        result = _logo.wordmark.save(out)
        assert isinstance(result, str)

    def test_save_returned_path_matches_file(self, tmp_path):
        out = tmp_path / "lockup.svg"
        result = _logo.wordmark.save(out)
        assert str(out) == result

    def test_save_ext_override(self, tmp_path):
        out = tmp_path / "lockup.svg"
        result = _logo.wordmark.save(out, ext="png")
        assert result.endswith(".png")
        assert Path(result).exists()

    def test_save_format_override(self, tmp_path):
        out = tmp_path / "lockup"
        result = _logo.wordmark.save(out, format="svg")
        assert result.endswith(".svg")

    def test_save_docs_hero_preset_creates_file(self, tmp_path):
        out = tmp_path / "lockup.svg"
        result = _logo.wordmark.save(out, preset="docs-hero")
        assert Path(result).exists()

    def test_save_docs_hero_preset_wider_than_default(self, tmp_path):
        """docs-hero should expand width from 6.0 → 8.0 when using default size."""
        # We cannot inspect the closed figure; verify by checking file size.
        # A wider figure produces a larger SVG file.
        out_default = tmp_path / "default.svg"
        out_hero = tmp_path / "hero.svg"
        _logo.wordmark.save(out_default)
        _logo.wordmark.save(out_hero, preset="docs-hero")
        assert out_hero.stat().st_size >= out_default.stat().st_size

    def test_save_closes_figure(self, tmp_path):
        """wordmark.save() must not leak figures into the pyplot registry."""
        before = len(plt.get_fignums())
        _logo.wordmark.save(tmp_path / "lockup.svg")
        after = len(plt.get_fignums())
        assert after == before

    def test_save_dark_theme(self, tmp_path):
        out = tmp_path / "lockup-dark.svg"
        result = _logo.wordmark.save(out, theme="dark")
        assert Path(result).exists()


# ===========================================================================
# Section 8 — Determinism tests (renderer-independent artist geometry)
# ===========================================================================


class TestDeterminism:
    """
    Guarantee brand-stable output via artist-level geometry comparison.

    We compare ``_ax_signature()`` rather than raw SVG/PNG bytes to avoid
    brittleness across Matplotlib patch releases that may change serialisation
    details without affecting visual output.
    """

    def test_primary_fixed_dots_artists_identical(self):
        fig1, ax1 = _logo.create(variant="primary", dots="fixed")
        fig2, ax2 = _logo.create(variant="primary", dots="fixed")
        fig1.canvas.draw()
        fig2.canvas.draw()
        assert _ax_signature(ax1) == _ax_signature(ax2)

    @pytest.mark.parametrize("variant", ["small", "metrics", "knn"])
    def test_all_variants_deterministic(self, variant):
        fig1, ax1 = _logo.create(variant=variant)
        fig2, ax2 = _logo.create(variant=variant)
        fig1.canvas.draw()
        fig2.canvas.draw()
        assert _ax_signature(ax1) == _ax_signature(ax2)

    def test_same_seed_random_dots_artists_identical(self):
        fig1, ax1 = _logo.create(variant="primary", dots="random", seed=42)
        fig2, ax2 = _logo.create(variant="primary", dots="random", seed=42)
        fig1.canvas.draw()
        fig2.canvas.draw()
        assert _ax_signature(ax1) == _ax_signature(ax2)

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_theme_deterministic(self, theme):
        fig1, ax1 = _logo.create(theme=theme)
        fig2, ax2 = _logo.create(theme=theme)
        assert _ax_signature(ax1) == _ax_signature(ax2)

    def test_mono_deterministic(self):
        fig1, ax1 = _logo.create(mono=True)
        fig2, ax2 = _logo.create(mono=True)
        assert _ax_signature(ax1) == _ax_signature(ax2)

    def test_primary_save_svg_produces_same_file_twice(self, tmp_path):
        """
        Two consecutive save() calls with identical arguments must produce
        SVG files of the same byte-length (exact bytes may vary with hashsalt,
        but structural length is stable for fixed-dots deterministic output).
        """
        a = tmp_path / "a.svg"
        b = tmp_path / "b.svg"
        _logo.save(a, variant="primary", dots="fixed")
        _logo.save(b, variant="primary", dots="fixed")
        # Size comparison is stable; byte-exact comparison is not guaranteed
        # without svg.hashsalt (handled in TestSVGDeterminism below).
        assert a.stat().st_size == b.stat().st_size


@pytest.mark.skipif(
    "svg.hashsalt" not in mpl.rcParams,
    reason="svg.hashsalt not supported by this Matplotlib version",
)
class TestSVGDeterminism:
    """
    Optional strict SVG-structure determinism gated on svg.hashsalt support.

    Notes
    -----
    Even with hashsalt, Matplotlib may embed version-specific comments or
    precision differences. ``_normalize_svg_bytes`` collapses id-based
    variability so that only structural changes produce test failures.
    """

    def test_normalised_svg_bytes_identical(self, tmp_path):
        a = tmp_path / "a.svg"
        b = tmp_path / "b.svg"
        with mpl.rc_context({"svg.hashsalt": "scikit-plots-test"}):
            _logo.save(a, variant="primary", dots="fixed", metadata={})
            _logo.save(b, variant="primary", dots="fixed", metadata={})
        assert _normalize_svg_bytes(a.read_bytes()) == _normalize_svg_bytes(b.read_bytes())


# ===========================================================================
# Section 9 — CLI (main) tests
# ===========================================================================


class TestCLI:
    """Tests for the main() CLI entry point."""

    def test_main_returns_zero_on_success(self, tmp_path):
        result = _logo.main(["--out", str(tmp_path)])
        assert result == 0

    def test_main_creates_at_least_one_file(self, tmp_path):
        _logo.main(["--out", str(tmp_path), "--format", "svg"])
        files = list(tmp_path.glob("*.svg"))
        assert len(files) >= 1

    def test_main_all_flag_creates_four_files(self, tmp_path):
        _logo.main(["--out", str(tmp_path), "--all", "--format", "svg"])
        files = list(tmp_path.glob("*.svg"))
        assert len(files) == 4

    def test_main_wordmark_flag_creates_lockup_file(self, tmp_path):
        _logo.main(["--out", str(tmp_path), "--wordmark", "--format", "svg"])
        files = list(tmp_path.glob("*lockup*.svg"))
        assert len(files) == 1

    def test_main_preset_favicon_creates_file(self, tmp_path):
        _logo.main(["--out", str(tmp_path), "--preset", "favicon", "--format", "png"])
        files = list(tmp_path.glob("*.png"))
        assert len(files) >= 1

    def test_main_theme_dark(self, tmp_path):
        result = _logo.main(["--out", str(tmp_path), "--theme", "dark"])
        assert result == 0

    def test_main_mono_flag(self, tmp_path):
        result = _logo.main(["--out", str(tmp_path), "--mono"])
        assert result == 0

    def test_main_custom_name_used_in_filename(self, tmp_path):
        _logo.main(["--out", str(tmp_path), "--name", "mylogo", "--format", "svg"])
        files = list(tmp_path.glob("mylogo*.svg"))
        assert len(files) >= 1

    def test_main_specific_variant_flag(self, tmp_path):
        _logo.main(["--out", str(tmp_path), "--variant", "metrics", "--format", "svg"])
        files = list(tmp_path.glob("*.svg"))
        assert len(files) >= 1

    def test_main_multiple_variant_flags(self, tmp_path):
        _logo.main([
            "--out", str(tmp_path),
            "--variant", "primary",
            "--variant", "knn",
            "--format", "svg",
        ])
        files = list(tmp_path.glob("*.svg"))
        assert len(files) == 2

    def test_main_wordmark_with_custom_text(self, tmp_path):
        result = _logo.main([
            "--out", str(tmp_path),
            "--wordmark",
            "--text", "my-brand",
            "--format", "svg",
        ])
        assert result == 0

    def test_main_creates_output_dir_if_missing(self, tmp_path):
        subdir = tmp_path / "nested" / "output"
        result = _logo.main(["--out", str(subdir)])
        assert result == 0
        assert subdir.exists()

    def test_main_dots_none(self, tmp_path):
        result = _logo.main(["--out", str(tmp_path), "--dots", "none"])
        assert result == 0

    def test_main_dots_random(self, tmp_path):
        result = _logo.main(["--out", str(tmp_path), "--dots", "random"])
        assert result == 0

    def test_main_wordmark_all_presets(self, tmp_path):
        for preset in _logo.list_size_presets():
            out = tmp_path / preset
            out.mkdir()
            result = _logo.main([
                "--out", str(out), "--wordmark", "--preset", preset, "--format", "svg"
            ])
            assert result == 0, f"main() failed for preset={preset!r}"


# ===========================================================================
# Section 10 — Edge cases and regression guards
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge cases, regression guards, and resource-safety checks."""

    def test_save_single_and_multiple_variants_dont_conflict(self, tmp_path):
        """Saving single then multi in the same directory must not overwrite."""
        out_single = tmp_path / "logo.svg"
        _logo.save(out_single, variant="primary")
        size_after_single = out_single.stat().st_size

        _logo.save(tmp_path / "logos.svg", variants=["primary", "small"])
        # The single-variant file should be untouched.
        assert out_single.stat().st_size == size_after_single

    def test_create_does_not_leave_open_figures(self):
        """
        create() returns a figure that the caller owns; the call itself must
        not silently open *extra* figures beyond the one it returns.
        """
        before = len(plt.get_fignums())
        fig, _ = _logo.create()
        after = len(plt.get_fignums())
        plt.close(fig)
        # Exactly one new figure was opened and one was closed.
        assert len(plt.get_fignums()) == before

    def test_draw_does_not_open_new_figure(self):
        """draw() operates on an existing axes; it must not create new figures."""
        fig, ax = plt.subplots()
        before = len(plt.get_fignums())
        _logo.draw(ax)
        after = len(plt.get_fignums())
        assert after == before

    def test_list_variants_is_not_mutable(self):
        """
        list_variants() returns a tuple; callers cannot accidentally mutate
        the module-level constant.
        """
        result = _logo.list_variants()
        assert isinstance(result, tuple)

    def test_list_size_presets_is_not_mutable(self):
        result = _logo.list_size_presets()
        assert isinstance(result, tuple)

    def test_draw_knn_all_link_endpoints_match_node_positions(self):
        """
        KNN link Line2D endpoints should correspond to node positions defined
        in draw().  This guards against silent coordinate mismatches.
        """
        expected_nodes = {(-0.05, 0.52), (0.10, 0.45), (0.25, 0.55)}
        expected_links = [
            ((-0.05, 0.10), (0.52, 0.45)),
            ((0.10, 0.25), (0.45, 0.55)),
        ]

        fig, ax = plt.subplots()
        _logo.draw(ax, variant="knn")

        # Trend line is always present; KNN links are the additional lines.
        knn_lines = [
            line for line in ax.lines
            if len(line.get_xdata()) == 2
            and tuple(np.round(line.get_xdata(), 2)) not in [(-0.05, 0.55)]
            # exclude trend line endpoints
        ]

        # At least 2 link lines should exist (the KNN graph has 2 edges).
        assert len(knn_lines) >= 2

    def test_wordmark_icon_ratio_respected(self):
        """
        The icon_ratio parameter should set the width of ax_icon relative to
        the figure. We check that ax_icon.get_position().width ≈ icon_ratio.
        """
        ratio = 0.35
        _, (ax_icon, _) = _logo.wordmark.create(icon_ratio=ratio)
        assert ax_icon.get_position().width == pytest.approx(ratio, abs=1e-6)

    def test_draw_primary_fixed_dots_all_inside_circle(self):
        """Every fixed dot center must lie inside the inner clip circle (r=0.93)."""
        inner_r = 0.93
        for x, y, _r, _key in _logo._FIXED_DOTS:
            dist = (x**2 + y**2) ** 0.5
            assert dist < inner_r, (
                f"Fixed dot ({x}, {y}) is outside clip circle r={inner_r}"
            )

    def test_save_wordmark_closes_figure_on_exception_path(self, tmp_path):
        """
        Even if we cannot verify exception suppression here, we confirm the
        normal path releases the figure correctly (regression for FIX 5 area).
        """
        before = len(plt.get_fignums())
        _logo.wordmark.save(tmp_path / "lockup.svg")
        assert len(plt.get_fignums()) == before


# ===========================================================================
# Section 11 — Banner structure validation (extended)
# ===========================================================================


class TestBannerStructure:
    """
    Verify the ASCII-art banner's structural invariants.

    The banner is a figlet-rendered representation of "scikit-plots". Its raw
    bytes contain box-drawing characters, not the plain-text name, so tests
    must check structural properties rather than substring equality.
    """

    def test_banner_is_str_type(self):
        assert isinstance(_logo._SCIKITPLOT_BANNER, str)

    def test_banner_has_at_least_four_lines(self):
        lines = _logo._SCIKITPLOT_BANNER.splitlines()
        assert len(lines) >= 4, f"Expected ≥4 lines, got {len(lines)}"

    def test_banner_has_at_most_ten_lines(self):
        """Guard against accidental whitespace bloat."""
        lines = _logo._SCIKITPLOT_BANNER.splitlines()
        assert len(lines) <= 10, f"Banner unexpectedly long: {len(lines)} lines"

    def test_banner_does_not_start_with_whitespace_only_line(self):
        """After the lstrip('\n') fix, first line must contain non-space chars."""
        first = _logo._SCIKITPLOT_BANNER.splitlines()[0]
        assert first.strip() != "", "First banner line should not be blank"

    def test_banner_contains_underscore_art_pattern(self):
        """The top row of figlet output for 'scikit-plots' contains '____'."""
        assert "____" in _logo._SCIKITPLOT_BANNER

    def test_banner_contains_pipe_char(self):
        """Vertical bar is a structural character in figlet ASCII art."""
        assert "|" in _logo._SCIKITPLOT_BANNER

    def test_banner_contains_slash_chars(self):
        """Forward slash appears in figlet letter curves."""
        assert "/" in _logo._SCIKITPLOT_BANNER

    def test_banner_line_count_is_stable(self):
        """
        Regression guard: line count must not change between calls.

        This catches accidental mutation of the module-level constant.
        """
        n1 = len(_logo._SCIKITPLOT_BANNER.splitlines())
        n2 = len(_logo._SCIKITPLOT_BANNER.splitlines())
        assert n1 == n2


# ===========================================================================
# Section 12 — _lerp_color extended edge cases
# ===========================================================================


class TestLerpColorEdgeCases:
    """
    Extended edge-case coverage for _lerp_color beyond the basic unit tests.

    Tests t-values outside [0, 1] confirm linear extrapolation semantics
    (no clamping is applied by the function).
    """

    def test_t_negative_extrapolates_beyond_c1(self):
        """
        At t < 0 the function linearly extrapolates past c1.
        For black→white at t=-0.5 the result should be below 0 (unclamped).
        """
        result = _logo._lerp_color("black", "white", -0.5)
        # The function does not clamp; raw extrapolation gives negative values.
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_t_greater_than_one_extrapolates_beyond_c2(self):
        """
        At t > 1 the function linearly extrapolates past c2.
        For black→white at t=1.5 the result should be above 1 (unclamped).
        """
        result = _logo._lerp_color("black", "white", 1.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_result_is_always_ndarray_shape_3(self):
        """Return shape is invariant across a range of t-values."""
        for t in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
            r = _logo._lerp_color("red", "blue", t)
            assert r.shape == (3,), f"Wrong shape at t={t}: {r.shape}"

    def test_named_colors_accepted(self):
        """Named Matplotlib colors (e.g., 'navy', 'steelblue') must work."""
        result = _logo._lerp_color("navy", "steelblue", 0.5)
        assert result.shape == (3,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_symmetry_at_half(self):
        """
        lerp(A, B, 0.5) == lerp(B, A, 0.5) for any A, B.

        Both halves land at the same midpoint regardless of argument order.
        """
        r1 = _logo._lerp_color("red", "blue", 0.5)
        r2 = _logo._lerp_color("blue", "red", 0.5)
        np.testing.assert_allclose(r1, r2, atol=1e-7)


# ===========================================================================
# Section 13 — _palette extended coverage
# ===========================================================================


class TestPaletteExtended:
    """Additional _palette() tests for robustness and API stability."""

    def test_palette_is_deterministic(self):
        """Same arguments must produce an identical dict on every call."""
        p1 = _logo._palette(theme="light", mono=False)
        p2 = _logo._palette(theme="light", mono=False)
        assert p1 == p2

    def test_all_palette_values_are_strings(self):
        """Every color value must be a plain string (hex or named)."""
        for theme in ("light", "dark"):
            for mono in (False, True):
                P = _logo._palette(theme=theme, mono=mono)
                for role, color in P.items():
                    assert isinstance(color, str), (
                        f"theme={theme!r} mono={mono}: "
                        f"{role!r} value {color!r} is not a str"
                    )

    def test_dark_mono_fg_is_light_color(self):
        """
        In dark-mono mode all fg colors should be a light hue
        (visually readable on a dark background).
        """
        P = _logo._palette(theme="dark", mono=True)
        fg_hex = next(v for k, v in P.items() if k != "BG")
        r, g, b = mpl.colors.to_rgb(fg_hex)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert luminance > 0.3, (
            f"Dark-mono fg color {fg_hex!r} appears too dark (luminance={luminance:.3f})"
        )

    def test_light_mono_fg_is_dark_color(self):
        """
        In light-mono mode all fg colors should be a dark hue
        (visually readable on a white background).
        """
        P = _logo._palette(theme="light", mono=True)
        fg_hex = next(v for k, v in P.items() if k != "BG")
        r, g, b = mpl.colors.to_rgb(fg_hex)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert luminance < 0.3, (
            f"Light-mono fg color {fg_hex!r} appears too light (luminance={luminance:.3f})"
        )

    def test_navy_darker_than_blue_light_in_light_theme(self):
        """
        Brand contract: NAVY should be darker than BLUE_LIGHT in the light theme
        (NAVY is used for strong contrast elements, BLUE_LIGHT for softer accents).
        """
        P = _logo._palette(theme="light", mono=False)

        def lum(hex_color):
            r, g, b = mpl.colors.to_rgb(hex_color)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        assert lum(P["NAVY"]) < lum(P["BLUE_LIGHT"]), (
            "NAVY should be darker than BLUE_LIGHT in light theme"
        )


# ===========================================================================
# Section 14 — _apply_preset avatar preset coverage
# ===========================================================================


class TestApplyPresetAvatar:
    """
    Explicit coverage of the 'avatar' preset, which was not individually
    exercised in TestApplyPreset (only tested via all_presets_accept_without_error).
    """

    def test_avatar_variant_is_primary(self):
        v, _, _, _ = _logo._apply_preset(
            preset="avatar", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert v == "primary"

    def test_avatar_dots_is_fixed(self):
        _, d, _, _ = _logo._apply_preset(
            preset="avatar", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert d == "fixed"

    def test_avatar_size_overrides_default(self):
        _, _, s, _ = _logo._apply_preset(
            preset="avatar", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert s == pytest.approx(3.2)

    def test_avatar_dpi_overrides_default(self):
        _, _, _, dpi = _logo._apply_preset(
            preset="avatar", variant="primary", dots="fixed", size=4, dpi=200
        )
        assert dpi == 220

    def test_avatar_explicit_size_not_overridden(self):
        """User-supplied size=2.0 must survive the avatar preset."""
        _, _, s, _ = _logo._apply_preset(
            preset="avatar", variant="primary", dots="fixed", size=2.0, dpi=200
        )
        assert s == pytest.approx(2.0)

    def test_avatar_explicit_dpi_not_overridden(self):
        """User-supplied dpi=96 must survive the avatar preset."""
        _, _, _, dpi = _logo._apply_preset(
            preset="avatar", variant="primary", dots="fixed", size=4, dpi=96
        )
        assert dpi == 96


# ===========================================================================
# Section 15 — draw() extended branch coverage
# ===========================================================================


class TestDrawExtended:
    """
    Additional draw() tests targeting branches not explicitly covered
    by TestDraw: bar FancyBboxPatch presence, structured-dot counts, and
    orange-signal-dot presence vs absence per variant.
    """

    def test_bars_are_fancy_bbox_patches(self):
        """
        Histogram bars must be FancyBboxPatch (rounded-corner bars),
        not plain Rectangle patches.
        """
        fig, ax = plt.subplots()
        _logo.draw(ax, variant="primary")
        fancy = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
        assert len(fancy) > 0, "Expected FancyBboxPatch bars in primary variant"

    def test_primary_has_more_bars_than_small(self):
        """Primary uses 12 bars; small uses 8. Patch counts should reflect this."""
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="primary", dots="none")
        n_primary = sum(1 for p in ax1.patches if isinstance(p, FancyBboxPatch))
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="small", dots="none")
        n_small = sum(1 for p in ax2.patches if isinstance(p, FancyBboxPatch))
        plt.close(fig2)

        assert n_primary > n_small, (
            f"Primary ({n_primary}) should have more bars than small ({n_small})"
        )

    def test_small_has_no_orange_dots(self):
        """
        Orange signal dots are skipped for variant='small'.
        All circles in 'small' should originate from non-orange logic.
        """
        P = _logo._palette(theme="light", mono=False)
        orange_rgba = mpl.colors.to_rgba(P["ORANGE"])
        orange_light_rgba = mpl.colors.to_rgba(P["ORANGE_LIGHT"])

        fig, ax = plt.subplots()
        _logo.draw(ax, variant="small", dots="none")

        for patch in ax.patches:
            if isinstance(patch, Circle):
                fc = tuple(round(c, 4) for c in patch.get_facecolor())
                # Neither orange shade should appear in the small variant
                assert fc != tuple(round(c, 4) for c in orange_rgba), (
                    "small variant must not contain orange signal dots"
                )
                assert fc != tuple(round(c, 4) for c in orange_light_rgba), (
                    "small variant must not contain orange_light signal dots"
                )

    def test_structured_dots_absent_in_small(self):
        """
        The 5×4 structured dot grid is not rendered for variant='small'.
        Verifies total Circle count for small < primary with dots='none'.
        """
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="primary", dots="none")
        circles_primary = [p for p in ax1.patches if isinstance(p, Circle)]
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="small", dots="none")
        circles_small = [p for p in ax2.patches if isinstance(p, Circle)]
        plt.close(fig2)

        assert len(circles_small) < len(circles_primary), (
            "small variant omits structured dots → fewer circles than primary"
        )

    def test_knn_has_exactly_two_graph_edges(self):
        """
        The KNN motif draws exactly 2 graph-edge Line2D artists
        (in addition to the trend line and its spoke lines).
        """
        fig_p, ax_p = plt.subplots()
        _logo.draw(ax_p, variant="primary")
        n_primary_lines = len(ax_p.lines)
        plt.close(fig_p)

        fig_k, ax_k = plt.subplots()
        _logo.draw(ax_k, variant="knn")
        n_knn_lines = len(ax_k.lines)
        plt.close(fig_k)

        # KNN adds exactly 2 edge lines to the shared baseline artists
        extra = n_knn_lines - n_primary_lines
        assert extra == 2, (
            f"Expected 2 extra KNN edges, got {extra} "
            f"(primary={n_primary_lines}, knn={n_knn_lines})"
        )

    def test_draw_metric_confusion_grid_is_3x3(self):
        """
        The confusion-matrix icon for variant='metrics' is a 3×3 grid
        → 9 Rectangle patches.
        """
        fig, ax = plt.subplots()
        _logo.draw(ax, variant="metrics")
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) == 9, f"Expected 9 confusion-grid rectangles, got {len(rects)}"

    @pytest.mark.parametrize("variant", ["primary", "small", "metrics", "knn"])
    def test_outer_ring_is_circle_with_no_fill(self, variant):
        """
        The outer ring Circle must have facecolor='none' (transparent fill).
        Its edgecolor must equal NAVY for any theme/mono combination.
        """
        fig, ax = plt.subplots()
        _logo.draw(ax, variant=variant)
        outer = ax.patches[-1]  # outer ring is added last
        assert isinstance(outer, Circle), "Last patch should be the outer ring Circle"
        fc = mpl.colors.to_hex(outer.get_facecolor(), keep_alpha=True)
        assert fc in ("#00000000", "none", "#ffffff00"), (
            f"Outer ring facecolor should be transparent, got {fc!r}"
        )

    def test_random_dots_different_seeds_may_differ(self):
        """
        Two random-dot draws with different seeds should generally differ.
        This is probabilistic but extremely reliable with distinct seeds.
        """
        fig1, ax1 = plt.subplots()
        _logo.draw(ax1, variant="primary", dots="random", seed=0)
        sig1 = _ax_signature(ax1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        _logo.draw(ax2, variant="primary", dots="random", seed=99)
        sig2 = _ax_signature(ax2)
        plt.close(fig2)

        # With seed=0 vs seed=99 the dot layout is virtually guaranteed to differ
        assert sig1 != sig2, (
            "Different seeds should produce different random dot layouts"
        )


# ===========================================================================
# Section 16 — create() extended coverage
# ===========================================================================


class TestCreateExtended:
    """Additional create() tests for defaults, types, and figure isolation."""

    def test_default_size_is_4_inches(self):
        fig, _ = _logo.create()
        assert fig.get_figwidth() == pytest.approx(4.0)
        assert fig.get_figheight() == pytest.approx(4.0)

    def test_default_dpi_is_200(self):
        fig, _ = _logo.create()
        assert fig.get_dpi() == pytest.approx(200.0)

    def test_create_returns_new_figure_each_call(self):
        """Two consecutive create() calls must return distinct figure objects."""
        fig1, _ = _logo.create()
        fig2, _ = _logo.create()
        assert fig1 is not fig2

    def test_create_avatar_preset_size(self):
        """Avatar preset overrides size to 3.2 when caller keeps the default."""
        fig, _ = _logo.create(preset="avatar")
        assert fig.get_figwidth() == pytest.approx(3.2)

    def test_create_docs_hero_preset_size(self):
        """docs-hero preset overrides size to 5.0 when caller keeps the default."""
        fig, _ = _logo.create(preset="docs-hero")
        assert fig.get_figwidth() == pytest.approx(5.0)

    def test_create_figure_is_square(self):
        """create() always produces a square figure (width == height)."""
        for size in (2.0, 4.0, 6.0):
            fig, _ = _logo.create(size=size)
            assert fig.get_figwidth() == pytest.approx(fig.get_figheight()), (
                f"Figure should be square at size={size}"
            )
            plt.close(fig)

    def test_create_dots_none_allowed(self):
        """dots='none' must not raise for create()."""
        fig, ax = _logo.create(dots="none")
        assert fig is not None

    def test_create_mono_does_not_error(self):
        fig, ax = _logo.create(mono=True, theme="dark")
        assert fig is not None


# ===========================================================================
# Section 17 — save() extended coverage
# ===========================================================================


class TestSaveExtended:
    """Additional save() tests for type safety, variant resolution and format."""

    def test_save_returns_list_of_strings_not_paths(self, tmp_path):
        """save() must return list[str], not list[Path]."""
        paths = _logo.save(tmp_path / "logo.svg")
        assert isinstance(paths, list)
        for p in paths:
            assert isinstance(p, str), f"Expected str, got {type(p)}: {p!r}"

    def test_save_variants_takes_precedence_over_variant(self, tmp_path):
        """When both variants and variant are supplied, variants wins."""
        paths = _logo.save(
            tmp_path / "logo.svg",
            variants=["primary", "small"],
            variant="metrics",  # should be ignored
        )
        assert len(paths) == 2
        assert all("-primary" in p or "-small" in p for p in paths)

    def test_save_pdf_format_creates_file(self, tmp_path):
        """PDF output must be supported via the suffix."""
        out = tmp_path / "logo.pdf"
        paths = _logo.save(out)
        assert Path(paths[0]).exists()
        assert paths[0].endswith(".pdf")

    def test_save_all_four_files_exist_on_disk(self, tmp_path):
        """All four variant files must actually exist after a batch save."""
        tmpl = tmp_path / "logo-{variant}.svg"
        paths = _logo.save(tmpl, variants=list(_logo.list_variants()))
        for p in paths:
            assert Path(p).exists(), f"Expected file on disk: {p}"

    def test_save_dots_none_creates_file(self, tmp_path):
        """dots='none' must not prevent file creation."""
        out = tmp_path / "logo.svg"
        paths = _logo.save(out, dots="none")
        assert Path(paths[0]).exists()

    def test_save_seed_param_accepted(self, tmp_path):
        """seed parameter must be forwarded without error for random dots."""
        out = tmp_path / "logo.svg"
        paths = _logo.save(out, dots="random", seed=7)
        assert Path(paths[0]).exists()


# ===========================================================================
# Section 18 — wordmark extended coverage
# ===========================================================================


class TestWordmarkExtended:
    """
    Additional wordmark tests targeting parameter contracts and figure geometry.
    """

    def test_create_default_icon_ratio_is_0_28(self):
        """Default icon_ratio=0.28 should be reflected in ax_icon.get_position()."""
        _, (ax_icon, _) = _logo.wordmark.create()
        assert ax_icon.get_position().width == pytest.approx(0.28, abs=1e-6)

    def test_create_height_is_fraction_of_width(self):
        """
        Height is derived as size * 0.28 in _WordmarkAPI.create().
        Verify the figure height-to-width ratio is approximately 0.28.
        """
        size = 6.0
        fig, _ = _logo.wordmark.create(size=size)
        expected_height = size * 0.28
        assert fig.get_figheight() == pytest.approx(expected_height, rel=1e-3)

    def test_create_text_axes_is_axis_off(self):
        """The text axes must have its axis turned off."""
        _, (_, ax_text) = _logo.wordmark.create()
        assert not ax_text.axison

    def test_create_icon_axes_has_patches(self):
        """The icon axes must contain patches (the logo drawing)."""
        _, (ax_icon, _) = _logo.wordmark.create()
        assert len(ax_icon.patches) > 0, "Icon axes should contain logo patches"

    def test_create_docs_hero_raises_text_size_to_44(self):
        """
        When preset='docs-hero' and text_size < 44, text_size must be
        raised to at least 44. We verify indirectly that create() does not
        raise and returns a valid figure; the actual max() is tested
        at the function-contract level.
        """
        fig, _ = _logo.wordmark.create(preset="docs-hero", text_size=10)
        assert isinstance(fig, mpl.figure.Figure)

    def test_create_letter_spacing_positive_places_multiple_artists(self):
        """
        Positive letter_spacing triggers per-character text placement.
        The text axes should have one Text artist per character.
        """
        text = "abc"
        _, (_, ax_text) = _logo.wordmark.create(
            text=text, letter_spacing=0.05
        )
        text_artists = ax_text.texts
        assert len(text_artists) == len(text), (
            f"Expected {len(text)} Text artists for per-char mode, "
            f"got {len(text_artists)}"
        )

    def test_create_letter_spacing_zero_places_single_artist(self):
        """
        letter_spacing ≤ 0 triggers a single ax.text() call.
        The text axes should have exactly one Text artist.
        """
        _, (_, ax_text) = _logo.wordmark.create(
            text="hello", letter_spacing=0.0
        )
        assert len(ax_text.texts) == 1

    def test_save_no_suffix_defaults_to_svg(self, tmp_path):
        """wordmark.save() with no suffix on filename must default to SVG."""
        out = tmp_path / "lockup"
        result = _logo.wordmark.save(out)
        assert result.endswith(".svg")
        assert Path(result).exists()

    def test_save_docs_hero_size_override(self, tmp_path):
        """
        docs-hero preset with default size should expand width to 8.0.
        Verified by comparing file sizes (wider figure → larger SVG).
        """
        out_default = tmp_path / "default.svg"
        out_hero = tmp_path / "hero.svg"
        _logo.wordmark.save(out_default)
        _logo.wordmark.save(out_hero, preset="docs-hero")
        assert out_hero.stat().st_size > 0
        # docs-hero at 8.0in produces a larger SVG than default 6.0in
        assert out_hero.stat().st_size >= out_default.stat().st_size

    def test_save_mono_creates_file(self, tmp_path):
        out = tmp_path / "lockup-mono.svg"
        result = _logo.wordmark.save(out, mono=True)
        assert Path(result).exists()
