# scikitplot/_brand/tests/test__eukrina.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot._brand._eukrina: the brand data, the logos, the builds."""

from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from .._eukrina import (
    EDITIONS,
    LAYOUTS,
    MARK,
    RULES,
    THEMES,
    Edition,
    asset_name,
    build,
    mark_bounds,
    palette,
    render,
    tokens,
)
from .._eukrina._render import GRAPHIC_CONTRAST, TEXT_CONTRAST, _mark_radius
from .._kit import _text, contrast_ratio, relative_luminance, verify_manifest

NS = "{http://www.w3.org/2000/svg}"
HEIGHT = 64.0
HALF_ALPHA = 127
AA_PIXELS = 2  # anti-aliasing at 4x zoom
SHA256_HEX = 64
RSVG = shutil.which("rsvg-convert")
ALL = list(itertools.product(LAYOUTS, EDITIONS, THEMES, ("full", "mono")))


def _parse(svg: str) -> ET.Element:
    # Parses the generator's own output, never untrusted input.
    return ET.fromstring(svg)  # noqa: S314


def _inter():
    try:
        return _text.find_font("Inter", 600)
    except LookupError:
        pytest.skip("Inter is not installed")


# ---------------------------------------------------------------------------
# Brand data
# ---------------------------------------------------------------------------


def test_mark_bounds_are_the_ink_box():
    # Extremes widened by half the 25-unit stroke.
    assert mark_bounds() == (42.5, 32.5, 227.5, 222.5)


def test_editions_hold_the_brand_colours():
    warm, blue = EDITIONS["warm-bronze"], EDITIONS["blue-green"]
    assert (warm.dark, warm.accent, warm.accent_light, warm.support) == (
        "#202729",
        "#B58B58",
        "#C6A477",
        "#96836D",
    )
    assert (blue.dark, blue.accent, blue.accent_light, blue.support) == (
        "#071B2D",
        "#1E90FF",
        "#00BFFF",
        "#228B22",
    )


def test_edition_validates_colours():
    with pytest.raises(ValueError, match="expected a colour"):
        Edition("x", "X", "navy", "#000", "#000", "#000", "#000", "#000")


# ---------------------------------------------------------------------------
# Palette and contrast rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edition", "theme", "color"),
    list(itertools.product(EDITIONS, THEMES, ("full", "mono"))),
)
def test_every_palette_meets_its_contrast_targets(edition, theme, color):
    pal = palette(edition, theme, color=color)
    for value in pal.u + pal.accent + (pal.title,):
        assert contrast_ratio(value, pal.background) >= GRAPHIC_CONTRAST
    assert contrast_ratio(pal.subtitle, pal.background) >= TEXT_CONTRAST


def test_dark_logos_keep_the_exact_accents():
    for edition in EDITIONS.values():
        assert palette(edition, "dark").accent == (edition.accent_light, edition.accent)


def test_contrast_off_keeps_the_edition_colours():
    pal = palette("warm-bronze", "light", contrast=False)
    assert pal.accent == ("#C6A477", "#B58B58")
    assert pal.subtitle == "#96836D"


@pytest.mark.parametrize(
    ("edition", "theme"), list(itertools.product(EDITIONS, THEMES))
)
def test_gradients_keep_their_direction(edition, theme):
    pal = palette(edition, theme)
    ed = EDITIONS[edition]
    given = relative_luminance(ed.accent_light) - relative_luminance(ed.accent)
    fitted = relative_luminance(pal.accent[0]) - relative_luminance(pal.accent[1])
    assert given * fitted >= 0


def test_palette_choices():
    for kwargs in ({"theme": "dim"}, {"color": "duo"}):
        with pytest.raises(ValueError, match="unknown"):
            palette("warm-bronze", **kwargs)
    with pytest.raises(ValueError, match="unknown edition"):
        palette("purple")


# ---------------------------------------------------------------------------
# Logos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("layout", "edition", "theme", "color"), ALL)
def test_every_logo_is_a_safe_well_formed_svg(layout, edition, theme, color):
    svg = render(layout, edition, theme, color=color)
    root = _parse(svg)
    assert root.tag == NS + "svg"
    assert root.get("role") == "img"
    assert root.find(NS + "title").text.startswith("Eukrina Mind AI")
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    for ref in re.findall(r"url\(#([^)]+)\)", svg):
        assert ref in ids
    assert not re.search(r"<script|href=|\son\w+=|style=", svg, flags=re.IGNORECASE)
    width, height = float(root.get("width")), float(root.get("height"))
    _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
    assert width / height == pytest.approx(vw / vh)


@pytest.mark.parametrize(("layout", "edition", "theme", "color"), ALL)
def test_render_is_deterministic(layout, edition, theme, color):
    assert render(layout, edition, theme, color=color) == render(
        layout, edition, theme, color=color
    )


def test_mono_cuts_a_gap_full_colour_does_not():
    assert "<mask" in render("symbol", color="mono")
    assert "<mask" not in render("symbol")
    assert "<mask" in render("symbol", gap=3)
    assert "<mask" not in render("symbol", color="mono", gap=0)
    with pytest.raises(ValueError, match="gap"):
        render(gap=-1)


def test_solid_colours_need_no_gradient():
    svg = render("symbol", "warm-bronze", "dark", color="mono")
    assert "linearGradient" not in svg


def test_two_documents_never_share_ids():
    a, b = (
        render("symbol", "warm-bronze", "light"),
        render("symbol", "blue-green", "light"),
    )
    ids = lambda s: {e.get("id") for e in _parse(s).iter() if e.get("id")}  # noqa: E731
    assert not ids(a) & ids(b)
    assert 'id="brand-' in render("symbol", id_prefix="brand")


def test_round_icon_keeps_the_mark_inside_its_safe_circle():
    root = _parse(render("round-icon"))
    group = next(e for e in root.iter(NS + "g") if e.get("transform"))
    tx, ty, scale = (float(v) for v in re.findall(r"-?[\d.]+", group.get("transform")))
    x0, y0, x1, y1 = mark_bounds()
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    reach = _mark_radius(MARK, center) * scale
    assert reach <= RULES["round_icon_fill"] * 128 + 1e-6
    assert (tx + center[0] * scale, ty + center[1] * scale) == pytest.approx(
        (128, 128), abs=1e-3
    )


def test_horizontal_centres_the_text_on_the_mark():
    root = _parse(render("horizontal"))
    texts = root.findall(NS + "text")
    assert [t.text for t in texts] == ["Eukrina", "MIND AI"]
    _, _, _, vh = (float(v) for v in root.get("viewBox").split())
    title, sub = texts
    cap = 1490 / 2048
    top = float(title.get("y")) - cap * float(title.get("font-size"))
    bottom = float(sub.get("y"))
    assert (top + bottom) / 2 == pytest.approx(vh / 2)


def test_title_and_subtitle_options():
    assert "MIND AI" not in render("horizontal", subtitle=None)
    assert render("horizontal", subtitle="MIND AI") == render("horizontal")
    with pytest.raises(ValueError, match="needs font="):
        render("horizontal", title="Other")
    with pytest.raises(ValueError, match="title or a subtitle"):
        render("wordmark", title=None, subtitle=None)
    # Layouts without text ignore custom text.
    assert render("symbol", title="Other") == render("symbol")


def test_height_scales_the_output():
    root = _parse(render("symbol", height=64))
    assert float(root.get("height")) == HEIGHT
    with pytest.raises(ValueError, match="height"):
        render(height=0)


@pytest.mark.parametrize(
    "kwargs", [{"layout": "banner"}, {"text": "bitmap"}, {"edition": "x"}]
)
def test_unknown_choices(kwargs):
    with pytest.raises(ValueError, match="unknown"):
        render(**kwargs)


def test_custom_edition():
    custom = Edition(
        "violet",
        "Violet",
        "#1B1033",
        "#2A1A4A",
        "#7C3AED",
        "#A78BFA",
        "#6D28D9",
        "#C4B5FD",
    )
    svg = render("horizontal", custom, "dark", id_prefix="violet")
    assert "#7C3AED" in svg
    assert "#A78BFA" in svg


# ---------------------------------------------------------------------------
# Outlined text (needs Inter)
# ---------------------------------------------------------------------------


def test_outline_has_no_text_and_the_same_canvas():
    font = _inter()
    live, outlined = (
        render("horizontal"),
        render("horizontal", text="outline", font=font),
    )
    assert "<text" not in outlined
    assert "<text" in live
    assert _parse(live).get("viewBox") == _parse(outlined).get("viewBox")


def test_custom_text_measured_with_a_font():
    font = _inter()
    svg = render("horizontal", title="Mind", font=font)
    assert ">Mind<" in svg


# ---------------------------------------------------------------------------
# Ink check with a real renderer (optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("rsvg-convert") is None, reason="rsvg-convert is not installed"
)
def test_rendered_ink_matches_mark_bounds(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    svg = tmp_path / "s.svg"
    svg.write_text(render("symbol", color="mono", gap=0, height=None))
    png = tmp_path / "s.png"
    # A fixed program (found by shutil.which) with fixed arguments.
    subprocess.run(  # noqa: S603
        [RSVG, "-z", "4", str(svg), "-o", str(png)], check=True
    )
    image = pil.open(png).convert("RGBA")
    box = image.getchannel("A").point(lambda a: 255 if a > HALF_ALPHA else 0).getbbox()
    root = _parse(svg.read_text())
    _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
    pad = RULES["symbol_pad"] * (mark_bounds()[3] - mark_bounds()[1])
    expected = (pad * 4, pad * 4, (vw - pad) * 4, (vh - pad) * 4)
    for got, want in zip(box, expected):
        assert abs(got - want) <= AA_PIXELS


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------


def test_build_is_idempotent_and_verified(tmp_path):
    first = build(tmp_path)
    again = build(tmp_path)
    assert len(first.files) == 2 * len(LAYOUTS) * 2 * 2 + 2 * 2 + 1
    assert first.changed == len(first.files)
    assert again.changed == 0
    assert verify_manifest(tmp_path) == []
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["meta"]["brand_version"] == "1"
    assert "time" not in json.dumps(manifest["meta"])


def test_build_subset_and_names(tmp_path):
    report = build(
        tmp_path,
        editions="blue-green",
        themes=["dark"],
        layouts=["symbol"],
        colors=["mono"],
        with_tokens=False,
    )
    assert [f.name for f in report.files] == [
        "blue-green/svg/eukrina-symbol-dark-mono.svg",
        "manifest.json",
    ]
    assert (
        asset_name("warm-bronze", "symbol", "light", "full", "outline")
        == "warm-bronze/svg/eukrina-symbol-light.svg"
    )
    assert asset_name("warm-bronze", "stacked", "light", "mono", "outline").endswith(
        "stacked-light-mono-outline.svg"
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"editions": []}, {"themes": ["dim"]}, {"layouts": ["banner"]}, {"text": "x"}],
)
def test_build_refuses_unknown_choices(tmp_path, kwargs):
    with pytest.raises(ValueError, match="unknown"):
        build(tmp_path, **kwargs)
    assert not any(tmp_path.iterdir())


def test_build_outline_records_the_font(tmp_path):
    font = _inter()
    build(
        tmp_path,
        editions="warm-bronze",
        layouts=["horizontal"],
        text="outline",
        font=font,
        with_tokens=False,
    )
    meta = json.loads((tmp_path / "manifest.json").read_text())["meta"]
    assert meta["text"] == "outline"
    assert meta["font"]["family"] == "Inter"
    assert len(meta["font"]["sha256"]) == SHA256_HEX


def test_tokens():
    data = tokens("blue-green")
    assert data["scales"]["accent"]["500"] == "#1E90FF"
    assert set(data["roles"]) == {"light", "dark"}
    assert "surface" not in data["scales"]


def test_fitted_stops_are_swapped_back_into_order():
    from .._eukrina._render import _keep_order  # noqa: PLC0415

    given = ("#FFFFFF", "#000000")  # light -> dark
    assert _keep_order(given, ("#111111", "#EEEEEE")) == ("#EEEEEE", "#111111")
    assert _keep_order(given, ("#EEEEEE", "#111111")) == ("#EEEEEE", "#111111")
    assert _keep_order(("#777777", "#777777"), ("#111111", "#EEEEEE")) == (
        "#111111",
        "#EEEEEE",
    )
