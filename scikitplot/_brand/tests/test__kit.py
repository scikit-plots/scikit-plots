# scikitplot/_brand/tests/test__kit.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot._brand._kit: colour, SVG writer, text, file output."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from .._kit import _color, _io, _text
from .._kit._color import (
    ContrastError,
    contrast_ratio,
    mix,
    normalize_hex,
    readable,
    relative_luminance,
    tonal_scale,
)
from .._kit._svg import Element, IdSpace, document, fmt, to_string, url_ref

GRAPHIC = 3.0  # WCAG contrast for graphics
SMALL_TEXT = 4.5  # WCAG contrast for small text
SHA256_HEX = 64  # hex digits of a SHA-256


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


class TestColor:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [("#1e90ff", "#1E90FF"), ("#FFF", "#FFFFFF"), ("#0aB", "#00AABB")],
    )
    def test_normalize(self, given, expected):
        assert normalize_hex(given) == expected

    @pytest.mark.parametrize(
        "bad", ["1E90FF", "#1E90F", "#1E90FFAA", "red", "#GGGGGG", "", " #FFFFFF"]
    )
    def test_normalize_refuses(self, bad):
        with pytest.raises(ValueError, match="expected a colour"):
            normalize_hex(bad)

    def test_normalize_type(self):
        with pytest.raises(TypeError):
            normalize_hex(0xFFFFFF)

    def test_mix(self):
        assert mix("#000000", "#FFFFFF", 0.0) == "#000000"
        assert mix("#000000", "#FFFFFF", 1.0) == "#FFFFFF"
        assert mix("#000000", "#FFFFFF", 0.5) == "#808080"
        for bad in (-0.01, 1.01, float("nan")):
            with pytest.raises(ValueError, match="amount must be in"):
                mix("#000000", "#FFFFFF", bad)

    def test_scale_has_base_and_runs_light_to_dark(self):
        scale = tonal_scale("#B58B58")
        assert list(scale) == list(_color.SCALE_STEPS)
        assert scale[500] == "#B58B58"
        lum = [relative_luminance(scale[s]) for s in _color.SCALE_STEPS]
        assert lum == sorted(lum, reverse=True)

    def test_contrast_known_values(self):
        assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
        assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)
        assert contrast_ratio("#202729", "#F7F6F3") == contrast_ratio(
            "#F7F6F3", "#202729"
        )

    def test_readable_keeps_a_colour_that_passes(self):
        assert readable("#202729", "#F7F6F3", 4.5) == "#202729"

    def test_readable_takes_the_nearest_passing_step(self):
        # 600 passes (3.62:1); 500 does not (2.86:1).
        assert readable("#B58B58", "#F7F6F3", 3.0) == tonal_scale("#B58B58")[600]
        assert contrast_ratio(readable("#B58B58", "#F7F6F3", 3.0), "#F7F6F3") >= GRAPHIC

    def test_readable_goes_lighter_on_dark(self):
        result = readable("#228B22", "#071B2D", 4.5)
        assert relative_luminance(result) > relative_luminance("#228B22")
        assert contrast_ratio(result, "#071B2D") >= SMALL_TEXT

    def test_readable_says_when_impossible(self):
        with pytest.raises(ContrastError, match="best is step"):
            readable("#808080", "#7F7F7F", 21.0)

    @pytest.mark.parametrize("bad", [0.5, 22, float("nan")])
    def test_readable_refuses_a_bad_target(self, bad):
        with pytest.raises(ValueError, match="minimum contrast"):
            readable("#000000", "#FFFFFF", bad)


# ---------------------------------------------------------------------------
# SVG writer
# ---------------------------------------------------------------------------


class TestSvg:
    @pytest.mark.parametrize(
        ("value", "text"),
        [(2, "2"), (2.5, "2.5"), (1 / 3, "0.333"), (-0.0001, "0"), (-1.25, "-1.25")],
    )
    def test_fmt(self, value, text):
        assert fmt(value) == text

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
    def test_fmt_refuses(self, bad):
        with pytest.raises((ValueError, TypeError)):
            fmt(bad)

    @pytest.mark.parametrize(
        "tag", ["script", "SCRIPT", "foreignObject", "image", "use", "a"]
    )
    def test_refused_tags(self, tag):
        with pytest.raises(ValueError, match="not allowed"):
            Element(tag)

    @pytest.mark.parametrize(
        "attr", ["onload", "onClick", "href", "xlink:href", "style", "src"]
    )
    def test_refused_attributes(self, attr):
        with pytest.raises(ValueError, match="not allowed"):
            Element("rect", {attr: "x"})

    @pytest.mark.parametrize(
        "value",
        ["url(http://example.invalid/x)", "url( #a)", "URL(data:x)", "url(#a) url(#b)"],
    )
    def test_only_local_references(self, value):
        with pytest.raises(ValueError, match="url"):
            Element("rect", {"fill": value})

    def test_local_reference_and_none(self):
        element = Element("rect", {"fill": url_ref("a-b"), "stroke": None})
        assert element.attrs == (("fill", "url(#a-b)"),)

    @pytest.mark.parametrize("name", ["", "1a", "a b", "a>b", 'a"'])
    def test_invalid_names(self, name):
        with pytest.raises(ValueError, match="not a valid SVG"):
            Element("rect", {name: "1"})

    def test_escaping_and_well_formed(self):
        root = document(
            (0, 0, 10, 10), [Element("text", {"x": 1}, text='<b>&"x"')], title="A & <B>"
        )
        text = to_string(root, declaration=True)
        # Parses this writer's own output, never untrusted input.
        parsed = ET.fromstring(text.split("\n", 1)[1])  # noqa: S314
        assert parsed.find("{http://www.w3.org/2000/svg}title").text == "A & <B>"
        assert parsed.find("{http://www.w3.org/2000/svg}text").text == '<b>&"x"'

    def test_deterministic(self):
        def make():
            return to_string(
                document(
                    (0, 0, 4, 3),
                    [Element("rect", {"width": 4, "height": 3})],
                    title="t",
                )
            )

        assert make() == make()

    def test_document_checks(self):
        with pytest.raises(ValueError, match="viewBox"):
            document((0, 0, 0, 10), [], title="t")
        with pytest.raises(ValueError, match="title"):
            document((0, 0, 1, 1), [], title="  ")

    def test_id_space(self):
        ids = IdSpace("p")
        assert [ids.new("a"), ids.new("a"), ids.new("b")] == ["p-a", "p-a-2", "p-b"]
        with pytest.raises(ValueError, match="not a valid SVG id"):
            IdSpace("1bad")

    def test_children_must_be_elements(self):
        with pytest.raises(TypeError):
            Element("g", {}, ["<script/>"])


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class TestIo:
    @pytest.mark.parametrize(
        "name",
        [
            "../x.svg",
            "/abs.svg",
            "a\\b.svg",
            "a/../b.svg",
            ".hidden",
            "a//b",
            "",
            "a/ b",
        ],
    )
    def test_unsafe_names(self, tmp_path, name):
        with pytest.raises(ValueError, match=r"asset names|unsafe part"):
            _io.safe_target(tmp_path, name)

    def test_symlink_escape(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links are not available")
        with pytest.raises(ValueError, match="outside"):
            _io.safe_target(root, "link/x.svg")

    def test_write_is_idempotent_and_atomic(self, tmp_path):
        first = _io.write_bytes(tmp_path, "a/b.svg", b"<svg/>")
        again = _io.write_bytes(tmp_path, "a/b.svg", b"<svg/>")
        other = _io.write_bytes(tmp_path, "a/b.svg", b"<svg />")
        assert (first.changed, again.changed, other.changed) == (True, False, True)
        assert (tmp_path / "a" / "b.svg").read_bytes() == b"<svg />"
        assert [p.name for p in (tmp_path / "a").iterdir()] == [
            "b.svg"
        ]  # no temporary file left

    def test_write_refuses_text(self, tmp_path):
        with pytest.raises(TypeError):
            _io.write_bytes(tmp_path, "a.svg", "<svg/>")

    def test_manifest_round_trip(self, tmp_path):
        results = [_io.write_bytes(tmp_path, n, n.encode()) for n in ("b.txt", "a.txt")]
        _io.write_manifest(tmp_path, results, {"k": 1})
        data = json.loads((tmp_path / _io.MANIFEST_NAME).read_text())
        assert [f["name"] for f in data["files"]] == ["a.txt", "b.txt"]
        assert _io.verify_manifest(tmp_path) == []
        (tmp_path / "a.txt").write_bytes(b"changed")
        (tmp_path / "b.txt").unlink()
        assert _io.verify_manifest(tmp_path) == ["changed: a.txt", "missing: b.txt"]

    def test_manifest_refuses_duplicates(self, tmp_path):
        result = _io.write_bytes(tmp_path, "a.txt", b"a")
        with pytest.raises(ValueError, match="same name"):
            _io.write_manifest(tmp_path, [result, result], {})


# ---------------------------------------------------------------------------
# Text (needs Matplotlib; the measurements need Inter)
# ---------------------------------------------------------------------------


def _inter():
    try:
        return _text.find_font("Inter", 600)
    except LookupError:
        pytest.skip("Inter is not installed")


class TestText:
    def test_missing_font_is_an_error(self):
        with pytest.raises(LookupError, match="never fall back"):
            _text.find_font("No Such Font Family 9b2c")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _text.measure("a", tmp_path / "none.otf")

    def test_measure_matches_tracking(self):
        font = _inter()
        plain = _text.measure("Eukrina", font)
        assert _text.measure("Eukrina", font, tracking=0.1) == pytest.approx(
            plain + 0.6
        )

    def test_outline_width_and_anchor(self):
        font = _inter()
        d, width = _text.outline(
            "MIND AI", font, x=0, baseline=0, size=10, tracking=0.48
        )
        assert width == pytest.approx(
            _text.measure("MIND AI", font, tracking=0.48) * 10
        )
        assert d.startswith("M")
        assert d.endswith("Z")
        d_mid, _ = _text.outline(
            "MIND AI",
            font,
            x=width / 2,
            baseline=0,
            size=10,
            tracking=0.48,
            anchor="middle",
        )
        assert d_mid == d

    @pytest.mark.parametrize("bad", ["", "a\nb"])
    def test_one_line_only(self, bad):
        with pytest.raises(ValueError, match=r"non-empty|one line"):
            _text.outline(bad, "unused.otf", x=0, baseline=0, size=1)

    def test_missing_glyph(self):
        with pytest.raises(ValueError, match="no glyph"):
            _text.measure(chr(0x4E00), _inter())  # a CJK ideograph Inter lacks

    def test_font_info(self):
        info = _text.font_info(_inter())
        assert info.family == "Inter"
        assert len(info.sha256) == SHA256_HEX
        assert info.units_per_em > 0
