"""
Tests for ``_sphinx_collection._presentation``: namespaced Sphinx Design
options with single-line and URL-scheme validation.

The option tables are derived from the installed Sphinx Design version, so the
tests assert the derivation and the added safety checks rather than a frozen
copy of the upstream option list.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]


def _presentation_module():
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection._presentation")


@pytest.fixture(scope="module")
def presentation():
    return _presentation_module()


URL_OPTIONS = ["card-link", "card-img-top", "card-img-bottom", "card-img-background"]


# -- option tables ------------------------------------------------------------


def test_specs_mirror_installed_sphinx_design(presentation):
    from sphinx_design.grids import GridDirective, GridItemCardDirective

    assert sorted(presentation.GRID_SPEC) == sorted(
        "grid-" + key for key in GridDirective.option_spec
    )
    assert sorted(presentation.CARD_SPEC) == sorted(
        "card-" + key for key in GridItemCardDirective.option_spec
    )
    assert presentation.GRID_SPEC
    assert presentation.CARD_SPEC


def test_spec_namespaces_do_not_overlap(presentation):
    assert not set(presentation.GRID_SPEC) & set(presentation.CARD_SPEC)
    assert all(key.startswith("grid-") for key in presentation.GRID_SPEC)
    assert all(key.startswith("card-") for key in presentation.CARD_SPEC)
    assert all(callable(value) for value in presentation.GRID_SPEC.values())
    assert all(callable(value) for value in presentation.CARD_SPEC.values())


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("grid-gutter", id="grid-gutter"),
        pytest.param("grid-margin", id="grid-margin"),
        pytest.param("grid-class-container", id="grid-class-container"),
        pytest.param("card-link", id="card-link"),
        pytest.param("card-img-top", id="card-img-top"),
        pytest.param("card-img-alt", id="card-img-alt"),
        pytest.param("card-shadow", id="card-shadow"),
        pytest.param("card-class-card", id="card-class-card"),
    ],
)
def test_expected_options_are_available(presentation, name):
    spec = presentation.GRID_SPEC if name.startswith("grid-") else presentation.CARD_SPEC
    assert name in spec


# -- value validation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "argument", "expected"),
    [
        pytest.param("grid-gutter", "2", "2", id="gutter-one-value"),
        pytest.param("grid-gutter", " 1 2 3 4 ", "1 2 3 4", id="gutter-trimmed"),
        pytest.param("grid-margin", "0 1 2 3", "0 1 2 3", id="margin-four-values"),
        pytest.param("grid-class-container", "a b", "a b", id="class-list"),
        pytest.param("card-shadow", "md", "md", id="choice"),
        pytest.param("card-text-align", "center", "center", id="alignment"),
        pytest.param("card-img-alt", "A  picture", "A  picture", id="free-text"),
    ],
)
def test_valid_values_keep_their_source_text(presentation, name, argument, expected):
    spec = {**presentation.GRID_SPEC, **presentation.CARD_SPEC}
    assert spec[name](argument) == expected


@pytest.mark.parametrize(
    ("name", "argument"),
    [
        pytest.param("grid-gutter", "9", id="gutter-out-of-range"),
        pytest.param("grid-gutter", "1 2", id="gutter-wrong-arity"),
        pytest.param("grid-gutter", "wide", id="gutter-not-a-number"),
        pytest.param("grid-margin", "1 2 3", id="margin-wrong-arity"),
        pytest.param("card-shadow", "huge", id="unknown-choice"),
        pytest.param("card-text-align", "<script>", id="markup-as-choice"),
        pytest.param("card-shadow", None, id="choice-without-value"),
        pytest.param("grid-gutter", None, id="gutter-without-value"),
        pytest.param("grid-gutter", "", id="gutter-empty"),
    ],
)
def test_upstream_validation_still_applies(presentation, name, argument):
    spec = {**presentation.GRID_SPEC, **presentation.CARD_SPEC}
    with pytest.raises(ValueError):
        spec[name](argument)


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("a\nb", id="newline"),
        pytest.param("a\rb", id="carriage-return"),
        pytest.param("a\r\nb", id="crlf"),
        pytest.param("a\x00b", id="nul"),
        pytest.param("x\n:link: javascript:alert(1)", id="option-injection"),
        pytest.param("x\n\n.. raw:: html\n\n   <script>", id="directive-injection"),
    ],
)
def test_every_option_rejects_multi_line_values(presentation, argument):
    spec = {**presentation.GRID_SPEC, **presentation.CARD_SPEC}
    accepted = []
    for name, validate in sorted(spec.items()):
        try:
            validate(argument)
        except ValueError as exc:
            if "single line" not in str(exc):
                accepted.append((name, str(exc)))
        else:
            accepted.append((name, "accepted"))
    assert accepted == []


def test_surrounding_newlines_are_trimmed_not_rejected(presentation):
    assert presentation.CARD_SPEC["card-img-alt"]("\n  alt text \n") == "alt text"


@pytest.mark.parametrize("name", URL_OPTIONS)
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("javascript:alert(1)", id="javascript"),
        pytest.param("JavaScript:alert(1)", id="javascript-mixed-case"),
        pytest.param("   javascript:alert(1)", id="javascript-leading-space"),
        pytest.param("java\tscript:alert(1)", id="javascript-embedded-tab"),
        pytest.param("data:text/html,<script>alert(1)</script>", id="data"),
        pytest.param("vbscript:msgbox(1)", id="vbscript"),
        pytest.param("file:///etc/passwd", id="file"),
        pytest.param("ftp://example.invalid/x", id="ftp"),
        pytest.param("blob:https://example.invalid/1", id="blob"),
        pytest.param("x-custom:thing", id="custom-scheme"),
    ],
)
def test_url_options_reject_active_schemes(presentation, name, url):
    with pytest.raises(ValueError, match=r"HTTP\(S\) URL or a relative"):
        presentation.CARD_SPEC[name](url)


@pytest.mark.parametrize("name", URL_OPTIONS)
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param("https://example.invalid/a?b=1#c", None, id="https"),
        pytest.param("HTTP://EXAMPLE.invalid/", None, id="http-uppercase"),
        pytest.param("mailto:someone@example.invalid", None, id="mailto"),
        pytest.param("docs/page", None, id="relative-path"),
        pytest.param("../_static/img.png", None, id="parent-relative"),
        pytest.param("/abs/img.png", None, id="root-relative"),
        pytest.param("page.html#frag", None, id="fragment"),
        pytest.param("  docs/page  ", "docs/page", id="trimmed"),
        pytest.param("images/ünï 日本.png", None, id="unicode-path"),
    ],
)
def test_url_options_accept_web_and_relative_targets(presentation, name, url, expected):
    assert presentation.CARD_SPEC[name](url) == (url if expected is None else expected)


def test_scheme_check_does_not_apply_to_plain_text_options(presentation):
    text = "javascript: the good parts"
    assert presentation.CARD_SPEC["card-img-alt"](text) == text
    assert presentation.CARD_SPEC["card-link-alt"](text) == text
    assert presentation.CARD_SPEC["card-class-card"]("data:x") == "data:x"


def test_very_long_single_line_value_is_kept(presentation):
    text = "word " * 50_000
    assert presentation.CARD_SPEC["card-img-alt"](text) == text.strip()


# -- flag semantics -----------------------------------------------------------


def _flag_options(presentation):
    from docutils.parsers.rst import directives

    from sphinx_design.grids import GridDirective, GridItemCardDirective

    found = ["grid-" + k for k, v in GridDirective.option_spec.items() if v is directives.flag]
    found += [
        "card-" + k
        for k, v in GridItemCardDirective.option_spec.items()
        if v is directives.flag
    ]
    return sorted(found)


def test_flag_options_preserve_none_for_canonical_flag_syntax(presentation):
    spec = {**presentation.GRID_SPEC, **presentation.CARD_SPEC}
    flags = _flag_options(presentation)
    assert flags, "installed Sphinx Design exposes no flag option to forward"
    for name in flags:
        assert spec[name](None) is None
        with pytest.raises(ValueError):
            spec[name]("unexpected value")


def test_valueless_text_option_becomes_empty_string(presentation):
    assert presentation.CARD_SPEC["card-img-alt"](None) == ""
    assert presentation.CARD_SPEC["card-link"](None) == ""


# -- forwarded ----------------------------------------------------------------


def test_forwarded_strips_grid_prefix(presentation):
    options = {
        "grid-gutter": "2",
        "grid-margin": "0",
        "card-shadow": "md",
        "columns": "3",
        "filter": "a=b",
    }
    assert presentation.forwarded(options, "grid-") == {"gutter": "2", "margin": "0"}


def test_forwarded_strips_card_prefix(presentation):
    options = {
        "grid-gutter": "2",
        "card-shadow": "md",
        "card-link": "docs/page",
        "card-class-card": "x",
        "title": "T",
    }
    assert presentation.forwarded(options, "card-") == {
        "shadow": "md",
        "link": "docs/page",
        "class-card": "x",
    }


@pytest.mark.parametrize("prefix", ["grid-", "card-"])
def test_forwarded_ignores_unrecognised_namespaced_options(presentation, prefix):
    options = {prefix + "not-a-real-option": "x", prefix.rstrip("-"): "y"}
    assert presentation.forwarded(options, prefix) == {}


@pytest.mark.parametrize("prefix", ["grid-", "card-"])
def test_forwarded_empty_options(presentation, prefix):
    assert presentation.forwarded({}, prefix) == {}


def test_forwarded_preserves_order_and_none_flags(presentation):
    flags = [name for name in _flag_options(presentation) if name.startswith("grid-")]
    assert flags
    options = {"grid-margin": "1", flags[0]: None, "grid-gutter": "3"}
    result = presentation.forwarded(options, "grid-")
    assert list(result) == ["margin", flags[0][len("grid-") :], "gutter"]
    assert result[flags[0][len("grid-") :]] is None


def test_forwarded_does_not_mutate_options(presentation):
    options = {"grid-gutter": "2", "card-shadow": "md"}
    before = dict(options)
    presentation.forwarded(options, "grid-")
    presentation.forwarded(options, "card-")
    assert options == before
