"""
Tests for the shared HTML→Markdown conversion rule table.

The table exists so the build-time converter and the browser Turndown rules
cannot diverge. These tests hold that property: the table's integrity, the
build-time behaviour of each active rule against real rendered HTML, and the
structural guarantee that the browser reads the same table rather than
duplicating it.

See Also
--------
:data:`CONVERSION_RULES` : the table itself.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INIT = _ROOT / "__init__.py"
_JS = _ROOT / "_static" / "ai-assistant.js"
_FIXTURES = _ROOT.parent / "_sphinxcontrib_youtube" / "tests" / "test_build"

bs4 = pytest.importorskip("bs4", reason="build-time conversion needs beautifulsoup4")
pytest.importorskip("markdownify", reason="build-time conversion needs markdownify")


def _load():
    spec = importlib.util.spec_from_file_location("_aia_rules_probe", _INIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aia = _load()


# ---------------------------------------------------------------- table shape

def test_every_rule_has_the_required_fields():
    for rule in aia.CONVERSION_RULES:
        assert rule.get("name"), rule
        assert rule.get("status") in {"implemented", "planned", "superseded"}, rule
        assert rule.get("kind") in aia.CONVERSION_KINDS, rule
        assert rule.get("evidence"), f"{rule['name']} has no evidence of where its selector was verified"


def test_rule_names_are_unique():
    names = [rule["name"] for rule in aia.CONVERSION_RULES]
    assert len(names) == len(set(names))


def test_implemented_rules_have_a_selector():
    for rule in aia._implemented_conversion_rules():
        assert rule.get("selector"), f"{rule['name']} is implemented but has no selector"


def test_planned_rules_are_inert():
    """
    A planned rule must not be able to fire.

    The list doubles as the tracking record for increment 5. If a planned entry
    could match, the table would be claiming coverage it does not have.
    """
    payload_names = {entry["name"] for entry in aia._conversion_rules_payload()}
    for rule in aia.CONVERSION_RULES:
        if rule["status"] in {"planned"}:
            assert not rule.get("selector"), f"{rule['name']} is planned but has a selector"
            assert rule["name"] not in payload_names


def test_every_implemented_selector_is_valid_css():
    soup = bs4.BeautifulSoup("<div></div>", "html.parser")
    for rule in aia._implemented_conversion_rules():
        soup.select(rule["selector"])  # raises on an invalid selector


# ---------------------------------------------------------------- payload

def test_payload_carries_only_what_the_browser_needs():
    for entry in aia._conversion_rules_payload():
        assert set(entry) == {"name", "selector", "kind", "label"}


def test_payload_preserves_table_order():
    """First match wins in the browser, so order is behaviour, not cosmetics."""
    payload = [entry["name"] for entry in aia._conversion_rules_payload()]
    implemented = [rule["name"] for rule in aia._implemented_conversion_rules()]
    assert payload == implemented


# ---------------------------------------------------------------- behaviour

def test_headerlink_is_dropped():
    out = aia.html_to_markdown('<h2>Title<a class="headerlink" href="#t">\u00b6</a></h2>')
    assert out.strip() == "## Title"


def test_ordinary_links_survive():
    """The drop rule must be selector-scoped, not a blanket anchor removal."""
    out = aia.html_to_markdown('<p>see <a href="/x">docs</a></p>')
    assert "[docs](/x)" in out


def test_admonition_becomes_a_block_quote():
    out = aia.html_to_markdown(
        '<div class="admonition note"><p class="admonition-title">Note</p><p>Body.</p></div>'
    )
    assert out.strip() == "> **Note**\n> Body."


def test_iframe_without_a_source_is_left_alone():
    rule = {"kind": "link", "label": "Video"}
    soup = bs4.BeautifulSoup("<iframe></iframe>", "html.parser")
    assert aia._render_conversion_rule(rule, soup.iframe, "fallback") == "fallback"


def test_pre_pass_is_a_no_op_without_rules(monkeypatch):
    monkeypatch.setattr(aia, "_implemented_conversion_rules", lambda: [])
    html = '<a class="headerlink" href="#t">x</a>'
    assert aia._apply_conversion_rules(html) == html


def test_pre_pass_returns_input_unchanged_when_nothing_matches():
    html = "<p>plain</p>"
    assert aia._apply_conversion_rules(html) == html


def test_pre_pass_drops_an_empty_heading_rather_than_emitting_empty_bold():
    html = '<div class="sd-card-header"></div><p>after</p>'
    out = aia.html_to_markdown(html)
    assert "**" not in out
    assert "after" in out


# ---------------------------------------------------------------- one table

def test_the_browser_reads_the_table_rather_than_duplicating_it():
    """
    The structural guarantee: no second rule set in the JavaScript.

    If a selector string from the table appeared literally in the JS, that would
    be a duplicated rule, and duplicated rules drift.
    """
    js = _JS.read_text(encoding="utf-8")
    assert "conversionRules" in js
    assert "_applyConfiguredRules" in js
    for rule in aia._implemented_conversion_rules():
        assert rule["selector"] not in js, (
            f"selector for {rule['name']} is hard-coded in the browser path; "
            "it must come from the config payload"
        )


def test_browser_driver_handles_every_kind_the_table_can_declare():
    js = _JS.read_text(encoding="utf-8")
    driver = js[js.index("function _applyConfiguredRules") :]
    driver = driver[: driver.index("\n    /**", 10)]
    for kind in aia.CONVERSION_KINDS:
        assert f"'{kind}'" in driver, f"browser driver has no branch for kind {kind!r}"


def test_config_payload_is_serialisable():
    import json

    json.dumps(aia._conversion_rules_payload())


def test_declared_kinds_match_what_the_renderer_implements():
    """`passthrough` returns the text unchanged; the rest transform it."""
    assert aia._render_conversion_rule({"kind": "passthrough"}, None, "keep") == "keep"
    assert aia._render_conversion_rule({"kind": "drop"}, None, "gone") == ""
    fenced = aia._render_conversion_rule({"kind": "fence"}, None, " body ")
    assert fenced.strip().startswith("```") and fenced.strip().endswith("```")


# ---------------------------------------------------------------- emphasis

def test_strong_produces_exactly_two_asterisks():
    """
    Regression: ``strong_em_symbol`` must be a single character.

    markdownify doubles the symbol for ``<strong>``, so passing ``"**"``
    produced ``****text****`` — literal asterisks around unbolded text, on every
    bold run in the documentation. Found by converting the published site: 40
    occurrences on the front page, 157 on one API page.
    """
    assert aia.html_to_markdown("<p><strong>Bold</strong></p>").strip() == "**Bold**"


def test_emphasis_produces_exactly_one_asterisk():
    assert aia.html_to_markdown("<p><em>it</em></p>").strip() == "*it*"


def test_no_quadruple_asterisks_in_mixed_emphasis():
    out = aia.html_to_markdown("<p><strong>a</strong> and <em>b</em></p>")
    assert "****" not in out


# ---------------------------------------------------------------- sphinx-design

def test_card_header_becomes_a_single_bold_line():
    html = (
        '<div class="sd-card docutils">'
        '<div class="sd-card-header docutils">'
        '<p class="sd-card-text"><strong>nearest neighbor</strong></p></div>'
        '<div class="sd-card-body docutils"><p>body</p></div></div>'
    )
    out = aia.html_to_markdown(html)
    assert "**nearest neighbor**" in out
    assert "****" not in out
    assert "body" in out


def test_tab_labels_become_bold_lines():
    """
    Verified against dev/install/installation.html.

    Without this, 'pip' and 'Windows' convert to bare paragraphs that a reader
    cannot distinguish from body text, losing the fact that they label
    alternatives.
    """
    html = (
        '<div class="sd-tab-set docutils">'
        '<input checked id="sd-tab-item-0" name="sd-tab-set-0" type="radio">'
        '<label class="sd-tab-label tab-6" for="sd-tab-item-0"> pip</label>'
        '<div class="sd-tab-content docutils"><p>body</p></div></div>'
    )
    out = aia.html_to_markdown(html)
    assert "**pip**" in out
    assert "body" in out


def test_dropdown_summary_becomes_a_bold_line():
    html = (
        '<details class="sd-dropdown sd-card"><summary class="sd-summary-title sd-card-header">'
        '<span class="sd-summary-text">Uploading artifacts</span></summary>'
        '<div class="sd-summary-content"><p>steps</p></div></details>'
    )
    out = aia.html_to_markdown(html)
    assert "**Uploading artifacts**" in out
    assert "steps" in out


def test_rubric_becomes_a_bold_line():
    """A rubric is semantically a heading but renders as an ordinary <p>."""
    out = aia.html_to_markdown('<p class="rubric">Preparation</p><ul><li><p>x</p></li></ul>')
    assert out.lstrip().startswith("**Preparation**")


def test_button_links_keep_text_and_href():
    html = '<a class="sd-sphinx-override sd-btn sd-btn-primary reference external" href="https://x/"><span>Visit →</span></a>'
    out = aia.html_to_markdown(html)
    assert "https://x/" in out and "Visit" in out


def test_layout_containers_are_transparent():
    """Rows, columns and containers carry no meaning a reader can use."""
    html = (
        '<div class="sd-container-fluid docutils">'
        '<div class="sd-row sd-row-cols-1 docutils">'
        '<div class="sd-col sd-d-flex-row docutils"><p>content</p></div></div></div>'
    )
    assert aia.html_to_markdown(html).strip() == "content"


def test_inheritance_diagram_keeps_its_descriptive_alt_text():
    html = '<img src="../../_images/inheritance-76d3b25.png" alt="Inheritance diagram of A, B">'
    out = aia.html_to_markdown(html)
    assert "Inheritance diagram of A, B" in out


# ---------------------------------------------------------------- video links

@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("https://www.youtube.com/embed/dQw4w9WgXcQ",
         ("YouTube video", "https://youtu.be/dQw4w9WgXcQ")),
        ("https://www.youtube-nocookie.com/embed/ID",
         ("YouTube video", "https://youtu.be/ID")),
        ("https://peertube.tv/videos/embed/327a21b3",
         ("PeerTube video", "https://peertube.tv/w/327a21b3")),
        # Vimeo's own _platform_url *is* the player URL, so no watch form is
        # invented — only the label changes.
        ("https://player.vimeo.com/video/148751763",
         ("Vimeo video", "https://player.vimeo.com/video/148751763")),
        ("https://example.org/embedded", ("Video", "https://example.org/embedded")),
        ("", ("Video", "")),
    ],
)
def test_embed_urls_map_to_their_watch_form(src, expected):
    assert aia._video_link(src) == expected


def test_query_parameters_survive_the_mapping():
    """`url_parameters` may carry a start time or playlist position."""
    label, href = aia._video_link("https://www.youtube.com/embed/ID?start=42&rel=0")
    assert href == "https://youtu.be/ID?start=42&rel=0"


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("youtube.html", "[YouTube video](https://youtu.be/dQw4w9WgXcQ)"),
        ("vimeo.html", "[Vimeo video](https://player.vimeo.com/video/148751763)"),
        ("peertube.html", "[PeerTube video](https://peertube.tv/w/"),
    ],
)
def test_submodule_fixtures_convert_to_watch_links(fixture, expected):
    """End-to-end against `_sphinxcontrib_youtube`'s own rendered output."""
    path = _ROOT.parent / "_sphinxcontrib_youtube" / "tests" / "test_build" / fixture
    if not path.is_file():
        pytest.skip(f"fixture not present: {path}")
    out = aia.html_to_markdown(path.read_text(encoding="utf-8"))
    assert expected in out


def test_video_prefix_map_is_shipped_to_the_browser():
    """The browser must resolve a video link identically, from the same data."""
    js = _JS.read_text(encoding="utf-8")
    assert "videoEmbedPrefixes" in js
    assert "_videoLink" in js
    for prefix, _watch, _label in aia.VIDEO_EMBED_PREFIXES:
        assert prefix not in js, "embed prefixes must come from the config, not be hard-coded"


# ---------------------------------------------------------------- layout tables

def test_hlist_converts_as_a_list_not_a_table():
    """
    Sphinx's ``hlist`` is a layout table, not tabular data.

    Converting it as a Markdown table produced a one-cell table with an empty
    header and every item flattened onto one line, because a Markdown cell
    cannot contain line breaks. `dev/learn/terminology/index.html` has 27 of
    them.
    """
    html = (
        '<table class="hlist"><tr><td><ul class="simple">'
        '<li><p><a href="a.html">Alpha</a></p></li>'
        '<li><p><a href="b.html">Beta</a></p></li></ul></td>'
        '<td><ul class="simple"><li><p><a href="c.html">Gamma</a></p></li></ul></td>'
        "</tr></table>"
    )
    out = aia.html_to_markdown(html)
    assert "| ---" not in out, "layout table must not survive as a Markdown table"
    for name, href in (("Alpha", "a.html"), ("Beta", "b.html"), ("Gamma", "c.html")):
        assert f"[{name}]({href})" in out
    # Each item on its own line, not '* A * B * C'.
    bullets = [ln for ln in out.splitlines() if ln.strip().startswith("*")]
    assert len(bullets) == 3


def test_a_real_data_table_still_converts_as_a_table():
    """The unwrap rule is selector-scoped and must not touch ordinary tables."""
    html = (
        "<table><thead><tr><th>h1</th><th>h2</th></tr></thead>"
        "<tbody><tr><td>a</td><td>b</td></tr></tbody></table>"
    )
    out = aia.html_to_markdown(html)
    assert "| --- |" in out or "---" in out
    assert "h1" in out and "b" in out


def test_topic_title_becomes_a_bold_line():
    html = '<nav class="contents"><p class="topic-title">Table of Contents</p><ul><li><p>x</p></li></ul></nav>'
    out = aia.html_to_markdown(html)
    assert out.lstrip().startswith("**Table of Contents**")
