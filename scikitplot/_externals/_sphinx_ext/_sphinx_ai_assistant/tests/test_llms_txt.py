"""
Tests for the structured ``llms.txt`` helpers.

These cover the pure functions only, so the suite runs without Sphinx, without a
built documentation tree, and without the optional Markdown dependencies. The
module is loaded from its file path rather than imported as
``scikitplot._externals...`` so the tests also work against a checkout that is
not installed.

See Also
--------
:func:`generate_llms_txt` : the Sphinx hook that composes these helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_INIT = Path(__file__).resolve().parents[1] / "__init__.py"


def _load():
    """Import the extension module directly from its path."""
    spec = importlib.util.spec_from_file_location("_aia_llms_probe", _INIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aia = _load()


# ---------------------------------------------------------------- titles

@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("# Getting started\n\nbody", "Getting started"),
        ("\n\n#  Padded  \n", "Padded"),
        ("no heading at all", "FALLBACK"),
        ("## Only level two\n", "FALLBACK"),
        ("#NoSpace\n", "FALLBACK"),
        ("# \n\ntext", "FALLBACK"),
        ("", "FALLBACK"),
    ],
)
def test_title_extraction(markdown, expected):
    assert aia._llms_entry_title(markdown, "FALLBACK") == expected


def test_title_stops_at_a_code_fence():
    """A ``#`` inside a fenced block is a comment, not a page title."""
    assert aia._llms_entry_title("```py\n# not a title\n```\n", "FALLBACK") == "FALLBACK"


def test_title_scan_is_bounded():
    """
    A heading far down a long page is not searched for.

    Whole-site generation must stay linear in page count, not page size.
    """
    markdown = "\n" * 200 + "# Late heading\n"
    assert aia._llms_entry_title(markdown, "FALLBACK") == "FALLBACK"


# ---------------------------------------------------------------- descriptions

def test_description_takes_the_first_prose_paragraph():
    md = "# Title\n\nFirst paragraph here.\n\nSecond paragraph.\n"
    assert aia._llms_entry_description(md) == "First paragraph here."


def test_description_joins_wrapped_lines():
    md = "# T\n\nWrapped across\ntwo source lines.\n"
    assert aia._llms_entry_description(md) == "Wrapped across two source lines."


@pytest.mark.parametrize(
    "markdown",
    [
        "# T\n\n```py\ncode\n```\n",
        "# T\n\n~~~\ncode\n~~~\n",
        "# T\n\n:::{note}\ntext\n:::\n",
        "# T\n\n- a list item\n",
        "# T\n\n* star item\n",
        "# T\n\n1. numbered\n",
        "# T\n\n> a block quote\n",
        "# T\n\n| a | table |\n",
        "# T\n\n.. directive::\n",
        "# T\n",
        "",
    ],
)
def test_description_is_empty_when_there_is_no_prose(markdown):
    """An empty description is a valid outcome; the caller omits the suffix."""
    assert aia._llms_entry_description(markdown) == ""


def test_description_does_not_read_inside_a_code_fence():
    """Regression: the first line inside a leading fence is not a summary."""
    assert aia._llms_entry_description("# T\n\n```py\nimport os\n```\n") == ""


def test_description_truncates_on_a_word_boundary():
    md = "# T\n\n" + ("word " * 100)
    out = aia._llms_entry_description(md, limit=40)
    assert len(out) <= 41
    assert out.endswith("\u2026")
    assert "  " not in out


def test_description_respects_an_explicit_limit():
    md = "# T\n\nexactly ten chars"
    assert aia._llms_entry_description(md, limit=500) == "exactly ten chars"


# ---------------------------------------------------------------- sections

@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("index.md", ""),
        ("user_guide/intro.md", "User Guide"),
        ("auto_examples/a/b.md", "Auto Examples"),
        ("api-reference/x.md", "Api Reference"),
        ("apis/deep/nested/page.md", "Apis"),
    ],
)
def test_section_derivation(rel, expected):
    assert aia._llms_section_for(rel) == expected


def test_section_mapping_overrides_the_derived_title():
    mapping = {"apis": "API Reference"}
    assert aia._llms_section_for("apis/metrics.md", mapping) == "API Reference"


def test_longest_matching_prefix_wins():
    """So a sub-tree can be separated from its parent."""
    mapping = {"apis": "API Reference", "apis/deprecated": "Deprecated API"}
    assert aia._llms_section_for("apis/deprecated/old.md", mapping) == "Deprecated API"
    assert aia._llms_section_for("apis/metrics.md", mapping) == "API Reference"


def test_mapping_tolerates_surrounding_slashes():
    assert aia._llms_section_for("apis/x.md", {"/apis/": "API"}) == "API"


def test_mapping_does_not_match_a_partial_segment():
    """``apis`` must not swallow ``apis_legacy``."""
    assert aia._llms_section_for("apis_legacy/x.md", {"apis": "API"}) == "Apis Legacy"


# ---------------------------------------------------------------- rendering

def _entry(section, title, url, description=""):
    return {"section": section, "title": title, "url": url, "description": description}


def test_render_emits_heading_summary_sections_and_entries():
    out = aia._render_llms_txt(
        "scikit-plots",
        "A short project summary.",
        [
            _entry("", "Home", "/index.md"),
            _entry("User Guide", "Intro", "/user_guide/intro.md", "How to start."),
            _entry("User Guide", "Plots", "/user_guide/plots.md"),
            _entry("API Reference", "metrics", "/apis/metrics.md", "Metrics API."),
        ],
    )
    assert out.startswith("# scikit-plots\n\n> A short project summary.\n")
    assert "## User Guide\n" in out
    assert "- [Intro](/user_guide/intro.md): How to start.\n" in out
    assert "- [Plots](/user_guide/plots.md)\n" in out  # no dangling colon
    assert out.endswith("\n")
    assert "\n\n\n" not in out


def test_render_omits_the_summary_when_absent():
    out = aia._render_llms_txt("P", None, [_entry("", "Home", "/index.md")])
    assert ">" not in out


def test_root_entries_precede_every_section():
    out = aia._render_llms_txt(
        "P",
        None,
        [
            _entry("User Guide", "Intro", "/u/i.md"),
            _entry("", "Home", "/index.md"),
        ],
    )
    assert out.index("- [Home]") < out.index("## User Guide")


def test_sections_keep_first_appearance_order():
    """
    Not alphabetical: the file should mirror toctree order.

    Sorting would put ``API Reference`` before ``Getting Started`` for every
    project, which is backwards for anyone meeting the project for the first
    time.
    """
    out = aia._render_llms_txt(
        "P",
        None,
        [
            _entry("Getting Started", "a", "/g/a.md"),
            _entry("API Reference", "b", "/api/b.md"),
            _entry("Getting Started", "c", "/g/c.md"),
        ],
    )
    assert out.index("## Getting Started") < out.index("## API Reference")
    # An interleaved entry rejoins its own section rather than starting a second one.
    assert out.count("## Getting Started") == 1


def test_render_falls_back_to_the_url_when_a_title_is_missing():
    out = aia._render_llms_txt("P", None, [_entry("", "", "/index.md")])
    assert "- [/index.md](/index.md)" in out


def test_render_handles_no_entries():
    out = aia._render_llms_txt("P", "summary", [])
    assert out == "# P\n\n> summary\n"


# ---------------------------------------------------------------- integration

def test_helpers_compose_into_a_valid_document():
    """End-to-end over the pure layer, as `generate_llms_txt` composes it."""
    pages = {
        "index.md": "# scikit-plots\n\nVisualisation for scikit-learn.\n",
        "user_guide/intro.md": "# Introduction\n\nStart here.\n",
        "apis/metrics.md": "# scikitplot.metrics\n\n```py\ncode\n```\n",
    }
    entries = []
    for rel, text in pages.items():
        entries.append(
            {
                "section": aia._llms_section_for(rel, {"apis": "API Reference"}),
                "title": aia._llms_entry_title(text, rel),
                "url": "https://example.org/" + rel,
                "description": aia._llms_entry_description(text),
            }
        )
    out = aia._render_llms_txt("scikit-plots", "Summary.", entries)

    lines = out.splitlines()
    assert lines[0] == "# scikit-plots"
    assert lines[2] == "> Summary."
    assert "- [scikit-plots](https://example.org/index.md): Visualisation for scikit-learn." in lines
    assert "## User Guide" in lines
    assert "## API Reference" in lines
    # The API page has only a code block, so it gets a link with no description.
    assert "- [scikitplot.metrics](https://example.org/apis/metrics.md)" in lines


def test_declared_formats_are_the_ones_the_writer_accepts():
    assert aia.LLMS_TXT_FORMATS == ("structured", "flat")
    assert "index.md" in aia.LLMS_TXT_ROOT_FIRST
