# scikitplot/corpus/_chunkers/tests/test__structure.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Structure-aware chunking (Run 4, R4-2): headings, code blocks, heading paths.
"""

from __future__ import annotations

import time

import pytest

from .._structure import StructureChunker, StructureChunkerConfig

MARKDOWN = """# Guide

Intro paragraph one.

Intro paragraph two.

## Install

Run this:

```bash
pip install x
# not a heading
```

Setext title
------------

Body under setext.

~~~~
unclosed fence
"""

RST = """=====
Guide
=====

Intro text here.

Install
-------

Run this::

    pip install x
    pip check

.. code-block:: python

   import x

Usage
-----

Body.

Deep
~~~~

Deeper body.
"""


#: Budgets and sizes the tests below assert against.
SMALL_BUDGET = 12
TITLE_BUDGET = 100
TITLE_WORDS = 2000
#: Generous bound: the quadratic ATX regex took ~3 minutes on these inputs.
LINEAR_SECONDS = 5.0


def _chunks(text, **kwargs):
    return StructureChunker(StructureChunkerConfig(**kwargs)).chunk(text).chunks


class TestConfig:
    def test_markup_is_given_not_guessed(self):
        with pytest.raises(ValueError, match="markup must be one of"):
            StructureChunkerConfig(markup="auto")

    def test_types_and_ranges(self):
        with pytest.raises(TypeError, match="include_heading"):
            StructureChunkerConfig(include_heading="yes")
        with pytest.raises(ValueError, match="chunk_size"):
            StructureChunkerConfig(chunk_size=0)
        assert StructureChunkerConfig(separators=["\n", ""]).separators == ("\n", "")


class TestMarkdown:
    def test_blocks_and_heading_paths(self):
        chunks = _chunks(MARKDOWN, chunk_size=200)
        got = [(c.metadata["block"], c.metadata["heading_path"]) for c in chunks]
        assert got == [
            ("prose", ("Guide",)),
            ("prose", ("Guide", "Install")),
            ("code", ("Guide", "Install")),
            ("prose", ("Guide", "Setext title")),
            ("code", ("Guide", "Setext title")),
        ]

    def test_a_hash_inside_a_fence_is_not_a_heading(self):
        code = [c for c in _chunks(MARKDOWN) if c.metadata["block"] == "code"]
        assert "# not a heading" in code[0].text
        assert code[1].text == "~~~~\nunclosed fence"  # runs to the end

    def test_paragraphs_of_a_section_share_a_chunk_within_the_budget(self):
        first = _chunks(MARKDOWN, chunk_size=200)[0]
        assert first.text == "# Guide\n\nIntro paragraph one.\n\nIntro paragraph two."
        assert first.context_span[0] == 0  # the heading starts the context
        assert MARKDOWN[first.start_char : first.end_char].startswith("Intro")

    def test_without_heading_context(self):
        heading, first = _chunks(MARKDOWN, include_heading=False)[:2]
        assert (heading.metadata["block"], heading.text) == ("heading", "# Guide")
        assert first.text.startswith("Intro paragraph one.")
        assert first.context_start is None

    def test_long_prose_is_split_by_the_budget(self):
        chunks = _chunks(MARKDOWN, chunk_size=SMALL_BUDGET)
        for chunk in chunks:
            assert chunk.end_char - chunk.start_char <= SMALL_BUDGET
            assert chunk.text == MARKDOWN[slice(*chunk.context_span)]


class TestRst:
    def test_titles_levels_and_code(self):
        chunks = _chunks(RST, markup="rst", chunk_size=200)
        got = [
            (
                c.metadata["block"],
                c.metadata["heading_path"],
                c.metadata["section_level"],
            )
            for c in chunks
        ]
        assert got == [
            ("prose", ("Guide",), 1),
            ("prose", ("Guide", "Install"), 2),
            ("code", ("Guide", "Install"), 2),
            ("code", ("Guide", "Install"), 2),
            ("prose", ("Guide", "Usage"), 2),
            ("prose", ("Guide", "Usage", "Deep"), 3),
        ]

    def test_a_literal_block_is_code(self):
        chunks = _chunks(RST, markup="rst", chunk_size=200)
        literal = chunks[2]
        assert literal.text == "    pip install x\n    pip check"

    def test_every_chunk_is_its_source_slice(self):
        for size in (5, 20, 200):
            for chunk in _chunks(RST, markup="rst", chunk_size=size):
                assert chunk.text == RST[slice(*chunk.context_span)]


class TestReviewFindings:
    """One test per finding of the 4b independent review."""

    def test_atx_lines_cost_linear_time(self):
        """Review 1: a long run of spaces or '#' backtracked quadratically."""
        chunker = StructureChunker()
        start = time.perf_counter()
        chunker.chunk("# a" + " " * 200_000 + "b")
        chunker.chunk("# " * 100_000 + "x")
        assert time.perf_counter() - start < LINEAR_SECONDS

    @pytest.mark.parametrize(
        ("line", "level", "title"),
        [
            ("# foo #", 1, "foo"),
            ("### Title#", 3, "Title#"),
            ("# foo \\#", 1, "foo \\#"),
            ("# ###", 1, ""),
            ("#", 1, ""),
            ("# #foo", 1, "#foo"),
            ("#\tTab", 1, "Tab"),
        ],
    )
    def test_atx_closing_sequence_follows_commonmark(self, line, level, title):
        """Review 8: a closing '#' needs a space or tab before it."""
        assert StructureChunker().blocks(line) == [
            ("heading", (0, len(line)), level, title)
        ]

    @pytest.mark.parametrize(
        "text", ["- item\n---", "> quote\n---", "1. item\n===", "    code\n---"]
    )
    def test_a_container_or_indented_line_is_not_underlined(self, text):
        """Review 4 and 8: these became headings, and headings were lost."""
        blocks = StructureChunker().blocks(text)
        assert {b[0] for b in blocks} == {"prose"}
        assert blocks[0][1][0] == 0
        assert blocks[-1][1][1] == len(text)

    def test_front_matter_is_its_own_chunk(self):
        """Review 4: the front matter became a heading and vanished."""
        text = "---\ntitle: My page\nlayout: post\n---\n\n# Real\n\nBody."
        chunks = _chunks(text)
        assert [(c.metadata["block"], c.text) for c in chunks] == [
            ("front_matter", "---\ntitle: My page\nlayout: post\n---"),
            ("prose", "# Real\n\nBody."),
        ]
        off = _chunks(text, front_matter=False)
        assert "title: My page" in "".join(c.text for c in off)
        with pytest.raises(TypeError, match="front_matter"):
            StructureChunkerConfig(front_matter=1)

    def test_an_unclosed_front_matter_is_commonmark(self):
        blocks = StructureChunker().blocks("---\nNo end here.\n")
        assert [b[0] for b in blocks] == ["prose", "prose"]  # a break, a paragraph

    @pytest.mark.parametrize("include_heading", [True, False])
    def test_no_heading_is_lost(self, include_heading):
        """Review 4: a heading before code, a heading or the end was dropped."""
        text = "# A\n\n## B\n\n```\nx\n```\n\n## C\n\nProse.\n\n## D\n"
        chunks = _chunks(text, include_heading=include_heading)
        joined = "\n".join(c.text for c in chunks)
        for title in ("# A", "## B", "## C", "## D", "Prose.", "x"):
            assert title in joined
        headings = [c.text for c in chunks if c.metadata["block"] == "heading"]
        expected = ["# A", "## B", "## D"] + ([] if include_heading else ["## C"])
        assert sorted(headings) == sorted(expected)

    def test_a_long_setext_title_is_kept_within_the_budget(self):
        """Review 4: a million-line title gave 0 chunks and no error."""
        text = "word\n" * TITLE_WORDS + "===\n"
        chunks = _chunks(text, chunk_size=TITLE_BUDGET)
        assert chunks
        assert all(c.metadata["block"] == "heading" for c in chunks)
        assert all(c.end_char - c.start_char <= TITLE_BUDGET for c in chunks)
        assert sum(c.text.count("word") for c in chunks) == TITLE_WORDS

    def test_a_section_that_starts_with_code_has_the_heading_alone(self):
        """Review 7: the docs claimed the first prose chunk held the heading."""
        chunks = _chunks("# H\n\n```\nx\n```\n\nFirst prose.")
        assert [(c.metadata["block"], c.text) for c in chunks] == [
            ("heading", "# H"),
            ("code", "```\nx\n```"),
            ("prose", "First prose."),
        ]

    def test_heading_path_is_immutable(self):
        """Review 6: a list inside the read-only metadata could be mutated."""
        chunk = _chunks(MARKDOWN)[0]
        assert chunk.metadata["heading_path"] == ("Guide",)
        with pytest.raises(AttributeError):
            chunk.metadata["heading_path"].append("x")

    def test_only_cr_lf_and_crlf_end_markdown_lines(self):
        """Review 8: U+2028 ended a line, making two headings."""
        assert StructureChunker().blocks("# A\u2028# B") == [
            ("heading", (0, 7), 1, "A\u2028# B")
        ]
        assert [b[0] for b in StructureChunker().blocks("# A\r\n# B\r# C")] == [
            "heading"
        ] * 3

    def test_rst_form_feed_is_not_a_line_break(self):
        rst = StructureChunker(StructureChunkerConfig(markup="rst"))
        assert [b[0] for b in rst.blocks("Title\x0cmore\n=====\n")] == ["prose"]

    @pytest.mark.parametrize("directive", ["note", "seealso", "toctree"])
    def test_a_directive_body_is_prose(self, directive):
        """Review 2: the body of '.. note::' was a literal block."""
        text = f"Intro.\n\n.. {directive}::\n\n   This is a body.\n"
        rst = StructureChunker(StructureChunkerConfig(markup="rst"))
        assert [b[0] for b in rst.blocks(text)] == ["prose"] * 3

    @pytest.mark.parametrize(
        ("text", "code", "after"),
        [
            (
                (
                    ".. note::\n\n   Run this::\n\n      pip install x\n\n"
                    "   Then more prose.\n"
                ),
                "      pip install x\n",
                "   Then more prose.\n",
            ),
            (
                "- Run::\n\n    code\n\n  continued item\n",
                "    code\n",
                "  continued item\n",
            ),
            ("1. Step::\n\n      x = 1\n\n   More.\n", "      x = 1\n", "   More.\n"),
        ],
    )
    def test_a_literal_block_ends_at_the_paragraphs_indentation(
        self, text, code, after
    ):
        """Review 3: a literal block ran on to the next line at indent 0."""
        rst = StructureChunker(StructureChunkerConfig(markup="rst"))
        blocks = [(k, text[a:b]) for k, (a, b), _, _ in rst.blocks(text)]
        assert ("code", code) in blocks
        assert blocks[-1] == ("prose", after)

    def test_documents_carry_the_structure_metadata(self, tmp_path):
        """Review 5: heading_path, section_level and block stopped at chunks."""
        from ..._base import DefaultFilter, DocumentReader  # noqa: PLC0415

        class KeepAll(DefaultFilter):  # the default drops short documents
            def include(self, doc):
                return True

        source = tmp_path / "guide.txt"  # the text reader: one raw chunk
        source.write_text(
            "# Guide\n\nIntro.\n\n## Install\n\n```\nx\n```\n", encoding="utf-8"
        )
        reader = DocumentReader.create(
            source, chunker=StructureChunker(), filter_=KeepAll()
        )
        got = [
            (
                d.metadata["block"],
                d.metadata["heading_path"],
                d.metadata["section_level"],
            )
            for d in reader.get_documents()
        ]
        assert got == [
            ("prose", ["Guide"], 1),
            ("heading", ["Guide", "Install"], 2),
            ("code", ["Guide", "Install"], 2),
        ]


@pytest.mark.parametrize(
    "line", ["* item", "- item", ".. note", ">>> 1 + 1", "| line", ":field: value"]
)
def test_an_rst_line_docutils_reads_otherwise_is_not_a_title(line):
    """Re-probe: '* item' over '------' became a title."""
    rst = StructureChunker(StructureChunkerConfig(markup="rst"))
    text = f"{line}\n{'-' * (len(line) + 2)}\n"
    assert "heading" not in [b[0] for b in rst.blocks(text)]


def test_an_enumerator_over_an_underline_is_a_title():
    """docutils: an enumerator not followed by an indented line is text."""
    rst = StructureChunker(StructureChunkerConfig(markup="rst"))
    assert rst.blocks("A. Einstein\n===========\n")[0][::3] == (
        "heading",
        "A. Einstein",
    )


def _rst_code(text):
    rst = StructureChunker(StructureChunkerConfig(markup="rst"))
    return [text[a:b] for k, (a, b), _, _ in rst.blocks(text) if k == "code"]


class TestReReviewFindings:
    """One test per finding of the 4b re-review; expectations are docutils'."""

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (
                ".. note::\n   Run this::\n\n      pip install x\n\n   Then prose.\n",
                ["      pip install x\n"],
            ),
            (
                ".. note:: This note ends with::\n\n      code here\n\n   More.\n",
                ["      code here\n"],
            ),
            (".. [1] See this::\n\n      code\n\n   more\n", ["      code\n"]),
            (".. [1] See this::\n\n      code\n\nAfter.\n", []),
            (".. a comment here::\n\n   not literal\n", []),
            (".. note::\n\n   A body.\n", []),
        ],
    )
    def test_literal_blocks_in_explicit_markup(self, text, code):
        """Re-review 1: a '::' in directive content was refused."""
        assert _rst_code(text) == code

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (
                (
                    "x : int\n    The value, e.g.::\n\n        f(1)\n\n"
                    "    Must be positive.\n"
                ),
                ["        f(1)\n"],
            ),
            (
                (
                    ":param x: first line\n    ends with::\n\n        code\n\n"
                    "    Still field.\n"
                ),
                ["        code\n"],
            ),
            ("- first line\n  second line::\n\n    code\n\n  more\n", ["    code\n"]),
        ],
    )
    def test_the_line_that_ends_with_colons_sets_the_indent(self, text, code):
        """Re-review 2: the term's indentation let the literal swallow prose."""
        assert _rst_code(text) == code

    def test_tabs_are_eight_columns(self):
        """docutils expands tabs: '-\tRun::' puts the item's text at column 8."""
        assert _rst_code("-\tRun::\n\n    code\n") == []
        assert _rst_code("- a::\n\n\tcode\n\n  more\n") == ["\tcode\n"]

    def test_a_quoted_literal_block(self):
        assert _rst_code("Para::\n\n> quoted\n> lines\n\nAfter.\n") == [
            "> quoted\n> lines\n"
        ]

    def test_front_matter_needs_a_line_after_the_opening(self):
        """Re-review 3: a document opening with a thematic break lost a title."""
        text = "---\n\n# Title\n\nSome text.\n\n---\n\nMore.\n"
        kinds = [b[0] for b in StructureChunker().blocks(text)]
        assert "front_matter" not in kinds
        assert ("Title",) in {c.metadata["heading_path"] for c in _chunks(text)}

    @pytest.mark.parametrize(
        ("text", "titles"),
        [
            ("---\na: 1\n---\n", ["a: 1"]),
            ("Intro\n\n***\nTitle\n===\n", ["Title"]),
            ("Para\n- - -\nNext\n", []),
        ],
    )
    def test_a_thematic_break_ends_a_paragraph(self, text, titles):
        """Re-review 4: '***' became part of a setext title."""
        chunker = StructureChunker(StructureChunkerConfig(front_matter=False))
        assert [b[3] for b in chunker.blocks(text) if b[0] == "heading"] == titles

    def test_thematic_breaks_cost_linear_time(self):
        start = time.perf_counter()
        StructureChunker().chunk("- " * 100_000 + "x\n" + "*" * 200_000 + " y\n")
        assert time.perf_counter() - start < LINEAR_SECONDS

    @pytest.mark.parametrize(
        "line", ["-v  verbose", "-a, --all  everything", "+-----+-+"]
    )
    def test_an_option_or_table_line_is_not_a_title(self, line):
        """Re-review 5: an option-list line over an underline was a title."""
        rst = StructureChunker(StructureChunkerConfig(markup="rst"))
        text = f"{line}\n{'=' * (len(line) + 2)}\n"
        assert "heading" not in [b[0] for b in rst.blocks(text)]

    @pytest.mark.parametrize("line", ["--long", "/V", "-v", "-a b", "-a, --all"])
    def test_an_option_without_a_description_is_a_title(self, line):
        """Re-review 2, F2: docutils reads a bodiless option as text."""
        rst = StructureChunker(StructureChunkerConfig(markup="rst"))
        text = f"{line}\n{'=' * (len(line) + 2)}\n"
        assert [b[3] for b in rst.blocks(text) if b[0] == "heading"] == [line]

    def test_min_length_may_drop_a_heading(self):
        """Re-review 6: the docs said no heading is ever lost."""
        chunks = _chunks("# A\n\n```\nlong code line\n```\n", min_length=5)
        assert [c.metadata["block"] for c in chunks] == ["code"]


class TestSecondReReviewFindings:
    """One test per finding of the second 4b re-review (docutils 0.22)."""

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (
                ":param x: Example::\n\n        code\n\n    more text\n",
                ["        code\n"],
            ),
            (
                ":returns: a list, e.g.::\n\n      [1, 2]\n\n   More.\n",
                ["      [1, 2]\n"],
            ),
            ("-v  verbose::\n\n      code\n\n    more\n", ["      code\n"]),
            (":param x: see::\n\n    code\n", []),
        ],
    )
    def test_field_and_option_bodies_set_the_literal_indent(self, text, code):
        """F1: a one-line field or option item swallowed the prose after it."""
        assert _rst_code(text) == code

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ("- .. note::\n\n     body\n", []),
            ("- .. note:: x::\n\n        code\n\n    more\n", ["        code\n"]),
            ("- - Run::\n\n      code\n\n    more\n", ["      code\n"]),
        ],
    )
    def test_explicit_markup_after_a_list_marker(self, text, code):
        """F3: '- .. note::' was not read as a directive."""
        assert _rst_code(text) == code

    @pytest.mark.parametrize("text", ["| a::\n\n  x\n", ">>> a::\n\n  x\n"])
    def test_a_line_block_or_doctest_starts_no_literal(self, text):
        """F4: docutils makes the indented text a block quote."""
        assert _rst_code(text) == []

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (".. image:: foo.png\n   :alt: Look::\n\n      not lit\n", []),
            (
                ".. note::\n   :class: x\n\n   Body::\n\n      lit\n\n   tail\n",
                ["      lit\n"],
            ),
        ],
    )
    def test_a_directive_option_ending_with_colons_is_a_value(self, text, code):
        """Probe: ':alt: Look::' started a literal block."""
        assert _rst_code(text) == code

    def test_explicit_markup_ends_at_a_dedented_line(self):
        """F6: a comment without a blank line swallowed the next paragraph."""
        assert _rst_code(".. foo bar::\nPara::\n\n  lit\n") == ["  lit\n"]

    def test_nested_explicit_markup_costs_linear_time(self):
        """F5: each explicit paragraph rescanned every deeper line."""
        rst = StructureChunker(StructureChunkerConfig(markup="rst"))
        text = "".join(" " * i + ".. a\n\n" for i in range(2000))
        start = time.perf_counter()
        rst.chunk(text)
        assert time.perf_counter() - start < LINEAR_SECONDS


#: reST inputs whose literal blocks and section titles must equal docutils'.
#: Left out: Sphinx-only directives and docutils' error recovery (it dumps
#: an invalid directive as a literal block), topic titles (not sections),
#: and titles with inline markup (docutils renders it; a title here is its
#: source text).
DOCUTILS_CASES = {
    "deflist_literal": (
        "term\n    Description::\n\n        code\n\n    More definition.\n\nNext.\n"
    ),
    "numpydoc_param": (
        "x : int\n    The value, e.g.::\n\n        f(1)\n\n    Must be positive.\n"
    ),
    "directive_inline_content": (
        ".. note:: This note ends with::\n\n      code here\n\n   More note.\n"
    ),
    "field_body_multiline": (
        ":param x: first line\n    ends with::\n\n        code\n\n    Still field.\n"
    ),
    "enum_title": "A. Einstein\n===========\n\nbody\n",
    "enum_item_literal": "1. Step one::\n\n      x = 1\n\n   More.\n",
    "roman_item": "ii. Second::\n\n       y\n\n    tail\n",
    "hanging_item": "- first line\n  second line::\n\n    code\n\n  more item\n",
    "bullet_title": "* item\n------\n",
    "option_title": "-v  verbose\n===========\n",
    "doctest_title": ">>> x\n=====\n",
    "pipe_title": "| line\n======\n",
    "field_title": ":field: value\n=============\n",
    "role_title": ":class:`Foo`\n============\n",
    "dotdot_title": "..not explicit\n==============\n",
    "plus_title": "+ plus\n======\n",
    "expanded_dd": "Para ends here::\n\n  lit\n\nAfter.\n",
    "quoted_literal": "Para::\n\n> quoted\n> lines\n\nAfter.\n",
    "colon_field_like": ":not a field\n============\n",
    "indented_para_lit": "  Indented para::\n\n    lit\n\n  more quote\n",
    "dir_comment_lit": "..\n   comment::\n\n      x\n",
    "note_no_blank": (
        ".. note::\n   Run this::\n\n      pip install x\n\n   Then prose.\n"
    ),
    "tab_item": "-\tRun::\n\n    code\n",
    "note_nob_literal": (
        ".. note::\n   Run this::\n\n      pip install x\n\n   Then prose.\n"
    ),
    "comment_colons": ".. a comment here::\n\n   not literal\n\nAfter.\n",
    "bare_dir_body": "Intro.\n\n.. note::\n\n   This is a note.\n",
    "footnote_lit": ".. [1] See this::\n\n      code\n\nAfter.\n",
    "nested_item": "- outer\n\n  - inner::\n\n      code\n\n    more inner\n",
    "nested_item_shallow": "- outer\n\n  - inner::\n\n     code\n\n    more inner\n",
    "item_cont_lit": "- first\n  cont::\n\n   code?\n\n  more\n",
    "item_cont_lit2": "- first\n  cont::\n\n    code\n\n  more\n",
    "enum_cont": "1. first\n   cont::\n\n      code\n\n   more\n",
    "enum_single_shallow": "1. Step::\n\n  x\n\n   more\n",
    "enum_paren": "(a) Step::\n\n     x\n\n    more\n",
    "field_next_line": (
        ":param x:\n    ends with::\n\n        code\n\n    Still field.\n"
    ),
    "field_single": ":param x: see::\n\n    code\n\n  more\n",
    "field_single2": ":param x: see::\n\n           code\n\n          more\n",
    "field_title_esc": ":a\\:b: v\n========\n",
    "option_next_line": "-v\n    verbose::\n\n        code\n\n    more\n",
    "option_single": "-v  verbose::\n\n      code\n\n    more\n",
    "option_title2": "--all=<x>  thing\n================\n",
    "option_title3": "-a  b\n=====\n",
    "option_title_single_space": "-a b\n====\n",
    "slash_title": "/V  x\n=====\n",
    "slash_word_title": "/usr/bin\n========\n",
    "dir_opt2": ".. note::\n   :class: x\n\n   Body::\n\n      lit\n\n   tail\n",
    "citation": ".. [CIT2002] See::\n\n      code\n\n   more\n",
    "citation_hash": ".. [#] Auto::\n\n      code\n\n   more\n",
    "citation_hashname": ".. [#note] Auto::\n\n      code\n\n   more\n",
    "citation_star": ".. [*] Sym::\n\n      code\n\n   more\n",
    "target": ".. _x:\n\nPara::\n\n   lit\n",
    "target_lit": ".. _x: http://a.b/c::\n\n   y\n",
    "target_anon": ".. __: http://x::\n\n   y\n",
    "anon": "__ http://x\n\nPara::\n\n  lit\n",
    "anon_title": "__ x\n====\n",
    "comment_multi": (
        ".. this is\n   a comment::\n\n   still comment\n\nAfter::\n\n  lit\n"
    ),
    "comment_empty_then": "..\n\nPara::\n\n  lit\n",
    "comment_dir_like_nospace": "..note::\n\n   x\n",
    "tab_lit": "Para::\n\n\tcode\n\nAfter.\n",
    "tab_in_item": "-\ta::\n\n\t  code\n\n\tmore\n",
    "tab_field": ":f:\tsee::\n\n\t code\n",
    "tab_dir": ".. note::\tx::\n\n\tcode\n",
    "quoted_hash": "Para::\n\n# a\n# b\n\nAfter.\n",
    "quoted_mixed": "Para::\n\n> a\n< b\n\nAfter.\n",
    "quoted_indented": "  Para::\n\n  > a\n  > b\n\n  After.\n",
    "quoted_in_item": "- Para::\n\n  > a\n\n  more\n",
    "quoted_alpha": "Para::\n\nabc\n",
    "quoted_dash": "Para::\n\n- a\n- b\n",
    "quoted_then_indent": "Para::\n\n> a\n  b\n",
    "eof_colons": "Para::",
    "eof_colons_nl": "Para::\n",
    "eof_colons_blank": "Para::\n\n",
    "eof_colons_ws": "Para::\n\n   \n",
    "expanded_form": "Para: ::\n\n  lit\n",
    "bare_colons": "::\n\n  lit\n",
    "colons_space_end": "Para::   \n\n  lit\n",
    "multiline_para": "first line\n  second::\n\n    lit\n\n  bq?\n",
    "multiline_para_shallow": "first line\n  second::\n\n lit\n",
    "deflist_classifier": "term : cls\n   def::\n\n      x\n\n   y\n",
    "blockquote_lit": "   Quote para::\n\n      lit\n\n   more\n",
    "line_block": "| a::\n\n  x\n",
    "doctest_lit": ">>> a::\n\n  x\n",
    "enum_title2": "1. Intro\n========\n",
    "enum_title3": "#. Intro\n========\n",
    "simple_table_top": "=====  =====\nA      B\n=====  =====\n",
    "table_title": "== ==\n=====\n",
    "bullet_deep_body": "- a::\n\n      code\n\n    more\n",
    "bullet_deep_only": "- a::\n\n      code\n",
    "enum_deep_body": "1. a::\n\n       code\n\n     more\n",
    "field_noafter": ":param x: see::\n\n    code\n",
    "field_sphinx": ":param x: Example::\n\n        code\n\n    more text\n",
    "field_numpy_like": ":returns: a list, e.g.::\n\n      [1, 2]\n\n   More.\n",
    "option_noafter": "-v  verbose::\n\n      code\n",
    "opt_long_title": "--long\n========\n",
    "opt_slash_title": "/V\n====\n",
    "opt_short_title": "-v\n===\n",
    "opt_arg_title": "-a b\n=====\n",
    "opt_two_title": "-a, --all\n=========\n",
    "opt_desc_title": "-a, --all  everything\n=========================\n",
    "opt_long_title_body": "--long\n========\n\n    indented\n",
    "explicit_in_item": "- .. note:: x::\n\n        code\n\n    more\n",
    "dir_multi_args": ".. note:: first\n   second::\n\n      lit\n\n   tail\n",
    "dir_multi_args_shallow": ".. note:: first\n   second::\n\n    lit\n\n   tail\n",
    "footnote_multi": ".. [1] first\n   second::\n\n      lit\n\n   tail\n",
    "comment_then_dedent_para": ".. foo bar::\nPara::\n\n  lit\n",
    "explicit_next_unindented": ".. note:: x::\n\nPara\n",
    "bare_dir_nested": "- .. note::\n\n     body\n",
    "field_in_item_multi": "- :f: a\n    b::\n\n      c\n\n    d\n",
    "opt_trailing_sp_title": "-v  \n====\n",
    "grid_top_title": "+-----+-+\n=========\n",
    "grid_top_title2": "+--+--+\n=======\n",
    "simple_top_title": "=== ===\n=======\n",
    "enum_letter_said": "A. Einstein said::\n\n  lit\n\nMore.\n",
    "enum_letter_said2": "A. Einstein said::\n\n     lit\n\n   More.\n",
    "letter_sentence": "e. g. this::\n\n  lit\n",
    "roman_sentence": "I. Intro::\n\n  lit\n",
    "paren_num": "(1) x::\n\n  lit\n",
    "num_no_list": "2001. A year::\n\n  lit\n",
    "field_lookalike": ":not a field: x::\n\n  lit\n",
    "dd_term_colons": "Term::\n    Definition.\n",
    "item_quoted": "- a::\n\n  > q\n\n  more\n",
    "quoted_deeper_indent_next": "Para::\n\n> a\n>b\nAfter\n",
    "field_indented_quote": ":f: a::\n\n   > q\n",
    "F1_field_sphinx": ":param x: Example::\n\n        code\n\n    more text\n",
    "F1_field_returns": ":returns: a list, e.g.::\n\n      [1, 2]\n\n   More.\n",
    "F1_field_noafter": ":param x: see::\n\n    code\n",
    "F1_option": "-v  verbose::\n\n      code\n\n    more\n",
    "F2_long": "--long\n========\n",
    "F2_slash": "/V\n====\n",
    "F2_short": "-v\n===\n",
    "F2_arg": "-a b\n=====\n",
    "F2_pair": "-a, --all\n=========\n",
    "F2_trailing": "-v  \n====\n",
    "F3_bare_dir_in_item": "- .. note::\n\n     body\n",
    "F3_dir_in_item": "- .. note:: x::\n\n        code\n\n    more\n",
    "F4_line_block": "| a::\n\n  x\n",
    "F4_doctest": ">>> a::\n\n  x\n",
    "F6_option_colons": ".. figure:: a.png\n   :alt: Look::\n\n   Caption.\n",
    "F6_comment_no_blank": ".. foo bar::\nPara::\n\n  lit\n",
}


@pytest.mark.parametrize("name", sorted(DOCUTILS_CASES))
def test_rst_literals_and_titles_match_docutils(name):
    """A differential against docutils, the reference implementation."""
    core = pytest.importorskip("docutils.core")
    nodes = pytest.importorskip("docutils.nodes")
    source = DOCUTILS_CASES[name]

    def norm(text):
        return "\n".join(x.strip() for x in text.strip("\n").splitlines()).strip()

    tree = core.publish_doctree(
        source, settings_overrides={"report_level": 5, "halt_level": 5}
    )
    expected = (
        sorted(norm(n.astext()) for n in tree.findall(nodes.literal_block)),
        sorted(n.astext() for n in tree.findall(nodes.title)),
    )
    blocks = StructureChunker(StructureChunkerConfig(markup="rst")).blocks(source)
    got = (
        sorted(norm(source[a:b]) for k, (a, b), _, _ in blocks if k == "code"),
        sorted(t for k, _, _, t in blocks if k == "heading"),
    )
    assert got == expected
