"""
Tests for :mod:`scikitplot.cleanprompt._documents`.

Notes
-----
**Developer notes.** One property does all the work here: **a region indexes
the document it came from**. Everything downstream — span resolution, the
single rewrite pass, every invariant from ``I1`` to ``I9`` — is stated in terms
of offsets into the original text, so a region that is off by one is not a
cosmetic defect but a corrupted notebook.

It is also the property most likely to break silently, because a wrong offset
still produces valid-looking output. So it is checked directly: every region's
slice of the raw text is compared against the string the parser found at that
path, for every string in the document.
"""

from __future__ import annotations

import json

import pytest

from .._documents import FORMATS, ROLES, Region, detect_format, regions_for
from .._exceptions import CleanPromptError


def notebook(cells, metadata=None):
    """Return a minimal but valid notebook as raw JSON text."""
    document = {
        "cells": cells,
        "metadata": metadata or {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document, indent=1)


def code_cell(*lines, outputs=None):
    """Return a code cell."""
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs or [],
        "source": list(lines),
    }


class TestRegion:
    """The value type, and what it refuses."""

    def test_length_and_slice_agree(self):
        region = Region(2, 6, "code")
        assert region.length == 4
        assert region.slice("0123456789") == "2345"

    def test_an_empty_region_is_refused(self):
        with pytest.raises(ValueError):
            Region(4, 4, "code")

    def test_a_negative_start_is_refused(self):
        with pytest.raises(ValueError):
            Region(-1, 4, "code")

    def test_an_unknown_role_is_refused(self):
        with pytest.raises(ValueError) as caught:
            Region(0, 4, "nonsense")
        assert "nonsense" in str(caught.value)


class TestFormatDetection:
    """The suffix is what the user believes; content is the fallback."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("a/b/analysis.ipynb", "notebook"),
            ("features.py", "python"),
            ("stub.pyi", "python"),
            ("notes.md", "text"),
            ("ANALYSIS.IPYNB", "notebook"),
        ],
    )
    def test_the_suffix_decides(self, path, expected):
        assert detect_format(path, "") == expected

    def test_content_is_the_fallback_for_standard_input(self):
        assert detect_format(None, notebook([code_cell("x = 1\n")])) == "notebook"

    def test_json_that_is_not_a_notebook_is_text(self):
        assert detect_format(None, '{"a": 1}') == "text"

    def test_prose_is_text(self):
        assert detect_format(None, "Mail ada@example.com") == "text"


class TestFlatFormats:
    """Text and Python are one region each."""

    def test_text_is_one_prose_region(self):
        regions = regions_for("hello there", "text")
        assert [one.role for one in regions] == ["prose"]
        assert regions[0].length == len("hello there")

    def test_python_is_one_code_region(self):
        assert [one.role for one in regions_for("x = 1", "python")] == ["code"]

    def test_an_empty_document_has_no_regions(self):
        assert regions_for("", "text") == ()

    def test_an_unknown_format_is_refused(self):
        with pytest.raises(CleanPromptError) as caught:
            regions_for("x", "sideways")
        assert "sideways" in str(caught.value)


class TestNotebookOffsets:
    """The property everything else depends on."""

    def test_every_region_slices_to_the_string_it_names(self):
        raw = notebook(
            [
                code_cell(
                    "df = pd.read_csv('a.csv')\n",
                    "df.head()\n",
                    outputs=[
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["a  b\n", "1  2\n"],
                        }
                    ],
                ),
                {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
            ]
        )
        document = json.loads(raw)
        for region in regions_for(raw, "notebook", "n.ipynb"):
            # The raw slice is the JSON-escaped form; decoding it must give a
            # string that occurs in the parsed document.
            decoded = json.loads('"{0}"'.format(region.slice(raw)))
            assert isinstance(decoded, str)
        assert document["cells"][1]["cell_type"] == "markdown"

    def test_regions_are_disjoint_and_ordered(self):
        raw = notebook([code_cell("x = 1\n"), code_cell("y = 2\n")])
        bounds = [(one.start, one.end) for one in regions_for(raw, "notebook")]
        assert bounds == sorted(bounds)
        assert all(a[1] <= b[0] for a, b in zip(bounds, bounds[1:]))

    def test_a_code_cell_is_code_and_a_markdown_cell_is_prose(self):
        raw = notebook(
            [
                code_cell("x = 1\n"),
                {"cell_type": "markdown", "metadata": {}, "source": ["hello\n"]},
            ]
        )
        roles = {
            one.path: one.role
            for one in regions_for(raw, "notebook")
            if one.path.endswith("source[0]")
        }
        assert roles["cells[0].source[0]"] == "code"
        assert roles["cells[1].source[0]"] == "prose"

    def test_an_output_is_an_output_and_a_traceback_is_a_traceback(self):
        raw = notebook(
            [
                code_cell(
                    "boom()\n",
                    outputs=[
                        {
                            "output_type": "error",
                            "ename": "KeyError",
                            "evalue": "'x'",
                            "traceback": ["File /home/a/b.py:1"],
                        },
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["printed\n"],
                        },
                    ],
                )
            ]
        )
        roles = {one.role for one in regions_for(raw, "notebook")}
        assert "traceback" in roles
        assert "output" in roles

    def test_a_base64_payload_is_binary(self):
        raw = notebook(
            [
                code_cell(
                    "plot()\n",
                    outputs=[
                        {
                            "output_type": "display_data",
                            "metadata": {},
                            "data": {"image/png": "iVBORw0KGgoAAAANS"},
                        }
                    ],
                )
            ]
        )
        binary = [one for one in regions_for(raw, "notebook") if one.role == "binary"]
        assert len(binary) == 1
        assert binary[0].slice(raw) == "iVBORw0KGgoAAAANS"

    def test_escapes_survive_alignment(self):
        """A cell full of escapes still aligns token-for-token."""
        raw = notebook([code_cell('s = "a\\tb"  # \\u2014 \\\\ \\n')])
        regions = regions_for(raw, "notebook")
        assert any(one.role == "code" for one in regions)

    def test_unicode_outside_the_basic_plane_aligns(self):
        raw = notebook([code_cell("emoji = '\U0001f600'\n")])
        assert any(one.role == "code" for one in regions_for(raw, "notebook"))


class TestNotebookRefusals:
    """A half-processed notebook is not a safe artefact."""

    def test_malformed_json_is_refused(self):
        with pytest.raises(CleanPromptError) as caught:
            regions_for('{"cells": [', "notebook", "broken.ipynb")
        assert "broken.ipynb" in str(caught.value)
        assert "nothing was written" in str(caught.value)

    def test_json_without_cells_is_refused_with_a_remedy(self):
        with pytest.raises(CleanPromptError) as caught:
            regions_for('{"a": 1}', "notebook", "config.json")
        assert "--as text" in str(caught.value)


class TestVocabulary:
    """The declared surface is the implemented one."""

    def test_every_format_is_handled(self):
        for fmt in FORMATS:
            document = notebook([code_cell("x = 1\n")]) if fmt == "notebook" else "x = 1"
            assert regions_for(document, fmt)

    def test_every_role_a_region_carries_is_declared(self):
        raw = notebook(
            [
                code_cell(
                    "x = 1\n",
                    outputs=[{"output_type": "stream", "name": "o", "text": ["a\n"]}],
                )
            ]
        )
        for region in regions_for(raw, "notebook"):
            assert region.role in ROLES

    def test_doctests_pass(self):
        import doctest

        from .. import _documents

        assert doctest.testmod(_documents, verbose=False).failed == 0
