# lean/tests/test__syntax.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The syntax core: names, scopes and flags as Lean reads the text
===============================================================

A «quoted name» is a name, not code, so a token inside one raises no flag;
``section A.B`` opens a scope per part, as ``namespace A.B`` does; an
``end`` Lean would refuse, or a file ending inside a comment or string, is
reported where the summary and the dossier show it, the same for any
number of workers. Projects are written in each test.
"""

from __future__ import annotations

import re
import textwrap

import pytest

from .._index import _NOT_UTF8, LeanIndex
from .._syntax import _code_matches, _token_pattern


def _index(tmp_path, source, **options):
    path = tmp_path / "Proj" / "A.lean"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return LeanIndex.from_folder(tmp_path, **options)


def _declarations(index):
    return {
        index._name[n]: index._flags.get(n, ()) for n in range(len(index._name))
    }


# ---------------------------------------------------------------------------
# «quoted names»
# ---------------------------------------------------------------------------


def test_a_flag_token_inside_a_quoted_name_is_not_code(tmp_path):
    index = _index(
        tmp_path,
        """
        theorem «a sorry b» : True := trivial
        theorem «sorry» : True := trivial
        theorem c : True := sorry
        """,
    )
    assert _declarations(index) == {"«a sorry b»": (), "«sorry»": (), "c": ("sorry",)}


@pytest.mark.parametrize(
    ("masked", "found"),
    [
        ("sorry x", ["sorry"]),
        ("«a sorry» sorry", ["sorry"]),  # only the one outside
        ("«a b» «c»", ["«a b»"]),  # a whole quoted name is kept
        ("«sorry»", []),
        ("«a» sorry «b»", ["sorry"]),
    ],
)
def test_code_matches_skip_text_inside_quoted_names(masked, found):
    pattern = _token_pattern(["sorry", "«a b»"])
    assert [m.group() for m in _code_matches(pattern, masked)] == found


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def test_a_dotted_section_keeps_the_namespace_around_it(tmp_path):
    index = _index(
        tmp_path,
        """
        namespace N
        section A.B
        theorem t : True := trivial
        end A.B
        theorem u : True := trivial
        end N
        theorem v : True := trivial
        """,
    )
    assert list(_declarations(index)) == ["N.t", "N.u", "v"]
    assert index.summary()["source_problems"]["count"] == 0


@pytest.mark.parametrize(
    ("source", "problems"),
    [
        ("namespace A\nend B\n", [(2, "`end B` does not match `namespace A`")]),
        ("namespace A\nend\n", [(2, "`end` does not match `namespace A`")]),
        ("section S\nend\n", [(2, "`end` does not match `section S`")]),
        ("end\n", [(1, "`end` closes no open scope")]),
        ("theorem t : True := trivial\n/- open\n", [(2, "the file ends inside a comment")]),
        ('def s := "open\n', [(1, "the file ends inside a string")]),
        ("namespace A.B\nend B\nend A\n", []),
        ("section\nend\nmutual\ndef f : Nat := 0\nend\n", []),
        ("namespace A\n", []),  # open at the end of the file: Lean closes it
    ],
    ids=[
        "other-name",
        "missing-name",
        "section-name",
        "nothing-open",
        "comment",
        "string",
        "dotted-in-parts",
        "anonymous",
        "left-open",
    ],
)
def test_source_problems_are_reported_by_line(tmp_path, source, problems):
    index = _index(tmp_path, source)
    items = index.summary()["source_problems"]["items"]
    assert [(i["line"], i["problem"]) for i in items] == problems
    assert {i["file"] for i in items} <= {"Proj/A.lean"}


def test_bytes_that_are_not_utf8_are_a_problem_and_the_rest_is_read(tmp_path):
    path = tmp_path / "Proj" / "A.lean"
    path.parent.mkdir(parents=True)
    # \r\n, a lone \r and \n each end one line, as the reader counts them.
    path.write_bytes(
        b"theorem a : True := trivial\r\ntheorem r : True := trivial\rdef s := \"caf\xe9\"\ntheorem b : True := sorry\n"
    )
    for workers in (1, 2):
        index = LeanIndex.from_folder(tmp_path, workers=workers)
        assert _declarations(index) == {"a": (), "r": (), "s": (), "b": ("sorry",)}
        assert list(index._start) == [1, 2, 3, 4]
        [item] = index.summary()["source_problems"]["items"]
        assert (item["line"], item["problem"]) == (3, _NOT_UTF8)


def test_valid_utf8_with_a_replacement_character_is_no_problem(tmp_path):
    index = _index(tmp_path, 'def s := "\ufffd"\n')
    assert index.summary()["source_problems"]["count"] == 0


def test_problems_are_the_same_for_any_number_of_workers(tmp_path):
    for name in ("B", "C"):
        path = tmp_path / "Proj" / f"{name}.lean"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"namespace {name}\nend X\n", encoding="utf-8")
    serial = LeanIndex.from_folder(tmp_path).summary()["source_problems"]
    parallel = LeanIndex.from_folder(tmp_path, workers=2).summary()["source_problems"]
    assert serial == parallel and serial["count"] == 2


def test_a_dossier_reviews_the_text_its_proof_depends_on(tmp_path):
    index = _index(
        tmp_path,
        """
        namespace A
        theorem t : True := trivial
        end B
        theorem u : True := t
        """,
    )
    report = index.dossier("u")
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["source_text"] == "review"
    assert report["source_problems"]["items"] == [
        {"file": "Proj/A.lean", "line": 3, "problem": "`end B` does not match `namespace A`"}
    ]
    detail = {c["check"]: c["detail"] for c in report["checks"]}["source_text"]
    assert re.search(r"1 place\(s\) Lean refuses to read", detail)
