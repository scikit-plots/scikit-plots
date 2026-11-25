# corpus/_readers/tests/test__lean.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
LeanReader: one chunk per Lean command, every line accounted for
================================================================

The lexer (comments, strings, characters, «names»), what starts a command,
qualified names and scopes, the audit flags, every option, Lean 3, and the
guarantee that every non-blank line lands in exactly one chunk whose text
is the file's own lines. Files are written in each test; no test reaches
the network.
"""

from __future__ import annotations

import random
import textwrap
import time
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers every reader)
from ..._base import DocumentReader
from .. import _lean
from .._code import CodeReader
from .._lean import LEAN_FLAGS, LeanReader, _Lexer


def _write(tmp_path, source, name="A.lean"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return path


def _read(tmp_path, source, name="A.lean", **options):
    return list(
        LeanReader(
            input_path=_write(tmp_path, source, name), **options
        ).get_raw_chunks()
    )


def _summary(chunks):
    return [
        (c["lean_kind"], c.get("declaration"), c["line_start"], c["line_end"])
        for c in chunks
    ]


def _assert_partition(path, chunks):
    """Every non-blank line in exactly one chunk, whose text is the file's lines."""
    lines = path.read_text(encoding="utf-8").split("\n")
    covered = []
    previous = 0
    for chunk in chunks:
        start, end = chunk["line_start"], chunk["line_end"]
        assert start > previous, f"overlap at line {start}"
        previous = end
        assert chunk["text"] == "\n".join(lines[start - 1 : end])
        covered.extend(range(start, end + 1))
    content = [n for n, line in enumerate(lines, 1) if line.strip()]
    assert set(content) <= set(covered)


MATHLIB_STYLE = """
/-
Copyright (c) 2026 Someone. All rights reserved.
-/
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

/-!
# Basic facts

Notes with -- no comment and "no string.
-/

namespace Foo.Bar

open Nat

variable {α : Type*} [Inhabited α]

/-- The double of `n`. Mentions sorry. -/
@[simp]
def double (n : ℕ) : ℕ := 2 * n

-- A helper lemma.
theorem double_eq (n : ℕ) : double n = n + n := by
  unfold double
  -- sorry in a comment
  omega

@[simp, norm_cast] private lemma h' (x : ℕ) : x = x := rfl

set_option maxHeartbeats 0 in
theorem slow : (let x := 1; x) = 1 := by
  sorry

instance : Inhabited Wrapper := ⟨⟨0⟩⟩

theorem _root_.Nat.my_lemma : ∀ n : ℕ, n = n
  | 0 => rfl
  | n + 1 => rfl

def msg : String := s!"v {double 2} \\"q\\" -- no /- no -/"

end Foo.Bar

example : 1 + 1 = 2 := by native_decide

axiom bigClaim : ∀ n : ℕ, n = n + 0

#exit
theorem fake : False := trivial
"""


# ---------------------------------------------------------------------------
# Registration and the whole-file shape
# ---------------------------------------------------------------------------


def test_lean_files_dispatch_to_lean_reader(tmp_path):
    reader = DocumentReader.create(_write(tmp_path, "def x := 1\n"))
    assert type(reader) is LeanReader
    assert isinstance(reader, CodeReader)


def test_a_mathlib_style_file_is_one_chunk_per_command(tmp_path):
    path = _write(tmp_path, MATHLIB_STYLE)
    chunks = list(LeanReader(input_path=path).get_raw_chunks())
    assert _summary(chunks) == [
        ("import", None, 1, 5),
        ("moduledoc", None, 7, 11),
        ("context", None, 13, 17),
        ("def", "Foo.Bar.double", 19, 21),
        ("theorem", "Foo.Bar.double_eq", 23, 27),
        ("lemma", "Foo.Bar.h'", 29, 29),
        ("theorem", "Foo.Bar.slow", 31, 33),
        ("instance", None, 35, 35),
        ("theorem", "Nat.my_lemma", 37, 39),
        ("def", "Foo.Bar.msg", 41, 41),
        ("context", None, 43, 43),
        ("example", None, 45, 45),
        ("axiom", "bigClaim", 47, 47),
        ("#exit", None, 49, 49),
        ("theorem", "fake", 50, 50),
    ]
    _assert_partition(path, chunks)
    by_line = {c["line_start"]: c for c in chunks}
    assert by_line[1]["imports"] == ["Mathlib.Data.Nat.Basic", "Mathlib.Tactic"]
    assert by_line[7]["section_type"] == "text"
    assert (
        by_line[19]["attributes"] == ["simp"] and by_line[19]["namespace"] == "Foo.Bar"
    )
    assert by_line[29]["modifiers"] == ["private"]
    assert by_line[29]["attributes"] == ["simp", "norm_cast"]
    assert by_line[31]["flags"] == ["sorry"]  # the proof's, not the comment's
    assert "flags" not in by_line[19] and "flags" not in by_line[23]
    assert by_line[45]["flags"] == ["native"]
    assert by_line[47]["flags"] == ["axiom"]
    assert by_line[50]["flags"] == ["after_exit"]
    assert {c["language"] for c in chunks} == {"lean"}
    assert all(c["source_type"] == "code" for c in chunks)


def test_documents_carry_the_metadata(tmp_path):
    path = _write(tmp_path, "namespace N\ntheorem t : True := trivial\nend N\n")
    docs = list(DocumentReader.create(path).get_documents())
    theorem = [d for d in docs if d.metadata.get("lean_kind") == "theorem"]
    assert len(theorem) == 1
    assert theorem[0].metadata["declaration"] == "N.t"
    assert (theorem[0].metadata["line_start"], theorem[0].metadata["line_end"]) == (
        2,
        2,
    )


# ---------------------------------------------------------------------------
# The lexer
# ---------------------------------------------------------------------------


def _mask(*lines):
    lexer = _Lexer()
    masked = [lexer.mask(line) for line in lines]
    for line, out in zip(lines, masked):
        assert len(out) == len(line)
    return masked, lexer.in_code()


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ('def x := "a -- b" -- c', "def x :="),
        ("/- a /- nested -/ still -/ def x", "def x"),
        ("/-- doc -/ theorem t", "theorem t"),
        ('def r := r#"he said "hi" -- no"# ++ y', "def r := ++ y"),
        ('def r := r"\\n" ++ y', "def r := ++ y"),
        ("def q := '\"' ++ x", "def q := ++ x"),
        ("def q := '-' ++ '\\'' ++ x", "def q := ++ ++ x"),
        ("h' h'' x' := y", "h' h'' x' := y"),
        ("def «a -- b» := 1", "def «a -- b» := 1"),
        ('def m := s!"n = {f "x"} and {g 1}" ++ z', "def m := s! f g 1 ++ z"),
        ('def e := "a \\" -- still string" ++ z', "def e := ++ z"),
        ('def b := "\\{ not code }" ++ z', "def b := ++ z"),
    ],
)
def test_the_lexer_keeps_only_code(line, code):
    (masked,), in_code = _mask(line)
    assert " ".join(masked.split()) == code
    assert in_code


def test_comments_strings_and_names_run_over_lines():
    masked, in_code = _mask(
        "/- open /- nested",
        "  -/ still open",
        "-/ def a := 1",
        'def s := "line one',
        'line two -- not a comment" ++ b',
        "def «multi",
        "line» := c",
        'def g := "gap \\',
        '   continued" ++ d',
    )
    assert [" ".join(m.split()) for m in masked] == [
        "",
        "",
        "def a := 1",
        "def s :=",
        "++ b",
        "def «multi",
        "line» := c",
        "def g :=",
        "++ d",
    ]
    assert in_code


def test_the_lexer_reports_what_is_left_open(tmp_path):
    lexer = _Lexer()
    lexer.mask("/- never closed")
    assert lexer.open_construct() == "comment"
    with mock.patch.object(_lean, "logger") as logger:
        chunks = _read(tmp_path, "theorem t : True := trivial\n/- open\nmore\n")
    message, _name, construct = logger.warning.call_args.args
    assert "ends inside a" in message and construct == "comment"
    assert chunks[-1]["line_end"] == 3


# ---------------------------------------------------------------------------
# Audit flags and watched names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "flag"),
    [
        ("theorem t : P := by sorry", "sorry"),
        ("theorem t : P := sorryAx _ false", "sorry"),
        ("theorem t : P := by admit", "admit"),
        ("theorem t : P := by native_decide", "native"),
        ("theorem t : P := by decide +native", "native"),
        ("theorem t : P := Lean.ofReduceBool a b c", "native"),
        ("unsafe def f : Nat := 1", "unsafe"),
        ("@[implemented_by g] def f : Nat := 1", "implemented_by"),
        ('@[extern "c_f"] opaque f : Nat', "extern"),
        ("set_option debug.skipKernelTC true in theorem t : P := h", "skip_kernel_tc"),
        ("def f : CoreM Unit := addDecl d", "add_decl"),
        ("axiom a : P", "axiom"),
    ],
)
def test_each_flag_is_raised_by_code(tmp_path, code, flag):
    (chunk,) = _read(tmp_path, code + "\n")
    assert flag in chunk["flags"]


def test_flags_ignore_comments_strings_and_longer_names(tmp_path):
    source = """
    /-- uses sorry, admit and native_decide -/
    theorem t : True := by
      -- sorry
      /- unsafe -/
      have h := "sorry admit"
      have k := sorry_free
      have u := unsafeCast
      trivial
    """
    (chunk,) = _read(tmp_path, source)
    assert "flags" not in chunk
    assert set(LEAN_FLAGS.values()) == {
        "sorry",
        "admit",
        "native",
        "unsafe",
        "implemented_by",
        "extern",
        "skip_kernel_tc",
        "add_decl",
    }


def test_watched_names_are_located(tmp_path):
    source = """
    theorem a : P := keyLemma
    theorem b : P := Main.keyLemma.symm
    theorem c : P := keyLemma' -- keyLemma in a comment
    theorem d : P := by exact "keyLemma"
    """
    chunks = _read(tmp_path, source, watch=["keyLemma", "Main.keyLemma"])
    assert [c.get("watched") for c in chunks] == [
        ["keyLemma"],
        ["Main.keyLemma"],
        None,
        None,
    ]


# ---------------------------------------------------------------------------
# Names and scopes
# ---------------------------------------------------------------------------


def test_names_follow_namespaces_sections_and_root(tmp_path):
    source = """
    namespace A.B
    section S
    def x := 1
    end S
    namespace C
    protected theorem y : True := trivial
    end C
    def _root_.z := 2
    end A.B
    def w := 3
    instance : Inhabited Nat := ⟨0⟩
    instance (priority := low) named : Inhabited Nat := ⟨0⟩
    macro (name := myMacro) "m" : term => `(1)
    example : True := trivial
    """
    names = [
        (c["lean_kind"], c.get("declaration"), c.get("namespace"))
        for c in _read(tmp_path, source)
        if c["lean_kind"] != "context"
    ]
    assert names == [
        ("def", "A.B.x", "A.B"),
        ("theorem", "A.B.C.y", "A.B.C"),
        ("def", "z", "A.B"),
        ("def", "w", None),
        ("instance", None, None),
        ("instance", "named", None),
        ("macro", "myMacro", None),
        ("example", None, None),
    ]


def test_an_unmatched_end_is_ignored(tmp_path):
    chunks = _read(tmp_path, "end Nothing\nend\ndef x := 1\n")
    assert chunks[-1]["declaration"] == "x"


def test_scope_depth_is_bounded(tmp_path):
    source = "".join(f"namespace N{i}\n" for i in range(5))
    with pytest.raises(ValueError, match="max_scope_depth=4"):
        _read(tmp_path, source, max_scope_depth=4)


def test_indented_commands_and_quoted_declarations(tmp_path):
    source = """
    namespace N
      /-- Indented style. -/
      theorem a : True := trivial

      def gen : MacroM Syntax := do
        `(command|
          def $name := 1
          theorem $other : True := trivial)

      def b := 2
    end N
    """
    chunks = _read(tmp_path, source)
    assert [(c["lean_kind"], c.get("declaration")) for c in chunks] == [
        ("context", None),
        ("theorem", "N.a"),
        ("def", "N.gen"),  # the quoted def and theorem are its code
        ("def", "N.b"),
        ("context", None),
    ]


def test_mutual_blocks_with_indented_declarations(tmp_path):
    source = """
    mutual
      def even : Nat → Bool
        | 0 => true
        | n + 1 => odd n
      def odd : Nat → Bool
        | 0 => false
        | n + 1 => even n
    end
    def after := 1
    """
    chunks = _read(tmp_path, source)
    assert [(c["lean_kind"], c.get("declaration")) for c in chunks] == [
        ("context", None),
        ("def", "even"),
        ("def", "odd"),
        ("context", None),
        ("def", "after"),
    ]


# ---------------------------------------------------------------------------
# Which lines belong to which command
# ---------------------------------------------------------------------------


def test_comments_docs_and_attributes_attach_by_rule(tmp_path):
    source = """
    def a := 1
      -- a's trailing remark, deeper than the next command
    -- apart from b by a blank line

    -- directly above b
    /-- Doc of b,

    with a blank line inside. -/
    @[simp,
      norm_cast]
    noncomputable
    def b := 2
    """
    path = _write(tmp_path, source)
    chunks = list(LeanReader(input_path=path).get_raw_chunks())
    assert _summary(chunks) == [("def", "a", 1, 3), ("def", "b", 5, 12)]
    assert chunks[1]["attributes"] == ["simp"]
    assert chunks[1]["modifiers"] == ["noncomputable"]
    _assert_partition(path, chunks)


def test_in_prefixes_join_the_command_they_prefix(tmp_path):
    source = """
    namespace N
    open Classical in
    set_option maxHeartbeats 0 in
    theorem t : True := trivial
    /-- info: 2 -/
    #guard_msgs in
    #eval 1 + 1
    """
    chunks = _read(tmp_path, source, watch=["maxHeartbeats"])
    assert _summary(chunks) == [
        ("context", None, 1, 1),
        ("theorem", "N.t", 2, 4),
        ("#eval", None, 5, 7),
    ]
    assert chunks[1]["watched"] == ["maxHeartbeats"]


def test_a_project_command_starts_a_chunk_only_when_named(tmp_path):
    source = "theorem a : True := trivial\ncopy_decls a b\ndef c := 1\n"
    assert [c["lean_kind"] for c in _read(tmp_path, source)] == ["theorem", "def"]
    assert [
        c["lean_kind"] for c in _read(tmp_path, source, commands=["copy_decls"])
    ] == [
        "theorem",
        "copy_decls",
        "def",
    ]


def test_library_build_and_claim_commands_stand_alone(tmp_path):
    source = """
    proof_wanted open_problem : ∀ n : ℕ, n = n
    grind_pattern Nat.find_spec => Nat.find H
    recommended_spelling "iff" for "↔" in [Iff]
    package mylib where
      leanOptions := #[]
    @[default_target]
    lean_lib MyLib where
    """
    chunks = _read(tmp_path, source, name="lakefile.lean")
    assert [(c["lean_kind"], c.get("declaration")) for c in chunks] == [
        ("proof_wanted", "open_problem"),
        ("grind_pattern", None),
        ("recommended_spelling", None),
        ("package", None),
        ("lean_lib", None),
    ]
    assert chunks[-1]["attributes"] == ["default_target"]


def test_byte_order_mark_and_crlf(tmp_path):
    path = tmp_path / "W.lean"
    path.write_bytes((chr(0xFEFF) + "def a := 1\r\n\r\ndef b := 2\r\n").encode("utf-8"))
    chunks = list(LeanReader(input_path=path).get_raw_chunks())
    assert [(c["text"], c["line_start"]) for c in chunks] == [
        ("def a := 1", 1),
        ("def b := 2", 3),
    ]


# ---------------------------------------------------------------------------
# Statements without proofs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "statement", "proof_lines"),
    [
        ("theorem t : P := by simp\n", "theorem t : P", (1, 1)),
        (
            "theorem t (x : Nat := 0) : P x\n    := by simp\n",
            "theorem t (x : Nat := 0) : P x",
            (2, 2),
        ),
        ("theorem t : f (h := a) = b := rfl\n", "theorem t : f (h := a) = b", (1, 1)),
        (
            "theorem t : let x := 1; x = 1 := rfl\n",
            "theorem t : let x := 1; x = 1",
            (1, 1),
        ),
        ("theorem t : P where\n  x := 1\n", "theorem t : P", (1, 2)),
        (
            "theorem t : ∀ n, P n\n  | 0 => a\n  | n + 1 => b\n",
            "theorem t : ∀ n, P n",
            (2, 3),
        ),
        (
            "theorem t (x : ℝ) :\n    |x| ≤ 1 := by\n  simp\n",
            "theorem t (x : ℝ) :\n    |x| ≤ 1",
            (2, 3),
        ),
        (
            "/-- Doc. -/\n@[simp] lemma t : P := by\n  simp\n",
            "/-- Doc. -/\n@[simp] lemma t : P",
            (2, 3),
        ),
    ],
)
def test_proofs_drop_keeps_the_statement(tmp_path, source, statement, proof_lines):
    (chunk,) = _read(tmp_path, source, proofs="drop")
    assert chunk["text"] == statement
    assert (chunk["proof_line_start"], chunk["proof_line_end"]) == proof_lines


def test_proofs_drop_keeps_definitions_and_flags_the_proof(tmp_path):
    source = """
    def f : Nat := 1 + 1
    theorem t : True := by
      sorry
    theorem u : True
    """
    chunks = _read(tmp_path, source, proofs="drop")
    assert [c["text"] for c in chunks] == [
        "def f : Nat := 1 + 1",
        "theorem t : True",
        "theorem u : True",  # no proof found: kept whole
    ]
    assert chunks[1]["flags"] == ["sorry"]
    assert "proof_line_start" not in chunks[0] and "proof_line_start" not in chunks[2]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_a_long_command_comes_in_parts(tmp_path):
    body = "".join(f"  have h{i} : True := trivial\n" for i in range(449))
    path = _write(tmp_path, "theorem big : True := by\n" + body + "  sorry\n")
    chunks = list(LeanReader(input_path=path, lines_per_chunk=200).get_raw_chunks())
    assert [(c["part"], c["line_start"], c["line_end"]) for c in chunks] == [
        (1, 1, 200),
        (2, 201, 400),
        (3, 401, 451),
    ]
    assert {c["declaration"] for c in chunks} == {"big"}
    assert [c.get("flags") for c in chunks] == [None, None, ["sorry"]]
    _assert_partition(path, chunks)
    (whole,) = LeanReader(input_path=path, lines_per_chunk=None).get_raw_chunks()
    assert "part" not in whole


def test_kinds_selects_chunks(tmp_path):
    chunks = _read(tmp_path, MATHLIB_STYLE, kinds={"theorem", "lemma"})
    assert {c["lean_kind"] for c in chunks} == {"theorem", "lemma"}
    assert len(chunks) == 5


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"proofs": "strip"}, ValueError, "proofs must be"),
        ({"dialect": "lean5"}, ValueError, "dialect must be"),
        ({"max_scope_depth": 0}, ValueError, "max_scope_depth must be"),
        ({"max_scope_depth": True}, ValueError, "max_scope_depth must be"),
        ({"kinds": "theorem"}, TypeError, "kinds must be a collection"),
        ({"kinds": [1]}, TypeError, "must hold str"),
        ({"kinds": [""]}, ValueError, "non-empty str"),
        ({"watch": ["two words"]}, ValueError, "single tokens"),
        ({"watch": "rh"}, TypeError, "watch must be a collection"),
        ({"commands": ["9lives"]}, ValueError, "commands holds identifiers"),
    ],
)
def test_invalid_options_are_refused(tmp_path, options, error, message):
    with pytest.raises(error, match=message):
        LeanReader(input_path=_write(tmp_path, "def x := 1\n"), **options)


# ---------------------------------------------------------------------------
# Lean 3
# ---------------------------------------------------------------------------


def test_lean3_blocks_equations_and_universes(tmp_path):
    source = """
    import data.nat.basic
    noncomputable theory
    universes u v
    namespace foo

    /-- A doc. -/
    structure {w} part (α : Type w) :=
    (val : α)

    lemma bar (n : ℕ) : n = n :=
    begin
      induction n,
      { refl },
      { refl }
    end

    theorem baz : ∀ n : ℕ, n = n
    | 0 := rfl
    | (n+1) := rfl

    end foo
    def after := 1
    """
    path = _write(tmp_path, source)
    chunks = list(LeanReader(input_path=path, dialect="lean3").get_raw_chunks())
    assert [(c["lean_kind"], c.get("declaration")) for c in chunks] == [
        ("import", None),
        ("context", None),
        ("structure", "foo.part"),
        ("lemma", "foo.bar"),
        ("theorem", "foo.baz"),
        ("context", None),
        ("def", "after"),
    ]
    _assert_partition(path, chunks)
    dropped = LeanReader(
        input_path=path, dialect="lean3", proofs="drop", kinds={"theorem"}
    )
    (theorem,) = dropped.get_raw_chunks()
    assert theorem["text"] == "theorem baz : ∀ n : ℕ, n = n"


# ---------------------------------------------------------------------------
# The line guarantee on assembled files, linear time, streaming memory
# ---------------------------------------------------------------------------

_SNIPPETS = [
    "theorem t{i} (x : Nat) : x = x := by\n  simp\n  -- note\n",
    "/-- Doc {i} with -- and /- nested -/ -/\n@[simp] lemma l{i} : True := trivial\n",
    'def s{i} : String := "a -- b /- c" ++ s!"{{x}} -- {{y}}"\n',
    'def r{i} := r#"raw "quoted" -/ "#\n',
    "namespace N{i}\n",
    "end N{i}\n",
    "section\nvariable (x : Nat)\nend\n",
    "\n\n",
    "-- comment {i}\n",
    "/- block\n\n   comment {i} -/\n",
    "open Nat in\ntheorem o{i} : True := trivial\n",
    "  theorem indented{i} : True := trivial\n",
    "instance : Inhabited (Fin {i}.succ) := ⟨0⟩\n",
    "def «odd -- name {i}» := 'a'\n",
    "theorem e{i} : ∀ n, n = n\n  | 0 => rfl\n  | n + 1 => rfl\n",
    "mutual\n  def p{i} : Nat := q{i}\n  def q{i} : Nat := 0\nend\n",
    "set_option maxHeartbeats 0 in\n",
    "@[simp]\n",
    "example : h' = h'' := rfl\n",
]


def test_every_line_is_in_exactly_one_chunk_of_assembled_files(tmp_path):
    rng = random.Random(1729)
    for case in range(300):
        source = "".join(
            rng.choice(_SNIPPETS).format(i=i) for i in range(rng.randint(1, 30))
        )
        path = tmp_path / f"F{case}.lean"
        path.write_text(source, encoding="utf-8")
        chunks = list(LeanReader(input_path=path, max_scope_depth=64).get_raw_chunks())
        _assert_partition(path, chunks)


def _seconds(path, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        for _chunk in LeanReader(
            input_path=path, max_scope_depth=10**6
        ).get_raw_chunks():
            pass
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize(
    "unit",
    [
        "/-",
        '"',
        "«",
        "'",
        "@[",
        'r#"',
        "(",
        "/--",
        's!"{',
        "namespace N\n",
        "  theorem t\n",
    ],
)
def test_reading_time_grows_linearly(tmp_path, unit):
    # Four times the input must take about four times as long; a quadratic
    # scan would take sixteen times as long.
    small = tmp_path / "small.lean"
    large = tmp_path / "large.lean"
    small.write_text(unit * (8000 // len(unit)), encoding="utf-8")
    large.write_text(unit * (32000 // len(unit)), encoding="utf-8")
    ratio = _seconds(large) / max(_seconds(small), 1e-4)
    assert ratio < 9, f"{unit!r}: x{ratio:.1f} for 4x the input"


def test_a_long_declaration_streams_in_bounded_memory(tmp_path):
    import tracemalloc

    path = tmp_path / "Big.lean"
    path.write_text(
        "theorem big : True := by\n" + "  skip\n" * 200_000 + "  trivial\n",
        encoding="utf-8",
    )
    size = path.stat().st_size
    tracemalloc.start()
    try:
        count = sum(1 for _ in LeanReader(input_path=path).get_raw_chunks())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert count == 1001  # 200 lines a part
    # The file is read once to choose the codec (its size); the command is
    # never held whole (that would take about 15 times its size).
    assert peak < 3 * size, f"peak {peak:,} bytes for a {size:,}-byte file"
