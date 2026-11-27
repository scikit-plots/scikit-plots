# lean/tests/test__shapes.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Statement shapes: equal exactly when statements differ only in bound names
==========================================================================

A shape keeps a result's code tokens with its bound names renamed in order
(``x0``, ``x1``, ...), each numeral as ``#`` and each string or char
literal as one ``"``; comments, attributes, modifiers, the keyword and the
result's own name are dropped. Binders without a type (``{x y}``,
``(hx)``) bind only in the signature: in the type, ``{x}`` is a set.
"""

from __future__ import annotations

import pytest

from .._shapes import _shape, fingerprint


def _print(statement):
    kind = "lemma" if " lemma " in f" {statement} " else "theorem"
    return fingerprint(statement, kind, None)


SAME = [
    ("theorem a (x y : Nat) : x + y = y + x", "theorem b (m n : Nat) : m + n = n + m"),
    ("theorem a : ∀ x y : Nat, x + y = y + x", "theorem b : ∀ p q : Nat, p + q = q + p"),
    ("theorem a (x : Nat) :\n    x = x", "theorem b (y : Nat) : y = y"),
    ("@[simp] theorem a (x : Nat) : x = x", "theorem b (y : Nat) : y = y"),
    ("/-- doc -/\ntheorem a (x : Nat) : x = x", "theorem b (y : Nat) : y = y"),
    ("protected theorem a (x : Nat) : x = x -- note", "lemma b (y : Nat) : y = y"),
    ("theorem a ⦃x : Nat⦄ : x = x", "theorem b ⦃z : Nat⦄ : z = z"),
    ("theorem a (x : Nat) : x * 2 = x + x", "theorem b (x : Nat) : x * 3 = x + x"),
    # binders without a type, in the signature
    ("theorem a {x y} : x = y → y = x", "theorem b {u v} : u = v → v = u"),
    ("theorem a (f : U) (y : X) (hy) : f y hy = y", "theorem b (f : U) (x : X) (hx) : f x hx = x"),
    ("theorem a {a : α} {l} (h : a ∈ l) : p a", "theorem b {a : α} {s} (h : a ∈ s) : p a"),
    # literals are abstracted as numerals are
    ('theorem a (s : String) : s = "a b"', 'theorem b (t : String) : t = "c"'),
    ("theorem a (c : Char) : c = 'x'", "theorem b (d : Char) : d = '\\n'"),
    ("theorem a (s : String) : s = \"x\"", 'theorem b (s : String) : s = r#"y"#'),
    ("theorem a [Group G] (g : G) : g = g", "theorem b [Group G] (h : G) : h = h"),
]

DIFFERENT = [
    ("theorem a (x y : Nat) : x + y = y + x", "theorem b (x y : Nat) : x + y = x + y"),
    ("theorem a (x : Nat) : x = x", "theorem b (x : Int) : x = x"),
    ("theorem a (x : Nat) : x = x", "theorem b (x : Nat) : x = x ∧ True"),
    # a literal is a token, not nothing
    ('theorem a : f "a" = 1', "theorem b : f = 1"),
    ("theorem a : c = 'x'", "theorem b : c = d"),
    # in the type, {x} is a set and (f x) an application: x is free
    ("theorem a : {x} = s", "theorem b : {y} = s"),
    ("theorem a (h : (f x) = y) : True", "theorem b (h : (g x) = y) : True"),
    # [Group G] binds no name: G is free
    ("theorem a [Group G] (g : G) : g = g", "theorem b [Group H] (g : H) : g = g"),
]


@pytest.mark.parametrize(("first", "second"), SAME)
def test_statements_differing_only_in_bound_names_share_a_shape(first, second):
    assert _print(first) is not None
    assert _print(first) == _print(second), (_shape(first, "a"), _shape(second, "b"))


@pytest.mark.parametrize(("first", "second"), DIFFERENT)
def test_statements_that_differ_otherwise_do_not(first, second):
    assert _print(first) != _print(second), _shape(first, "a")


def test_a_comment_leaves_nothing_and_a_literal_one_token():
    assert _shape('theorem a : f "a b" = 1 -- "note"', "a") == ': f " = #'
    assert _shape("theorem a : f = 1 /- x -/", "a") == ": f = #"
