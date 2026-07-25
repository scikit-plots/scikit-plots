# scikitplot/cython/tests/test__sanitize_collision.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-API-003.

``sanitize`` collapsed distinct inputs (``"a-b"``, ``"a.b"``) to one name and
leaked non-ASCII letters through (``str.isalnum`` accepts them), violating the
documented ASCII-only contract.  It now replaces non-ASCII, keeps already-valid
identifiers unchanged, and appends a content-hash suffix when it alters the
input so distinct inputs stay distinct.
"""
from __future__ import annotations

import pytest

from .._utils import sanitize


class TestNoCollisions:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("a-b", "a.b"),
            ("a b", "a/b"),
            ("foo!", "foo?"),
            ("x@1", "x#1"),
            ("café", "cafe"),  # non-ASCII vs ASCII near-miss
        ],
    )
    def test_distinct_inputs_distinct_outputs(self, a: str, b: str) -> None:
        assert sanitize(a) != sanitize(b)


class TestAsciiOnly:
    @pytest.mark.parametrize("s", ["café", "αβγ", "naïve", "Ä", "①②③", "Ω"])
    def test_output_is_ascii(self, s: str) -> None:
        out = sanitize(s)
        assert out.isascii()
        assert out.isidentifier()


class TestValidNamesUnchanged:
    @pytest.mark.parametrize(
        "s",
        ["hello", "hello_world", "_private", "__dunder__", "abc123", "MyModule", "a_b_c"],
    )
    def test_already_valid_unchanged(self, s: str) -> None:
        assert sanitize(s) == s


class TestDeterminism:
    def test_stable_across_calls(self) -> None:
        assert sanitize("a-b") == sanitize("a-b")
        assert sanitize("café") == sanitize("café")

    def test_empty_and_type(self) -> None:
        assert sanitize("") == "_"
        with pytest.raises(TypeError):
            sanitize(123)  # type: ignore[arg-type]
