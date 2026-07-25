# scikitplot/cython/tests/test__builder_names.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CYTHON-PKG-001.

Package builds validated module short names only as "non-empty, no dot" and did
not validate ``package_name`` at all — so a malformed name (``..``, separators,
leading digit, keyword) produced invalid import names or, via
``package_name.replace(".", os.sep)``, wrote build files outside the intended
tree.  Names are now validated segment-by-segment as ASCII, non-keyword Python
identifiers before any path is built.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._builder import (
    _validate_dotted_name,
    _validate_module_short_name,
    build_extension_package_from_code_result,
)


class TestValidateDottedName:
    @pytest.mark.parametrize(
        "ok", ["foo", "foo.bar", "foo.bar.baz", "_private", "a1.b2", "_1.c"]
    )
    def test_valid(self, ok: str) -> None:
        _validate_dotted_name(ok, what="package name")  # no raise

    @pytest.mark.parametrize(
        "bad",
        [
            "",              # empty
            "../evil",       # traversal / separators
            "foo/bar",       # separator
            "foo\\bar",      # windows separator
            "1foo",          # leading digit
            "foo..bar",      # empty segment
            ".foo",          # leading dot
            "foo.",          # trailing dot
            "foo.class",     # keyword segment
            "foo bar",       # space
            "foo-bar",       # hyphen
            "café",          # non-ASCII
        ],
    )
    def test_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _validate_dotted_name(bad, what="package name")


class TestValidateModuleShortName:
    @pytest.mark.parametrize("ok", ["mymod", "_m", "a1"])
    def test_valid(self, ok: str) -> None:
        _validate_module_short_name(ok)

    @pytest.mark.parametrize(
        "bad", ["", "foo.bar", "1mod", "mod-x", "class", "with space", "évil"]
    )
    def test_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _validate_module_short_name(bad)


class TestPackageBuildRejectsBadNames:
    @pytest.mark.parametrize("bad_pkg", ["../evil", "foo/bar", "1foo", "foo.class"])
    def test_bad_package_name_rejected_before_build(
        self, bad_pkg: str, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            build_extension_package_from_code_result(
                {"m": "def f(): return 1"},
                package_name=bad_pkg,
                cache_dir=tmp_path,
            )

    def test_bad_module_short_name_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            build_extension_package_from_code_result(
                {"bad-name": "def f(): return 1"},
                package_name="pkg",
                cache_dir=tmp_path,
            )

    def test_no_files_written_outside_tree(self, tmp_path: Path) -> None:
        """A traversal package name must not create files outside the cache."""
        outside = tmp_path.parent / "evil"
        with pytest.raises(ValueError):
            build_extension_package_from_code_result(
                {"m": "def f(): return 1"},
                package_name="../evil",
                cache_dir=tmp_path,
            )
        assert not outside.exists()
