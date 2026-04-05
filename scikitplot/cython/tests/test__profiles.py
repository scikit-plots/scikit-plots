# scikitplot/cython/tests/test__profiles.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._profiles`.

Covers
------
- ``ProfileDefaults``   : dataclass defaults, frozen
- ``resolve_profile()`` : None → empty defaults; fast-debug / release / annotate
                          on Linux and Windows (mocked); unknown → ValueError
- ``apply_profile()``   : annotate precedence (explicit False wins), user args
                          override profile, language passthrough, merged directives
- ``is_windows()``      : returns bool
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from .._profiles import ProfileDefaults, apply_profile, is_windows, resolve_profile
import sys


class TestProfileDefaults:
    """Tests for :class:`~scikitplot.cython._profiles.ProfileDefaults`."""

    def test_default_values(self) -> None:
        p = ProfileDefaults()
        assert p.annotate is False
        assert p.compiler_directives == {}
        assert p.language is None

    def test_frozen(self) -> None:
        p = ProfileDefaults()
        with pytest.raises((TypeError, AttributeError)):
            p.annotate = True  # type: ignore[misc]


class TestResolveProfile:
    """Tests for :func:`~scikitplot.cython._profiles.resolve_profile`."""

    def test_none_returns_empty_defaults(self) -> None:
        d = resolve_profile(None)
        assert d.annotate is False
        assert d.compiler_directives == {}
        assert d.extra_compile_args == ()
        assert d.language is None

    def test_fast_debug_directives(self) -> None:
        d = resolve_profile("fast-debug")
        assert d.compiler_directives["boundscheck"] is True
        assert d.compiler_directives["wraparound"] is True
        assert len(d.extra_compile_args) > 0

    def test_release_directives(self) -> None:
        d = resolve_profile("release")
        assert d.compiler_directives["boundscheck"] is False
        assert d.compiler_directives["wraparound"] is False
        assert len(d.extra_compile_args) > 0

    def test_annotate_profile(self) -> None:
        d = resolve_profile("annotate")
        assert d.annotate is True

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_profile("nonexistent")

    @pytest.mark.parametrize("profile", ["fast-debug", "release", "annotate"])
    def test_all_profiles_are_valid(self, profile: str) -> None:
        d = resolve_profile(profile)
        assert isinstance(d, ProfileDefaults)

    def test_windows_compiler_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scikitplot.cython._profiles.is_windows", lambda: True)
        d = resolve_profile("fast-debug")
        assert any("O" in a for a in d.extra_compile_args)


class TestApplyProfile:
    """Tests for :func:`~scikitplot.cython._profiles.apply_profile`."""

    def _apply(self, **kw) -> tuple:
        defaults = dict(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        defaults.update(kw)
        return apply_profile(**defaults)

    # --- annotate precedence fix regression tests ---

    def test_annotate_false_wins_over_annotate_profile(self) -> None:
        """Regression: user annotate=False must beat profile='annotate'."""
        out_annotate, *_ = self._apply(profile="annotate", annotate=False)
        assert out_annotate is False, (
            "BUG: profile overrode user annotate=False — fix regressed"
        )

    def test_annotate_true_is_kept(self) -> None:
        out_annotate, *_ = self._apply(profile=None, annotate=True)
        assert out_annotate is True

    def test_annotate_true_kept_with_profile(self) -> None:
        out_annotate, *_ = self._apply(profile="release", annotate=True)
        assert out_annotate is True

    def test_annotate_false_no_profile(self) -> None:
        out_annotate, *_ = self._apply(profile=None, annotate=False)
        assert out_annotate is False

    # --- directive merging ---

    def test_none_directives_use_profile_defaults(self) -> None:
        _, directives, *_ = self._apply(profile="fast-debug", compiler_directives=None)
        assert directives is not None
        assert directives["boundscheck"] is True

    def test_user_directives_override_profile(self) -> None:
        _, directives, *_ = self._apply(
            profile="fast-debug",
            compiler_directives={"boundscheck": False},
        )
        assert directives["boundscheck"] is False

    def test_user_directives_merged_with_profile(self) -> None:
        _, directives, *_ = self._apply(
            profile="fast-debug",
            compiler_directives={"my_custom": True},
        )
        assert directives["my_custom"] is True
        assert "boundscheck" in directives  # profile default still present

    def test_none_profile_none_directives(self) -> None:
        _, directives, *_ = self._apply(profile=None, compiler_directives=None)
        assert directives is None

    # --- compile args ---

    def test_user_compile_args_override_profile(self) -> None:
        _, _, cargs, *_ = self._apply(
            profile="release", extra_compile_args=["-O0"]
        )
        assert list(cargs) == ["-O0"]

    def test_profile_compile_args_used_when_none(self) -> None:
        _, _, cargs, *_ = self._apply(profile="release", extra_compile_args=None)
        assert cargs is not None
        assert len(cargs) > 0

    # --- language ---

    def test_user_language_wins(self) -> None:
        *_, lang = self._apply(profile=None, language="c++")
        assert lang == "c++"

    def test_none_language_stays_none_without_profile(self) -> None:
        *_, lang = self._apply(profile=None, language=None)
        assert lang is None


class TestIsWindows:
    """Tests for :func:`~scikitplot.cython._profiles.is_windows`."""

    def test_returns_bool(self) -> None:
        assert isinstance(is_windows(), bool)

    def test_linux_not_windows(self) -> None:
        if sys.platform.startswith("linux"):
            assert is_windows() is False


class TestResolvProfileBranches:
    """Cover all profile resolution branches."""

    def test_none_returns_empty_defaults(self) -> None:
        d = resolve_profile(None)
        assert isinstance(d, ProfileDefaults)
        assert d.annotate is False
        assert d.compiler_directives == {}
        assert d.language is None

    def test_fast_debug_linux_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            d = resolve_profile("fast-debug")
        assert "-O0" in d.extra_compile_args
        assert d.compiler_directives["boundscheck"] is True

    def test_fast_debug_windows_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=True):
            d = resolve_profile("fast-debug")
        assert "/Od" in d.extra_compile_args

    def test_release_linux_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            d = resolve_profile("release")
        assert "-O3" in d.extra_compile_args
        assert d.compiler_directives["boundscheck"] is False

    def test_release_windows_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=True):
            d = resolve_profile("release")
        assert "/O2" in d.extra_compile_args

    def test_annotate_profile_sets_annotate_true(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            d = resolve_profile("annotate")
        assert d.annotate is True
        assert d.compiler_directives["boundscheck"] is True

    def test_annotate_windows_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=True):
            d = resolve_profile("annotate")
        assert "/Od" in d.extra_compile_args

    def test_unknown_profile_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_profile("turbo")

    def test_is_windows_returns_bool(self) -> None:
        assert isinstance(is_windows(), bool)


class TestApplyProfileBranches:
    """Cover apply_profile precedence rules."""

    def test_none_profile_none_directives_returns_none(self) -> None:
        _, directives, _, _, _ = apply_profile(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert directives is None

    def test_user_directives_override_profile_defaults(self) -> None:
        _, directives, _, _, _ = apply_profile(
            profile="fast-debug",
            annotate=False,
            compiler_directives={"boundscheck": False},
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        # User override takes precedence; merged with profile defaults
        assert directives["boundscheck"] is False

    def test_user_compile_args_override_profile(self) -> None:
        _, _, cargs, _, _ = apply_profile(
            profile="fast-debug",
            annotate=False,
            compiler_directives=None,
            extra_compile_args=["-O2"],
            extra_link_args=None,
            language=None,
        )
        assert cargs == ["-O2"]

    def test_profile_compile_args_used_when_none(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            _, _, cargs, _, _ = apply_profile(
                profile="fast-debug",
                annotate=False,
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert "-O0" in cargs

    def test_user_language_wins_over_profile(self) -> None:
        _, _, _, _, lang = apply_profile(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language="c++",
        )
        assert lang == "c++"

    def test_annotate_false_always_wins(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            annotate_out, _, _, _, _ = apply_profile(
                profile="annotate",
                annotate=False,  # explicit False wins
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert annotate_out is False

    def test_annotate_true_is_kept(self) -> None:
        annotate_out, _, _, _, _ = apply_profile(
            profile=None,
            annotate=True,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert annotate_out is True

    def test_profile_directives_applied_when_user_none(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            _, directives, _, _, _ = apply_profile(
                profile="release",
                annotate=False,
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert directives["boundscheck"] is False

    def test_empty_profile_compile_args_returns_none(self) -> None:
        """Profile with empty extra_compile_args falls back to None."""
        _, _, cargs, _, _ = apply_profile(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert cargs is None

    def test_user_link_args_override_profile(self) -> None:
        _, _, _, largs, _ = apply_profile(
            profile="fast-debug",
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=["-lz"],
            language=None,
        )
        assert largs == ["-lz"]


@pytest.mark.parametrize(
    "profile",
    ["fast-debug", "release", "annotate", None],
)
def test_resolve_profile_returns_profile_defaults(profile: str | None) -> None:
    with patch("scikitplot.cython._profiles.is_windows", return_value=False):
        result = resolve_profile(profile)
    assert isinstance(result, ProfileDefaults)


@pytest.mark.parametrize(
    "profile",
    ["fast-debug", "release", "annotate", None],
)
def test_resolve_profile_all_values(profile) -> None:
    d = resolve_profile(profile)
    assert isinstance(d, ProfileDefaults)
