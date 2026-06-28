# scikitplot/cython/tests/test__profiles.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._profiles`.

Covers
------
- ``ProfileDefaults``   : dataclass defaults, frozen/immutable
- ``resolve_profile()`` : None -> empty defaults; fast-debug / release / annotate
                          under MSVC and GCC toolchains (mocked); unknown ->
                          ValueError
- ``apply_profile()``   : three-state annotate precedence (unset inherits,
                          explicit wins), user args override profile, language
                          passthrough, merged directives, tuple/NamedTuple result
- ``is_windows()``      : host-OS predicate, returns bool
- ``_is_msvc()``        : toolchain predicate (the correct basis for flags),
                          including the Windows+MinGW case
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from .._profiles import (
    AppliedProfile,
    ProfileDefaults,
    _is_msvc,
    apply_profile,
    is_windows,
    resolve_profile,
)

# Fully-qualified patch targets for the toolchain predicate.
_MSVC = "scikitplot.cython._profiles._is_msvc"


class TestProfileDefaults:
    """Tests for :class:`~scikitplot.cython._profiles.ProfileDefaults`."""

    def test_default_values(self) -> None:
        p = ProfileDefaults()
        assert p.annotate is False
        assert p.compiler_directives == {}
        assert p.extra_compile_args == ()
        assert p.extra_link_args == ()
        assert p.language is None

    def test_frozen(self) -> None:
        p = ProfileDefaults()
        with pytest.raises((TypeError, AttributeError)):
            p.annotate = True  # type: ignore[misc]

    def test_rejects_attribute_injection(self) -> None:
        """slots=True must prevent setting unknown attributes (hardening)."""
        p = ProfileDefaults()
        with pytest.raises((TypeError, AttributeError)):
            p.injected = "x"  # type: ignore[attr-defined]


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

    def test_msvc_compiler_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flags are selected by the toolchain predicate, not the host OS."""
        monkeypatch.setattr(_MSVC, lambda: True)
        d = resolve_profile("fast-debug")
        assert "/Od" in d.extra_compile_args

    def test_extra_compile_args_is_tuple(self) -> None:
        """Resolved flags are always an immutable tuple."""
        for name in ("fast-debug", "release", "annotate", None):
            assert isinstance(resolve_profile(name).extra_compile_args, tuple)


class TestApplyProfile:
    """Tests for :func:`~scikitplot.cython._profiles.apply_profile`."""

    def _apply(self, **kw) -> AppliedProfile:
        # NOTE: annotate defaults to None here to model "user did not specify";
        # tests that need an explicit value pass it via **kw.
        defaults = dict(
            profile=None,
            annotate=None,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        defaults.update(kw)
        return apply_profile(**defaults)

    # --- annotate three-state precedence ---

    def test_annotate_profile_applies_when_unset(self) -> None:
        """profile='annotate' with annotate unset (None) must yield True."""
        out_annotate, *_ = self._apply(profile="annotate", annotate=None)
        assert out_annotate is True, (
            "BUG: annotate profile ignored when annotate is unset"
        )

    def test_annotate_false_wins_over_annotate_profile(self) -> None:
        """Regression: explicit annotate=False must beat profile='annotate'."""
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

    def test_annotate_unset_no_profile_is_false(self) -> None:
        out_annotate, *_ = self._apply(profile=None, annotate=None)
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
        _, _, cargs, *_ = self._apply(profile="release", extra_compile_args=["-O0"])
        assert list(cargs) == ["-O0"]
        assert isinstance(cargs, tuple)  # normalized to immutable tuple

    def test_profile_compile_args_used_when_none(self) -> None:
        _, _, cargs, *_ = self._apply(profile="release", extra_compile_args=None)
        assert cargs is not None
        assert len(cargs) > 0

    def test_empty_profile_compile_args_returns_empty_tuple(self) -> None:
        """No profile + no user args -> empty tuple (never None)."""
        _, _, cargs, *_ = self._apply(profile=None, extra_compile_args=None)
        assert cargs == ()
        assert isinstance(cargs, tuple)

    # --- language ---

    def test_user_language_wins(self) -> None:
        *_, lang = self._apply(profile=None, language="c++")
        assert lang == "c++"

    def test_none_language_stays_none_without_profile(self) -> None:
        *_, lang = self._apply(profile=None, language=None)
        assert lang is None

    # --- result type / shape ---

    def test_result_is_named_tuple(self) -> None:
        result = self._apply(profile="release")
        assert isinstance(result, AppliedProfile)
        assert isinstance(result, tuple)  # backward-compatible with bare tuple
        # named access
        assert result.annotate is False
        assert isinstance(result.extra_compile_args, tuple)
        assert isinstance(result.extra_link_args, tuple)
        # positional unpacking still works (historical contract)
        annotate, directives, cargs, largs, lang = result
        assert annotate is False


class TestIsWindows:
    """Tests for :func:`~scikitplot.cython._profiles.is_windows`."""

    def test_returns_bool(self) -> None:
        assert isinstance(is_windows(), bool)

    def test_linux_not_windows(self) -> None:
        if sys.platform.startswith("linux"):
            assert is_windows() is False


class TestIsMsvc:
    """Tests for :func:`~scikitplot.cython._profiles._is_msvc` (toolchain)."""

    def test_false_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setattr("sys.platform", "linux")
        assert _is_msvc() is False

    def test_true_on_windows_with_cl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("shutil.which", lambda cmd: r"C:\\MSVC\\cl.exe")
        assert _is_msvc() is True

    def test_false_on_windows_mingw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows + MinGW (no cl.exe on PATH) must NOT be detected as MSVC."""
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        assert _is_msvc() is False


class TestResolveProfileBranches:
    """Cover all profile resolution branches against both toolchains."""

    def test_none_returns_empty_defaults(self) -> None:
        d = resolve_profile(None)
        assert isinstance(d, ProfileDefaults)
        assert d.annotate is False
        assert d.compiler_directives == {}
        assert d.language is None

    def test_fast_debug_gcc_args(self) -> None:
        with patch(_MSVC, return_value=False):
            d = resolve_profile("fast-debug")
        assert "-O0" in d.extra_compile_args
        assert d.compiler_directives["boundscheck"] is True

    def test_fast_debug_msvc_args(self) -> None:
        with patch(_MSVC, return_value=True):
            d = resolve_profile("fast-debug")
        assert "/Od" in d.extra_compile_args

    def test_release_gcc_args(self) -> None:
        with patch(_MSVC, return_value=False):
            d = resolve_profile("release")
        assert "-O3" in d.extra_compile_args
        assert d.compiler_directives["boundscheck"] is False

    def test_release_msvc_args(self) -> None:
        with patch(_MSVC, return_value=True):
            d = resolve_profile("release")
        assert "/O2" in d.extra_compile_args

    def test_release_mingw_uses_gcc_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BUG-001 regression: Windows host + GCC must receive GCC flags."""
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("shutil.which", lambda cmd: None)  # MinGW, no cl.exe
        d = resolve_profile("release")
        assert "-O3" in d.extra_compile_args
        assert "/O2" not in d.extra_compile_args

    def test_annotate_profile_sets_annotate_true(self) -> None:
        with patch(_MSVC, return_value=False):
            d = resolve_profile("annotate")
        assert d.annotate is True
        assert d.compiler_directives["boundscheck"] is True

    def test_annotate_msvc_args(self) -> None:
        with patch(_MSVC, return_value=True):
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
            annotate=None,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert directives is None

    def test_user_directives_override_profile_defaults(self) -> None:
        _, directives, _, _, _ = apply_profile(
            profile="fast-debug",
            annotate=None,
            compiler_directives={"boundscheck": False},
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        # User override takes precedence; merged with profile defaults.
        assert directives["boundscheck"] is False

    def test_user_compile_args_override_profile(self) -> None:
        _, _, cargs, _, _ = apply_profile(
            profile="fast-debug",
            annotate=None,
            compiler_directives=None,
            extra_compile_args=["-O2"],
            extra_link_args=None,
            language=None,
        )
        assert list(cargs) == ["-O2"]

    def test_profile_compile_args_used_when_none(self) -> None:
        with patch(_MSVC, return_value=False):
            _, _, cargs, _, _ = apply_profile(
                profile="fast-debug",
                annotate=None,
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert "-O0" in cargs

    def test_user_language_wins_over_profile(self) -> None:
        _, _, _, _, lang = apply_profile(
            profile=None,
            annotate=None,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language="c++",
        )
        assert lang == "c++"

    def test_annotate_false_always_wins(self) -> None:
        with patch(_MSVC, return_value=False):
            annotate_out, _, _, _, _ = apply_profile(
                profile="annotate",
                annotate=False,  # explicit False wins
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert annotate_out is False

    def test_annotate_unset_applies_profile(self) -> None:
        with patch(_MSVC, return_value=False):
            annotate_out, _, _, _, _ = apply_profile(
                profile="annotate",
                annotate=None,  # unset -> inherit profile default (True)
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert annotate_out is True

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
        with patch(_MSVC, return_value=False):
            _, directives, _, _, _ = apply_profile(
                profile="release",
                annotate=None,
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert directives["boundscheck"] is False

    def test_empty_profile_compile_args_returns_empty_tuple(self) -> None:
        """No profile + no user args -> empty tuple (never None)."""
        _, _, cargs, _, _ = apply_profile(
            profile=None,
            annotate=None,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert cargs == ()
        assert isinstance(cargs, tuple)

    def test_user_link_args_override_profile(self) -> None:
        _, _, _, largs, _ = apply_profile(
            profile="fast-debug",
            annotate=None,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=["-lz"],
            language=None,
        )
        assert list(largs) == ["-lz"]


@pytest.mark.parametrize("profile", ["fast-debug", "release", "annotate", None])
def test_resolve_profile_returns_profile_defaults(profile: str | None) -> None:
    with patch(_MSVC, return_value=False):
        result = resolve_profile(profile)
    assert isinstance(result, ProfileDefaults)


@pytest.mark.parametrize("profile", ["fast-debug", "release", "annotate", None])
def test_resolve_profile_all_values(profile) -> None:
    d = resolve_profile(profile)
    assert isinstance(d, ProfileDefaults)
