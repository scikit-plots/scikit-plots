# scikitplot/cython/_profiles.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Build profile presets for :mod:`scikitplot.cython`.

Profiles provide canonical, reproducible presets for common developer workflows.

Design goals:

- Deterministic: a profile maps to a fixed set of defaults.
- Strict precedence: explicit user arguments always override profile defaults.
- Cross-platform: defaults are chosen for POSIX and Windows toolchains.

Notes
-----
Profiles are applied in the public API layer before calling the builder. This
keeps the builder strictly "mechanical" and reduces the chance of accidental
signature drift.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProfileDefaults:
    """
    Normalized defaults produced by applying a profile.

    Parameters
    ----------
    annotate : bool
        Default for the Cython annotate flag.
    compiler_directives : Mapping[str, Any]
        Default Cython compiler directives merged on top of the baseline.
    extra_compile_args : Sequence[str]
        Default compiler flags for the C/C++ compiler.
    extra_link_args : Sequence[str]
        Default linker flags.
    language : {'c', 'c++'} or None
        Optional default language, or None to leave unspecified.
    """

    # NOTE: Defaults exist to satisfy documentation tooling that expects class
    # attributes to have defaults. The public API always returns fully-filled
    # values produced by ``resolve_profile``.
    annotate: bool = False
    compiler_directives: Mapping[str, Any] = field(default_factory=dict)
    extra_compile_args: Sequence[str] = field(default_factory=tuple)
    extra_link_args: Sequence[str] = field(default_factory=tuple)
    language: str | None = None

    def __repr__(self) -> str:  # pragma: no cover
        # Keep a stable, all-fields repr for logging and doc tooling.
        directives = dict(self.compiler_directives)
        cargs = tuple(self.extra_compile_args)
        largs = tuple(self.extra_link_args)
        return (
            "ProfileDefaults("
            f"annotate={self.annotate!r}, "
            f"compiler_directives={directives!r}, "
            f"extra_compile_args={cargs!r}, "
            f"extra_link_args={largs!r}, "
            f"language={self.language!r}"
            ")"
        )


_PROFILE_NAMES = {"fast-debug", "release", "annotate"}


def is_windows() -> bool:
    """Return True if running on Windows."""
    return os.name == "nt" or sys.platform.startswith("win")


def resolve_profile(profile: str | None) -> ProfileDefaults:
    """
    Resolve a profile name to deterministic defaults.

    Parameters
    ----------
    profile : str or None
        One of: ``"fast-debug"``, ``"release"``, ``"annotate"``, or None.

    Returns
    -------
    ProfileDefaults
        Deterministic defaults for the requested profile.

    Raises
    ------
    ValueError
        If ``profile`` is not recognized.
    """
    if profile is None:
        return ProfileDefaults(
            annotate=False,
            compiler_directives={},
            extra_compile_args=(),
            extra_link_args=(),
            language=None,
        )

    if profile not in _PROFILE_NAMES:
        raise ValueError(
            f"Unknown profile: {profile!r}. Expected one of: {sorted(_PROFILE_NAMES)!r}"
        )

    win = is_windows()

    if profile == "fast-debug":
        # Canonical debug-ish defaults: minimal optimization + debug symbols.
        cargs = ("/Od", "/Zi") if win else ("-O0", "-g")
        largs = () if win else ()  # noqa: RUF034
        directives: Mapping[str, Any] = {
            "boundscheck": True,
            "wraparound": True,
            "initializedcheck": True,
            "cdivision": False,
        }
        return ProfileDefaults(
            annotate=False,
            compiler_directives=directives,
            extra_compile_args=cargs,
            extra_link_args=largs,
            language=None,
        )

    if profile == "release":
        # Canonical release defaults: optimization + remove asserts.
        cargs = ("/O2",) if win else ("-O3", "-DNDEBUG")
        largs = () if win else ()  # noqa: RUF034
        directives = {
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        }
        return ProfileDefaults(
            annotate=False,
            compiler_directives=directives,
            extra_compile_args=cargs,
            extra_link_args=largs,
            language=None,
        )

    # profile == "annotate"
    # Generate the HTML annotation and keep compiler settings developer-friendly.
    cargs = ("/Od", "/Zi") if win else ("-O0", "-g")
    largs = () if win else ()  # noqa: RUF034
    directives = {
        "boundscheck": True,
        "wraparound": True,
        "initializedcheck": True,
    }
    return ProfileDefaults(
        annotate=True,
        compiler_directives=directives,
        extra_compile_args=cargs,
        extra_link_args=largs,
        language=None,
    )


def apply_profile(
    *,
    profile: str | None,
    annotate: bool,
    compiler_directives: Mapping[str, Any] | None,
    extra_compile_args: Sequence[str] | None,
    extra_link_args: Sequence[str] | None,
    language: str | None,
) -> tuple[
    bool, dict[str, Any] | None, Sequence[str] | None, Sequence[str] | None, str | None
]:
    """
    Apply a profile with strict precedence rules.

    Precedence
    ----------
    - If an explicit argument is provided by the user, it is kept unchanged.
    - Otherwise, the profile default is applied.

    Parameters
    ----------
    profile : str or None
        Profile name.
    annotate : bool
        User-provided annotate flag.
    compiler_directives : Mapping[str, Any] or None
        User-provided compiler directives.
    extra_compile_args : Sequence[str] or None
        User-provided compiler args.
    extra_link_args : Sequence[str] or None
        User-provided link args.
    language : {'c', 'c++'} or None
        User-provided language.

    Returns
    -------
    tuple
        (annotate, compiler_directives, extra_compile_args, extra_link_args, language)
        with profile defaults applied where the user did not specify a value.
    """
    defaults = resolve_profile(profile)

    out_annotate = annotate if profile is None else (annotate or defaults.annotate)

    out_directives: dict[str, Any] | None
    if compiler_directives is None:
        out_directives = (
            dict(defaults.compiler_directives) if defaults.compiler_directives else None
        )
    else:
        # Merge: user directives override defaults.
        merged = dict(defaults.compiler_directives)
        merged.update(dict(compiler_directives))
        out_directives = merged

    out_cargs = (
        extra_compile_args
        if extra_compile_args is not None
        else (defaults.extra_compile_args or None)
    )
    out_largs = (
        extra_link_args
        if extra_link_args is not None
        else (defaults.extra_link_args or None)
    )
    out_lang = language if language is not None else defaults.language
    return out_annotate, out_directives, out_cargs, out_largs, out_lang
