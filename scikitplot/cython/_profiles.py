# scikitplot/cython/_profiles.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Build profile presets for :mod:`scikitplot.cython`.

Profiles provide canonical, reproducible presets for common developer workflows.

Design goals
------------
- Deterministic: a profile maps to a fixed, hashable set of defaults.
- Strict precedence: explicit user arguments always override profile defaults;
  an *unset* argument (``None``) inherits the profile default.
- Cross-platform: flags are selected by the active **compiler toolchain**, not
  by the host **operating system** (see :func:`_is_msvc`).
- Immutable outputs: every returned structure is frozen / tuple-based so a
  resolved build configuration cannot be mutated after the fact, before it
  reaches the content-addressed cache key.

Security model
--------------
This module is intentionally narrow and is *not* a security boundary on its own,
but it is written so that it cannot become an attack surface:

- **No argument injection via profiles.** Every compiler/linker flag a profile
  emits is a hard-coded string literal drawn from a closed set
  (``-O3``/``-O0``/``-g``/``-DNDEBUG`` or ``/O2``/``/Od``/``/Zi``). No
  user-controlled value is ever interpolated into a flag string, so a profile
  cannot be coerced into emitting a malicious flag.
- **User-supplied flags are passed through verbatim and validated elsewhere.**
  ``extra_compile_args`` / ``extra_link_args`` provided by the caller are *not*
  trusted or rewritten here; they are normalized to tuples and forwarded. The
  build layer (:mod:`scikitplot.cython._security` —
  :func:`validate_build_inputs`, :func:`is_safe_compiler_arg`,
  :func:`is_safe_macro_name`, :func:`is_safe_path`) is the single place that
  vets them before compilation. Re-implementing that here would duplicate the
  policy and risk drift, so this module deliberately does not.
- **Toolchain detection never executes anything.** :func:`_is_msvc` only probes
  ``PATH`` with :func:`shutil.which`; it never runs the compiler. Its result is
  advisory, and trust in ``PATH`` is the host environment's responsibility.
- **Closed, immutable profile set.** The set of accepted profile names is a
  :class:`frozenset`; it cannot be extended or mutated at runtime.

Notes
-----
Profiles are applied in the public API layer before calling the builder. This
keeps the builder strictly "mechanical" and reduces the chance of accidental
signature drift.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Sequence

__all__ = [
    "AppliedProfile",
    "ProfileDefaults",
    "apply_profile",
    "is_windows",
    "resolve_profile",
]


@dataclass(frozen=True, slots=True)
class ProfileDefaults:
    """
    Normalized defaults produced by resolving a profile.

    The instance is frozen and slotted: it is immutable and rejects attribute
    injection, so a resolved profile cannot be tampered with after creation.

    Parameters
    ----------
    annotate : bool, default=False
        Default for the Cython ``annotate`` flag.
    compiler_directives : Mapping[str, Any]
        Default Cython compiler directives merged on top of the baseline.
    extra_compile_args : tuple of str
        Default compiler flags for the C/C++ compiler. Always a tuple.
    extra_link_args : tuple of str
        Default linker flags. Always a tuple.
    language : {'c', 'c++'} or None, default=None
        Optional default language, or ``None`` to leave unspecified.

    Notes
    -----
    Defaults exist only to satisfy documentation tooling that expects class
    attributes to be defaulted. The public API always returns fully-filled
    values produced by :func:`resolve_profile`.

    .. note::
        ``slots=True`` requires Python >= 3.10. The project's
        ``requires-python`` floor should not be lowered below that without
        removing this argument.
    """

    annotate: bool = False
    compiler_directives: Mapping[str, Any] = field(default_factory=dict)
    extra_compile_args: tuple[str, ...] = field(default_factory=tuple)
    extra_link_args: tuple[str, ...] = field(default_factory=tuple)
    language: str | None = None

    def __repr__(self) -> str:  # pragma: no cover
        # Stable, all-fields repr for logging and doc tooling.
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


class AppliedProfile(NamedTuple):
    """
    Result of applying a profile with user-precedence rules.

    This is the return type of :func:`apply_profile`. It is a
    :class:`typing.NamedTuple`, so it is fully backward-compatible with the
    historical 5-tuple return (positional unpacking
    ``annotate, directives, cargs, largs, lang = apply_profile(...)`` still
    works) while also exposing named, self-documenting fields and remaining
    immutable.

    Parameters
    ----------
    annotate : bool
        Resolved Cython ``annotate`` flag.
    compiler_directives : dict[str, Any] or None
        Resolved Cython directives, or ``None`` to fall back to Cython defaults.
    extra_compile_args : tuple of str
        Resolved compiler flags. Always a tuple (possibly empty), never
        ``None`` -- empty means "no extra flags".
    extra_link_args : tuple of str
        Resolved linker flags. Always a tuple (possibly empty), never ``None``.
    language : {'c', 'c++'} or None
        Resolved language, or ``None`` to leave unspecified.

    Notes
    -----
    Prefer named-field access (``result.extra_compile_args``) in new code.
    Positional unpacking of all five fields remains supported for existing
    callers, but if a sixth field is ever appended, fixed-arity unpackers
    (``a, b, c, d, e = ...``) must migrate to named access or ``*rest``.
    """

    annotate: bool
    compiler_directives: dict[str, Any] | None
    extra_compile_args: tuple[str, ...]
    extra_link_args: tuple[str, ...]
    language: str | None


# Closed, immutable set of accepted profile names. Frozen so it cannot be
# extended or mutated at runtime.
_PROFILE_NAMES: frozenset[str] = frozenset({"fast-debug", "release", "annotate"})


def is_windows() -> bool:
    """
    Return ``True`` if the host operating system is native Windows.

    .. warning::
        This detects the **host OS**, not the active **compiler toolchain**, and
        is therefore *not* the correct predicate for selecting compiler flags.
        On a native-Windows host using MinGW-w64 or an MSYS2 ``MINGW64`` shell,
        the active compiler is GCC, yet this function still returns ``True``.
        Passing MSVC flags (e.g. ``/O2``) to GCC causes a build failure or a
        silently unoptimized build. Use :func:`_is_msvc` to select flags.

    Returns
    -------
    bool
        ``True`` if ``os.name == "nt"`` or ``sys.platform`` starts with
        ``"win"``; ``False`` otherwise (including WSL, Cygwin, and
        Linux-to-Windows cross-compilation).

    See Also
    --------
    _is_msvc : Toolchain predicate; the correct basis for flag selection.

    Notes
    -----
    Retained in the public API for backward compatibility and for callers that
    legitimately need host-OS detection (e.g. path handling). It intentionally
    no longer governs the compiler-flag branches in :func:`resolve_profile`.
    """
    return os.name == "nt" or sys.platform.startswith("win")


def _is_msvc() -> bool:
    """
    Return ``True`` if MSVC (``cl.exe``) is the active C/C++ compiler.

    This is the correct predicate for selecting compiler flags. It returns
    ``False`` on Windows + MinGW/MSYS2 (where GCC-style flags must be used) and
    ``False`` on every non-Windows host.

    Returns
    -------
    bool
        ``True`` if and only if the host OS is Windows *and* ``cl.exe`` is
        resolvable on ``PATH``.

    See Also
    --------
    is_windows : Host-OS predicate (does not imply MSVC).

    Notes
    -----
    :func:`shutil.which` only searches ``PATH`` for an executable named ``cl``;
    it never runs it, so this probe has no side effects and cannot execute
    attacker-controlled code. ``PATH`` ordering is honoured, so a host with both
    MSVC and MinGW installed controls the result deterministically via its
    environment. Trust in ``PATH`` itself is the host's responsibility.
    """
    if not (os.name == "nt" or sys.platform.startswith("win")):
        return False
    return shutil.which("cl") is not None


def resolve_profile(profile: str | None) -> ProfileDefaults:
    """
    Resolve a profile name to deterministic defaults.

    Parameters
    ----------
    profile : str or None
        One of ``"fast-debug"``, ``"release"``, ``"annotate"``, or ``None``.

    Returns
    -------
    ProfileDefaults
        Deterministic, immutable defaults for the requested profile.

    Raises
    ------
    ValueError
        If ``profile`` is not ``None`` and is not a recognized name.

    Notes
    -----
    Compiler-flag branches key off :func:`_is_msvc` (active toolchain), not
    :func:`is_windows` (host OS). This is the fix for the toolchain/OS mismatch:
    a native-Windows host using GCC (MinGW/MSYS2) now receives GCC flags instead
    of MSVC flags. On every non-Windows host, and on Windows-with-MSVC, the
    resolved flags are unchanged from the historical behaviour, so existing
    content-addressed cache keys are preserved there.

    Every flag below is a hard-coded literal; no user input flows into flag
    selection, so this function cannot emit an injected argument.
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
            f"Unknown profile: {profile!r}. "
            f"Expected one of: {sorted(_PROFILE_NAMES)!r} or None."
        )

    # Select flags by the *active compiler*, never by the host OS.
    use_msvc = _is_msvc()

    if profile == "fast-debug":
        # Canonical debug-ish defaults: minimal optimization + debug symbols.
        cargs: tuple[str, ...] = ("/Od", "/Zi") if use_msvc else ("-O0", "-g")
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
            extra_link_args=(),
            language=None,
        )

    if profile == "release":
        # Canonical release defaults: optimization + remove asserts.
        cargs = ("/O2",) if use_msvc else ("-O3", "-DNDEBUG")
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
            extra_link_args=(),
            language=None,
        )

    # profile == "annotate"
    # Generate the HTML annotation and keep compiler settings developer-friendly.
    cargs = ("/Od", "/Zi") if use_msvc else ("-O0", "-g")
    directives = {
        "boundscheck": True,
        "wraparound": True,
        "initializedcheck": True,
    }
    return ProfileDefaults(
        annotate=True,
        compiler_directives=directives,
        extra_compile_args=cargs,
        extra_link_args=(),
        language=None,
    )


def apply_profile(
    *,
    profile: str | None,
    annotate: bool | None = None,
    compiler_directives: Mapping[str, Any] | None,
    extra_compile_args: Sequence[str] | None,
    extra_link_args: Sequence[str] | None,
    language: str | None,
) -> AppliedProfile:
    """
    Apply a profile with strict, three-state precedence.

    Every field follows the same contract:

    - The user passed an explicit value (not ``None``) -> the user value wins.
    - The user left the value unset (``None``) -> the profile default applies.

    For ``annotate`` this is the precedence fix: the parameter is ``bool | None``
    so "user did not specify" (``None``) is distinguishable from "user
    explicitly disabled" (``False``). Previously ``annotate`` was a plain
    ``bool`` defaulting to ``False``, so the ``"annotate"`` profile -- whose sole
    purpose is to enable annotation -- could never take effect unless the caller
    *also* passed ``annotate=True``, which defeated the profile.

    Parameters
    ----------
    profile : str or None
        Profile name, validated by :func:`resolve_profile`.
    annotate : bool or None, default=None
        ``None`` inherits the profile default; ``True``/``False`` are explicit
        and always win.
    compiler_directives : Mapping[str, Any] or None
        ``None`` inherits the profile default; a mapping is merged on top of the
        profile default (user keys win).
    extra_compile_args : Sequence[str] or None
        ``None`` inherits the profile default; otherwise normalized to a tuple.
    extra_link_args : Sequence[str] or None
        ``None`` inherits the profile default; otherwise normalized to a tuple.
    language : {'c', 'c++'} or None
        ``None`` inherits the profile default.

    Returns
    -------
    AppliedProfile
        Named 5-tuple ``(annotate, compiler_directives, extra_compile_args,
        extra_link_args, language)``. ``extra_compile_args`` and
        ``extra_link_args`` are always tuples (empty means "no flags", never
        ``None``); ``compiler_directives`` is ``None`` only when neither the
        profile nor the user supplied any.

    Notes
    -----
    The result is an :class:`AppliedProfile` (a ``NamedTuple``), so positional
    unpacking remains identical to the historical bare-tuple return while adding
    named access. User-supplied flag sequences are normalized to tuples but
    otherwise forwarded verbatim; argument *safety* is enforced by the security
    layer at build time, not here.
    """
    defaults = resolve_profile(profile)

    # Three-state precedence for annotate:
    #   None  -> inherit the profile default
    #   True  -> user explicitly enabled  (wins)
    #   False -> user explicitly disabled (wins)
    out_annotate = bool(defaults.annotate) if annotate is None else bool(annotate)

    out_directives: dict[str, Any] | None
    if compiler_directives is None:
        out_directives = (
            dict(defaults.compiler_directives) if defaults.compiler_directives else None
        )
    else:
        # Merge: user directives override profile defaults.
        merged = dict(defaults.compiler_directives)
        merged.update(dict(compiler_directives))
        out_directives = merged

    # Normalize flags to immutable tuples. Empty tuple (not None) means
    # "no extra flags", removing the historical ``() or None`` ambiguity.
    out_cargs: tuple[str, ...] = (
        tuple(extra_compile_args)
        if extra_compile_args is not None
        else tuple(defaults.extra_compile_args)
    )
    out_largs: tuple[str, ...] = (
        tuple(extra_link_args)
        if extra_link_args is not None
        else tuple(defaults.extra_link_args)
    )

    out_lang = language if language is not None else defaults.language

    return AppliedProfile(
        annotate=out_annotate,
        compiler_directives=out_directives,
        extra_compile_args=out_cargs,
        extra_link_args=out_largs,
        language=out_lang,
    )
