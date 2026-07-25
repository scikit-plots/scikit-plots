# scikitplot/cython/_profiles.py
#
# Flake8: noqa: D213
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
    "PlatformCapabilities",
    "ProfileDefaults",
    "ResolvedToolchain",
    "RuntimeCapabilities",
    "UnsupportedRuntimeError",
    "apply_profile",
    "check_runtime_supported",
    "is_windows",
    "platform_capabilities",
    "resolve_profile",
    "resolved_toolchain",
    "runtime_capabilities",
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


class PlatformCapabilities(NamedTuple):
    """Declarative contract describing what the current platform supports.

    Runtime Cython compilation requires an out-of-process C/C++ compiler.  In a
    browser/WASM runtime (Emscripten — e.g. Pyodide / JupyterLite / xeus) there
    is no subprocess compiler, so on-the-fly compilation is unavailable and only
    **prebuilt** ``emscripten-wasm32`` extensions can be imported.  Exposing
    this as an explicit contract lets callers branch deterministically instead
    of discovering the limitation through an opaque build failure
    (CYTHON-WASM-001).

    Parameters
    ----------
    platform : str
        The value of :data:`sys.platform`.
    is_browser_wasm : bool
        ``True`` on an Emscripten/WASM runtime (browser).
    can_compile_at_runtime : bool
        ``True`` when an out-of-process compiler toolchain can be invoked.
        Always ``False`` on browser/WASM.
    prebuilt_only : bool
        ``True`` when only prebuilt extensions may be imported (no runtime
        build).  The complement of :attr:`can_compile_at_runtime`.
    wasm_package_suffix : str or None
        The platform tag for prebuilt browser packages (``"emscripten-wasm32"``)
        on a browser runtime, else ``None``.

    See Also
    --------
    platform_capabilities : Construct the contract for the current process.

    Examples
    --------
    >>> caps = platform_capabilities()
    >>> if not caps.can_compile_at_runtime:
    ...     ...  # import a prebuilt artifact instead of compiling
    """

    platform: str
    is_browser_wasm: bool
    can_compile_at_runtime: bool
    prebuilt_only: bool
    wasm_package_suffix: str | None


def platform_capabilities() -> PlatformCapabilities:
    """Return the :class:`PlatformCapabilities` contract for this process.

    Browser/WASM is detected via :data:`sys.platform` (``"emscripten"`` or a
    ``"wasm"`` prefix) and the presence of the Emscripten build sysconfig
    platform.  Detection never executes a subprocess.

    Returns
    -------
    PlatformCapabilities
        The capability contract for the current runtime.

    See Also
    --------
    PlatformCapabilities : The returned contract type.
    """
    plat = sys.platform
    is_wasm = plat == "emscripten" or plat.startswith("wasm")
    # Pyodide/xeus set sys.platform == "emscripten"; guard with sysconfig too.
    if not is_wasm:
        try:
            import sysconfig  # noqa: PLC0415

            host = (sysconfig.get_platform() or "").lower()
            is_wasm = "emscripten" in host or "wasm" in host
        except Exception:  # noqa: BLE001
            pass
    return PlatformCapabilities(
        platform=plat,
        is_browser_wasm=is_wasm,
        can_compile_at_runtime=not is_wasm,
        prebuilt_only=is_wasm,
        wasm_package_suffix="emscripten-wasm32" if is_wasm else None,
    )


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


class ResolvedToolchain(NamedTuple):
    """The compiler the build backend *actually* selects (CYTHON-PORT-001).

    Host/toolchain detection based on ``PATH`` (e.g. "is ``cl`` present?") is a
    guess, not the effective contract: setuptools/distutils decides the real
    compiler.  This queries that decision so cache entries can be keyed from the
    resolved plan rather than a heuristic.

    Parameters
    ----------
    compiler_type : str
        The distutils compiler type actually selected (``"unix"``, ``"msvc"``,
        ``"mingw32"``, ...), or ``"unknown"`` if it could not be determined.
    cc : str
        The resolved C compiler executable (argv[0] of the compile command), or
        ``""`` if unavailable.
    cxx : str
        The resolved C++ compiler executable, or ``""`` if unavailable.
    linker : str
        The resolved shared linker executable, or ``""`` if unavailable.

    See Also
    --------
    resolved_toolchain : Construct the contract for the current backend.

    Notes
    -----
    This inspects the backend's configured commands; it does **not** invoke the
    compiler, so it has no side effects.  It degrades gracefully to
    ``ResolvedToolchain("unknown", "", "", "")`` if the backend cannot be
    interrogated.
    """

    compiler_type: str
    cc: str
    cxx: str
    linker: str


def _first_token(value: Any) -> str:
    """Return argv[0] of a distutils command list/string, else ``""``."""
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    # A plain string command: take the first whitespace-separated token.
    return str(value).split()[0] if str(value).strip() else ""


def resolved_toolchain() -> ResolvedToolchain:
    """Return the compiler the build backend actually selects (CYTHON-PORT-001).

    Returns
    -------
    ResolvedToolchain
        The resolved compiler type and executables.  Never raises: on any
        failure it returns ``ResolvedToolchain("unknown", "", "", "")``.

    Notes
    -----
    Reads the configured commands from a fresh, customised distutils compiler;
    no subprocess is spawned.  On Windows this reports ``"msvc"`` and the real
    ``cl``/``link`` rather than whatever happens to be on ``PATH``.
    """
    try:
        from setuptools._distutils.ccompiler import (  # ruff:ignore[import-outside-top-level]
            new_compiler,
        )
        from setuptools._distutils.sysconfig import (  # ruff:ignore[import-outside-top-level]
            customize_compiler,
        )

        comp = new_compiler()
        ctype = getattr(comp, "compiler_type", "unknown") or "unknown"
        try:  # ruff:ignore[suppressible-exception]
            customize_compiler(comp)
        except Exception:  # noqa: BLE001 - customization is best-effort
            pass
        # Unix-like compilers expose command lists; MSVC exposes attributes only
        # after initialize(), which we avoid (it can spawn vcvars).  Fall back to
        # the type alone when executables aren't statically available.
        cc = _first_token(
            getattr(comp, "compiler_so", None) or getattr(comp, "cc", None)
        )
        cxx = _first_token(
            getattr(comp, "compiler_cxx", None) or getattr(comp, "cxx", None)
        )
        linker = _first_token(getattr(comp, "linker_so", None))
        return ResolvedToolchain(compiler_type=ctype, cc=cc, cxx=cxx, linker=linker)
    except Exception:  # noqa: BLE001 - never let detection break a build
        return ResolvedToolchain("unknown", "", "", "")


class UnsupportedRuntimeError(RuntimeError):
    """Raised when a runtime lifecycle operation is not supported (CYTHON-ABI-001)."""


class RuntimeCapabilities(NamedTuple):
    """Declared contract for native-extension runtime lifecycle (CYTHON-ABI-001).

    Native extensions cannot generally be unloaded safely, and behavior under a
    free-threaded (GIL-disabled) interpreter, a non-main subinterpreter, or
    after ``fork`` is not universally guaranteed.  Rather than let these
    manifest as opaque crashes or silent state loss, this contract declares them
    explicitly so callers (and :func:`check_runtime_supported`) can branch and
    fail fast with a clear message.

    Parameters
    ----------
    gil_enabled : bool
        Whether the GIL is currently enabled.  ``False`` on a free-threaded
        (``Py_GIL_DISABLED``) build — where thread-safety of arbitrary
        extensions is not guaranteed.
    free_threaded_build : bool
        Whether CPython was built free-threaded (``Py_GIL_DISABLED``).
    in_main_interpreter : bool
        Whether the current interpreter is the main one.  Extensions without
        per-interpreter GIL support must not run in a subinterpreter.
    supports_unload : bool
        Whether a loaded extension can be safely unloaded/replaced.  Always
        ``False``: CPython cannot generally unload native modules.
    supports_fork_after_load : bool
        Whether ``fork`` after loading an extension is supported (POSIX only;
        the child must not rely on threads created pre-fork).
    platform : str
        The value of :data:`sys.platform`.

    See Also
    --------
    runtime_capabilities : Construct the contract for the current process.
    check_runtime_supported : Fail fast on an unsupported configuration.
    """

    gil_enabled: bool
    free_threaded_build: bool
    in_main_interpreter: bool
    supports_unload: bool
    supports_fork_after_load: bool
    platform: str


def runtime_capabilities() -> RuntimeCapabilities:
    """Return the :class:`RuntimeCapabilities` contract for this process."""
    import sysconfig  # noqa: PLC0415

    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    # Best-effort main-interpreter check: interpreter id 0 is the main one on
    # CPython; absence of the API implies the (single) main interpreter.
    try:
        import _interpreters  # pyright: ignore[reportMissingImports] # noqa: PLC0415

        in_main = _interpreters.get_current() == _interpreters.get_main()
    except Exception:  # noqa: BLE001 - no subinterpreter API => main interpreter
        in_main = True
    return RuntimeCapabilities(
        gil_enabled=bool(gil_enabled),
        free_threaded_build=free_threaded,
        in_main_interpreter=bool(in_main),
        supports_unload=False,  # CPython cannot generally unload native modules
        supports_fork_after_load=hasattr(os, "fork"),
        platform=sys.platform,
    )


def check_runtime_supported(
    *,
    allow_free_threaded: bool = False,
    allow_subinterpreter: bool = False,
) -> None:
    """Fail fast on an unsupported runtime configuration (CYTHON-ABI-001).

    Safe default: raise :class:`UnsupportedRuntimeError` on a free-threaded
    interpreter or in a non-main subinterpreter, because arbitrary compiled
    extensions do not universally support those.  Callers who have verified
    their extensions can opt in via the ``allow_*`` flags.

    Parameters
    ----------
    allow_free_threaded : bool, default=False
        Permit running under a free-threaded (GIL-disabled) interpreter.
    allow_subinterpreter : bool, default=False
        Permit running in a non-main subinterpreter.

    Raises
    ------
    UnsupportedRuntimeError
        If the current runtime is unsupported and not explicitly allowed.
    """
    caps = runtime_capabilities()
    if caps.free_threaded_build and not allow_free_threaded:
        raise UnsupportedRuntimeError(
            "runtime compilation/import on a free-threaded (GIL-disabled) "
            "CPython is not supported by default because arbitrary compiled "
            "extensions may not be thread-safe; pass allow_free_threaded=True "
            "if your extensions are verified free-thread-safe."
        )
    if not caps.in_main_interpreter and not allow_subinterpreter:
        raise UnsupportedRuntimeError(
            "runtime compilation/import in a non-main subinterpreter is not "
            "supported by default because extensions without per-interpreter "
            "GIL support cannot run there; pass allow_subinterpreter=True if "
            "your extensions declare multi-interpreter support."
        )


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
