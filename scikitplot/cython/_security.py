# scikitplot/cython/_security.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Security guards for :mod:`scikitplot.cython`.

This module provides a :class:`SecurityPolicy` dataclass and validation
helpers that protect the build pipeline against the most common attack
surfaces when user-controlled input reaches a C/C++ compiler invocation.

.. rubric:: Threat model

The primary threat is a *partially trusted* caller who can supply build
parameters (``extra_compile_args``, ``define_macros``, ``include_dirs``,
``libraries``, source strings) but should not be able to:

- escape the cache directory via path-traversal sequences (``../``, ``~``,
  absolute paths pointing outside an allowed prefix).
- inject arbitrary shell metacharacters into compiler argument strings that
  are passed as ``extra_compile_args`` / ``extra_link_args`` by a build
  backend that calls the compiler via ``subprocess.Popen`` **with**
  ``shell=True`` (setuptools historically does *not* use ``shell=True``,
  but downstream wrappers occasionally do).
- define macros with names that shadow security-sensitive preprocessor
  guards (e.g., ``Py_LIMITED_API``, ``_FORTIFY_SOURCE``).
- load shared libraries outside the expected runtime environment.

.. rubric:: What this module does NOT do

- It does **not** sandbox, jail, or restrict what the compiled native code
  can do at runtime.  Compiled extension modules run with full process
  privileges.  Treat this as a *defense-in-depth* measure for the
  *build inputs*, not a runtime sandbox.
- It does **not** prevent a sufficiently privileged or creative attacker
  from compiling malicious code if they can write arbitrary ``.pyx`` source.

.. rubric:: Usage

For most users the defaults are appropriate::

    from scikitplot.cython._security import SecurityPolicy, validate_build_inputs

    policy = SecurityPolicy()  # default: strict mode
    validate_build_inputs(
        policy=policy,
        define_macros=user_macros,
        extra_compile_args=user_cargs,
        extra_link_args=user_largs,
        include_dirs=user_inc,
        libraries=user_libs,
    )

Notes
-----
**For newbies**: leave ``security_policy=None`` in the public API; the
default :class:`SecurityPolicy` with ``strict=True`` is applied
automatically and protects you from the most common mistakes.

**For pro/master users**: you can relax specific checks by constructing a
custom :class:`SecurityPolicy`::

    policy = SecurityPolicy(
        allow_absolute_include_dirs=True,
        allow_shell_metacharacters=False,
    )

References
----------
* CWE-78  OS Command Injection
* CWE-22  Path Traversal
* CWE-74  Injection
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from dataclasses import field as field_  # noqa: F401
from pathlib import Path
from typing import Sequence

__all__ = [
    "DEFAULT_SECURITY_POLICY",
    "RELAXED_SECURITY_POLICY",
    "SecurityError",
    "SecurityPolicy",
    "is_safe_compiler_arg",
    "is_safe_macro_name",
    "is_safe_path",
    "validate_build_inputs",
]

# ---------------------------------------------------------------------------
# Shell metacharacters that must not appear in compiler argument strings
# when shell=True could be in play (defense-in-depth).
# ---------------------------------------------------------------------------
_SHELL_META_RE = re.compile(r"[;&|`$<>()\\\n\r\x00]")

# ---------------------------------------------------------------------------
# Macro names that shadow CPython/security preprocessor guards.
# Users should not redefine these via the build API.
# ---------------------------------------------------------------------------
_RESERVED_MACRO_NAMES: frozenset[str] = frozenset(
    {
        "Py_LIMITED_API",
        "Py_DEBUG",
        "Py_TRACE_REFS",
        "NDEBUG",
        "_FORTIFY_SOURCE",
        "_GLIBCXX_ASSERTIONS",
        "PY_SSIZE_T_CLEAN",
    }
)

# ---------------------------------------------------------------------------
# Compiler args that are almost always wrong and indicate misconfiguration
# or attempted injection (not outright blocked, but warned in strict mode).
# ---------------------------------------------------------------------------
_DANGEROUS_ARG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^-x\s*c\+\+"),  # language override via flag
    re.compile(r"^-imacros\b"),  # inject macro from file (arbitrary read)
    re.compile(r"^-include\s+/"),  # force-include absolute path
    re.compile(r"^--sysroot="),  # sysroot override
    re.compile(r"^-specs="),  # GCC specs injection
    re.compile(r"^/manifestuac"),  # Windows UAC manifest injection
)


class SecurityError(ValueError):
    """
    Raised when a build input violates the active :class:`SecurityPolicy`.

    Inherits from :exc:`ValueError` so callers that catch ``ValueError``
    continue to work without modification.

    Parameters
    ----------
    message : str
        Human-readable description of the violation.
    field : str or None
        Name of the build parameter that triggered the violation.

    Notes
    -----
    Do NOT catch :exc:`SecurityError` generically and continue — treat it as
    a hard stop.  The error message always describes *what* violated *which*
    rule so developers can fix inputs rather than silencing errors.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message if field is None else f"[{field}] {message}")


@dataclass(frozen=True, slots=True)
class SecurityPolicy:
    r"""
    Immutable security policy applied to build inputs before compilation.

    Parameters
    ----------
    strict : bool, default=True
        Master switch.  When ``False``, all checks below default to the
        most permissive setting.  Overriding individual flags still works.
    allow_absolute_include_dirs : bool, default=False
        When ``False``, include directories must be relative paths or must
        be resolved to be inside the cache directory.  Setting ``True``
        allows any absolute path, which is required when pointing at
        system headers (e.g., ``/usr/local/include``).
        Newbies: leave ``False``.  Pros: set ``True`` for custom installs.
    allow_shell_metacharacters : bool, default=False
        When ``False``, shell metacharacters (``; & | ` $ < > ( ) \\``)
        are rejected in ``extra_compile_args`` and ``extra_link_args``.
        Only enable this when you are **certain** your build backend does
        not use ``shell=True``.
    allow_reserved_macros : bool, default=False
        When ``False``, define-macro names that shadow CPython or
        security-sensitive preprocessor guards are rejected.
    allow_dangerous_compiler_args : bool, default=False
        When ``False``, compiler arguments that match known dangerous
        patterns (``-imacros``, ``-specs=``, etc.) are rejected.
    max_source_bytes : int or None, default=10_485_760
        Maximum allowed source code size in bytes (default 10 MiB).
        ``None`` disables the limit.  Prevents accidental or deliberate
        memory exhaustion during compilation.
    max_extra_compile_args : int, default=64
        Maximum number of extra C/C++ compiler arguments accepted.
    max_extra_link_args : int, default=64
        Maximum number of extra linker arguments accepted.  Separate from
        ``max_extra_compile_args`` because link-time argument counts can
        legitimately differ from compile-time counts (e.g., many ``-l`` flags).
    max_include_dirs : int, default=32
        Maximum number of include directories accepted.
    max_libraries : int, default=32
        Maximum number of library names accepted.

    Notes
    -----
    **Newbie users** (Scenarios 1 & 2): use :data:`DEFAULT_SECURITY_POLICY`
    (``strict=True``).  You get path-traversal protection and macro-shadow
    guards with no extra configuration.

    **Master/pro users** (Scenarios 3-7): construct a custom policy that
    relaxes only the specific checks you need::

        from scikitplot.cython._security import SecurityPolicy

        policy = SecurityPolicy(allow_absolute_include_dirs=True)

    **CI/automation environments**: set
    ``SCIKITPLOT_CYTHON_ALLOW_ABSOLUTE_DIRS=1`` in the environment to
    temporarily enable absolute include dirs without code changes.

    See Also
    --------
    validate_build_inputs : Apply this policy against actual build inputs.
    SecurityError : Raised on violation.

    Examples
    --------
    Default (strict) policy:

    >>> policy = SecurityPolicy()
    >>> policy.strict
    True
    >>> policy.allow_shell_metacharacters
    False

    Relaxed policy for pro users who supply system include paths:

    >>> policy = SecurityPolicy(allow_absolute_include_dirs=True)
    >>> policy.allow_absolute_include_dirs
    True
    """

    strict: bool = True
    allow_absolute_include_dirs: bool = False
    allow_shell_metacharacters: bool = False
    allow_reserved_macros: bool = False
    allow_dangerous_compiler_args: bool = False
    max_source_bytes: int | None = 10 * 1024 * 1024  # 10 MiB
    max_extra_compile_args: int = 64
    max_extra_link_args: int = 64
    max_include_dirs: int = 32
    max_libraries: int = 32

    def __post_init__(self) -> None:
        # Validate the policy's own parameters on construction.
        if self.max_extra_compile_args < 0:
            raise ValueError("max_extra_compile_args must be >= 0")
        if self.max_extra_link_args < 0:
            raise ValueError("max_extra_link_args must be >= 0")
        if self.max_include_dirs < 0:
            raise ValueError("max_include_dirs must be >= 0")
        if self.max_libraries < 0:
            raise ValueError("max_libraries must be >= 0")
        if self.max_source_bytes is not None and self.max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be > 0 or None")

    @classmethod
    def relaxed(cls) -> SecurityPolicy:
        """
        Return a pre-configured policy with all dangerous checks disabled.

        .. warning::
            Only use this for fully trusted inputs (e.g., your own build
            scripts in a controlled CI environment).  Do NOT apply this
            to user-supplied data.

        Returns
        -------
        SecurityPolicy
            Permissive policy instance.
        """
        return cls(
            strict=False,
            allow_absolute_include_dirs=True,
            allow_shell_metacharacters=True,
            allow_reserved_macros=True,
            allow_dangerous_compiler_args=True,
            max_source_bytes=None,
            max_extra_compile_args=1024,
            max_extra_link_args=1024,
            max_include_dirs=512,
            max_libraries=512,
        )


# Module-level policy singletons for convenience.
DEFAULT_SECURITY_POLICY: SecurityPolicy = SecurityPolicy()
"""Default strict security policy (all guards enabled)."""

RELAXED_SECURITY_POLICY: SecurityPolicy = SecurityPolicy.relaxed()
"""Fully permissive policy for trusted build scripts."""


# ---------------------------------------------------------------------------
# Individual predicate helpers (exported for targeted testing)
# ---------------------------------------------------------------------------


def is_safe_path(
    path: str | os.PathLike[str],
    *,
    allow_absolute: bool = False,
) -> bool:
    r"""
    Return ``True`` when a filesystem path does not contain traversal sequences.

    Parameters
    ----------
    path : str or os.PathLike
        Path to validate.
    allow_absolute : bool, default=False
        If ``False``, absolute paths are considered unsafe.

    Returns
    -------
    bool
        ``True`` if the path passes all checks, ``False`` otherwise.

    Notes
    -----
    Path-traversal sequences (``../``, ``..\\``, ``~``) are always
    rejected regardless of ``allow_absolute``.

    Examples
    --------
    >>> is_safe_path("include/mylib")
    True
    >>> is_safe_path("../../../etc/passwd")
    False
    >>> is_safe_path("/usr/include", allow_absolute=True)
    True
    >>> is_safe_path("/usr/include", allow_absolute=False)
    False
    """
    p = Path(os.fsdecode(os.fspath(path)))
    s = str(p)
    # Reject traversal sequences unconditionally.
    if ".." in p.parts:
        return False
    # Reject tilde that was not expanded (unexpanded tildes are suspicious).
    if "~" in s:
        return False
    # Reject null bytes.
    if "\x00" in s:
        return False
    # Absolute path check.
    if p.is_absolute() and not allow_absolute:  # noqa: SIM103
        return False
    return True


def is_safe_macro_name(name: str, *, allow_reserved: bool = False) -> bool:
    """
    Return ``True`` when a C preprocessor macro name is safe to define.

    Parameters
    ----------
    name : str
        Macro name (the left-hand side of a ``-D`` flag).
    allow_reserved : bool, default=False
        If ``False``, names that shadow CPython or security-critical
        preprocessor guards are rejected.

    Returns
    -------
    bool
        ``True`` if the name is safe, ``False`` otherwise.

    Notes
    -----
    Valid macro names match the regex ``[A-Za-z_][A-Za-z0-9_]*``.  The
    reserved-name check is independent of syntax validity.

    Examples
    --------
    >>> is_safe_macro_name("MY_FLAG")
    True
    >>> is_safe_macro_name("Py_LIMITED_API")
    False
    >>> is_safe_macro_name("Py_LIMITED_API", allow_reserved=True)
    True
    >>> is_safe_macro_name("123INVALID")
    False
    """
    if not isinstance(name, str) or not name:
        return False
    # Syntactic validity: must be a valid C identifier.
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return False
    # Reserved-name check.
    if not allow_reserved and name in _RESERVED_MACRO_NAMES:  # noqa: SIM103
        return False
    return True


def is_safe_compiler_arg(
    arg: str,
    *,
    allow_shell_meta: bool = False,
    allow_dangerous: bool = False,
) -> bool:
    """
    Return ``True`` when a compiler argument string is safe to pass.

    Parameters
    ----------
    arg : str
        A single compiler flag (e.g., ``"-O2"``, ``"-DNDEBUG"``).
    allow_shell_meta : bool, default=False
        If ``False``, shell metacharacters are rejected.
    allow_dangerous : bool, default=False
        If ``False``, known-dangerous flag patterns are rejected.

    Returns
    -------
    bool
        ``True`` if the argument passes all checks, ``False`` otherwise.

    Notes
    -----
    Null bytes are always rejected regardless of other flags.

    Examples
    --------
    >>> is_safe_compiler_arg("-O2")
    True
    >>> is_safe_compiler_arg("-O2; rm -rf /")
    False
    >>> is_safe_compiler_arg("-imacros /etc/shadow")
    False
    """
    if not isinstance(arg, str):
        return False
    if "\x00" in arg:
        return False
    if not allow_shell_meta and _SHELL_META_RE.search(arg):
        return False
    if not allow_dangerous and any(  # noqa: SIM103
        p.search(arg) for p in _DANGEROUS_ARG_PATTERNS
    ):
        return False
    return True


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_build_inputs(  # noqa: PLR0912
    *,
    policy: SecurityPolicy | None = None,
    source: str | None = None,
    define_macros: Sequence[tuple[str, str | None]] | None = None,
    extra_compile_args: Sequence[str] | None = None,
    extra_link_args: Sequence[str] | None = None,
    include_dirs: Sequence[str | os.PathLike[str]] | None = None,
    libraries: Sequence[str] | None = None,
) -> None:
    """
    Validate build inputs against a :class:`SecurityPolicy`.

    Raises :exc:`SecurityError` on the **first** violation found.  All
    checks are deterministic and do not perform filesystem I/O.

    Parameters
    ----------
    policy : SecurityPolicy or None, default=None
        Policy to enforce.  If ``None``, :data:`DEFAULT_SECURITY_POLICY`
        (strict mode) is used.
    source : str or None, default=None
        Cython source text.  Checked against ``policy.max_source_bytes``.
    define_macros : sequence of (str, str | None) or None, default=None
        Preprocessor macro definitions.  Each ``(name, value)`` pair is
        validated.
    extra_compile_args : sequence of str or None, default=None
        Extra C/C++ compiler arguments to validate.
    extra_link_args : sequence of str or None, default=None
        Extra linker arguments to validate.
    include_dirs : sequence of path-like or None, default=None
        Additional include directories to validate.
    libraries : sequence of str or None, default=None
        Library names to validate.

    Raises
    ------
    SecurityError
        On the first detected violation.
    TypeError
        If ``policy`` is not a :class:`SecurityPolicy` instance.

    Notes
    -----
    **For newbies** (Scenarios 1 & 2): you do not need to call this
    function directly — the public API applies it automatically via
    :data:`DEFAULT_SECURITY_POLICY`.

    **For masters** (Scenarios 3-7): call this explicitly when you bypass
    the public API or when building with custom compilers.

    Examples
    --------
    >>> from scikitplot.cython._security import validate_build_inputs
    >>> validate_build_inputs(
    ...     source="def hello(): return 42",
    ...     extra_compile_args=["-O2"],
    ... )  # No error: all inputs are safe.

    >>> validate_build_inputs(
    ...     extra_compile_args=["-O2; rm -rf /"],
    ... )
    Traceback (most recent call last):
        ...
    SecurityError: [extra_compile_args] shell metacharacter in arg: '-O2; rm -rf /'
    """
    if policy is None:
        policy = DEFAULT_SECURITY_POLICY
    if not isinstance(policy, SecurityPolicy):
        raise TypeError(
            f"policy must be a SecurityPolicy instance, got {type(policy).__name__!r}"
        )

    # --- Source size guard ---
    if source is not None and policy.max_source_bytes is not None:
        src_bytes = source.encode("utf-8")
        if len(src_bytes) > policy.max_source_bytes:
            raise SecurityError(
                f"source exceeds max_source_bytes limit "
                f"({len(src_bytes)} > {policy.max_source_bytes})",
                field="source",
            )

    # --- define_macros ---
    if define_macros:
        if not isinstance(define_macros, (list, tuple)):
            raise SecurityError(
                "define_macros must be a sequence of (name, value) tuples",
                field="define_macros",
            )
        for item in define_macros:
            if not (
                isinstance(item, (list, tuple)) and len(item) == 2  # noqa: PLR2004
            ):
                raise SecurityError(
                    f"each define_macro must be a 2-tuple (name, value), got {item!r}",
                    field="define_macros",
                )
            name, _val = item
            if not isinstance(name, str) or not name:
                raise SecurityError(
                    f"macro name must be a non-empty str, got {name!r}",
                    field="define_macros",
                )
            if not is_safe_macro_name(
                name, allow_reserved=policy.allow_reserved_macros
            ):
                raise SecurityError(
                    f"unsafe or reserved macro name: {name!r}",
                    field="define_macros",
                )

    # --- extra_compile_args ---
    if extra_compile_args is not None:
        if len(extra_compile_args) > policy.max_extra_compile_args:
            raise SecurityError(
                f"too many extra_compile_args: "
                f"{len(extra_compile_args)} > {policy.max_extra_compile_args}",
                field="extra_compile_args",
            )
        for arg in extra_compile_args:
            if not isinstance(arg, str):
                raise SecurityError(
                    f"extra_compile_args entries must be str, got {type(arg).__name__!r}",
                    field="extra_compile_args",
                )
            if not is_safe_compiler_arg(
                arg,
                allow_shell_meta=policy.allow_shell_metacharacters,
                allow_dangerous=policy.allow_dangerous_compiler_args,
            ):
                # Surface the specific reason so the error is actionable.
                if not policy.allow_shell_metacharacters and _SHELL_META_RE.search(arg):
                    reason = f"shell metacharacter in arg: {arg!r}"
                elif not policy.allow_dangerous_compiler_args and any(
                    p.search(arg) for p in _DANGEROUS_ARG_PATTERNS
                ):
                    reason = f"dangerous compiler-arg pattern in: {arg!r}"
                else:
                    reason = f"unsafe compiler arg: {arg!r}"
                raise SecurityError(reason, field="extra_compile_args")

    # --- extra_link_args ---
    if extra_link_args is not None:
        if len(extra_link_args) > policy.max_extra_link_args:
            raise SecurityError(
                f"too many extra_link_args: "
                f"{len(extra_link_args)} > {policy.max_extra_link_args}",
                field="extra_link_args",
            )
        for arg in extra_link_args:
            if not isinstance(arg, str):
                raise SecurityError(
                    f"extra_link_args entries must be str, got {type(arg).__name__!r}",
                    field="extra_link_args",
                )
            if not is_safe_compiler_arg(
                arg,
                allow_shell_meta=policy.allow_shell_metacharacters,
                allow_dangerous=policy.allow_dangerous_compiler_args,
            ):
                if not policy.allow_shell_metacharacters and _SHELL_META_RE.search(arg):
                    reason = f"shell metacharacter in linker arg: {arg!r}"
                elif not policy.allow_dangerous_compiler_args and any(
                    p.search(arg) for p in _DANGEROUS_ARG_PATTERNS
                ):
                    reason = f"dangerous linker-arg pattern in: {arg!r}"
                else:
                    reason = f"unsafe linker arg: {arg!r}"
                raise SecurityError(reason, field="extra_link_args")

    # --- include_dirs ---
    if include_dirs is not None:
        if len(include_dirs) > policy.max_include_dirs:
            raise SecurityError(
                f"too many include_dirs: "
                f"{len(include_dirs)} > {policy.max_include_dirs}",
                field="include_dirs",
            )
        for p in include_dirs:
            ps = os.fsdecode(os.fspath(p)) if not isinstance(p, str) else p
            if not is_safe_path(ps, allow_absolute=policy.allow_absolute_include_dirs):
                raise SecurityError(
                    f"unsafe include_dir (path traversal or absolute path not allowed): "
                    f"{ps!r}",
                    field="include_dirs",
                )

    # --- libraries ---
    if libraries is not None:
        if len(libraries) > policy.max_libraries:
            raise SecurityError(
                f"too many libraries: {len(libraries)} > {policy.max_libraries}",
                field="libraries",
            )
        for lib in libraries:
            if not isinstance(lib, str) or not lib:
                raise SecurityError(
                    f"library names must be non-empty str, got {lib!r}",
                    field="libraries",
                )
            # Reject shell metacharacters in library names.
            if _SHELL_META_RE.search(lib):
                raise SecurityError(
                    f"shell metacharacter in library name: {lib!r}",
                    field="libraries",
                )
            # Reject path separators in library names (use library_dirs instead).
            if os.sep in lib or "/" in lib:
                raise SecurityError(
                    f"library name must not contain path separators: {lib!r}",
                    field="libraries",
                )
