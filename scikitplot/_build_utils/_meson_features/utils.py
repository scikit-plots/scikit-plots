# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Copyright (c) 2023, NumPy Developers.
#
# ============================================================================
# COMPATIBILITY NOTICE — stock mesonbuild vs. numpy/meson fork
# ============================================================================
#
# Origin
# ------
# This file originates from the NumPy project's vendored Meson fork:
#   https://github.com/numpy/meson/tree/main-numpymeson/mesonbuild/modules/features
#
# That fork predates a breaking API change that was later adopted by upstream
# (stock) mesonbuild.  The single fix in this file brings test_code() into
# alignment with stock meson >= ~1.5.0.
#
# Breaking API change fixed here
# --------------------------------
# Compiler.cached_compile(code, *, extra_args, ...)
#
# numpy/meson fork — original positional signature:
#   cached_compile(code, coredata, extra_args=args)
#   ^                   ^^^^^^^^^
#   code is the source  coredata was passed explicitly so the method could
#   look up and update the compile-result cache stored in CoreData.
#
# Stock meson >= ~1.5.0 — keyword-only signature:
#   cached_compile(code, *, extra_args=None, mode=CompileCheckMode.LINK,
#                  temp_dir=None) -> Iterator[CompileResult]
#   ^                  ^^
#   coredata removed   all remaining params are now keyword-only
#
# Why upstream removed coredata
# ------------------------------
# The numpy/meson fork was written when Compiler instances were short-lived
# helpers that did not hold a reference to their own build environment.
# Callers had to supply coredata explicitly so the method could reach the
# shared compile-result cache.
#
# Stock meson later introduced a stable self.environment attribute on every
# Compiler instance, set once during compiler detection in environment.py
# (detect_compiler_for / _detect_c_or_cpp_compiler).  From that point on,
# every method that previously needed coredata could simply use
# self.environment.coredata internally, making the external parameter
# redundant.
#
# The remaining keyword-only parameters (mode, temp_dir) and the context
# manager protocol (yields CompileResult) are unchanged.  The CompileResult
# attributes used by this package (.cached, .returncode, .stderr) are also
# unchanged in stock meson 1.10.1.
#
# What the fix is
# ---------------
# Drop  state.environment.coredata  from the call in test_code().
# That is the only change.  All return-value handling is identical.
#
# How to verify after a meson upgrade
# ------------------------------------
# Run:
#   python -c "
#   import inspect
#   from mesonbuild.compilers.compilers import Compiler
#   print(inspect.signature(Compiler.cached_compile))
#   "
# Expected (stock meson 1.10.1):
#   (self, code, *, extra_args=None, mode=<CompileCheckMode.LINK: 'link'>,
#    temp_dir=None) -> Iterator[CompileResult]
#
# If a future meson version reintroduces coredata (unlikely) or renames
# extra_args, update test_code() below and record the change in this header.
# ============================================================================
#
# ============================================================================
# PYTHON VERSION COMPATIBILITY
# ============================================================================
#
# from __future__ import annotations  [Python 3.8 – 3.15+]
# Makes all annotations strings at runtime (PEP 563 / 749).
# The meson decorator system uses KwargInfo runtime objects for type
# checking — not Python annotations — so this is safe.
# ============================================================================
from __future__ import annotations

import hashlib
from typing import Tuple, List, Union, Any, TYPE_CHECKING
from ...mesonlib import MesonException, MachineChoice # type: ignore[]

if TYPE_CHECKING:
    from ...compilers import Compiler # type: ignore[]
    from ...mesonlib import File # type: ignore[]
    from .. import ModuleState


def get_compiler(state: 'ModuleState') -> 'Compiler':
    for_machine = MachineChoice.HOST
    clist = state.environment.coredata.compilers[for_machine]
    for cstr in ('c', 'cpp'):
        try:
            compiler = clist[cstr]
            break
        except KeyError:
            raise MesonException(
                'Unable to get compiler for C or C++ language '
                'try to specify a valid C/C++ compiler via option "compiler".'
            )
    return compiler


def test_code(state: 'ModuleState', compiler: 'Compiler',
              args: List[str], code: 'Union[str, File]'
              ) -> Tuple[bool, bool, str]:
    # TODO: Add option to treat warnings as errors
    #
    # COMPAT FIX [stock meson >= ~1.5.0]
    #
    # numpy/meson fork — original call:
    #   compiler.cached_compile(code, state.environment.coredata, extra_args=args)
    #                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                 This positional arg was removed upstream.
    #
    # Stock meson no longer accepts coredata as a positional argument because
    # the compiler resolves it internally via self.environment.coredata.
    # Passing it causes:
    #   TypeError: cached_compile() takes 2 positional arguments but 3 were given
    #
    # The context manager still yields CompileResult.  The attributes we use:
    #   p.cached      -> bool  (True if result was served from cache)
    #   p.returncode  -> int   (0 = compilation succeeded)
    #   p.stderr      -> str   (compiler diagnostic output)
    # are defined identically in stock meson 1.10.1.
    with compiler.cached_compile(
        code, extra_args=args
    ) as p:
        return p.cached, p.returncode == 0, p.stderr


def generate_hash(*args: Any) -> str:
    # usedforsecurity=False was added in Python 3.9 (PEP 644).
    # On FIPS-enforced systems (e.g. RHEL/Fedora in FIPS mode), calling
    # hashlib.sha1() without usedforsecurity=False raises:
    #   ValueError: [digital envelope routines] unsupported
    # because SHA-1 is prohibited for security-sensitive use cases, but
    # the FIPS provider still allows it for non-security uses (checksums,
    # cache keys) when the flag is explicitly False.
    #
    # We use a try/except for Python 3.8 compatibility: Python 3.8 does
    # not accept the keyword argument and raises TypeError, so we fall
    # back to the plain call which works on all non-FIPS 3.8 systems.
    try:
        hasher = hashlib.sha1(usedforsecurity=False)   # Python 3.9+ safe
    except TypeError:
        hasher = hashlib.sha1()                        # Python 3.8 fallback
    for a in args:
        hasher.update(bytes(str(a), encoding='utf-8'))
    return hasher.hexdigest()
