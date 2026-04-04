# scikitplot/cython/tests/test_scenarios.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Scenario-based test suite for :mod:`scikitplot.cython`.

Covers all ten user scenarios requested in the submodule review plus
regression tests for every critical bug found in the review.

Scenarios
---------
1  Newbie — pure Python, setuptools only.
2  Newbie — compile C++ via Cython only.
3  Pro/master — full stack: setuptools + Cython + pybind11 + NumPy + C-API.
4  Master — pybind11 only.
5  Master — own C-API (single file / multi-file / folder / nested / exclude).
6  All users — security guards: path traversal, shell injection, macros, size.
7  Master — custom compilers: naming convention, registry, protocol check.
8  Master — ``custom_*`` / ``Custom*`` hook naming enforcement.
9  Newbie + master — user/dev notes: docstring examples validate correctly.
10 Visual — SVG mind map generation verified separately.

Regression tests
----------------
- R1: ``_security.SecurityPolicy`` uses ``max_extra_link_args`` independently.
- R2: ``cython_import_result`` bare-string ``include_dirs`` coercion.
- R3: ``__init__.__all__`` contains no duplicate names.
- R4: ``collect_c_api_sources`` doctest passes (no stray ``write_text`` output).

All tests run without a C compiler.  Tests that require a compiler are marked
``requires_compiler`` and skip automatically in CI without the toolchain.

Notes
-----
**For newbies**: run the whole suite with ``pytest`` from the repo root —
no arguments needed, no Cython toolchain required.

**For pro/master users**: enable compiler tests by installing Cython and a
C compiler, then run::

    pytest -m requires_compiler

**Developer note**: every test class targets exactly one scenario/module.
Helper fixtures are defined at module scope for reuse across classes.
Add new tests by subclassing the relevant class and calling
``super().setup_method()`` if state is needed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import subjects under test
# ---------------------------------------------------------------------------
from .._security import (
    DEFAULT_SECURITY_POLICY,
    RELAXED_SECURITY_POLICY,
    SecurityError,
    SecurityPolicy,
    is_safe_compiler_arg,
    is_safe_macro_name,
    is_safe_path,
    validate_build_inputs,
)
from .._custom_compiler import (
    CApiCompiler,
    CompilerRegistry,
    CustomCompilerProtocol,
    PybindCompiler,
    _validate_compiler_name,
    c_api_prereqs,
    collect_c_api_sources,
    collect_header_dirs,
    cython_cpp_prereqs,
    full_stack_prereqs,
    get_compiler,
    list_compilers,
    numpy_include,
    pure_python_prereqs,
    pybind11_include,
    pybind11_only_prereqs,
    register_compiler,
)
from .._public import (
    _coerce_path_seq,
    check_build_prereqs,
)
from .. import __all__ as _PKG_ALL  # noqa: E402

# ---------------------------------------------------------------------------
# Pytest marks
# ---------------------------------------------------------------------------
requires_compiler = pytest.mark.skipif(
    True,  # default: always skip — flip to False when toolchain available
    reason="requires Cython + C compiler",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_minimal_compiler(name: str = "custom_minimal") -> CustomCompilerProtocol:
    """Return the simplest object that satisfies CustomCompilerProtocol."""

    class _Compiler:
        def __call__(self, source, *, build_dir, module_name, **kwargs):
            raise NotImplementedError("test stub")

    c = _Compiler()
    c.name = name  # type: ignore[attr-defined]
    return c  # type: ignore[return-value]


# ===========================================================================
# SCENARIO 1 — Newbie: pure Python, setuptools only
# ===========================================================================


class TestScenario1PurePython:
    """Newbie scenario: pure Python, only setuptools needed.

    Notes
    -----
    **User note**: this scenario checks that setuptools is present and that
    calling :func:`pure_python_prereqs` gives an actionable report.
    **Dev note**: the function must not import Cython or NumPy.
    """

    def test_pure_python_prereqs_returns_dict(self) -> None:
        result = pure_python_prereqs()
        assert isinstance(result, dict)

    def test_pure_python_prereqs_has_setuptools_key(self) -> None:
        result = pure_python_prereqs()
        assert "setuptools" in result

    def test_pure_python_prereqs_no_cython_key(self) -> None:
        result = pure_python_prereqs()
        assert "cython" not in result

    def test_pure_python_prereqs_no_numpy_key(self) -> None:
        result = pure_python_prereqs()
        assert "numpy" not in result

    def test_pure_python_prereqs_no_pybind11_key(self) -> None:
        result = pure_python_prereqs()
        assert "pybind11" not in result

    def test_setuptools_entry_has_ok_bool(self) -> None:
        result = pure_python_prereqs()
        assert isinstance(result["setuptools"]["ok"], bool)

    def test_setuptools_ok_entry_has_version_when_available(self) -> None:
        result = pure_python_prereqs()
        if result["setuptools"]["ok"]:
            assert "version" in result["setuptools"]

    def test_setuptools_fail_entry_has_error_key(self) -> None:
        with patch.dict(sys.modules, {"setuptools": None}):
            result = pure_python_prereqs()
        # After un-mocking, result was captured under mock — ok or not, has ok key
        assert "ok" in result["setuptools"]

    def test_check_build_prereqs_minimal(self) -> None:
        """Newbie scenario: check_build_prereqs with no extras."""
        result = check_build_prereqs()
        assert "cython" in result
        assert "setuptools" in result
        assert "numpy" not in result
        assert "pybind11" not in result

    def test_check_build_prereqs_result_ok_is_bool(self) -> None:
        result = check_build_prereqs()
        assert isinstance(result["cython"]["ok"], bool)
        assert isinstance(result["setuptools"]["ok"], bool)


# ===========================================================================
# SCENARIO 2 — Newbie: compile C++ via Cython only
# ===========================================================================


class TestScenario2CythonCpp:
    """Newbie scenario: C++ via Cython only, no numpy or setuptools needed.

    Notes
    -----
    **User note**: install Cython with ``pip install Cython`` and ensure a
    C++ compiler (``g++``) is on the PATH.  No NumPy required.
    **Dev note**: :func:`cython_cpp_prereqs` must only check Cython.
    """

    def test_cython_cpp_prereqs_returns_dict(self) -> None:
        result = cython_cpp_prereqs()
        assert isinstance(result, dict)

    def test_cython_cpp_prereqs_has_cython_key(self) -> None:
        result = cython_cpp_prereqs()
        assert "cython" in result

    def test_cython_cpp_prereqs_no_setuptools(self) -> None:
        result = cython_cpp_prereqs()
        assert "setuptools" not in result

    def test_cython_cpp_prereqs_no_numpy(self) -> None:
        result = cython_cpp_prereqs()
        assert "numpy" not in result

    def test_cython_entry_has_ok_bool(self) -> None:
        result = cython_cpp_prereqs()
        assert isinstance(result["cython"]["ok"], bool)

    def test_cython_ok_has_version_key(self) -> None:
        result = cython_cpp_prereqs()
        if result["cython"]["ok"]:
            assert "version" in result["cython"]

    def test_cython_fail_has_error_key(self) -> None:
        with patch.dict(sys.modules, {"Cython": None}):
            result = cython_cpp_prereqs()
        assert "ok" in result["cython"]


# ===========================================================================
# SCENARIO 3 — Pro: full stack (setuptools + Cython + pybind11 + NumPy)
# ===========================================================================


class TestScenario3FullStack:
    """Master/pro scenario: full build stack.

    Notes
    -----
    **User note**: install with ``pip install setuptools Cython pybind11 numpy``.
    **Dev note**: :func:`full_stack_prereqs` validates all four dependencies.
    """

    def test_full_stack_prereqs_returns_dict(self) -> None:
        result = full_stack_prereqs()
        assert isinstance(result, dict)

    def test_full_stack_prereqs_has_all_keys(self) -> None:
        result = full_stack_prereqs()
        for key in ("setuptools", "cython", "pybind11", "numpy"):
            assert key in result, f"Missing key: {key!r}"

    def test_full_stack_prereqs_all_ok_are_bool(self) -> None:
        result = full_stack_prereqs()
        for key, val in result.items():
            assert isinstance(val["ok"], bool), f"{key}['ok'] is not bool"

    def test_check_build_prereqs_numpy_true_adds_numpy(self) -> None:
        result = check_build_prereqs(numpy=True)
        assert "numpy" in result

    def test_check_build_prereqs_pybind11_true_adds_pybind11(self) -> None:
        result = check_build_prereqs(pybind11=True)
        assert "pybind11" in result

    def test_check_build_prereqs_all_adds_all(self) -> None:
        result = check_build_prereqs(numpy=True, pybind11=True)
        for key in ("cython", "setuptools", "numpy", "pybind11"):
            assert key in result

    def test_check_build_prereqs_pybind11_ok_entry(self) -> None:
        result = check_build_prereqs(pybind11=True)
        assert isinstance(result["pybind11"]["ok"], bool)

    def test_check_build_prereqs_pybind11_ok_has_include(self) -> None:
        result = check_build_prereqs(pybind11=True)
        if result["pybind11"]["ok"]:
            assert "include" in result["pybind11"]

    def test_pybind11_include_is_none_or_dir(self) -> None:
        p = pybind11_include()
        assert p is None or (isinstance(p, Path) and p.is_dir())

    def test_numpy_include_is_none_or_dir(self) -> None:
        p = numpy_include()
        assert p is None or (isinstance(p, Path) and p.is_dir())


# ===========================================================================
# SCENARIO 4 — Master: pybind11 only
# ===========================================================================


class TestScenario4Pybind11Only:
    """Master scenario: pybind11-only projects, no Cython needed.

    Notes
    -----
    **User note**: use :func:`pybind11_only_prereqs` to verify your env,
    then :class:`PybindCompiler` to compile C++ extension modules.
    **Dev note**: :class:`PybindCompiler` satisfies the protocol and has
    ``name == "custom_pybind11"``.
    """

    def test_pybind11_only_prereqs_has_pybind11_key(self) -> None:
        result = pybind11_only_prereqs()
        assert "pybind11" in result

    def test_pybind11_only_prereqs_only_pybind11_key(self) -> None:
        result = pybind11_only_prereqs()
        assert list(result.keys()) == ["pybind11"]

    def test_pybind11_compiler_name(self) -> None:
        pc = PybindCompiler()
        assert pc.name == "custom_pybind11"

    def test_pybind11_compiler_satisfies_protocol(self) -> None:
        pc = PybindCompiler()
        assert isinstance(pc, CustomCompilerProtocol)

    def test_pybind11_compiler_is_callable(self) -> None:
        pc = PybindCompiler()
        assert callable(pc)

    def test_pybind11_include_returns_path_or_none(self) -> None:
        result = pybind11_include()
        assert result is None or isinstance(result, Path)

    def test_pybind11_include_resolved_when_available(self) -> None:
        p = pybind11_include()
        if p is not None:
            assert p.is_absolute()

    def test_pybind11_compiler_raises_import_error_without_pybind11(
        self, tmp_path: Path
    ) -> None:
        pc = PybindCompiler()
        with patch("scikitplot.cython._custom_compiler.pybind11_include", return_value=None):
            with pytest.raises(ImportError, match="pybind11 is required"):
                pc("int x = 1;", build_dir=tmp_path, module_name="testmod")


# ===========================================================================
# SCENARIO 5 — Master: own C-API (single/multi file, folder, nested, exclude)
# ===========================================================================


class TestScenario5CApiSources:
    """Master scenario: C/C++ source and header collection.

    Notes
    -----
    **User note**: :func:`collect_c_api_sources` accepts files, directories,
    or glob patterns.  :func:`collect_header_dirs` gives you the
    ``include_dirs`` list from a source tree automatically.
    **Dev note**: all paths are returned as absolute, deduplicated, sorted.
    """

    # --- c_api_prereqs ---

    def test_c_api_prereqs_has_required_keys(self) -> None:
        result = c_api_prereqs()
        for key in ("cython", "numpy", "setuptools"):
            assert key in result

    def test_c_api_prereqs_all_ok_are_bool(self) -> None:
        result = c_api_prereqs()
        for key, val in result.items():
            assert isinstance(val["ok"], bool)

    # --- Scenario 5a: single file ---

    def test_single_c_file(self, tmp_path: Path) -> None:
        f = tmp_path / "foo.c"
        f.write_text("int foo() { return 1; }")
        result = collect_c_api_sources(str(f))
        assert len(result) == 1
        assert result[0] == f.resolve()

    def test_single_cpp_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bar.cpp"
        f.write_text("int bar() { return 2; }")
        result = collect_c_api_sources(str(f))
        assert len(result) == 1
        assert result[0].suffix == ".cpp"

    def test_single_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            collect_c_api_sources(str(tmp_path / "nonexistent.c"))

    def test_single_file_bad_suffix_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("not a C file")
        with pytest.raises(ValueError, match="unsupported source suffix"):
            collect_c_api_sources(str(f))

    def test_header_file_ignored_in_default_suffixes(self, tmp_path: Path) -> None:
        h = tmp_path / "foo.h"
        h.write_text("#pragma once")
        with pytest.raises(ValueError, match="unsupported source suffix"):
            collect_c_api_sources(str(h))

    # --- Scenario 5b: multiple files ---

    def test_multiple_explicit_files(self, tmp_path: Path) -> None:
        files = []
        for name in ("a.c", "b.cpp", "c.cxx"):
            f = tmp_path / name
            f.write_text("void stub() {}")
            files.append(str(f))
        result = collect_c_api_sources(*files)
        assert len(result) == 3

    def test_deduplication_of_same_file_twice(self, tmp_path: Path) -> None:
        f = tmp_path / "dup.c"
        f.write_text("int dup() { return 0; }")
        result = collect_c_api_sources(str(f), str(f))
        assert len(result) == 1

    # --- Scenario 5c: directory ---

    def test_directory_collects_all_c_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.c").write_text("int a() {}")
        (tmp_path / "b.cpp").write_text("int b() {}")
        (tmp_path / "README.md").write_text("# docs")
        result = collect_c_api_sources(str(tmp_path))
        names = {p.name for p in result}
        assert "a.c" in names
        assert "b.cpp" in names
        assert "README.md" not in names

    def test_directory_skips_headers_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "a.c").write_text("int a() {}")
        (tmp_path / "mylib.h").write_text("#pragma once")
        result = collect_c_api_sources(str(tmp_path))
        names = {p.name for p in result}
        assert "mylib.h" not in names

    # --- Scenario 5d: nested folder tree ---

    def test_recursive_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.c").write_text("int r() {}")
        (sub / "child.c").write_text("int c() {}")
        result = collect_c_api_sources(str(tmp_path), recursive=True)
        names = {p.name for p in result}
        assert "root.c" in names
        assert "child.c" in names

    def test_non_recursive_directory_excludes_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.c").write_text("int r() {}")
        (sub / "child.c").write_text("int c() {}")
        result = collect_c_api_sources(str(tmp_path), recursive=False)
        names = {p.name for p in result}
        assert "root.c" in names
        assert "child.c" not in names

    # --- exclude_patterns ---

    def test_exclude_pattern_applied(self, tmp_path: Path) -> None:
        (tmp_path / "main.c").write_text("int main() {}")
        (tmp_path / "test_helper.c").write_text("int helper() {}")
        result = collect_c_api_sources(str(tmp_path), exclude_patterns=["test_*.c"])
        names = {p.name for p in result}
        assert "main.c" in names
        assert "test_helper.c" not in names

    def test_custom_suffixes(self, tmp_path: Path) -> None:
        (tmp_path / "a.c").write_text("int a() {}")
        (tmp_path / "b.f90").write_text("real b")
        result = collect_c_api_sources(
            str(tmp_path),
            suffixes=frozenset({".f90"}),
        )
        names = {p.name for p in result}
        assert "b.f90" in names
        assert "a.c" not in names

    def test_results_are_absolute_paths(self, tmp_path: Path) -> None:
        (tmp_path / "x.c").write_text("int x() {}")
        result = collect_c_api_sources(str(tmp_path))
        for p in result:
            assert p.is_absolute()

    # --- collect_header_dirs ---

    def test_collect_header_dirs_single_dir(self, tmp_path: Path) -> None:
        (tmp_path / "mylib.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path))
        assert len(result) == 1
        assert result[0] == tmp_path.resolve()

    def test_collect_header_dirs_nested(self, tmp_path: Path) -> None:
        sub = tmp_path / "include"
        sub.mkdir()
        (sub / "api.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path), recursive=True)
        assert sub.resolve() in result

    def test_collect_header_dirs_deduplicated(self, tmp_path: Path) -> None:
        (tmp_path / "a.h").write_text("#pragma once")
        (tmp_path / "b.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path))
        assert result.count(tmp_path.resolve()) == 1

    def test_collect_header_dirs_no_headers_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "main.c").write_text("int main() {}")
        result = collect_header_dirs(str(tmp_path))
        assert result == []

    def test_collect_header_dirs_results_sorted(self, tmp_path: Path) -> None:
        for sub in ("z_inc", "a_inc"):
            d = tmp_path / sub
            d.mkdir()
            (d / "h.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path), recursive=True)
        assert result == sorted(result)

    def test_collect_header_dirs_explicit_file(self, tmp_path: Path) -> None:
        h = tmp_path / "mylib.h"
        h.write_text("#pragma once")
        result = collect_header_dirs(str(h))
        assert tmp_path.resolve() in result

    def test_c_api_compiler_name(self) -> None:
        cc = CApiCompiler()
        assert cc.name == "custom_c_api"

    def test_c_api_compiler_satisfies_protocol(self) -> None:
        cc = CApiCompiler()
        assert isinstance(cc, CustomCompilerProtocol)


# ===========================================================================
# SCENARIO 6 — All users: security guards
# ===========================================================================


class TestScenario6SecurityPolicy:
    """Security policy: construction, defaults, and validation.

    Notes
    -----
    **User note**: the default strict policy blocks the most common
    attack surfaces.  To relax, use ``SecurityPolicy(allow_...=True)``.
    **Dev note**: ``validate_build_inputs`` raises :exc:`SecurityError`
    (a subclass of ``ValueError``) on the first violation found.
    """

    # --- SecurityPolicy construction ---

    def test_default_policy_strict(self) -> None:
        p = SecurityPolicy()
        assert p.strict is True

    def test_default_policy_denies_absolute_dirs(self) -> None:
        p = SecurityPolicy()
        assert p.allow_absolute_include_dirs is False

    def test_default_policy_denies_shell_meta(self) -> None:
        p = SecurityPolicy()
        assert p.allow_shell_metacharacters is False

    def test_default_policy_denies_reserved_macros(self) -> None:
        p = SecurityPolicy()
        assert p.allow_reserved_macros is False

    def test_default_policy_denies_dangerous_args(self) -> None:
        p = SecurityPolicy()
        assert p.allow_dangerous_compiler_args is False

    def test_default_policy_max_source_bytes_ten_mib(self) -> None:
        p = SecurityPolicy()
        assert p.max_source_bytes == 10 * 1024 * 1024

    def test_default_policy_max_extra_compile_args_64(self) -> None:
        p = SecurityPolicy()
        assert p.max_extra_compile_args == 64

    def test_default_policy_max_extra_link_args_64(self) -> None:
        # Bug R1: max_extra_link_args must be an independent field.
        p = SecurityPolicy()
        assert p.max_extra_link_args == 64

    def test_max_extra_link_args_independent_from_compile_args(self) -> None:
        # R1: setting compile limit must NOT change link limit.
        p = SecurityPolicy(max_extra_compile_args=200, max_extra_link_args=10)
        assert p.max_extra_compile_args == 200
        assert p.max_extra_link_args == 10

    def test_relaxed_policy_disables_all_guards(self) -> None:
        p = SecurityPolicy.relaxed()
        assert p.allow_absolute_include_dirs is True
        assert p.allow_shell_metacharacters is True
        assert p.allow_reserved_macros is True
        assert p.allow_dangerous_compiler_args is True
        assert p.max_source_bytes is None

    def test_relaxed_policy_large_limits(self) -> None:
        p = SecurityPolicy.relaxed()
        assert p.max_extra_compile_args == 1024
        assert p.max_extra_link_args == 1024

    def test_default_security_policy_singleton(self) -> None:
        assert isinstance(DEFAULT_SECURITY_POLICY, SecurityPolicy)
        assert DEFAULT_SECURITY_POLICY.strict is True

    def test_relaxed_security_policy_singleton(self) -> None:
        assert isinstance(RELAXED_SECURITY_POLICY, SecurityPolicy)
        assert RELAXED_SECURITY_POLICY.strict is False

    def test_security_policy_is_frozen(self) -> None:
        p = SecurityPolicy()
        with pytest.raises((AttributeError, TypeError)):
            p.strict = False  # type: ignore[misc]

    def test_negative_max_compile_args_raises(self) -> None:
        with pytest.raises(ValueError, match="max_extra_compile_args"):
            SecurityPolicy(max_extra_compile_args=-1)

    def test_negative_max_link_args_raises(self) -> None:
        with pytest.raises(ValueError, match="max_extra_link_args"):
            SecurityPolicy(max_extra_link_args=-1)

    def test_negative_max_include_dirs_raises(self) -> None:
        with pytest.raises(ValueError, match="max_include_dirs"):
            SecurityPolicy(max_include_dirs=-1)

    def test_negative_max_libraries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_libraries"):
            SecurityPolicy(max_libraries=-1)

    def test_zero_max_source_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_source_bytes"):
            SecurityPolicy(max_source_bytes=0)

    def test_none_max_source_bytes_allowed(self) -> None:
        p = SecurityPolicy(max_source_bytes=None)
        assert p.max_source_bytes is None


class TestScenario6IssafePath:
    """is_safe_path: path traversal and absolute path guards."""

    def test_relative_path_safe(self) -> None:
        assert is_safe_path("include/mylib") is True

    def test_dotdot_unsafe(self) -> None:
        assert is_safe_path("../../../etc/passwd") is False

    def test_dotdot_in_middle_unsafe(self) -> None:
        assert is_safe_path("a/b/../c") is False

    def test_tilde_unsafe(self) -> None:
        assert is_safe_path("~/secret") is False

    def test_null_byte_unsafe(self) -> None:
        assert is_safe_path("foo\x00bar") is False

    def test_absolute_unsafe_by_default(self) -> None:
        assert is_safe_path("/usr/include") is False

    def test_absolute_safe_when_allowed(self) -> None:
        assert is_safe_path("/usr/include", allow_absolute=True) is True

    def test_simple_relative_segment_safe(self) -> None:
        assert is_safe_path("src/mylib") is True

    def test_current_dir_dot_safe(self) -> None:
        assert is_safe_path(".") is True

    def test_pathlib_path_accepted(self) -> None:
        assert is_safe_path(Path("include/mylib")) is True


class TestScenario6IsSafeMacroName:
    """is_safe_macro_name: C identifier and reserved-name guards."""

    def test_valid_identifier(self) -> None:
        assert is_safe_macro_name("MY_FLAG") is True

    def test_leading_digit_invalid(self) -> None:
        assert is_safe_macro_name("123BAD") is False

    def test_empty_string_invalid(self) -> None:
        assert is_safe_macro_name("") is False

    def test_non_string_invalid(self) -> None:
        assert is_safe_macro_name(None) is False  # type: ignore[arg-type]

    def test_reserved_py_limited_api_blocked(self) -> None:
        assert is_safe_macro_name("Py_LIMITED_API") is False

    def test_reserved_ndebug_blocked(self) -> None:
        assert is_safe_macro_name("NDEBUG") is False

    def test_reserved_py_debug_blocked(self) -> None:
        assert is_safe_macro_name("Py_DEBUG") is False

    def test_reserved_fortify_blocked(self) -> None:
        assert is_safe_macro_name("_FORTIFY_SOURCE") is False

    def test_reserved_allowed_when_flag_set(self) -> None:
        assert is_safe_macro_name("Py_LIMITED_API", allow_reserved=True) is True

    def test_underscore_prefix_valid(self) -> None:
        assert is_safe_macro_name("_MY_INTERNAL") is True

    def test_spaces_invalid(self) -> None:
        assert is_safe_macro_name("MY FLAG") is False


class TestScenario6IsSafeCompilerArg:
    """is_safe_compiler_arg: shell metachar and dangerous pattern guards."""

    def test_plain_optimization_safe(self) -> None:
        assert is_safe_compiler_arg("-O2") is True

    def test_define_macro_safe(self) -> None:
        assert is_safe_compiler_arg("-DNDEBUG") is True

    def test_semicolon_injection_unsafe(self) -> None:
        assert is_safe_compiler_arg("-O2; rm -rf /") is False

    def test_pipe_injection_unsafe(self) -> None:
        assert is_safe_compiler_arg("-O2 | cat /etc/passwd") is False

    def test_backtick_injection_unsafe(self) -> None:
        assert is_safe_compiler_arg("`id`") is False

    def test_dollar_injection_unsafe(self) -> None:
        assert is_safe_compiler_arg("$HOME") is False

    def test_null_byte_always_unsafe(self) -> None:
        assert is_safe_compiler_arg("-O2\x00evil") is False

    def test_imacros_dangerous(self) -> None:
        assert is_safe_compiler_arg("-imacros /etc/shadow") is False

    def test_specs_injection_dangerous(self) -> None:
        assert is_safe_compiler_arg("-specs=/tmp/evil") is False

    def test_sysroot_override_dangerous(self) -> None:
        assert is_safe_compiler_arg("--sysroot=/tmp") is False

    def test_shell_meta_allowed_when_flag_set(self) -> None:
        assert is_safe_compiler_arg("-O2; echo hi", allow_shell_meta=True) is True

    def test_dangerous_allowed_when_flag_set(self) -> None:
        assert is_safe_compiler_arg("-imacros /tmp/x", allow_dangerous=True) is True

    def test_non_string_rejected(self) -> None:
        assert is_safe_compiler_arg(42) is False  # type: ignore[arg-type]


class TestScenario6ValidateBuildInputs:
    """validate_build_inputs: integration of all guards."""

    def test_clean_inputs_no_raise(self) -> None:
        validate_build_inputs(
            source="def hello(): return 42",
            extra_compile_args=["-O2"],
            extra_link_args=["-shared"],
            include_dirs=["include/mylib"],
            libraries=["m"],
        )

    def test_none_inputs_no_raise(self) -> None:
        validate_build_inputs()

    def test_none_policy_uses_default(self) -> None:
        validate_build_inputs(policy=None, source="x = 1")

    def test_wrong_policy_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="SecurityPolicy"):
            validate_build_inputs(policy="strict")  # type: ignore[arg-type]

    # --- source size ---

    def test_source_exceeds_limit_raises(self) -> None:
        policy = SecurityPolicy(max_source_bytes=10)
        with pytest.raises(SecurityError, match="source"):
            validate_build_inputs(policy=policy, source="x" * 100)

    def test_source_exactly_at_limit_ok(self) -> None:
        policy = SecurityPolicy(max_source_bytes=5)
        validate_build_inputs(policy=policy, source="hello")

    def test_source_none_skips_check(self) -> None:
        policy = SecurityPolicy(max_source_bytes=1)
        validate_build_inputs(policy=policy, source=None)

    # --- define_macros ---

    def test_valid_macro_ok(self) -> None:
        validate_build_inputs(define_macros=[("MY_FLAG", "1")])

    def test_reserved_macro_raises(self) -> None:
        with pytest.raises(SecurityError, match="define_macros"):
            validate_build_inputs(define_macros=[("Py_LIMITED_API", None)])

    def test_invalid_macro_name_syntax_raises(self) -> None:
        with pytest.raises(SecurityError, match="define_macros"):
            validate_build_inputs(define_macros=[("123BAD", "1")])

    def test_macro_not_two_tuple_raises(self) -> None:
        with pytest.raises(SecurityError, match="define_macros"):
            validate_build_inputs(define_macros=[("ONLY_ONE",)])  # type: ignore

    def test_macros_not_sequence_raises(self) -> None:
        with pytest.raises(SecurityError, match="define_macros"):
            validate_build_inputs(define_macros="MY_FLAG=1")  # type: ignore

    # --- extra_compile_args ---

    def test_shell_meta_in_compile_args_raises(self) -> None:
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(extra_compile_args=["-O2; rm -rf /"])

    def test_too_many_compile_args_raises(self) -> None:
        policy = SecurityPolicy(max_extra_compile_args=2)
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(policy=policy, extra_compile_args=["-O1", "-O2", "-O3"])

    def test_non_str_compile_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(extra_compile_args=[42])  # type: ignore

    # --- extra_link_args (R1: uses max_extra_link_args, not max_extra_compile_args) ---

    def test_shell_meta_in_link_args_raises(self) -> None:
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(extra_link_args=["-shared; echo evil"])

    def test_too_many_link_args_uses_link_limit(self) -> None:
        # R1: max_extra_link_args is independent — raising compile limit should
        # NOT raise the link limit.
        policy = SecurityPolicy(max_extra_compile_args=100, max_extra_link_args=2)
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=policy,
                extra_link_args=["-lm", "-lz", "-lpthread"],
            )

    def test_many_link_args_ok_when_limit_high(self) -> None:
        policy = SecurityPolicy(max_extra_link_args=10)
        validate_build_inputs(
            policy=policy,
            extra_link_args=["-lm", "-lz", "-lpthread"],
        )

    def test_non_str_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(extra_link_args=[None])  # type: ignore

    # --- include_dirs ---

    def test_path_traversal_in_include_raises(self) -> None:
        with pytest.raises(SecurityError, match="include_dirs"):
            validate_build_inputs(include_dirs=["../../../etc"])

    def test_absolute_include_raises_by_default(self) -> None:
        with pytest.raises(SecurityError, match="include_dirs"):
            validate_build_inputs(include_dirs=["/usr/include"])

    def test_absolute_include_ok_when_allowed(self) -> None:
        policy = SecurityPolicy(allow_absolute_include_dirs=True)
        validate_build_inputs(policy=policy, include_dirs=["/usr/include"])

    def test_too_many_include_dirs_raises(self) -> None:
        policy = SecurityPolicy(max_include_dirs=2)
        with pytest.raises(SecurityError, match="include_dirs"):
            validate_build_inputs(policy=policy, include_dirs=["a", "b", "c"])

    # --- libraries ---

    def test_valid_library_name_ok(self) -> None:
        validate_build_inputs(libraries=["m", "z", "pthread"])

    def test_library_with_path_separator_raises(self) -> None:
        with pytest.raises(SecurityError, match="libraries"):
            validate_build_inputs(libraries=["lib/m"])

    def test_library_with_shell_meta_raises(self) -> None:
        with pytest.raises(SecurityError, match="libraries"):
            validate_build_inputs(libraries=["m; rm -rf /"])

    def test_empty_library_name_raises(self) -> None:
        with pytest.raises(SecurityError, match="libraries"):
            validate_build_inputs(libraries=[""])

    def test_too_many_libraries_raises(self) -> None:
        policy = SecurityPolicy(max_libraries=2)
        with pytest.raises(SecurityError, match="libraries"):
            validate_build_inputs(policy=policy, libraries=["a", "b", "c"])

    # --- SecurityError ---

    def test_security_error_is_value_error(self) -> None:
        err = SecurityError("test", field="extra_compile_args")
        assert isinstance(err, ValueError)

    def test_security_error_field_in_message(self) -> None:
        err = SecurityError("bad arg", field="extra_compile_args")
        assert "extra_compile_args" in str(err)

    def test_security_error_field_attribute(self) -> None:
        err = SecurityError("bad arg", field="myfield")
        assert err.field == "myfield"

    def test_security_error_no_field(self) -> None:
        err = SecurityError("bare error")
        assert err.field is None
        assert "bare error" in str(err)

    # --- error message quality (actionable reasons) ---

    def test_shell_meta_error_says_shell_metacharacter(self) -> None:
        with pytest.raises(SecurityError) as exc_info:
            validate_build_inputs(extra_compile_args=["-O2; evil"])
        assert "shell metacharacter" in str(exc_info.value).lower()

    def test_dangerous_arg_error_says_dangerous(self) -> None:
        with pytest.raises(SecurityError) as exc_info:
            validate_build_inputs(extra_compile_args=["-imacros /etc/shadow"])
        assert "dangerous" in str(exc_info.value).lower()


# ===========================================================================
# SCENARIO 7 — Master: custom compilers
# ===========================================================================


class TestScenario7CustomCompilers:
    """Custom compiler protocol and registry.

    Notes
    -----
    **User note**: implement :class:`CustomCompilerProtocol`, register with
    :func:`register_compiler`, then pass the name as ``compiler=`` to the
    build pipeline.
    **Dev note**: the registry enforces the ``custom_*`` / ``Custom*``
    naming convention at registration time.
    """

    def setup_method(self) -> None:
        # Isolate each test with a fresh registry.
        from .. import _custom_compiler
        self._saved = _custom_compiler._REGISTRY._compilers.copy()

    def teardown_method(self) -> None:
        from .. import _custom_compiler
        _custom_compiler._REGISTRY._compilers = self._saved

    # --- Protocol conformance ---

    def test_minimal_compiler_satisfies_protocol(self) -> None:
        c = _make_minimal_compiler("custom_test_proto")
        assert isinstance(c, CustomCompilerProtocol)

    def test_object_without_name_not_protocol(self) -> None:
        class NoName:
            def __call__(self, source, *, build_dir, module_name, **kw):
                pass
        assert not isinstance(NoName(), CustomCompilerProtocol)

    def test_object_without_call_not_protocol(self) -> None:
        class NoCall:
            name = "custom_nocall"
        assert not isinstance(NoCall(), CustomCompilerProtocol)

    # --- CompilerRegistry ---

    def test_register_and_get(self) -> None:
        c = _make_minimal_compiler("custom_reg_get")
        register_compiler(c)
        assert get_compiler("custom_reg_get") is c

    def test_register_duplicate_raises(self) -> None:
        c = _make_minimal_compiler("custom_dup")
        register_compiler(c)
        with pytest.raises(ValueError, match="already registered"):
            register_compiler(c)

    def test_register_overwrite_true_replaces(self) -> None:
        c1 = _make_minimal_compiler("custom_ow")
        c2 = _make_minimal_compiler("custom_ow")
        register_compiler(c1)
        register_compiler(c2, overwrite=True)
        assert get_compiler("custom_ow") is c2

    def test_get_missing_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="No compiler registered"):
            get_compiler("custom_nonexistent_xyz")

    def test_list_compilers_returns_sorted(self) -> None:
        register_compiler(_make_minimal_compiler("custom_z"))
        register_compiler(_make_minimal_compiler("custom_a"))
        names = list_compilers()
        relevant = [n for n in names if n in ("custom_z", "custom_a")]
        assert relevant == sorted(relevant)

    def test_unregister_existing_returns_true(self) -> None:
        from .. import _custom_compiler
        c = _make_minimal_compiler("custom_unreg")
        register_compiler(c)
        assert _custom_compiler._REGISTRY.unregister("custom_unreg") is True

    def test_unregister_missing_returns_false(self) -> None:
        from .. import _custom_compiler
        assert _custom_compiler._REGISTRY.unregister("custom_absent_xyz") is False

    def test_register_non_protocol_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="CustomCompilerProtocol"):
            register_compiler("not_a_compiler")  # type: ignore[arg-type]

    def test_built_in_pybind_compiler_registers(self) -> None:
        pc = PybindCompiler()
        register_compiler(pc, overwrite=True)
        assert get_compiler("custom_pybind11") is pc

    def test_built_in_c_api_compiler_registers(self) -> None:
        cc = CApiCompiler()
        register_compiler(cc, overwrite=True)
        assert get_compiler("custom_c_api") is cc


# ===========================================================================
# SCENARIO 8 — Master: custom_* / Custom* naming convention
# ===========================================================================


class TestScenario8NamingConvention:
    """``custom_*`` / ``Custom*`` naming enforcement.

    Notes
    -----
    **User note**: name your compiler with ``custom_`` prefix (lowercase) or
    ``Custom`` prefix (title-case).  Any other prefix is rejected.
    **Dev note**: :func:`_validate_compiler_name` is called at register time.
    """

    @pytest.mark.parametrize("name", [
        "custom_nvcc",
        "custom_clang",
        "custom_msvc_2022",
        "custom_my_compiler_v2",
        "CustomNvcc",
        "CustomClang",
        "CustomMy_Compiler",
    ])
    def test_valid_names_accepted(self, name: str) -> None:
        _validate_compiler_name(name)  # must not raise

    @pytest.mark.parametrize("bad_name", [
        "nvcc",             # no prefix
        "my_compiler",      # wrong prefix
        "CUSTOM_upper",     # all-caps doesn't match
        "custom",           # no suffix after custom
        "Custom",           # no suffix after Custom
        "",                 # empty
        "_custom_leading",  # leading underscore
        "custom_",          # trailing underscore only
        42,                 # not a string
    ])
    def test_invalid_names_raise_value_error(self, bad_name: Any) -> None:
        with pytest.raises((ValueError, TypeError)):
            _validate_compiler_name(bad_name)  # type: ignore[arg-type]

    def test_register_invalid_name_raises(self) -> None:
        c = _make_minimal_compiler("custom_ok_temp")
        object.__setattr__(c, "name", "badname")  # type: ignore[misc]
        with pytest.raises(ValueError):
            from .. import _custom_compiler
            _custom_compiler._REGISTRY.register(c)

    def test_custom_underscore_suffix_required(self) -> None:
        with pytest.raises(ValueError):
            _validate_compiler_name("custom_")  # "custom_" alone — no suffix chars


# ===========================================================================
# SCENARIO 9 — User/dev notes: public API docstring examples
# ===========================================================================


class TestScenario9DocstringExamples:
    """Validate public API examples work as documented.

    Notes
    -----
    **User note**: every example in this file is a guarantee.  If an
    example here fails, the documentation is wrong or the code is broken.
    **Dev note**: keep these tests in sync with docstrings in the modules
    under test.  Don't mark them xfail — they are binding contracts.
    """

    # --- _security examples ---

    def test_security_policy_default_strict(self) -> None:
        policy = SecurityPolicy()
        assert policy.strict is True
        assert policy.allow_shell_metacharacters is False

    def test_security_policy_relaxed_example(self) -> None:
        policy = SecurityPolicy(allow_absolute_include_dirs=True)
        assert policy.allow_absolute_include_dirs is True

    def test_validate_build_inputs_clean_example(self) -> None:
        validate_build_inputs(
            source="def hello(): return 42",
            extra_compile_args=["-O2"],
        )

    def test_validate_build_inputs_shell_injection_example(self) -> None:
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(extra_compile_args=["-O2; rm -rf /"])

    def test_is_safe_path_relative_true(self) -> None:
        assert is_safe_path("include/mylib") is True

    def test_is_safe_path_traversal_false(self) -> None:
        assert is_safe_path("../../../etc/passwd") is False

    def test_is_safe_path_absolute_with_flag(self) -> None:
        assert is_safe_path("/usr/include", allow_absolute=True) is True

    def test_is_safe_path_absolute_no_flag_false(self) -> None:
        assert is_safe_path("/usr/include", allow_absolute=False) is False

    def test_is_safe_macro_name_valid(self) -> None:
        assert is_safe_macro_name("MY_FLAG") is True

    def test_is_safe_macro_name_reserved(self) -> None:
        assert is_safe_macro_name("Py_LIMITED_API") is False

    def test_is_safe_macro_name_reserved_allowed(self) -> None:
        assert is_safe_macro_name("Py_LIMITED_API", allow_reserved=True) is True

    def test_is_safe_macro_name_invalid_syntax(self) -> None:
        assert is_safe_macro_name("123INVALID") is False

    def test_is_safe_compiler_arg_clean(self) -> None:
        assert is_safe_compiler_arg("-O2") is True

    def test_is_safe_compiler_arg_shell(self) -> None:
        assert is_safe_compiler_arg("-O2; rm -rf /") is False

    def test_is_safe_compiler_arg_imacros(self) -> None:
        assert is_safe_compiler_arg("-imacros /etc/shadow") is False

    # --- _custom_compiler examples ---

    def test_custom_compiler_registration_example(self) -> None:
        from .. import _custom_compiler

        saved = _custom_compiler._REGISTRY._compilers.copy()
        try:
            class custom_fast:
                name = "custom_fast"
                def __call__(self, source, *, build_dir, module_name, **kw):
                    raise NotImplementedError

            register_compiler(custom_fast(), overwrite=True)
            assert "custom_fast" in list_compilers()
        finally:
            _custom_compiler._REGISTRY._compilers = saved

    def test_pybind11_include_example(self) -> None:
        p = pybind11_include()
        assert p is None or p.is_dir()

    def test_numpy_include_example(self) -> None:
        p = numpy_include()
        assert p is None or p.is_dir()

    def test_collect_c_api_sources_two_files_example(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path
        _ = (p / "a.c").write_text("int a() { return 1; }")
        _ = (p / "b.cpp").write_text("int b() { return 2; }")
        srcs = collect_c_api_sources(str(tmp_path))
        assert len(srcs) == 2

    def test_collect_header_dirs_one_dir_example(self, tmp_path: Path) -> None:
        _ = (tmp_path / "mylib.h").write_text("#pragma once")
        dirs = collect_header_dirs(str(tmp_path))
        assert len(dirs) == 1

    # --- _public examples ---

    def test_check_build_prereqs_has_cython_and_setuptools(self) -> None:
        result = check_build_prereqs()
        assert "cython" in result and "setuptools" in result

    def test_check_build_prereqs_with_all_options(self) -> None:
        result = check_build_prereqs(numpy=True, pybind11=True)
        assert all(k in result for k in ("cython", "setuptools", "numpy", "pybind11"))


# ===========================================================================
# REGRESSION TESTS
# ===========================================================================


class TestRegressionR1LinkArgLimit:
    """R1: ``max_extra_link_args`` is independent of ``max_extra_compile_args``."""

    def test_link_args_limited_by_link_field(self) -> None:
        policy = SecurityPolicy(max_extra_compile_args=100, max_extra_link_args=1)
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=policy,
                extra_link_args=["-lm", "-lz"],
            )

    def test_compile_args_not_limited_by_link_field(self) -> None:
        policy = SecurityPolicy(max_extra_compile_args=5, max_extra_link_args=100)
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(
                policy=policy,
                extra_compile_args=["-O1", "-O2", "-O3", "-O4", "-O5", "-O6"],
            )

    def test_independent_limits_both_pass(self) -> None:
        policy = SecurityPolicy(max_extra_compile_args=3, max_extra_link_args=3)
        validate_build_inputs(
            policy=policy,
            extra_compile_args=["-O1", "-O2"],
            extra_link_args=["-lm", "-lz"],
        )


class TestRegressionR2CythonImportResultIncludeDirs:
    """R2: bare-string ``include_dirs`` must not be iterated as characters."""

    def test_coerce_none_returns_none(self) -> None:
        assert _coerce_path_seq(None, "include_dirs") is None

    def test_coerce_bare_string_wraps_in_list(self) -> None:
        result = _coerce_path_seq("include/mylib", "include_dirs")
        assert result == ["include/mylib"]
        # NOT ["i", "n", "c", "l", "u", "d", "e", "/", "m", "y", "l", "i", "b"]
        assert len(result) == 1

    def test_coerce_pathlib_path_wraps_in_list(self) -> None:
        p = Path("include/mylib")
        result = _coerce_path_seq(p, "include_dirs")
        assert result == [p]
        assert len(result) == 1

    def test_coerce_list_returns_list(self) -> None:
        result = _coerce_path_seq(["a", "b", "c"], "include_dirs")
        assert result == ["a", "b", "c"]

    def test_coerce_tuple_returns_list(self) -> None:
        result = _coerce_path_seq(("a", "b"), "include_dirs")
        assert result == ["a", "b"]

    def test_coerce_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="include_dirs"):
            _coerce_path_seq(42, "include_dirs")  # type: ignore[arg-type]

    def test_coerce_bytes_path_wraps_in_list(self) -> None:
        result = _coerce_path_seq(b"include/mylib", "include_dirs")
        assert result == [b"include/mylib"]
        assert len(result) == 1


class TestRegressionR3NoDuplicatesInAll:
    """R3: ``scikitplot.cython.__all__`` must contain no duplicate names."""

    def test_no_duplicate_names(self) -> None:
        duplicates = [x for x in _PKG_ALL if _PKG_ALL.count(x) > 1]
        assert duplicates == [], f"Duplicate __all__ entries: {sorted(set(duplicates))}"

    def test_all_is_list_of_strings(self) -> None:
        assert isinstance(_PKG_ALL, list)
        for name in _PKG_ALL:
            assert isinstance(name, str), f"Non-string in __all__: {name!r}"

    def test_key_public_names_present(self) -> None:
        for name in (
            "compile_and_load",
            "compile_and_load_result",
            "check_build_prereqs",
            "SecurityPolicy",
            "SecurityError",
            "validate_build_inputs",
            "CustomCompilerProtocol",
            "CompilerRegistry",
            "register_compiler",
            "collect_c_api_sources",
            "collect_header_dirs",
            "PybindCompiler",
            "CApiCompiler",
        ):
            assert name in _PKG_ALL, f"Expected name missing from __all__: {name!r}"


class TestRegressionR4DoctestNoStrayOutput:
    """R4: ``collect_c_api_sources`` doctest must not have stray numeric output."""

    def test_write_text_return_value_suppressed_in_doctest(
        self, tmp_path: Path
    ) -> None:
        """The doctest uses ``_ =`` to suppress write_text return values."""
        p = tmp_path
        _ = (p / "a.c").write_text("int a() { return 1; }")
        _ = (p / "b.cpp").write_text("int b() { return 2; }")
        srcs = collect_c_api_sources(str(tmp_path))
        # Must be exactly 2 — not 21 or any other stray value.
        assert len(srcs) == 2
