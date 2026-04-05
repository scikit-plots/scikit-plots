# scikitplot/cython/tests/test__custom_compiler.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._custom_compiler`.

Covers
------
- Scenario 4 — pybind11 only     : ``PybindCompiler`` protocol, include path,
                                   ImportError when pybind11 absent
- Scenario 7 — custom compilers  : ``CompilerRegistry`` register / get / list /
                                   unregister; protocol duck-typing; built-in
                                   compilers; non-protocol rejection
- Scenario 8 — naming convention : ``custom_*`` / ``Custom*`` prefix enforcement,
                                   ``_validate_compiler_name`` edge cases
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .. import _custom_compiler
from .._custom_compiler import (
    CApiCompiler,
    CompilerRegistry,
    CustomCompilerProtocol,
    PybindCompiler,
    _REGISTRY,
    _validate_compiler_name,
    get_compiler,
    list_compilers,
    pybind11_include,
    pybind11_only_prereqs,
    register_compiler,
)


def _make_minimal_compiler(name: str = "custom_minimal") -> CustomCompilerProtocol:
    """Return the simplest object that satisfies ``CustomCompilerProtocol``."""

    class _Compiler:
        def __call__(self, source, *, build_dir, module_name, **kwargs):
            raise NotImplementedError("test stub")

    c = _Compiler()
    c.name = name  # type: ignore[attr-defined]
    return c  # type: ignore[return-value]


class TestScenario4Pybind11Only:
    """
    Master scenario: pybind11-only projects, no Cython needed.

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


class TestScenario7CustomCompilers:
    """
    Custom compiler protocol and registry.

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
        self._saved = _custom_compiler._REGISTRY._compilers.copy()

    def teardown_method(self) -> None:
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
        c = _make_minimal_compiler("custom_unreg")
        register_compiler(c)
        assert _custom_compiler._REGISTRY.unregister("custom_unreg") is True

    def test_unregister_missing_returns_false(self) -> None:
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


class TestScenario8NamingConvention:
    """
    ``custom_*`` / ``Custom*`` naming enforcement.

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
    def test_invalid_names_raise_value_error(self, bad_name: any) -> None:
        with pytest.raises((ValueError, TypeError)):
            _validate_compiler_name(bad_name)  # type: ignore[arg-type]

    def test_register_invalid_name_raises(self) -> None:
        c = _make_minimal_compiler("custom_ok_temp")
        object.__setattr__(c, "name", "badname")  # type: ignore[misc]
        with pytest.raises(ValueError):
            _custom_compiler._REGISTRY.register(c)

    def test_custom_underscore_suffix_required(self) -> None:
        with pytest.raises(ValueError):
            _validate_compiler_name("custom_")  # "custom_" alone — no suffix chars
