# scikitplot/_testing/tests/test_pytesttester.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive tests for :mod:`scikitplot._testing._pytesttester`.

Coverage targets
----------------
* :func:`scikitplot._testing._pytesttester._show_numpy_info`
    - prints NumPy version line
    - prints relaxed-strides line
    - version string matches numpy.__version__

* :class:`scikitplot._testing._pytesttester.PytestTester`
    - __init__: stores module_name, rejects non-str
    - __repr__: correct format
    - __call__: argument assembly (label, verbose, extra_argv, coverage,
      durations, tests, base flags)
    - __call__: doctests=True raises ValueError
    - __call__: pytest.main return values -> True / False
    - __call__: SystemExit code normalisation (int, None, empty-str, str)
    - __call__: module with __path__ (package)
    - __call__: module without __path__ (single-file)
    - __call__: module without __path__ or __file__

Notes
-----
Developer note: ``pytest`` is imported *inside* ``PytestTester.__call__``
with a plain ``import pytest`` statement.  The canonical way to mock a
deferred local import is to temporarily replace the entry in ``sys.modules``
via ``unittest.mock.patch.dict``.  Using ``patch("scikitplot._testing._pytesttester.pytest")``
would raise ``AttributeError`` because the attribute does not exist at the
module level until the first call executes.

The helper ``_run_call`` handles this consistently for every test.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from .._pytesttester import PytestTester, _show_numpy_info  # noqa: TID252


# ===========================================================================
# Helpers
# ===========================================================================


def _make_package_module(name: str, path: str = "/fake/pkg") -> types.ModuleType:
    """Return a fake *package* module with ``__path__`` set."""
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__file__ = f"{path}/__init__.py"
    return mod


def _make_file_module(name: str, file_path: str = "/fake/mod.py") -> types.ModuleType:
    """Return a fake *single-file* module (no ``__path__``)."""
    mod = types.ModuleType(name)
    mod.__file__ = file_path
    return mod


def _run_call(mod_name: str, mod: types.ModuleType, **call_kwargs):
    """Invoke ``PytestTester(mod_name)(**call_kwargs)`` with mocked pytest.

    ``pytest`` is imported inside ``__call__`` via a local ``import pytest``.
    We mock it by temporarily replacing ``sys.modules["pytest"]`` so the local
    import picks up our :class:`~unittest.mock.MagicMock`.

    Parameters
    ----------
    mod_name : str
        Module name to register in sys.modules.
    mod : types.ModuleType
        Fake module object.
    **call_kwargs
        Forwarded to ``PytestTester.__call__``.

    Returns
    -------
    result : bool
        Return value of ``PytestTester.__call__``.
    mock_pytest : MagicMock
        The mock that replaced ``pytest`` during the call.
    """
    mock_pytest = MagicMock()
    mock_pytest.main.return_value = 0
    with patch.dict(sys.modules, {"pytest": mock_pytest, mod_name: mod}):
        with patch("scikitplot._testing._pytesttester._show_numpy_info"):
            result = PytestTester(mod_name)(**call_kwargs)
    return result, mock_pytest


def _pytest_args(mod_name: str, mod: types.ModuleType, **call_kwargs) -> list:
    """Return the pytest_args list that would be passed to pytest.main."""
    _, mock_pytest = _run_call(mod_name, mod, **call_kwargs)
    return mock_pytest.main.call_args[0][0]


# ===========================================================================
# _show_numpy_info
# ===========================================================================


class TestShowNumpyInfo:
    """Output lines must contain expected content."""

    def test_prints_numpy_version_line(self, capsys) -> None:
        _show_numpy_info()
        assert "NumPy version" in capsys.readouterr().out

    def test_prints_relaxed_strides_line(self, capsys) -> None:
        _show_numpy_info()
        assert "relaxed strides" in capsys.readouterr().out

    def test_output_contains_actual_version_number(self, capsys) -> None:
        import numpy as np
        _show_numpy_info()
        assert np.__version__ in capsys.readouterr().out

    def test_outputs_two_lines(self, capsys) -> None:
        _show_numpy_info()
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 2


# ===========================================================================
# PytestTester — __init__
# ===========================================================================


class TestPytestTesterInit:
    def test_stores_module_name(self) -> None:
        assert PytestTester("my.pkg").module_name == "my.pkg"

    def test_accepts_dotted_name(self) -> None:
        t = PytestTester("scikitplot.decomposition")
        assert t.module_name == "scikitplot.decomposition"

    def test_non_str_int_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="module_name"):
            PytestTester(42)  # type: ignore[arg-type]

    def test_non_str_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="module_name"):
            PytestTester(None)  # type: ignore[arg-type]

    def test_non_str_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            PytestTester(["pkg"])  # type: ignore[arg-type]


# ===========================================================================
# PytestTester — __repr__
# ===========================================================================


class TestPytestTesterRepr:
    def test_repr_contains_class_name(self) -> None:
        assert "PytestTester" in repr(PytestTester("m"))

    def test_repr_contains_module_name(self) -> None:
        assert "mymod" in repr(PytestTester("mymod"))

    def test_repr_exact_format(self) -> None:
        assert repr(PytestTester("a.b")) == "PytestTester(module_name='a.b')"


# ===========================================================================
# PytestTester — __call__: label variants
# ===========================================================================

_PKG = "_tp_label"
_MOD = _make_package_module(_PKG)


class TestPytestTesterLabel:
    def test_fast_adds_not_slow_marker(self) -> None:
        args = _pytest_args(_PKG, _MOD, label="fast")
        assert "-m" in args
        assert args[args.index("-m") + 1] == "not slow"

    def test_full_omits_marker(self) -> None:
        args = _pytest_args(_PKG, _MOD, label="full")
        assert "-m" not in args

    def test_custom_label_forwarded_as_marker(self) -> None:
        args = _pytest_args(_PKG, _MOD, label="integration")
        assert "-m" in args
        assert args[args.index("-m") + 1] == "integration"

    def test_empty_string_label_added_as_marker(self) -> None:
        # Edge case: empty-string label is NOT "fast" and NOT "full"
        args = _pytest_args(_PKG, _MOD, label="")
        assert "-m" in args


# ===========================================================================
# PytestTester — __call__: verbose
# ===========================================================================


class TestPytestTesterVerbose:
    def test_verbose_1_no_dash_v(self) -> None:
        args = _pytest_args(_PKG, _MOD, verbose=1)
        assert not any(a.startswith("-v") for a in args)

    def test_verbose_2_adds_single_v(self) -> None:
        args = _pytest_args(_PKG, _MOD, verbose=2)
        assert "-v" in args

    def test_verbose_3_adds_double_v(self) -> None:
        args = _pytest_args(_PKG, _MOD, verbose=3)
        assert "-vv" in args

    def test_verbose_4_adds_triple_v(self) -> None:
        args = _pytest_args(_PKG, _MOD, verbose=4)
        assert "-vvv" in args


# ===========================================================================
# PytestTester — __call__: extra_argv
# ===========================================================================


class TestPytestTesterExtraArgv:
    def test_none_extra_argv_adds_nothing_extra(self) -> None:
        args = _pytest_args(_PKG, _MOD, extra_argv=None)
        assert "--tb=short" not in args

    def test_extra_args_appended(self) -> None:
        args = _pytest_args(_PKG, _MOD, extra_argv=["--tb=short", "-x"])
        assert "--tb=short" in args
        assert "-x" in args

    def test_extra_args_empty_list_ok(self) -> None:
        # empty list is falsy — should not add anything
        args = _pytest_args(_PKG, _MOD, extra_argv=[])
        assert "--tb=short" not in args


# ===========================================================================
# PytestTester — __call__: coverage
# ===========================================================================


class TestPytestTesterCoverage:
    def test_coverage_false_no_cov_flag(self) -> None:
        args = _pytest_args(_PKG, _MOD, coverage=False)
        assert not any(a.startswith("--cov") for a in args)

    def test_coverage_true_adds_cov_flag(self) -> None:
        args = _pytest_args(_PKG, _MOD, coverage=True)
        assert any(a.startswith("--cov=") for a in args)

    def test_cov_flag_contains_path(self) -> None:
        args = _pytest_args(_PKG, _MOD, coverage=True)
        cov_args = [a for a in args if a.startswith("--cov=")]
        assert len(cov_args) == 1
        assert len(cov_args[0]) > len("--cov=")  # path is non-empty


# ===========================================================================
# PytestTester — __call__: durations
# ===========================================================================


class TestPytestTesterDurations:
    def test_negative_durations_omits_flag(self) -> None:
        args = _pytest_args(_PKG, _MOD, durations=-1)
        assert not any(a.startswith("--durations") for a in args)

    def test_zero_durations_adds_flag(self) -> None:
        assert "--durations=0" in _pytest_args(_PKG, _MOD, durations=0)

    def test_positive_durations_adds_flag(self) -> None:
        assert "--durations=5" in _pytest_args(_PKG, _MOD, durations=5)

    def test_durations_minus_2_omits_flag(self) -> None:
        args = _pytest_args(_PKG, _MOD, durations=-2)
        assert not any(a.startswith("--durations") for a in args)


# ===========================================================================
# PytestTester — __call__: tests / pyargs
# ===========================================================================


class TestPytestTesterTests:
    def test_default_tests_uses_module_name(self) -> None:
        args = _pytest_args(_PKG, _MOD)
        assert "--pyargs" in args
        assert _PKG in args

    def test_explicit_tests_forwarded(self) -> None:
        args = _pytest_args(_PKG, _MOD, tests=["a.b", "a.c"])
        assert "a.b" in args
        assert "a.c" in args

    def test_pyargs_precedes_tests(self) -> None:
        args = _pytest_args(_PKG, _MOD, tests=["x.y"])
        idx_pyargs = args.index("--pyargs")
        idx_test = args.index("x.y")
        assert idx_pyargs < idx_test


# ===========================================================================
# PytestTester — __call__: base flags always present
# ===========================================================================


class TestPytestTesterBaseFlags:
    def test_dash_l_always_present(self) -> None:
        assert "-l" in _pytest_args(_PKG, _MOD)

    def test_dash_q_always_present(self) -> None:
        assert "-q" in _pytest_args(_PKG, _MOD)

    def test_ignore_numpy_dtype_filter_present(self) -> None:
        args = _pytest_args(_PKG, _MOD)
        assert any("numpy.dtype size changed" in a for a in args)

    def test_ignore_numpy_ufunc_filter_present(self) -> None:
        args = _pytest_args(_PKG, _MOD)
        assert any("numpy.ufunc size changed" in a for a in args)


# ===========================================================================
# PytestTester — __call__: doctests raises ValueError
# ===========================================================================


class TestPytestTesterDoctests:
    def test_doctests_true_raises_value_error(self) -> None:
        mod = _make_package_module("_tp_docs")
        mock_pytest = MagicMock()
        with patch.dict(sys.modules, {"pytest": mock_pytest, "_tp_docs": mod}):
            with patch("scikitplot._testing._pytesttester._show_numpy_info"):
                with pytest.raises(ValueError, match="not supported"):
                    PytestTester("_tp_docs")(doctests=True)

    def test_doctests_false_does_not_raise(self) -> None:
        _run_call("_tp_no_docs", _make_package_module("_tp_no_docs"), doctests=False)


# ===========================================================================
# PytestTester — __call__: return value mapping
# ===========================================================================


class TestPytestTesterReturnValues:
    """pytest.main exit codes -> True/False."""

    _NAME = "_tp_retval"
    _MOD = _make_package_module(_NAME)

    def _run(self, retval):
        mock_pytest = MagicMock()
        mock_pytest.main.return_value = retval
        with patch.dict(sys.modules, {"pytest": mock_pytest, self._NAME: self._MOD}):
            with patch("scikitplot._testing._pytesttester._show_numpy_info"):
                return PytestTester(self._NAME)()

    def test_exit_0_returns_true(self) -> None:
        assert self._run(0) is True

    def test_exit_1_returns_false(self) -> None:
        assert self._run(1) is False

    def test_exit_2_returns_false(self) -> None:
        assert self._run(2) is False

    def test_pytest_exitcode_ok_returns_true(self) -> None:
        import pytest as real_pytest
        assert self._run(real_pytest.ExitCode.OK) is True

    def test_pytest_exitcode_tests_failed_returns_false(self) -> None:
        import pytest as real_pytest
        assert self._run(real_pytest.ExitCode.TESTS_FAILED) is False

    def test_pytest_exitcode_no_tests_collected_returns_false(self) -> None:
        import pytest as real_pytest
        assert self._run(real_pytest.ExitCode.NO_TESTS_COLLECTED) is False


# ===========================================================================
# PytestTester — __call__: SystemExit normalisation
# ===========================================================================


class TestPytestTesterSystemExit:
    """SystemExit.code normalisation: int / None / empty-str / non-empty str."""

    _NAME = "_tp_sysexit"
    _MOD = _make_package_module(_NAME)

    def _run_sysexit(self, code):
        mock_pytest = MagicMock()
        mock_pytest.main.side_effect = SystemExit(code)
        with patch.dict(sys.modules, {"pytest": mock_pytest, self._NAME: self._MOD}):
            with patch("scikitplot._testing._pytesttester._show_numpy_info"):
                return PytestTester(self._NAME)()

    def test_int_0_returns_true(self) -> None:
        assert self._run_sysexit(0) is True

    def test_int_1_returns_false(self) -> None:
        assert self._run_sysexit(1) is False

    def test_int_2_returns_false(self) -> None:
        assert self._run_sysexit(2) is False

    def test_none_returns_true(self) -> None:
        # POSIX: exit() with no arg -> success
        assert self._run_sysexit(None) is True

    def test_empty_string_returns_true(self) -> None:
        # Empty string is falsy -> treated as success
        assert self._run_sysexit("") is True

    def test_non_empty_string_returns_false(self) -> None:
        assert self._run_sysexit("error!") is False

    def test_non_empty_string_msg_returns_false(self) -> None:
        assert self._run_sysexit("Collection error") is False


# ===========================================================================
# PytestTester — __call__: module path resolution
# ===========================================================================


class TestPytestTesterModulePath:
    """__path__ and __file__ resolution must not crash for any module type."""

    _NAME = "_tp_path"

    def _get_args(self, mod, **kwargs):
        return _pytest_args(self._NAME, mod, **kwargs)

    def test_package_module_resolves_without_error(self) -> None:
        mod = _make_package_module(self._NAME, path="/my/pkg")
        args = self._get_args(mod)
        assert "--pyargs" in args

    def test_single_file_module_resolves_without_error(self) -> None:
        mod = _make_file_module(self._NAME, file_path="/my/mod.py")
        args = self._get_args(mod)
        assert "--pyargs" in args

    def test_module_without_path_or_file_does_not_crash(self) -> None:
        # Bare module: no __path__, no __file__ -> falls back to "."
        mod = types.ModuleType(self._NAME)
        args = self._get_args(mod)
        assert "--pyargs" in args

    def test_coverage_flag_has_non_empty_path(self) -> None:
        mod = _make_package_module(self._NAME, path="/my/pkg")
        args = self._get_args(mod, coverage=True)
        cov = next(a for a in args if a.startswith("--cov="))
        assert cov != "--cov="

    def test_single_file_coverage_has_non_empty_path(self) -> None:
        mod = _make_file_module(self._NAME, file_path="/my/mod.py")
        args = self._get_args(mod, coverage=True)
        cov = next(a for a in args if a.startswith("--cov="))
        assert cov != "--cov="

    def test_show_numpy_info_called_once_per_invocation(self) -> None:
        mod = _make_package_module(self._NAME)
        mock_pytest = MagicMock()
        mock_pytest.main.return_value = 0
        with patch.dict(sys.modules, {"pytest": mock_pytest, self._NAME: mod}):
            with patch("scikitplot._testing._pytesttester._show_numpy_info") as mock_info:
                PytestTester(self._NAME)()
        mock_info.assert_called_once()
