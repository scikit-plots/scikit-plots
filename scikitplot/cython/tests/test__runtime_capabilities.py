# scikitplot/cython/tests/test__runtime_capabilities.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-ABI-001.

The subinterpreter / free-threaded / unload / fork / finalization contracts were
undefined.  ``runtime_capabilities()`` now declares them, and
``check_runtime_supported()`` fails fast (``UnsupportedRuntimeError``) on a
free-threaded interpreter or a non-main subinterpreter unless explicitly allowed.
Because the sandbox is a standard GIL, main-interpreter build, the unsupported
branches are exercised by simulating each configuration.
"""
from __future__ import annotations

from unittest import mock

import pytest

from .. import _profiles
from .._profiles import (
    RuntimeCapabilities,
    UnsupportedRuntimeError,
    check_runtime_supported,
    runtime_capabilities,
)


class TestRuntimeCapabilities:
    def test_returns_namedtuple(self) -> None:
        caps = runtime_capabilities()
        assert isinstance(caps, RuntimeCapabilities)
        assert caps._fields == (
            "gil_enabled",
            "free_threaded_build",
            "in_main_interpreter",
            "supports_unload",
            "supports_fork_after_load",
            "platform",
        )

    def test_unload_always_false(self) -> None:
        # CPython cannot generally unload native modules.
        assert runtime_capabilities().supports_unload is False

    def test_fields_typed(self) -> None:
        caps = runtime_capabilities()
        assert isinstance(caps.gil_enabled, bool)
        assert isinstance(caps.free_threaded_build, bool)
        assert isinstance(caps.in_main_interpreter, bool)
        assert isinstance(caps.supports_fork_after_load, bool)
        assert isinstance(caps.platform, str)


class TestSafeDefaults:
    def test_standard_interpreter_passes(self) -> None:
        # The sandbox is a standard GIL, main interpreter → no error.
        check_runtime_supported()

    def _caps(self, **overrides) -> RuntimeCapabilities:
        base = RuntimeCapabilities(
            gil_enabled=True,
            free_threaded_build=False,
            in_main_interpreter=True,
            supports_unload=False,
            supports_fork_after_load=True,
            platform="linux",
        )
        return base._replace(**overrides)

    def test_free_threaded_rejected_by_default(self) -> None:
        with mock.patch.object(
            _profiles,
            "runtime_capabilities",
            return_value=self._caps(free_threaded_build=True, gil_enabled=False),
        ):
            with pytest.raises(UnsupportedRuntimeError, match="free-threaded"):
                check_runtime_supported()

    def test_free_threaded_opt_in(self) -> None:
        with mock.patch.object(
            _profiles,
            "runtime_capabilities",
            return_value=self._caps(free_threaded_build=True, gil_enabled=False),
        ):
            check_runtime_supported(allow_free_threaded=True)

    def test_subinterpreter_rejected_by_default(self) -> None:
        with mock.patch.object(
            _profiles,
            "runtime_capabilities",
            return_value=self._caps(in_main_interpreter=False),
        ):
            with pytest.raises(UnsupportedRuntimeError, match="subinterpreter"):
                check_runtime_supported()

    def test_subinterpreter_opt_in(self) -> None:
        with mock.patch.object(
            _profiles,
            "runtime_capabilities",
            return_value=self._caps(in_main_interpreter=False),
        ):
            check_runtime_supported(allow_subinterpreter=True)

    def test_both_unsupported_free_threaded_reported_first(self) -> None:
        with mock.patch.object(
            _profiles,
            "runtime_capabilities",
            return_value=self._caps(
                free_threaded_build=True, in_main_interpreter=False
            ),
        ):
            with pytest.raises(UnsupportedRuntimeError, match="free-threaded"):
                check_runtime_supported()


class TestUnsupportedRuntimeError:
    def test_is_runtime_error(self) -> None:
        assert issubclass(UnsupportedRuntimeError, RuntimeError)
