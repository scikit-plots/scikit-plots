# scikitplot/cython/tests/test__public_security_gate.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-SEC-001.

Originally only ``compile_and_load_result`` applied the security policy; the
package build entrypoints (``build_package_from_code_result`` and
``build_package_from_paths_result``) reached the compiler without validation.
These tests pin that *every* stable build path now routes through the same gate
via the shared ``_validate_build_security`` helper: a dangerous compile arg is
rejected before any compiler work, and a permissive policy is honoured.

The malicious inputs raise during validation, so no real compiler is invoked —
these tests are fast and toolchain-independent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._public import (
    _validate_build_security,
    build_package_from_code_result,
    build_package_from_paths_result,
)
from .._security import SecurityError, SecurityPolicy


# A shell metacharacter in a compile arg is rejected by the strict default
# policy (see validate_build_inputs docstring example).
BAD_ARG = "-O2; rm -rf /"


class TestSharedHelper:
    def test_rejects_bad_compile_arg(self) -> None:
        with pytest.raises(SecurityError):
            _validate_build_security(
                security_policy=None,
                sources=("def f(): pass",),
                extra_compile_args=[BAD_ARG],
            )

    def test_rejects_non_policy_type(self) -> None:
        with pytest.raises(TypeError):
            _validate_build_security(security_policy=object(), sources=(None,))

    def test_permissive_policy_allows_bad_arg(self) -> None:
        # NOTE: as of R19 (CYTHON-SEC-002), strict=False also relaxes this
        # guard; here we use the specific allow_* flag (which has always worked
        # and still overrides strict) to keep this test narrowly scoped to R5.
        _validate_build_security(
            security_policy=SecurityPolicy(allow_shell_metacharacters=True),
            sources=(None,),
            extra_compile_args=[BAD_ARG],
        )

    def test_validates_every_source(self) -> None:
        """A violation in ANY module source is caught (per-source loop)."""
        small = SecurityPolicy(max_source_bytes=10)
        with pytest.raises(SecurityError, match="max_source_bytes"):
            _validate_build_security(
                security_policy=small,
                sources=("ok", "x" * 999),  # second source too large
            )


class TestPackageFromCodeGate:
    def test_bad_arg_rejected_before_compile(self) -> None:
        with pytest.raises(SecurityError):
            build_package_from_code_result(
                {"m": "def f(): return 1"},
                package_name="pkg",
                extra_compile_args=[BAD_ARG],
            )

    def test_oversized_module_source_rejected(self) -> None:
        with pytest.raises(SecurityError, match="max_source_bytes"):
            build_package_from_code_result(
                {"m": "x" * 5000},
                package_name="pkg",
                security_policy=SecurityPolicy(max_source_bytes=100),
            )


class TestPackageFromPathsGate:
    def test_bad_arg_rejected_before_compile(self, tmp_path: Path) -> None:
        pyx = tmp_path / "m.pyx"
        pyx.write_text("def f(): return 1\n", encoding="utf-8")
        with pytest.raises(SecurityError):
            build_package_from_paths_result(
                {"m": pyx},
                package_name="pkg",
                extra_compile_args=[BAD_ARG],
            )

    def test_oversized_source_file_rejected(self, tmp_path: Path) -> None:
        pyx = tmp_path / "big.pyx"
        pyx.write_text("x" * 5000, encoding="utf-8")
        with pytest.raises(SecurityError, match="max_source_bytes"):
            build_package_from_paths_result(
                {"big": pyx},
                package_name="pkg",
                security_policy=SecurityPolicy(max_source_bytes=100),
            )
