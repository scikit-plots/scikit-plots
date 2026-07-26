# scikitplot/cython/tests/test__security_strict.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-SEC-002.

``SecurityPolicy.strict`` was descriptive only: the individual guards checked
the ``allow_*`` flags directly and ignored ``strict``, so ``strict=False`` did
not actually relax anything (surfaced while writing the R5 tests).  ``strict`` is
now operative — an unset ``allow_*`` flag follows ``strict`` — while an explicit
per-flag value always overrides it.
"""
from __future__ import annotations

import pytest

from .._security import SecurityError, SecurityPolicy, validate_build_inputs

BAD_ARG = "-O2; rm -rf /"  # shell metacharacter
ALLOW_FLAGS = [
    "allow_absolute_include_dirs",
    "allow_shell_metacharacters",
    "allow_reserved_macros",
    "allow_dangerous_compiler_args",
]


class TestStrictResolvesFlags:
    def test_strict_true_flags_restrictive(self) -> None:
        p = SecurityPolicy(strict=True)
        for flag in ALLOW_FLAGS:
            assert getattr(p, flag) is False, flag

    def test_strict_false_flags_permissive(self) -> None:
        p = SecurityPolicy(strict=False)
        for flag in ALLOW_FLAGS:
            assert getattr(p, flag) is True, flag

    def test_default_is_strict(self) -> None:
        p = SecurityPolicy()
        assert p.strict is True
        assert p.allow_shell_metacharacters is False


class TestStrictIsOperative:
    def test_strict_true_rejects_shell_meta(self) -> None:
        with pytest.raises(SecurityError):
            validate_build_inputs(
                policy=SecurityPolicy(strict=True), extra_compile_args=[BAD_ARG]
            )

    def test_strict_false_permits_shell_meta(self) -> None:
        # The core fix: strict=False now actually relaxes the guard.
        validate_build_inputs(
            policy=SecurityPolicy(strict=False), extra_compile_args=[BAD_ARG]
        )


class TestExplicitOverrideWins:
    def test_strict_false_but_relock_one_flag(self) -> None:
        p = SecurityPolicy(strict=False, allow_shell_metacharacters=False)
        assert p.allow_shell_metacharacters is False
        # Other flags still permissive.
        assert p.allow_absolute_include_dirs is True
        with pytest.raises(SecurityError):
            validate_build_inputs(policy=p, extra_compile_args=[BAD_ARG])

    def test_strict_true_but_allow_one_flag(self) -> None:
        p = SecurityPolicy(strict=True, allow_shell_metacharacters=True)
        assert p.allow_shell_metacharacters is True
        assert p.allow_absolute_include_dirs is False
        # No SecurityError from the shell-meta guard now.
        validate_build_inputs(policy=p, extra_compile_args=[BAD_ARG])


class TestRelaxedStillWorks:
    def test_relaxed_permits(self) -> None:
        validate_build_inputs(
            policy=SecurityPolicy.relaxed(), extra_compile_args=[BAD_ARG]
        )
        for flag in ALLOW_FLAGS:
            assert getattr(SecurityPolicy.relaxed(), flag) is True
