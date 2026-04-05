# scikitplot/cython/tests/test__security.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._security`.

Covers
------
- ``SecurityPolicy``        : construction validation, frozen, defaults, singletons
- ``SecurityError``         : field attribute, ValueError subclass
- ``is_safe_path()``        : relative/absolute/tilde/null-byte/dotdot
- ``is_safe_macro_name()``  : identifier syntax, reserved names, allow_reserved flag
- ``is_safe_compiler_arg()``: safe flags, shell meta, dangerous patterns, non-string
- ``validate_build_inputs()``: source size, macros, compile/link args, include dirs,
                               libraries, policy type checking, error messages
"""
from __future__ import annotations

from pathlib import Path

import pytest

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
import os
import sys


class TestSecurityPolicyConstruction:
    def test_negative_max_compile_args_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_extra_compile_args=-1)

    def test_negative_max_link_args_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_extra_link_args=-1)

    def test_negative_max_include_dirs_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_include_dirs=-1)

    def test_negative_max_libraries_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_libraries=-1)

    def test_zero_max_source_bytes_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_source_bytes=0)

    def test_none_max_source_bytes_allowed(self) -> None:
        p = SecurityPolicy(max_source_bytes=None)
        assert p.max_source_bytes is None

    def test_policy_is_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            DEFAULT_SECURITY_POLICY.strict = False  # type: ignore[misc]

    def test_default_policy_strict(self) -> None:
        assert DEFAULT_SECURITY_POLICY.strict is True

    def test_relaxed_policy_disables_guards(self) -> None:
        assert RELAXED_SECURITY_POLICY.strict is False
        assert RELAXED_SECURITY_POLICY.allow_shell_metacharacters is True

    def test_default_policy_max_source_10mib(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_source_bytes == 10 * 1024 * 1024

    def test_default_policy_max_compile_args_64(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_extra_compile_args == 64

    def test_default_policy_max_link_args_64(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_extra_link_args == 64


class TestScenario6SecurityPolicy:
    """
    Security policy: construction, defaults, and validation.

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


class TestIsSafePathEdgeCases:
    def test_dot_current_dir_safe(self) -> None:
        assert is_safe_path(".") is True

    def test_simple_relative_safe(self) -> None:
        assert is_safe_path("include/mylib") is True

    def test_dotdot_unsafe(self) -> None:
        assert is_safe_path("../secret") is False

    def test_dotdot_in_middle_unsafe(self) -> None:
        assert is_safe_path("include/../etc") is False

    def test_tilde_unsafe(self) -> None:
        assert is_safe_path("~/secret") is False

    def test_absolute_unsafe_by_default(self) -> None:
        assert is_safe_path("/usr/local") is False

    def test_absolute_safe_when_allowed(self) -> None:
        assert is_safe_path("/usr/local", allow_absolute=True) is True

    def test_pathlib_path_accepted(self) -> None:
        assert is_safe_path(Path("include/mylib")) is True


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


class TestSecurityUncoveredBranches:
    """Cover the 'else' reason branches in validate_build_inputs (lines 595, 619-624)."""

    def test_null_byte_in_compile_arg_else_branch(self) -> None:
        """A null byte is unsafe but not a shell meta or dangerous pattern — hits else."""
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_compile_args=["-Dfoo\x00bar"],
            )

    def test_non_str_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=[42],  # type: ignore[list-item]
            )

    def test_shell_meta_in_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="shell metacharacter"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=["-L/tmp; rm -rf /"],
            )

    def test_dangerous_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="dangerous"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=["--specs=/tmp/evil.specs"],
            )

    def test_null_byte_in_link_arg_else_branch(self) -> None:
        """Null byte in link arg hits the 'else' reason branch."""
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=["-lfoo\x00bar"],
            )

    def test_too_many_link_args_independent_limit(self) -> None:
        policy = SecurityPolicy(max_extra_link_args=2)
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=policy,
                source=None,
                extra_link_args=["-la", "-lb", "-lc"],
            )

    def test_relaxed_policy_allows_shell_meta_in_link(self) -> None:
        validate_build_inputs(
            policy=RELAXED_SECURITY_POLICY,
            source=None,
            extra_link_args=["-L/tmp; echo hi"],
        )  # Should NOT raise

    def test_is_safe_path_null_byte(self) -> None:
        assert is_safe_path("foo\x00bar") is False

    def test_is_safe_macro_name_with_allowed_reserved(self) -> None:
        assert is_safe_macro_name("NDEBUG", allow_reserved=True) is True

    def test_is_safe_compiler_arg_non_string(self) -> None:
        assert is_safe_compiler_arg(42) is False  # type: ignore[arg-type]

    def test_validate_build_inputs_source_exactly_at_limit(self) -> None:
        policy = SecurityPolicy(max_source_bytes=10)
        validate_build_inputs(
            policy=policy,
            source="a" * 10,
        )  # Exactly at limit — should not raise

    def test_validate_build_inputs_source_over_limit(self) -> None:
        policy = SecurityPolicy(max_source_bytes=5)
        with pytest.raises(SecurityError, match="source"):
            validate_build_inputs(policy=policy, source="a" * 6)

    def test_validate_build_inputs_none_source_skips_check(self) -> None:
        policy = SecurityPolicy(max_source_bytes=1)
        validate_build_inputs(policy=policy, source=None)  # should not raise

    def test_validate_build_inputs_wrong_policy_type(self) -> None:
        with pytest.raises(TypeError, match="SecurityPolicy"):
            validate_build_inputs(policy="strict", source=None)  # type: ignore[arg-type]

    def test_security_error_field_attribute(self) -> None:
        err = SecurityError("msg", field="extra_compile_args")
        assert err.field == "extra_compile_args"

    def test_security_error_no_field(self) -> None:
        err = SecurityError("msg")
        assert err.field is None


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


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("-O2", True),
        ("-DFOO=1", True),
        ("-L/tmp; rm -rf /", False),
        ("--specs=/tmp/evil", False),
        ("-lfoo\x00bar", False),
        ("--imacros=/etc/shadow", False),
    ],
)
def test_is_safe_compiler_arg_parametric(arg: str, expected: bool) -> None:
    assert is_safe_compiler_arg(arg) is expected


@pytest.mark.parametrize(
    "name,valid",
    [
        ("MY_MACRO", True),
        ("_PRIVATE", True),
        ("", False),
        ("1INVALID", False),
        ("NDEBUG", False),       # reserved
        ("PY_LIMITED_API", False),
        ("PY_DEBUG", False),
    ],
)
def test_is_safe_macro_name_parametric(name: str, valid: bool) -> None:
    assert is_safe_macro_name(name) is valid
