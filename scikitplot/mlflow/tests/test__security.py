# scikitplot/mlflow/tests/test__security.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._security.

Naming convention: test__<module_name>.py

Covers (target: 90%+ from 36% baseline)
----------------------------------------
- _is_cloud_metadata_host  : IPv4 link-local, IANA shared, provider DNS names, normal hosts
- _has_path_traversal      : POSIX and Windows traversal patterns, safe paths
- SecurityPolicy           : __post_init__ validation, all validate_* methods,
                             DEFAULT_SECURITY_POLICY, RELAXED_SECURITY_POLICY
- get_security_policy      : returns None by default
- set_security_policy      : sets/clears, raises on wrong type
- security_policy          : context manager restores on exit and on exception

Notes
-----
All tests are pure-Python with no external dependencies.
Global policy state is reset via the security_policy context manager to isolate tests.
"""

from __future__ import annotations

import pytest

from scikitplot.mlflow._errors import SecurityPolicyViolationError
from scikitplot.mlflow._security import (
    DEFAULT_SECURITY_POLICY,
    RELAXED_SECURITY_POLICY,
    SecurityPolicy,
    _has_path_traversal,
    _is_cloud_metadata_host,
    get_security_policy,
    security_policy,
    set_security_policy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_policy():
    """Reset global policy to None before and after every test."""
    set_security_policy(None)
    yield
    set_security_policy(None)


# ===========================================================================
# _is_cloud_metadata_host
# ===========================================================================


class TestIsCloudMetadataHost:
    """Tests for the _is_cloud_metadata_host internal helper."""

    def test_aws_link_local_ip(self) -> None:
        assert _is_cloud_metadata_host("169.254.169.254") is True

    def test_link_local_other_ip(self) -> None:
        # Any address in 169.254.0.0/16 is considered metadata
        assert _is_cloud_metadata_host("169.254.1.1") is True

    def test_iana_shared_space(self) -> None:
        # 100.64.0.0/10 (IANA shared address space)
        assert _is_cloud_metadata_host("100.64.0.1") is True

    def test_google_metadata_hostname(self) -> None:
        assert _is_cloud_metadata_host("metadata.google.internal") is True

    def test_instance_data_hostname(self) -> None:
        assert _is_cloud_metadata_host("instance-data") is True

    def test_link_local_hostname(self) -> None:
        assert _is_cloud_metadata_host("link-local") is True

    def test_normal_hostname_is_safe(self) -> None:
        assert _is_cloud_metadata_host("localhost") is False
        assert _is_cloud_metadata_host("example.com") is False
        assert _is_cloud_metadata_host("127.0.0.1") is False

    def test_regular_public_ip_is_safe(self) -> None:
        assert _is_cloud_metadata_host("8.8.8.8") is False
        assert _is_cloud_metadata_host("192.168.1.1") is False

    def test_case_insensitive(self) -> None:
        assert _is_cloud_metadata_host("METADATA.GOOGLE.INTERNAL") is True
        assert _is_cloud_metadata_host("Metadata.Google.Internal") is True

    def test_invalid_ip_string_is_safe(self) -> None:
        # Invalid IP-like string that is not a DNS alias → safe
        assert _is_cloud_metadata_host("999.999.999.999") is False

    def test_empty_string_is_safe(self) -> None:
        assert _is_cloud_metadata_host("") is False


# ===========================================================================
# _has_path_traversal
# ===========================================================================


class TestHasPathTraversal:
    """Tests for the _has_path_traversal internal helper."""

    def test_simple_traversal(self) -> None:
        assert _has_path_traversal("../secret") is True

    def test_nested_traversal(self) -> None:
        assert _has_path_traversal("/home/user/../../etc/passwd") is True

    def test_traversal_at_start(self) -> None:
        assert _has_path_traversal("../foo") is True

    def test_traversal_at_end(self) -> None:
        assert _has_path_traversal("/foo/..") is True

    def test_windows_backslash_traversal(self) -> None:
        assert _has_path_traversal("foo\\..\\bar") is True

    def test_safe_path_no_traversal(self) -> None:
        assert _has_path_traversal("/home/user/project") is False
        assert _has_path_traversal("relative/path/file.txt") is False

    def test_dotfile_is_safe(self) -> None:
        """A single dot component like .git or .env is NOT traversal."""
        assert _has_path_traversal("/home/user/.git") is False
        assert _has_path_traversal(".env") is False

    def test_double_dot_in_filename_is_safe(self) -> None:
        """'file..txt' is not a traversal sequence."""
        assert _has_path_traversal("/path/file..txt") is False

    def test_empty_path_is_safe(self) -> None:
        assert _has_path_traversal("") is False


# ===========================================================================
# SecurityPolicy.__post_init__ validation
# ===========================================================================


class TestSecurityPolicyPostInit:
    """Tests for SecurityPolicy dataclass __post_init__ validation."""

    def test_valid_default_construction(self) -> None:
        """Default construction must succeed without errors."""
        policy = SecurityPolicy()
        assert policy.max_env_value_length == 65536
        assert policy.max_env_pairs == 256

    def test_max_env_value_length_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_env_value_length"):
            SecurityPolicy(max_env_value_length=0)

    def test_max_env_value_length_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_env_value_length"):
            SecurityPolicy(max_env_value_length=-1)

    def test_max_env_pairs_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_env_pairs"):
            SecurityPolicy(max_env_pairs=0)

    def test_max_env_pairs_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_env_pairs"):
            SecurityPolicy(max_env_pairs=-1)

    def test_blocked_prefix_non_string_raises(self) -> None:
        with pytest.raises(TypeError, match="blocked_env_key_prefixes"):
            SecurityPolicy(blocked_env_key_prefixes=frozenset({123}))  # type: ignore[arg-type]

    def test_valid_custom_policy(self) -> None:
        policy = SecurityPolicy(
            max_env_value_length=1024,
            max_env_pairs=10,
            blocked_env_key_prefixes=frozenset({"LD_", "DYLD_"}),
        )
        assert policy.max_env_value_length == 1024
        assert policy.max_env_pairs == 10


# ===========================================================================
# SecurityPolicy.validate_tracking_uri
# ===========================================================================


class TestValidateTrackingUri:
    """Tests for SecurityPolicy.validate_tracking_uri."""

    _policy = DEFAULT_SECURITY_POLICY

    def test_http_uri_passes(self) -> None:
        self._policy.validate_tracking_uri("http://localhost:5000")

    def test_https_uri_passes(self) -> None:
        self._policy.validate_tracking_uri("https://mlflow.example.com")

    def test_file_uri_passes(self) -> None:
        self._policy.validate_tracking_uri("file:///home/user/mlruns")

    def test_sqlite_uri_passes(self) -> None:
        self._policy.validate_tracking_uri("sqlite:///mlflow.db")

    def test_disallowed_scheme_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="scheme"):
            self._policy.validate_tracking_uri("ftp://badserver/mlruns")

    def test_cloud_metadata_host_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metadata"):
            self._policy.validate_tracking_uri("http://169.254.169.254/mlruns")

    def test_path_traversal_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="traversal"):
            self._policy.validate_tracking_uri("file:///home/../etc/passwd")

    def test_empty_scheme_set_allows_all(self) -> None:
        """When allowed_tracking_uri_schemes is empty, no scheme enforcement."""
        p = SecurityPolicy(allowed_tracking_uri_schemes=frozenset())
        p.validate_tracking_uri("ftp://anything")

    def test_context_label_in_error_message(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="my-context"):
            self._policy.validate_tracking_uri("ftp://bad", context="my-context")

    def test_cloud_metadata_only_for_http_schemes(self) -> None:
        """Cloud metadata check applies only to http/https URIs."""
        # file:// with metadata-looking path doesn't trigger SSRF check
        p = SecurityPolicy(allowed_tracking_uri_schemes=frozenset({"file"}))
        # Should not raise cloud metadata error (only scheme/path checks apply)
        p.validate_tracking_uri("file:///169.254.169.254/data")


# ===========================================================================
# SecurityPolicy.validate_server_config
# ===========================================================================


class TestValidateServerConfig:
    """Tests for SecurityPolicy.validate_server_config."""

    def _make_server_cfg(self, **overrides):
        """Build a minimal server config namespace."""
        defaults = {
            "dev": False,
            "disable_security_middleware": False,
            "cors_allowed_origins": None,
            "allowed_hosts": None,
            "x_frame_options": None,
            "gunicorn_opts": None,
            "uvicorn_opts": None,
            "waitress_opts": None,
            "extra_args": None,
        }
        defaults.update(overrides)
        from types import SimpleNamespace
        return SimpleNamespace(**defaults)

    def test_valid_default_cfg_passes(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_server_config(self._make_server_cfg())

    def test_dev_mode_blocked_by_default(self) -> None:
        cfg = self._make_server_cfg(dev=True)
        with pytest.raises(SecurityPolicyViolationError, match="dev=True"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_dev_mode_allowed_by_relaxed(self) -> None:
        cfg = self._make_server_cfg(dev=True)
        RELAXED_SECURITY_POLICY.validate_server_config(cfg)

    def test_disable_security_middleware_blocked(self) -> None:
        cfg = self._make_server_cfg(disable_security_middleware=True)
        with pytest.raises(SecurityPolicyViolationError, match="disable_security_middleware"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_cors_wildcard_blocked(self) -> None:
        cfg = self._make_server_cfg(cors_allowed_origins="*")
        with pytest.raises(SecurityPolicyViolationError, match="wildcard"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_cors_wildcard_allowed_by_relaxed(self) -> None:
        cfg = self._make_server_cfg(cors_allowed_origins="*")
        RELAXED_SECURITY_POLICY.validate_server_config(cfg)

    def test_header_injection_in_cors_raises(self) -> None:
        cfg = self._make_server_cfg(cors_allowed_origins="good\r\nX-Injected: evil")
        with pytest.raises(SecurityPolicyViolationError, match="header injection"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_header_injection_in_x_frame_options_raises(self) -> None:
        cfg = self._make_server_cfg(x_frame_options="DENY\nX-Evil: yes")
        with pytest.raises(SecurityPolicyViolationError, match="header injection"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_header_injection_in_allowed_hosts_raises(self) -> None:
        cfg = self._make_server_cfg(allowed_hosts="host\r\nX-Evil: yes")
        with pytest.raises(SecurityPolicyViolationError, match="header injection"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_shell_meta_in_gunicorn_opts_raises(self) -> None:
        cfg = self._make_server_cfg(gunicorn_opts="--workers 4; rm -rf /")
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_shell_meta_in_extra_args_raises(self) -> None:
        cfg = self._make_server_cfg(extra_args=["--workers", "4; rm -rf /"])
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_shell_meta_allowed_by_relaxed(self) -> None:
        cfg = self._make_server_cfg(gunicorn_opts="--workers 4; echo hi")
        RELAXED_SECURITY_POLICY.validate_server_config(cfg)

    def test_context_in_error_message(self) -> None:
        cfg = self._make_server_cfg(dev=True)
        with pytest.raises(SecurityPolicyViolationError, match="myctx"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg, context="myctx")


# ===========================================================================
# SecurityPolicy.validate_session_config
# ===========================================================================


class TestValidateSessionConfig:
    """Tests for SecurityPolicy.validate_session_config."""

    def _make_session_cfg(self, **overrides):
        from types import SimpleNamespace
        defaults = {
            "tracking_uri": None,
            "registry_uri": None,
            "env_file": None,
            "extra_env": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_none_fields_pass(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_session_config(self._make_session_cfg())

    def test_valid_http_tracking_uri_passes(self) -> None:
        cfg = self._make_session_cfg(tracking_uri="http://localhost:5000")
        DEFAULT_SECURITY_POLICY.validate_session_config(cfg)

    def test_disallowed_tracking_scheme_raises(self) -> None:
        cfg = self._make_session_cfg(tracking_uri="ftp://bad-server")
        with pytest.raises(SecurityPolicyViolationError):
            DEFAULT_SECURITY_POLICY.validate_session_config(cfg)

    def test_disallowed_registry_scheme_raises(self) -> None:
        cfg = self._make_session_cfg(registry_uri="ftp://bad-registry")
        with pytest.raises(SecurityPolicyViolationError):
            DEFAULT_SECURITY_POLICY.validate_session_config(cfg)

    def test_path_traversal_in_env_file_raises(self) -> None:
        cfg = self._make_session_cfg(env_file="../../../etc/secret.env")
        with pytest.raises(SecurityPolicyViolationError, match="traversal"):
            DEFAULT_SECURITY_POLICY.validate_session_config(cfg)

    def test_blocked_env_key_raises(self) -> None:
        cfg = self._make_session_cfg(extra_env={"LD_PRELOAD": "/evil.so"})
        with pytest.raises(SecurityPolicyViolationError, match="LD_"):
            DEFAULT_SECURITY_POLICY.validate_session_config(cfg)

    def test_safe_extra_env_passes(self) -> None:
        cfg = self._make_session_cfg(extra_env={"MY_APP_KEY": "value"})
        DEFAULT_SECURITY_POLICY.validate_session_config(cfg)


# ===========================================================================
# SecurityPolicy.validate_env_item
# ===========================================================================


class TestValidateEnvItem:
    """Tests for SecurityPolicy.validate_env_item."""

    def test_safe_item_passes(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_env_item("MY_KEY", "my_value")

    def test_ld_preload_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="LD_"):
            DEFAULT_SECURITY_POLICY.validate_env_item("LD_PRELOAD", "/evil.so")

    def test_ld_library_path_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="LD_"):
            DEFAULT_SECURITY_POLICY.validate_env_item("LD_LIBRARY_PATH", "/evil")

    def test_value_too_long_raises(self) -> None:
        p = SecurityPolicy(max_env_value_length=10)
        with pytest.raises(SecurityPolicyViolationError, match="max_env_value_length"):
            p.validate_env_item("KEY", "x" * 11)

    def test_value_at_max_length_passes(self) -> None:
        p = SecurityPolicy(max_env_value_length=5)
        p.validate_env_item("KEY", "x" * 5)

    def test_crlf_in_value_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_env_item("KEY", "value\r\nX-Evil: yes")

    def test_lf_in_value_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_env_item("KEY", "value\nevil")

    def test_context_in_error_message(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="myenv"):
            DEFAULT_SECURITY_POLICY.validate_env_item(
                "LD_PRELOAD", "/evil.so", context="myenv"
            )


# ===========================================================================
# SecurityPolicy.validate_env_mapping
# ===========================================================================


class TestValidateEnvMapping:
    """Tests for SecurityPolicy.validate_env_mapping."""

    def test_empty_mapping_passes(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_env_mapping({})

    def test_safe_mapping_passes(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_env_mapping({"KEY1": "val1", "KEY2": "val2"})

    def test_exceeds_max_pairs_raises(self) -> None:
        p = SecurityPolicy(max_env_pairs=2)
        env = {f"KEY{i}": "v" for i in range(3)}
        with pytest.raises(SecurityPolicyViolationError, match="max_env_pairs"):
            p.validate_env_mapping(env)

    def test_at_max_pairs_passes(self) -> None:
        p = SecurityPolicy(max_env_pairs=2)
        env = {"KEY1": "v1", "KEY2": "v2"}
        p.validate_env_mapping(env)

    def test_blocked_key_in_mapping_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="LD_"):
            DEFAULT_SECURITY_POLICY.validate_env_mapping(
                {"SAFE": "ok", "LD_PRELOAD": "/evil.so"}
            )


# ===========================================================================
# SecurityPolicy.validate_path
# ===========================================================================


class TestValidatePath:
    """Tests for SecurityPolicy.validate_path."""

    def test_safe_path_passes(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_path("/home/user/project")

    def test_traversal_path_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="traversal"):
            DEFAULT_SECURITY_POLICY.validate_path("/home/user/../etc/passwd")

    def test_traversal_disabled_passes(self) -> None:
        p = SecurityPolicy(block_path_traversal=False)
        p.validate_path("../any/path")  # must not raise

    def test_context_in_error(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="my_path_ctx"):
            DEFAULT_SECURITY_POLICY.validate_path("../secret", context="my_path_ctx")


# ===========================================================================
# SecurityPolicy.validate_cli_arg_value
# ===========================================================================


class TestValidateCliArgValue:
    """Tests for SecurityPolicy.validate_cli_arg_value."""

    def test_safe_value_passes(self) -> None:
        DEFAULT_SECURITY_POLICY.validate_cli_arg_value("--workers 4")

    def test_semicolon_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("value; rm -rf /")

    def test_pipe_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("value | cat /etc/passwd")

    def test_backtick_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("`id`")

    def test_dollar_paren_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("$(id)")

    def test_crlf_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("good\r\nevil")

    def test_metacharacters_disabled_passes(self) -> None:
        p = SecurityPolicy(block_shell_metacharacters_in_args=False)
        p.validate_cli_arg_value("value; rm -rf /")  # must not raise

    def test_context_in_error(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="gunicorn_opts"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value(
                "value; evil", context="gunicorn_opts"
            )


# ===========================================================================
# Built-in presets: DEFAULT_SECURITY_POLICY and RELAXED_SECURITY_POLICY
# ===========================================================================


class TestBuiltinPresets:
    """Tests for DEFAULT_SECURITY_POLICY and RELAXED_SECURITY_POLICY."""

    def test_default_policy_is_security_policy_instance(self) -> None:
        assert isinstance(DEFAULT_SECURITY_POLICY, SecurityPolicy)

    def test_relaxed_policy_is_security_policy_instance(self) -> None:
        assert isinstance(RELAXED_SECURITY_POLICY, SecurityPolicy)

    def test_default_blocks_dev(self) -> None:
        assert DEFAULT_SECURITY_POLICY.allow_dev_mode is False

    def test_default_blocks_cors_wildcard(self) -> None:
        assert DEFAULT_SECURITY_POLICY.allow_cors_wildcard is False

    def test_default_blocks_ld_prefix(self) -> None:
        assert "LD_" in DEFAULT_SECURITY_POLICY.blocked_env_key_prefixes

    def test_relaxed_allows_dev(self) -> None:
        assert RELAXED_SECURITY_POLICY.allow_dev_mode is True

    def test_relaxed_allows_cors_wildcard(self) -> None:
        assert RELAXED_SECURITY_POLICY.allow_cors_wildcard is True

    def test_relaxed_has_empty_blocked_prefixes(self) -> None:
        assert RELAXED_SECURITY_POLICY.blocked_env_key_prefixes == frozenset()

    def test_relaxed_has_larger_env_value_length(self) -> None:
        assert RELAXED_SECURITY_POLICY.max_env_value_length > DEFAULT_SECURITY_POLICY.max_env_value_length

    def test_default_includes_standard_schemes(self) -> None:
        for scheme in ("http", "https", "file", "sqlite"):
            assert scheme in DEFAULT_SECURITY_POLICY.allowed_tracking_uri_schemes

    def test_relaxed_includes_db_schemes(self) -> None:
        for scheme in ("postgresql", "mysql", "mssql"):
            assert scheme in RELAXED_SECURITY_POLICY.allowed_tracking_uri_schemes


# ===========================================================================
# get_security_policy / set_security_policy
# ===========================================================================


class TestGetSetSecurityPolicy:
    """Tests for global policy get/set functions."""

    def test_default_is_none(self) -> None:
        assert get_security_policy() is None

    def test_set_policy_makes_it_active(self) -> None:
        set_security_policy(DEFAULT_SECURITY_POLICY)
        assert get_security_policy() is DEFAULT_SECURITY_POLICY

    def test_set_none_clears_policy(self) -> None:
        set_security_policy(DEFAULT_SECURITY_POLICY)
        set_security_policy(None)
        assert get_security_policy() is None

    def test_set_non_policy_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="SecurityPolicy"):
            set_security_policy("not-a-policy")  # type: ignore[arg-type]

    def test_set_integer_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            set_security_policy(42)  # type: ignore[arg-type]


# ===========================================================================
# security_policy context manager
# ===========================================================================


class TestSecurityPolicyContextManager:
    """Tests for the security_policy context manager."""

    def test_activates_policy_inside_block(self) -> None:
        assert get_security_policy() is None
        with security_policy(DEFAULT_SECURITY_POLICY):
            assert get_security_policy() is DEFAULT_SECURITY_POLICY

    def test_restores_previous_policy_on_exit(self) -> None:
        set_security_policy(None)
        with security_policy(DEFAULT_SECURITY_POLICY):
            pass
        assert get_security_policy() is None

    def test_restores_on_exception(self) -> None:
        set_security_policy(None)
        with pytest.raises(ValueError):
            with security_policy(DEFAULT_SECURITY_POLICY):
                raise ValueError("boom")
        assert get_security_policy() is None

    def test_none_disables_enforcement_inside_block(self) -> None:
        set_security_policy(DEFAULT_SECURITY_POLICY)
        with security_policy(None):
            assert get_security_policy() is None
        assert get_security_policy() is DEFAULT_SECURITY_POLICY

    def test_nested_context_managers(self) -> None:
        """Nested security_policy blocks must each restore their predecessor."""
        set_security_policy(None)
        with security_policy(DEFAULT_SECURITY_POLICY):
            assert get_security_policy() is DEFAULT_SECURITY_POLICY
            with security_policy(RELAXED_SECURITY_POLICY):
                assert get_security_policy() is RELAXED_SECURITY_POLICY
            assert get_security_policy() is DEFAULT_SECURITY_POLICY
        assert get_security_policy() is None


# ===========================================================================
# Additional edge-case tests (coverage gap fill)
# ===========================================================================


class TestIsCloudMetadataHostEdgeCases:
    """Edge cases for _is_cloud_metadata_host not covered by the primary suite."""

    def test_iana_shared_space_last_address(self) -> None:
        """100.127.255.255 is the last address in 100.64.0.0/10 → metadata."""
        assert _is_cloud_metadata_host("100.127.255.255") is True

    def test_just_outside_iana_shared_space(self) -> None:
        """100.128.0.0 is the first address OUTSIDE 100.64.0.0/10 → safe."""
        assert _is_cloud_metadata_host("100.128.0.0") is False

    def test_link_local_first_address(self) -> None:
        """169.254.0.1 is inside 169.254.0.0/16 → metadata."""
        assert _is_cloud_metadata_host("169.254.0.1") is True

    def test_link_local_last_address(self) -> None:
        """169.254.255.254 is still in the link-local range → metadata."""
        assert _is_cloud_metadata_host("169.254.255.254") is True

    def test_whitespace_around_ip_is_normalised(self) -> None:
        """Leading/trailing whitespace is stripped before classification."""
        assert _is_cloud_metadata_host("  169.254.169.254  ") is True

    def test_ipv6_loopback_is_safe(self) -> None:
        """IPv6 addresses are not in the checked ranges → safe."""
        assert _is_cloud_metadata_host("::1") is False

    def test_ipv6_address_is_safe(self) -> None:
        """Arbitrary IPv6 addresses are treated as safe (no v6 CIDR checks)."""
        assert _is_cloud_metadata_host("2001:db8::1") is False


class TestHasPathTraversalEdgeCases:
    """Edge cases for _has_path_traversal not covered by the primary suite."""

    def test_triple_dot_is_safe(self) -> None:
        """'...' (three dots) is not a traversal component."""
        assert _has_path_traversal("/path/.../file") is False

    def test_dotdot_prefix_in_component_is_safe(self) -> None:
        """A filename that starts with '..' but is longer is not traversal."""
        assert _has_path_traversal("/path/..hidden") is False

    def test_consecutive_traversals(self) -> None:
        """Multiple back-to-back traversals are all detected."""
        assert _has_path_traversal("../../etc/passwd") is True

    def test_mixed_slash_backslash_traversal(self) -> None:
        """Mixed separators are normalised before checking."""
        assert _has_path_traversal("a/..\\b") is True


class TestSecurityPolicyPostInitEdgeCases:
    """Additional __post_init__ boundary tests."""

    def test_max_env_value_length_one_passes(self) -> None:
        """Minimum positive value is valid."""
        p = SecurityPolicy(max_env_value_length=1)
        assert p.max_env_value_length == 1

    def test_max_env_pairs_one_passes(self) -> None:
        """Minimum positive value is valid."""
        p = SecurityPolicy(max_env_pairs=1)
        assert p.max_env_pairs == 1

    def test_empty_blocked_prefixes_passes(self) -> None:
        """An empty frozenset for blocked_env_key_prefixes is valid."""
        p = SecurityPolicy(blocked_env_key_prefixes=frozenset())
        assert p.blocked_env_key_prefixes == frozenset()

    def test_multiple_blocked_prefixes_stored(self) -> None:
        """Multiple valid prefixes are all stored."""
        p = SecurityPolicy(blocked_env_key_prefixes=frozenset({"LD_", "DYLD_", "ASAN_"}))
        assert "LD_" in p.blocked_env_key_prefixes
        assert "DYLD_" in p.blocked_env_key_prefixes
        assert "ASAN_" in p.blocked_env_key_prefixes


class TestValidateCliArgValueEdgeCases:
    """Edge cases and flag-interaction tests for validate_cli_arg_value."""

    # ------------------------------------------------------------------
    # CRLF takes priority over generic shell-meta when both flags are on
    # ------------------------------------------------------------------

    def test_lf_only_raises_cr_or_lf(self) -> None:
        """A lone LF character must produce a 'CR or LF' message."""
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("line1\nline2")

    def test_cr_only_raises_cr_or_lf(self) -> None:
        """A lone CR character must produce a 'CR or LF' message."""
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("line1\rline2")

    def test_crlf_raises_cr_or_lf_not_metacharacters(self) -> None:
        """CRLF must be reported as 'CR or LF', not 'metacharacters' (ordering test)."""
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("good\r\nevil")

    # ------------------------------------------------------------------
    # Only block_header_injection=True, shell-meta disabled
    # ------------------------------------------------------------------

    def test_crlf_raises_when_only_header_injection_blocked(self) -> None:
        """CRLF raises even when shell-meta checking is disabled."""
        p = SecurityPolicy(block_shell_metacharacters_in_args=False, block_header_injection=True)
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            p.validate_cli_arg_value("good\r\nevil")

    def test_shell_meta_passes_when_only_shell_meta_disabled(self) -> None:
        """Shell metacharacters are accepted when block_shell_metacharacters_in_args=False."""
        p = SecurityPolicy(block_shell_metacharacters_in_args=False, block_header_injection=True)
        p.validate_cli_arg_value("value; rm -rf /")  # must not raise

    # ------------------------------------------------------------------
    # block_header_injection=False paths
    # ------------------------------------------------------------------

    def test_crlf_passes_when_header_injection_disabled(self) -> None:
        """CRLF is accepted when block_header_injection=False."""
        p = SecurityPolicy(block_header_injection=False)
        p.validate_cli_arg_value("good\r\nevil")  # must not raise

    def test_both_disabled_passes_all_dangerous_chars(self) -> None:
        """With both checks off, any value is accepted."""
        p = SecurityPolicy(
            block_shell_metacharacters_in_args=False,
            block_header_injection=False,
        )
        p.validate_cli_arg_value("val; `rm -rf /`\r\nX-Evil: yes")  # must not raise

    # ------------------------------------------------------------------
    # Rejected value appears in error repr
    # ------------------------------------------------------------------

    def test_rejected_value_in_error_repr(self) -> None:
        """The rejected value must be included verbatim in the error message."""
        bad = "value; evil"
        with pytest.raises(SecurityPolicyViolationError) as exc_info:
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value(bad)
        assert repr(bad) in str(exc_info.value)

    # ------------------------------------------------------------------
    # Additional shell metacharacter coverage
    # ------------------------------------------------------------------

    def test_ampersand_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("cmd1 & cmd2")

    def test_double_ampersand_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("cmd1 && cmd2")

    def test_double_pipe_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("cmd1 || cmd2")

    def test_redirect_in_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("cmd < /etc/passwd")

    def test_redirect_out_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("cmd > /tmp/out")

    def test_dollar_sign_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("$HOME/path")

    def test_brace_expansion_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_cli_arg_value("{a,b}")

    def test_empty_value_passes(self) -> None:
        """Empty string contains no metacharacters."""
        DEFAULT_SECURITY_POLICY.validate_cli_arg_value("")

    def test_plain_flag_value_passes(self) -> None:
        """A typical CLI flag value with numbers and dashes must pass."""
        DEFAULT_SECURITY_POLICY.validate_cli_arg_value("--timeout=30")


class TestValidateServerConfigEdgeCases:
    """Additional coverage for validate_server_config."""

    def _make_server_cfg(self, **overrides):
        from types import SimpleNamespace
        defaults = {
            "dev": False,
            "disable_security_middleware": False,
            "cors_allowed_origins": None,
            "allowed_hosts": None,
            "x_frame_options": None,
            "gunicorn_opts": None,
            "uvicorn_opts": None,
            "waitress_opts": None,
            "extra_args": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_shell_meta_in_uvicorn_opts_raises(self) -> None:
        cfg = self._make_server_cfg(uvicorn_opts="--loop uvloop; evil")
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_shell_meta_in_waitress_opts_raises(self) -> None:
        cfg = self._make_server_cfg(waitress_opts="--threads 4 | evil")
        with pytest.raises(SecurityPolicyViolationError, match="metacharacters"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_empty_extra_args_list_passes(self) -> None:
        """An empty extra_args list produces no errors."""
        cfg = self._make_server_cfg(extra_args=[])
        DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_safe_extra_args_list_passes(self) -> None:
        """Valid extra_args strings produce no errors."""
        cfg = self._make_server_cfg(extra_args=["--workers", "4", "--timeout", "30"])
        DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_crlf_in_extra_arg_raises_cr_or_lf(self) -> None:
        """A CRLF in extra_args must raise 'CR or LF', not metacharacters."""
        cfg = self._make_server_cfg(extra_args=["good\r\nevil"])
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_crlf_in_gunicorn_opts_raises_cr_or_lf(self) -> None:
        """CRLF in gunicorn_opts must raise 'CR or LF' message."""
        cfg = self._make_server_cfg(gunicorn_opts="--workers 4\r\nX-Injected: evil")
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_lf_in_x_frame_options_raises_header_injection(self) -> None:
        """LF in x_frame_options is caught by the header-injection check."""
        cfg = self._make_server_cfg(x_frame_options="DENY\nX-Evil: yes")
        with pytest.raises(SecurityPolicyViolationError, match="header injection"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_cors_non_wildcard_safe_value_passes(self) -> None:
        """A specific CORS origin without injection characters passes."""
        cfg = self._make_server_cfg(cors_allowed_origins="https://app.example.com")
        DEFAULT_SECURITY_POLICY.validate_server_config(cfg)

    def test_missing_attrs_are_ignored(self) -> None:
        """validate_server_config uses getattr with None defaults so missing fields are safe."""
        from types import SimpleNamespace
        # A completely empty namespace must not crash
        DEFAULT_SECURITY_POLICY.validate_server_config(SimpleNamespace())

    def test_context_propagated_to_extra_args_error(self) -> None:
        """The context label must appear in errors raised for extra_args."""
        cfg = self._make_server_cfg(extra_args=["evil; cmd"])
        with pytest.raises(SecurityPolicyViolationError, match="myctx"):
            DEFAULT_SECURITY_POLICY.validate_server_config(cfg, context="myctx")


class TestValidateTrackingUriEdgeCases:
    """Additional coverage for validate_tracking_uri."""

    def test_cloud_metadata_check_disabled_allows_metadata_ip(self) -> None:
        """When block_cloud_metadata_hosts=False the SSRF guard is bypassed."""
        p = SecurityPolicy(block_cloud_metadata_hosts=False)
        p.validate_tracking_uri("http://169.254.169.254/api")  # must not raise

    def test_path_traversal_check_disabled_allows_dotdot(self) -> None:
        """When block_path_traversal=False traversal sequences are accepted."""
        p = SecurityPolicy(block_path_traversal=False)
        p.validate_tracking_uri("file:///home/../etc/passwd")  # must not raise

    def test_sqlite_with_traversal_raises(self) -> None:
        """sqlite:// URIs are also subject to path-traversal checks."""
        with pytest.raises(SecurityPolicyViolationError, match="traversal"):
            DEFAULT_SECURITY_POLICY.validate_tracking_uri("sqlite:///../../../evil.db")

    def test_empty_uri_path_does_not_raise_traversal(self) -> None:
        """A URI with an empty path component must not trigger traversal error."""
        DEFAULT_SECURITY_POLICY.validate_tracking_uri("http://localhost:5000")

    def test_https_with_metadata_ip_raises(self) -> None:
        """https:// URIs also trigger the cloud-metadata SSRF guard."""
        with pytest.raises(SecurityPolicyViolationError, match="metadata"):
            DEFAULT_SECURITY_POLICY.validate_tracking_uri("https://169.254.169.254/")


class TestValidateEnvItemEdgeCases:
    """Additional coverage for validate_env_item."""

    def test_dyld_prefix_not_blocked_by_default(self) -> None:
        """By default only 'LD_' is blocked, not 'DYLD_'."""
        DEFAULT_SECURITY_POLICY.validate_env_item("DYLD_INSERT_LIBRARIES", "lib.dylib")

    def test_dyld_blocked_when_prefix_added(self) -> None:
        """Custom policy can extend blocked prefixes."""
        p = SecurityPolicy(blocked_env_key_prefixes=frozenset({"LD_", "DYLD_"}))
        with pytest.raises(SecurityPolicyViolationError, match="DYLD_"):
            p.validate_env_item("DYLD_INSERT_LIBRARIES", "lib.dylib")

    def test_value_exactly_at_limit_passes(self) -> None:
        """A value whose length equals the limit exactly must pass."""
        p = SecurityPolicy(max_env_value_length=8)
        p.validate_env_item("KEY", "a" * 8)  # must not raise

    def test_value_one_over_limit_raises(self) -> None:
        """A value one byte over the limit must raise."""
        p = SecurityPolicy(max_env_value_length=8)
        with pytest.raises(SecurityPolicyViolationError, match="max_env_value_length"):
            p.validate_env_item("KEY", "a" * 9)

    def test_cr_in_value_raises_cr_or_lf(self) -> None:
        """A bare CR in the value triggers the CR/LF guard."""
        with pytest.raises(SecurityPolicyViolationError, match="CR or LF"):
            DEFAULT_SECURITY_POLICY.validate_env_item("KEY", "value\revil")

    def test_header_injection_disabled_crlf_passes(self) -> None:
        """CRLF in env values is accepted when block_header_injection=False."""
        p = SecurityPolicy(block_header_injection=False)
        p.validate_env_item("KEY", "line1\r\nline2")  # must not raise

    def test_empty_value_passes(self) -> None:
        """Empty string is a valid env value."""
        DEFAULT_SECURITY_POLICY.validate_env_item("MY_KEY", "")


class TestValidateEnvMappingEdgeCases:
    """Additional coverage for validate_env_mapping."""

    def test_exactly_at_max_pairs_passes(self) -> None:
        p = SecurityPolicy(max_env_pairs=3)
        p.validate_env_mapping({"A": "1", "B": "2", "C": "3"})

    def test_one_over_max_pairs_raises(self) -> None:
        p = SecurityPolicy(max_env_pairs=3)
        with pytest.raises(SecurityPolicyViolationError, match="max_env_pairs"):
            p.validate_env_mapping({"A": "1", "B": "2", "C": "3", "D": "4"})

    def test_context_propagated_to_env_item_error(self) -> None:
        """Context label from validate_env_mapping appears in nested item errors."""
        with pytest.raises(SecurityPolicyViolationError, match="my_mapping_ctx"):
            DEFAULT_SECURITY_POLICY.validate_env_mapping(
                {"LD_PRELOAD": "/evil.so"}, context="my_mapping_ctx"
            )


class TestValidatePathEdgeCases:
    """Additional coverage for validate_path."""

    def test_windows_backslash_traversal_raises(self) -> None:
        with pytest.raises(SecurityPolicyViolationError, match="traversal"):
            DEFAULT_SECURITY_POLICY.validate_path("a\\..\\b")

    def test_empty_path_passes(self) -> None:
        """An empty path string does not contain traversal components."""
        DEFAULT_SECURITY_POLICY.validate_path("")

    def test_dotfile_path_passes(self) -> None:
        """A path containing a dotfile (.env) is safe."""
        DEFAULT_SECURITY_POLICY.validate_path("/project/.env")

    def test_path_traversal_disabled_windows_passes(self) -> None:
        """Windows-style traversal is accepted when block_path_traversal=False."""
        p = SecurityPolicy(block_path_traversal=False)
        p.validate_path("a\\..\\b")  # must not raise


class TestBuiltinPresetsEdgeCases:
    """Additional attribute assertions for the built-in policy presets."""

    def test_default_blocks_shell_metacharacters(self) -> None:
        assert DEFAULT_SECURITY_POLICY.block_shell_metacharacters_in_args is True

    def test_default_blocks_header_injection(self) -> None:
        assert DEFAULT_SECURITY_POLICY.block_header_injection is True

    def test_default_blocks_path_traversal(self) -> None:
        assert DEFAULT_SECURITY_POLICY.block_path_traversal is True

    def test_default_blocks_cloud_metadata(self) -> None:
        assert DEFAULT_SECURITY_POLICY.block_cloud_metadata_hosts is True

    def test_default_disallows_disable_security_middleware(self) -> None:
        assert DEFAULT_SECURITY_POLICY.allow_disable_security_middleware is False

    def test_relaxed_allows_disable_security_middleware(self) -> None:
        assert RELAXED_SECURITY_POLICY.allow_disable_security_middleware is True

    def test_relaxed_disables_shell_meta_check(self) -> None:
        assert RELAXED_SECURITY_POLICY.block_shell_metacharacters_in_args is False

    def test_relaxed_disables_header_injection_check(self) -> None:
        assert RELAXED_SECURITY_POLICY.block_header_injection is False

    def test_relaxed_disables_path_traversal_check(self) -> None:
        assert RELAXED_SECURITY_POLICY.block_path_traversal is False

    def test_relaxed_disables_cloud_metadata_check(self) -> None:
        assert RELAXED_SECURITY_POLICY.block_cloud_metadata_hosts is False

    def test_default_allows_spawn_server(self) -> None:
        assert DEFAULT_SECURITY_POLICY.allow_spawn_server is True

    def test_relaxed_max_env_pairs_larger_than_default(self) -> None:
        assert RELAXED_SECURITY_POLICY.max_env_pairs > DEFAULT_SECURITY_POLICY.max_env_pairs

    def test_default_env_value_length_is_64kib(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_env_value_length == 65536

    def test_relaxed_env_value_length_is_1mib(self) -> None:
        assert RELAXED_SECURITY_POLICY.max_env_value_length == 1_048_576

    def test_default_env_pairs_is_256(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_env_pairs == 256

    def test_default_policy_is_frozen(self) -> None:
        """SecurityPolicy is a frozen dataclass; mutation must raise."""
        with pytest.raises((AttributeError, TypeError)):
            DEFAULT_SECURITY_POLICY.allow_dev_mode = True  # type: ignore[misc]

    def test_relaxed_policy_is_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            RELAXED_SECURITY_POLICY.allow_dev_mode = False  # type: ignore[misc]


class TestSecurityPolicyViolationErrorType:
    """SecurityPolicyViolationError is a PermissionError subclass."""

    def test_is_permission_error(self) -> None:
        from scikitplot.mlflow._errors import SecurityPolicyViolationError as SPVE
        assert issubclass(SPVE, PermissionError)

    def test_catchable_as_permission_error(self) -> None:
        with pytest.raises(PermissionError):
            with security_policy(DEFAULT_SECURITY_POLICY):
                DEFAULT_SECURITY_POLICY.validate_cli_arg_value("bad; cmd")
