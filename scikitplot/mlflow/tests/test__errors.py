# scikitplot/mlflow/tests/test__errors.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._errors.

Naming convention: test__<module_name>.py

Covers
------
- MlflowIntegrationError     : is RuntimeError, message round-trip, catchable
- MlflowNotInstalledError    : is ImportError, message round-trip, catchable
- MlflowCliIncompatibleError : is ValueError, message round-trip, catchable
- MlflowServerStartError     : is RuntimeError, message round-trip, catchable
- SecurityPolicyViolationError : is PermissionError (and OSError), message round-trip,
                                  catchable as both precise type and broad base
"""

from __future__ import annotations

import errno

import pytest

from scikitplot.mlflow._errors import (
    MlflowCliIncompatibleError,
    MlflowIntegrationError,
    MlflowNotInstalledError,
    MlflowServerStartError,
    SecurityPolicyViolationError,
)


# ===========================================================================
# MlflowIntegrationError
# ===========================================================================


class TestMlflowIntegrationError:
    """Tests for MlflowIntegrationError."""

    def test_is_runtime_error(self) -> None:
        assert issubclass(MlflowIntegrationError, RuntimeError)

    def test_message_round_trip(self) -> None:
        msg = "integration went wrong"
        assert str(MlflowIntegrationError(msg)) == msg

    def test_catchable_as_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise MlflowIntegrationError("boom")

    def test_catchable_as_exact_type(self) -> None:
        with pytest.raises(MlflowIntegrationError):
            raise MlflowIntegrationError("boom")

    def test_empty_construction(self) -> None:
        exc = MlflowIntegrationError()
        assert isinstance(exc, RuntimeError)

    def test_args_preserved(self) -> None:
        exc = MlflowIntegrationError("a", "b")
        assert exc.args == ("a", "b")


# ===========================================================================
# MlflowNotInstalledError
# ===========================================================================


class TestMlflowNotInstalledError:
    """Tests for MlflowNotInstalledError."""

    def test_is_import_error(self) -> None:
        assert issubclass(MlflowNotInstalledError, ImportError)

    def test_message_round_trip(self) -> None:
        msg = "MLflow is not installed."
        assert str(MlflowNotInstalledError(msg)) == msg

    def test_catchable_as_import_error(self) -> None:
        with pytest.raises(ImportError):
            raise MlflowNotInstalledError("missing")

    def test_catchable_as_exact_type(self) -> None:
        with pytest.raises(MlflowNotInstalledError):
            raise MlflowNotInstalledError("missing")

    def test_not_runtime_error(self) -> None:
        """Must not accidentally subclass RuntimeError."""
        assert not issubclass(MlflowNotInstalledError, RuntimeError)


# ===========================================================================
# MlflowCliIncompatibleError
# ===========================================================================


class TestMlflowCliIncompatibleError:
    """Tests for MlflowCliIncompatibleError."""

    def test_is_value_error(self) -> None:
        assert issubclass(MlflowCliIncompatibleError, ValueError)

    def test_message_round_trip(self) -> None:
        msg = "Unsupported flag --workers"
        assert str(MlflowCliIncompatibleError(msg)) == msg

    def test_catchable_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise MlflowCliIncompatibleError("bad flag")

    def test_catchable_as_exact_type(self) -> None:
        with pytest.raises(MlflowCliIncompatibleError):
            raise MlflowCliIncompatibleError("bad flag")

    def test_not_runtime_error(self) -> None:
        assert not issubclass(MlflowCliIncompatibleError, RuntimeError)

    def test_not_import_error(self) -> None:
        assert not issubclass(MlflowCliIncompatibleError, ImportError)


# ===========================================================================
# MlflowServerStartError
# ===========================================================================


class TestMlflowServerStartError:
    """Tests for MlflowServerStartError."""

    def test_is_runtime_error(self) -> None:
        assert issubclass(MlflowServerStartError, RuntimeError)

    def test_message_round_trip(self) -> None:
        msg = "Server failed to start within 30 seconds."
        assert str(MlflowServerStartError(msg)) == msg

    def test_catchable_as_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise MlflowServerStartError("timeout")

    def test_catchable_as_exact_type(self) -> None:
        with pytest.raises(MlflowServerStartError):
            raise MlflowServerStartError("timeout")

    def test_distinct_from_integration_error(self) -> None:
        """The two RuntimeError subclasses must not catch each other."""
        assert not issubclass(MlflowServerStartError, MlflowIntegrationError)
        assert not issubclass(MlflowIntegrationError, MlflowServerStartError)

    def test_args_preserved(self) -> None:
        exc = MlflowServerStartError("host unreachable", "127.0.0.1:5000")
        assert "host unreachable" in exc.args


# ===========================================================================
# SecurityPolicyViolationError
# ===========================================================================


class TestSecurityPolicyViolationError:
    """Tests for SecurityPolicyViolationError."""

    def test_is_permission_error(self) -> None:
        assert issubclass(SecurityPolicyViolationError, PermissionError)

    def test_is_os_error(self) -> None:
        """PermissionError is a subclass of OSError — must propagate correctly."""
        assert issubclass(SecurityPolicyViolationError, OSError)

    def test_message_round_trip(self) -> None:
        msg = "dev mode blocked by policy"
        assert str(SecurityPolicyViolationError(msg)) == msg

    def test_catchable_as_permission_error(self) -> None:
        with pytest.raises(PermissionError):
            raise SecurityPolicyViolationError("denied")

    def test_catchable_as_os_error(self) -> None:
        with pytest.raises(OSError):
            raise SecurityPolicyViolationError("denied")

    def test_catchable_as_exact_type(self) -> None:
        with pytest.raises(SecurityPolicyViolationError):
            raise SecurityPolicyViolationError("denied")

    def test_not_value_error(self) -> None:
        assert not issubclass(SecurityPolicyViolationError, ValueError)

    def test_not_import_error(self) -> None:
        assert not issubclass(SecurityPolicyViolationError, ImportError)

    def test_empty_construction(self) -> None:
        exc = SecurityPolicyViolationError()
        assert isinstance(exc, PermissionError)

    def test_errno_strerror_positional_args(self) -> None:
        """PermissionError supports (errno, strerror) construction."""
        exc = SecurityPolicyViolationError(errno.EPERM, "Operation not permitted")
        assert exc.args[0] == errno.EPERM


# ===========================================================================
# Cross-type isolation
# ===========================================================================


class TestErrorHierarchyIsolation:
    """
    Verify that unrelated exception types do not accidentally intercept each other.

    Notes
    -----
    This guards against accidental multiple-inheritance or mis-ordering in the MRO
    that would cause one handler to swallow a different exception class.
    """

    def test_server_start_not_caught_as_not_installed(self) -> None:
        assert not issubclass(MlflowServerStartError, MlflowNotInstalledError)
        assert not issubclass(MlflowNotInstalledError, MlflowServerStartError)

    def test_not_installed_not_caught_as_cli_incompatible(self) -> None:
        assert not issubclass(MlflowNotInstalledError, MlflowCliIncompatibleError)
        assert not issubclass(MlflowCliIncompatibleError, MlflowNotInstalledError)

    def test_security_violation_not_caught_as_integration_error(self) -> None:
        assert not issubclass(SecurityPolicyViolationError, MlflowIntegrationError)
        assert not issubclass(MlflowIntegrationError, SecurityPolicyViolationError)

    def test_all_exported_in_module_all(self) -> None:
        """__all__ must list all five public exception names."""
        import scikitplot.mlflow._errors as m

        expected = {
            "MlflowCliIncompatibleError",
            "MlflowIntegrationError",
            "MlflowNotInstalledError",
            "MlflowServerStartError",
            "SecurityPolicyViolationError",
        }
        assert expected == set(m.__all__)
