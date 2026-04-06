from __future__ import annotations

__all__ = [
    "MlflowCliIncompatibleError",
    "MlflowIntegrationError",
    "MlflowNotInstalledError",
    "MlflowServerStartError",
    "SecurityPolicyViolationError",
]


class MlflowIntegrationError(RuntimeError):
    """
    Base exception for scikitplot.mlflow errors.
    """


class MlflowNotInstalledError(ImportError):
    """
    Raised when MLflow is required but not installed.
    """


class MlflowCliIncompatibleError(ValueError):
    """
    Raised when a requested `mlflow server` option is not supported by the installed MLflow.
    """


class MlflowServerStartError(RuntimeError):
    """
    Raised when the managed MLflow server fails to start or exits prematurely.
    """


class SecurityPolicyViolationError(PermissionError):
    """
    Raised when an operation is rejected by the active :class:`SecurityPolicy`.

    Notes
    -----
    This is a subclass of :class:`PermissionError` so callers can catch it with either
    ``SecurityPolicyViolationError`` (precise) or ``PermissionError`` (broad).
    """
