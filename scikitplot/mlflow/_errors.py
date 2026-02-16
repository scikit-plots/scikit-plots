from __future__ import annotations


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
