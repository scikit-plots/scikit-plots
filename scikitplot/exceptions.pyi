# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This stub file covers all public symbols exported by scikitplot.exceptions.
# Upstream references:
#   https://github.com/numpy/numpy/blob/main/numpy/exceptions.pyi
#   https://github.com/astropy/astropy/blob/main/astropy/utils/exceptions.py
#   https://github.com/mlflow/mlflow/blob/master/mlflow/exceptions.py

from typing import Any, overload

__all__: list[str]

# ---------------------------------------------------------------------------
# NumPy-derived exceptions and warnings
# ---------------------------------------------------------------------------

class TooHardError(RuntimeError): ...

class ComplexWarning(RuntimeWarning): ...

class RankWarning(RuntimeWarning): ...

class DTypePromotionError(TypeError): ...

class ModuleDeprecationWarning(DeprecationWarning): ...

class VisibleDeprecationWarning(UserWarning): ...

class AxisError(ValueError, IndexError):
    axis: int | None
    ndim: int | None

    @overload
    def __init__(
        self,
        axis: str,
        ndim: None = ...,
        msg_prefix: None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        axis: int,
        ndim: int,
        msg_prefix: str | None = ...,
    ) -> None: ...
    def __str__(self) -> str: ...

# ---------------------------------------------------------------------------
# Astropy-derived warnings
# ---------------------------------------------------------------------------

class ScikitplotWarning(Warning): ...

class ScikitplotUserWarning(UserWarning): ...

class ScikitplotDeprecationWarning(DeprecationWarning): ...

class ScikitplotPendingDeprecationWarning(PendingDeprecationWarning): ...

class ScikitplotBackwardsIncompatibleChangeWarning(ScikitplotWarning): ...

class DuplicateRepresentationWarning(ScikitplotWarning): ...

# ---------------------------------------------------------------------------
# MLflow-derived exceptions
# ---------------------------------------------------------------------------

class ScikitplotException(Exception):
    error_code: int | str
    message: str
    json_kwargs: dict[str, Any]

    def __init__(
        self,
        message: str | Exception,
        error_code: int | str = ...,
        **kwargs: Any,
    ) -> None: ...
    def serialize_as_json(self) -> str: ...
    def get_http_status_code(self) -> int: ...
    @classmethod
    def invalid_parameter_value(
        cls,
        message: str,
        **kwargs: Any,
    ) -> ScikitplotException: ...

class CommandError(ScikitplotException): ...

class ExecutionException(ScikitplotException): ...

class MissingConfigException(ScikitplotException): ...

class InvalidUrlException(ScikitplotException): ...

class ScikitplotTracingException(ScikitplotException):
    def __init__(
        self,
        message: str | Exception,
        error_code: int | str = ...,
    ) -> None: ...

class ScikitplotTraceDataException(ScikitplotTracingException):
    ctx: str

    def __init__(
        self,
        error_code: str,
        request_id: str | None = ...,
        artifact_path: str | None = ...,
    ) -> None: ...

class ScikitplotTraceDataNotFound(ScikitplotTraceDataException):
    def __init__(
        self,
        request_id: str | None = ...,
        artifact_path: str | None = ...,
    ) -> None: ...

class ScikitplotTraceDataCorrupted(ScikitplotTraceDataException):
    def __init__(
        self,
        request_id: str | None = ...,
        artifact_path: str | None = ...,
    ) -> None: ...

# ---------------------------------------------------------------------------
# Module-level __getattr__ for deprecated ErfaError / ErfaWarning aliases
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any: ...
