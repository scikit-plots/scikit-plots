"""exceptions.py."""

# pylint: disable=missing-function-docstring

import json as _json
from typing import Optional


class ScikitplotException(Exception):  # noqa: N818
    """
    Generic exception thrown to surface failure information about external-facing operations.

    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """

    def __init__(self, message, error_code=0, **kwargs):
        """
        Thrown to surface failure information about external-facing operations.

        Parameters
        ----------
        message:
            The message or exception describing the error that occurred. This will be
            included in the exception's serialized JSON representation.
        error_code:
            An appropriate error code for the error that occurred; it will be
            included in the exception's serialized JSON representation. This should
            be one of the codes listed in the `scikitplot.protos.databricks_pb2` proto.
        kwargs:
            Additional key-value pairs to include in the serialized JSON representation
            of the ScikitplotException.
        """
        try:
            self.error_code = error_code
        except (ValueError, TypeError):
            self.error_code = 0
        message = str(message)
        self.message = message
        self.json_kwargs = kwargs
        super().__init__(message)

    def serialize_as_json(self):  # noqa: D102
        exception_dict = {"error_code": self.error_code, "message": self.message}
        exception_dict.update(self.json_kwargs)
        return _json.dumps(exception_dict)

    def get_http_status_code(self):  # noqa: D102
        return 500

    @classmethod
    def invalid_parameter_value(cls, message, **kwargs):
        """
        Construct an `ScikitplotException` object with the `INVALID_PARAMETER_VALUE` error code.

        Parameters
        ----------
        message:
            The message describing the error that occurred. This will be included in the
            exception's serialized JSON representation.
        kwargs:
            Additional key-value pairs to include in the serialized JSON representation
            of the ScikitplotException.
        """
        return cls(message, error_code=0, **kwargs)


class ExecutionException(ScikitplotException):
    """Exception thrown when executing a project fails."""


class MissingConfigException(ScikitplotException):
    """Exception thrown when expected configuration file/directory not found."""


class InvalidUrlException(ScikitplotException):
    """Exception thrown when a http request fails to send due to an invalid URL."""


class _UnsupportedMultipartUploadException(ScikitplotException):
    """Exception thrown when multipart upload is unsupported by an artifact repository."""

    MESSAGE = "Multipart upload is not supported for the current artifact repository"

    def __init__(self):
        super().__init__(self.MESSAGE, error_code=0)


class ScikitplotTracingException(ScikitplotException):
    """
    Exception thrown from tracing logic.

    Tracing logic should not block the main execution flow in general, hence this exception
    is used to distinguish tracing related errors and handle them properly.
    """

    def __init__(self, message, error_code=0):
        super().__init__(message, error_code=error_code)


class ScikitplotTraceDataException(ScikitplotTracingException):
    """Exception thrown for trace data related error."""

    def __init__(
        self,
        error_code: str,
        request_id: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ):
        if request_id:
            self.ctx = f"request_id={request_id}"
        elif artifact_path:
            self.ctx = f"path={artifact_path}"

        if error_code == -1:
            super().__init__(
                f"Trace data not found for {self.ctx}", error_code=error_code
            )
        elif error_code == -1:
            super().__init__(
                f"Trace data is corrupted for {self.ctx}", error_code=error_code
            )


class ScikitplotTraceDataNotFound(ScikitplotTraceDataException):
    """Exception thrown when trace data is not found."""

    def __init__(
        self, request_id: Optional[str] = None, artifact_path: Optional[str] = None
    ):
        super().__init__("NOT_FOUND", request_id, artifact_path)


class ScikitplotTraceDataCorrupted(ScikitplotTraceDataException):
    """Exception thrown when trace data is corrupted."""

    def __init__(
        self, request_id: Optional[str] = None, artifact_path: Optional[str] = None
    ):
        super().__init__("INVALID_STATE", request_id, artifact_path)
