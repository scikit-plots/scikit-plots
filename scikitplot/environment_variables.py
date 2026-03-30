# scikitplot/environment_variables.py
#
# flake8: noqa: D213
# pylint: disable=line-too-long
# noqa: E501
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# This module was adapted from the mlflow project.
# https://github.com/mlflow/mlflow/blob/master/mlflow/environment_variables.py

"""
Defines environment variables used in scikit-plots.

scikit-plots's environment variables adhere to the following naming conventions:

- **Public variables**: environment variable names begin with ``SKPLT_``.
  The corresponding Python module-level name is identical to the env var string.
- **Internal-use variables**: names start with ``_SKPLT_``.
  The corresponding Python module-level name is identical to the env var string.

Invariant
---------
For every ``EnvironmentVariable`` or ``BooleanEnvironmentVariable`` instance
declared at module level, the Python attribute name **must** equal the ``name``
argument passed to the constructor.  Violations of this invariant make it
impossible to discover the correct env var from the Python symbol (or vice
versa) and are treated as bugs.

Adapted from MLflow project.
- https://github.com/mlflow/mlflow/blob/master/mlflow/environment_variables.py
"""

import os as _os
import tempfile as _tempfile
from pathlib import Path as _Path


class EnvironmentVariable:
    """Represent a typed, defaulted environment variable.

    Parameters
    ----------
    name : str
        The exact environment variable name (e.g. ``"SKPLT_TRACKING_URI"``).
        Must be a non-empty string.  The module-level Python attribute that
        holds this instance **must** have the same name.
    type_ : type
        Callable used to coerce the raw string value obtained from the
        environment.  Common values: ``str``, ``int``, ``float``.
    default : object
        Value returned by :meth:`get` when the variable is not set.  May be
        ``None`` regardless of ``type_``.

    Raises
    ------
    TypeError
        If ``name`` is not a non-empty ``str``, or if ``type_`` is not
        callable.

    Notes
    -----
    **Developer note**: this class intentionally avoids caching the resolved
    value so that changes to ``os.environ`` made after import are always
    reflected correctly.  Call :meth:`get` at the point of use, not at module
    import time.
    """

    def __init__(self, name: str, type_: type, default):
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"'name' must be a non-empty str, got {name!r} ({type(name).__name__})"
            )
        if not callable(type_):
            raise TypeError(
                f"'type_' must be callable, got {type_!r} ({type(type_).__name__})"
            )
        self.name = name
        self.type = type_
        self.default = default

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def defined(self) -> bool:
        """Return ``True`` when the variable is present in the environment.

        Returns
        -------
        bool
            ``True`` if ``name`` exists as a key in ``os.environ``,
            ``False`` otherwise.
        """
        return self.name in _os.environ

    def is_set(self) -> bool:
        """Return ``True`` when the variable is present in the environment.

        Alias for the :attr:`defined` property retained for API compatibility.

        Returns
        -------
        bool
            Equivalent to :attr:`defined`.
        """
        return self.defined

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get_raw(self):
        """Return the raw string value from the environment, or ``None``.

        Returns
        -------
        str or None
            The string stored in ``os.environ`` for this variable, or
            ``None`` if the variable is absent.
        """
        return _os.getenv(self.name)

    def get(self):
        """Return the typed value of the environment variable.

        If the variable is set, its raw string value is cast to
        :attr:`type`.  If it is absent, :attr:`default` is returned
        unchanged (no coercion is applied to the default).

        Returns
        -------
        object
            Coerced value of type :attr:`type`, or :attr:`default`.

        Raises
        ------
        ValueError
            If the raw string value cannot be converted to :attr:`type`.

        Examples
        --------
        >>> import os
        >>> var = EnvironmentVariable("EXAMPLE_INT_VAR", int, 5)
        >>> os.environ["EXAMPLE_INT_VAR"] = "42"
        >>> var.get()
        42
        >>> del os.environ["EXAMPLE_INT_VAR"]
        >>> var.get()
        5
        """
        if (val := self.get_raw()) is not None:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert {val!r} to {self.type} for {self.name}: {e}"
                ) from e
        return self.default

    def set(self, value) -> None:
        """Store *value* in the environment as a string.

        Parameters
        ----------
        value : object
            The value to store.  It is coerced to ``str`` via ``str(value)``
            before being written to ``os.environ``.

        Notes
        -----
        No type-validation is performed here; the value is stored verbatim
        as a string.  Call :meth:`get` to retrieve it with type coercion.
        """
        _os.environ[self.name] = str(value)

    def unset(self) -> None:
        """Remove the variable from the environment if it is present.

        Notes
        -----
        This is a no-op when the variable is absent; it never raises.
        """
        _os.environ.pop(self.name, None)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare by ``name`` for equality.

        Two ``EnvironmentVariable`` instances are considered equal when
        their ``name`` attributes are identical strings.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when ``other`` is an ``EnvironmentVariable`` with the
            same :attr:`name`, otherwise ``NotImplemented``.
        """
        if isinstance(other, EnvironmentVariable):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on :attr:`name`, consistent with :meth:`__eq__`.

        Returns
        -------
        int
            Hash of :attr:`name`.
        """
        return hash(self.name)

    def __str__(self) -> str:  # noqa: D105
        return f"{self.name} (default: {self.default}, type: {self.type.__name__})"

    def __repr__(self) -> str:  # noqa: D105
        return repr(self.name)

    def __format__(self, format_spec: str) -> str:  # noqa: D105
        return self.name.__format__(format_spec)


class BooleanEnvironmentVariable(EnvironmentVariable):
    """Represent a boolean environment variable.

    Accepts the following case-insensitive string values:

    - ``"true"`` or ``"1"`` → ``True``
    - ``"false"`` or ``"0"`` → ``False``

    Any other value raises :exc:`ValueError` at read time.

    Parameters
    ----------
    name : str
        The exact environment variable name.  See :class:`EnvironmentVariable`.
    default : bool or None
        Default value returned when the variable is absent.  Must be one of
        ``True``, ``False``, or ``None``.  Integer values such as ``1`` or
        ``0`` are rejected to avoid identity-based pitfalls (``1 in [True]``
        evaluates to ``True`` in Python).

    Raises
    ------
    ValueError
        If *default* is not one of ``True``, ``False``, or ``None``.
    TypeError
        Propagated from :class:`EnvironmentVariable` if *name* is invalid.
    """

    def __init__(self, name: str, default):
        # `default not in [True, False, None]` doesn't work because `1 in [True]`
        # (or `0 in [False]`) returns True.  Use identity checks instead.
        if not (default is True or default is False or default is None):
            raise ValueError(
                f"{name} default value must be one of [True, False, None], "
                f"got {default!r} ({type(default).__name__})"
            )
        super().__init__(name, bool, default)

    def get(self):
        """Return the boolean value of the environment variable.

        Parameters
        ----------
        (none)

        Returns
        -------
        bool or None
            Parsed boolean value, or :attr:`default` when the variable is
            absent.

        Raises
        ------
        ValueError
            If the raw string value is not one of ``"true"``, ``"false"``,
            ``"1"``, or ``"0"`` (case-insensitive).

        Examples
        --------
        >>> import os
        >>> var = BooleanEnvironmentVariable("EXAMPLE_BOOL_VAR", False)
        >>> os.environ["EXAMPLE_BOOL_VAR"] = "true"
        >>> var.get()
        True
        >>> os.environ["EXAMPLE_BOOL_VAR"] = "TRUE"
        >>> var.get()
        True
        >>> del os.environ["EXAMPLE_BOOL_VAR"]
        >>> var.get()
        False
        """
        if not self.defined:
            return self.default

        val = self.get_raw()  # delegates to EnvironmentVariable.get_raw()
        lowercased = val.lower()
        if lowercased not in {"true", "false", "1", "0"}:
            raise ValueError(
                f"{self.name} value must be one of ['true', 'false', '1', '0'] "
                f"(case-insensitive), but got {val!r}"
            )
        return lowercased in {"true", "1"}


# ---------------------------------------------------------------------------
# Module-level private helpers (not part of the public namespace)
# ---------------------------------------------------------------------------

#: Secure temp directory used as the default DFS temporary path.
_default_tmp_path = _Path(_tempfile.gettempdir()) / "scikitplot"

# ---------------------------------------------------------------------------
# Public environment variables  (prefix: SKPLT_)
# ---------------------------------------------------------------------------

#: Specifies the tracking URI.
#: (default: ``None``)
SKPLT_TRACKING_URI = EnvironmentVariable("SKPLT_TRACKING_URI", str, None)

#: Specifies the registry URI.
#: (default: ``None``)
SKPLT_REGISTRY_URI = EnvironmentVariable("SKPLT_REGISTRY_URI", str, None)

#: Specifies the ``dfs_tmpdir`` parameter to use for ``mlflow.spark.save_model``,
#: ``mlflow.spark.log_model`` and ``mlflow.spark.load_model``. See
#: https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.save_model
#: for more information.
#: (default: ``/tmp/scikitplot``)
SKPLT_DFS_TMP = EnvironmentVariable("SKPLT_DFS_TMP", str, str(_default_tmp_path))

#: Specifies the maximum number of retries with exponential backoff for MLflow HTTP requests
#: (default: ``7``)
SKPLT_HTTP_REQUEST_MAX_RETRIES = EnvironmentVariable(
    "SKPLT_HTTP_REQUEST_MAX_RETRIES",
    int,
    # Important: It's common for MLflow backends to rate limit requests for more than 1 minute.
    # To remain resilient to rate limiting, the MLflow client needs to retry for more than 1
    # minute. Assuming 2 seconds per retry, 7 retries with backoff will take ~ 4 minutes,
    # which is appropriate for most rate limiting scenarios
    7,
)

#: Specifies the backoff increase factor between MLflow HTTP request failures
#: (default: ``2``)
SKPLT_HTTP_REQUEST_BACKOFF_FACTOR = EnvironmentVariable(
    "SKPLT_HTTP_REQUEST_BACKOFF_FACTOR", int, 2
)

#: Specifies the backoff jitter between MLflow HTTP request failures
#: (default: ``1.0``)
SKPLT_HTTP_REQUEST_BACKOFF_JITTER = EnvironmentVariable(
    "SKPLT_HTTP_REQUEST_BACKOFF_JITTER", float, 1.0
)

#: Specifies the timeout in seconds for MLflow HTTP requests
#: (default: ``120``)
SKPLT_HTTP_REQUEST_TIMEOUT = EnvironmentVariable("SKPLT_HTTP_REQUEST_TIMEOUT", int, 120)

#: Specifies whether to respect Retry-After header on status codes defined as
#: Retry.RETRY_AFTER_STATUS_CODES or not for MLflow HTTP request
#: (default: ``True``)
SKPLT_HTTP_RESPECT_RETRY_AFTER_HEADER = BooleanEnvironmentVariable(
    "SKPLT_HTTP_RESPECT_RETRY_AFTER_HEADER", True
)

#: Internal-only configuration that sets an upper bound to the allowable maximum
#: retries for HTTP requests
#: (default: ``10``)
_SKPLT_HTTP_REQUEST_MAX_RETRIES_LIMIT = EnvironmentVariable(
    "_SKPLT_HTTP_REQUEST_MAX_RETRIES_LIMIT", int, 10
)

#: Internal-only configuration that sets the upper bound for an HTTP backoff_factor
#: (default: ``120``)
_SKPLT_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT = EnvironmentVariable(
    "_SKPLT_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT", int, 120
)

#: Specifies whether MLflow HTTP requests should be signed using AWS signature V4. It will overwrite
#: (default: ``False``). When set, it will overwrite the "Authorization" HTTP header.
#: See https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html for more information.
SKPLT_TRACKING_AWS_SIGV4 = BooleanEnvironmentVariable("SKPLT_TRACKING_AWS_SIGV4", False)

#: Specifies the auth provider to sign the MLflow HTTP request
#: (default: ``None``). When set, it will overwrite the "Authorization" HTTP header.
SKPLT_TRACKING_AUTH = EnvironmentVariable("SKPLT_TRACKING_AUTH", str, None)

#: Specifies the chunk size to use when downloading a file from GCS
#: (default: ``None``). If None, the chunk size is automatically determined by the
#: ``google-cloud-storage`` package.
SKPLT_GCS_DOWNLOAD_CHUNK_SIZE = EnvironmentVariable(
    "SKPLT_GCS_DOWNLOAD_CHUNK_SIZE", int, None
)

#: Specifies the chunk size to use when uploading a file to GCS.
#: (default: ``None``). If None, the chunk size is automatically determined by the
#: ``google-cloud-storage`` package.
SKPLT_GCS_UPLOAD_CHUNK_SIZE = EnvironmentVariable(
    "SKPLT_GCS_UPLOAD_CHUNK_SIZE", int, None
)

#: Specifies whether to disable model logging and loading via mlflowdbfs.
#: Internal-use variable.
#: (default: ``None``)
#:
#: .. note::
#:    FIX: was ``_DISABLE_SKPLTDBFS`` / ``"DISABLE_SKPLTDBFS"`` — both the Python
#:    name and env var string violated the ``_SKPLT_`` private-prefix convention.
_SKPLT_DISABLE_DBFS = EnvironmentVariable("_SKPLT_DISABLE_DBFS", str, None)

#: Specifies the S3 endpoint URL to use for S3 artifact operations.
#: (default: ``None``)
SKPLT_S3_ENDPOINT_URL = EnvironmentVariable("SKPLT_S3_ENDPOINT_URL", str, None)

#: Specifies whether or not to skip TLS certificate verification for S3 artifact operations.
#: (default: ``False``)
SKPLT_S3_IGNORE_TLS = BooleanEnvironmentVariable("SKPLT_S3_IGNORE_TLS", False)

#: Specifies extra arguments for S3 artifact uploads.
#: (default: ``None``)
SKPLT_S3_UPLOAD_EXTRA_ARGS = EnvironmentVariable(
    "SKPLT_S3_UPLOAD_EXTRA_ARGS", str, None
)

#: Specifies the location of a Kerberos ticket cache to use for HDFS artifact operations.
#: (default: ``None``)
SKPLT_KERBEROS_TICKET_CACHE = EnvironmentVariable(
    "SKPLT_KERBEROS_TICKET_CACHE", str, None
)

#: Specifies a Kerberos user for HDFS artifact operations.
#: (default: ``None``)
SKPLT_KERBEROS_USER = EnvironmentVariable("SKPLT_KERBEROS_USER", str, None)

#: Specifies extra pyarrow configurations for HDFS artifact operations.
#: (default: ``None``)
SKPLT_PYARROW_EXTRA_CONF = EnvironmentVariable("SKPLT_PYARROW_EXTRA_CONF", str, None)

#: Specifies the ``pool_size`` parameter to use for ``sqlalchemy.create_engine`` in the SQLAlchemy
#: tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.pool_size
#: for more information.
#: (default: ``None``)
SKPLT_SQLALCHEMYSTORE_POOL_SIZE = EnvironmentVariable(
    "SKPLT_SQLALCHEMYSTORE_POOL_SIZE", int, None
)

#: Specifies the ``pool_recycle`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.pool_recycle
#: for more information.
#: (default: ``None``)
SKPLT_SQLALCHEMYSTORE_POOL_RECYCLE = EnvironmentVariable(
    "SKPLT_SQLALCHEMYSTORE_POOL_RECYCLE", int, None
)

#: Specifies the ``max_overflow`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.max_overflow
#: for more information.
#: (default: ``None``)
SKPLT_SQLALCHEMYSTORE_MAX_OVERFLOW = EnvironmentVariable(
    "SKPLT_SQLALCHEMYSTORE_MAX_OVERFLOW", int, None
)

#: Specifies the ``echo`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.echo
#: for more information.
#: (default: ``False``)
SKPLT_SQLALCHEMYSTORE_ECHO = BooleanEnvironmentVariable(
    "SKPLT_SQLALCHEMYSTORE_ECHO", False
)

#: Specifies whether or not to print a warning when `--env-manager=conda` is specified.
#: (default: ``False``)
SKPLT_DISABLE_ENV_MANAGER_CONDA_WARNING = BooleanEnvironmentVariable(
    "SKPLT_DISABLE_ENV_MANAGER_CONDA_WARNING", False
)

#: Specifies the ``poolclass`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.poolclass
#: for more information.
#: (default: ``None``)
SKPLT_SQLALCHEMYSTORE_POOLCLASS = EnvironmentVariable(
    "SKPLT_SQLALCHEMYSTORE_POOLCLASS", str, None
)

#: Specifies the ``timeout_seconds`` for MLflow Model dependency inference operations.
#: (default: ``120``)
SKPLT_REQUIREMENTS_INFERENCE_TIMEOUT = EnvironmentVariable(
    "SKPLT_REQUIREMENTS_INFERENCE_TIMEOUT", int, 120
)

#: Specifies the MLflow Model Scoring server request timeout in seconds
#: (default: ``60``)
SKPLT_SCORING_SERVER_REQUEST_TIMEOUT = EnvironmentVariable(
    "SKPLT_SCORING_SERVER_REQUEST_TIMEOUT", int, 60
)

#: (Experimental, may be changed or removed)
#: Specifies the timeout to use when uploading or downloading a file
#: (default: ``None``). If None, individual artifact stores will choose defaults.
SKPLT_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT = EnvironmentVariable(
    "SKPLT_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", int, None
)

#: Specifies the timeout for model inference with input example(s) when logging/saving a model.
#: MLflow runs a few inference requests against the model to infer model signature and pip
#: requirements. Sometimes the prediction hangs for a long time, especially for a large model.
#: This timeout limits the allowable time for performing a prediction for signature inference
#: and will abort the prediction, falling back to the default signature and pip requirements.
SKPLT_INPUT_EXAMPLE_INFERENCE_TIMEOUT = EnvironmentVariable(
    "SKPLT_INPUT_EXAMPLE_INFERENCE_TIMEOUT", int, 180
)


#: Specifies the device intended for use in the predict function - can be used
#: to override behavior where the GPU is used by default when available by
#: setting this environment variable to be ``cpu``. Currently, this
#: variable is only supported for the MLflow PyTorch and HuggingFace flavors.
#: For the HuggingFace flavor, note that device must be parseable as an integer.
SKPLT_DEFAULT_PREDICTION_DEVICE = EnvironmentVariable(
    "SKPLT_DEFAULT_PREDICTION_DEVICE", str, None
)

#: Specifies to Huggingface whether to use the automatic device placement logic of
#: HuggingFace accelerate. If it's set to false, the low_cpu_mem_usage flag will not be
#: set to True and device_map will not be set to "auto".
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_DISABLE_HUGGINGFACE_ACCELERATE_FEATURES"``
#:    (word-order mismatch with Python name). Corrected to match Python symbol.
SKPLT_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES = BooleanEnvironmentVariable(
    "SKPLT_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES", False
)

#: Specifies to Huggingface whether to use the automatic device placement logic of
#: HuggingFace accelerate. If it's set to false, the low_cpu_mem_usage flag will not be
#: set to True and device_map will not be set to "auto". Default to False.
SKPLT_HUGGINGFACE_USE_DEVICE_MAP = BooleanEnvironmentVariable(
    "SKPLT_HUGGINGFACE_USE_DEVICE_MAP", False
)

#: Specifies to Huggingface to use the automatic device placement logic of HuggingFace accelerate.
#: This can be set to values supported by the version of HuggingFace Accelerate being installed.
SKPLT_HUGGINGFACE_DEVICE_MAP_STRATEGY = EnvironmentVariable(
    "SKPLT_HUGGINGFACE_DEVICE_MAP_STRATEGY", str, "auto"
)

#: Specifies to Huggingface to use the low_cpu_mem_usage flag powered by HuggingFace accelerate.
#: If it's set to false, the low_cpu_mem_usage flag will be set to False.
SKPLT_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE = BooleanEnvironmentVariable(
    "SKPLT_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE", True
)

#: Specifies the max_shard_size to use when mlflow transformers flavor saves the model checkpoint.
#: This can be set to override the 500MB default.
SKPLT_HUGGINGFACE_MODEL_MAX_SHARD_SIZE = EnvironmentVariable(
    "SKPLT_HUGGINGFACE_MODEL_MAX_SHARD_SIZE", str, "500MB"
)

#: Specifies the name of the Databricks secret scope to use for storing OpenAI API keys.
SKPLT_OPENAI_SECRET_SCOPE = EnvironmentVariable("SKPLT_OPENAI_SECRET_SCOPE", str, None)

#: (Experimental, may be changed or removed)
#: Specifies the download options to be used by pip wheel when `add_libraries_to_model` is used to
#: create and log model dependencies as model artifacts. The default behavior only uses dependency
#: binaries and no source packages.
#: (default: ``--only-binary=:all:``).
SKPLT_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS = EnvironmentVariable(
    "SKPLT_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS", str, "--only-binary=:all:"
)

#: Specifies whether or not to use multipart download when downloading a large file on Databricks.
SKPLT_ENABLE_MULTIPART_DOWNLOAD = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_MULTIPART_DOWNLOAD", True
)

#: Specifies whether or not to use multipart upload when uploading large artifacts.
SKPLT_ENABLE_MULTIPART_UPLOAD = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_MULTIPART_UPLOAD", True
)

#: Specifies whether or not to use multipart upload for proxied artifact access.
#: (default: ``False``)
SKPLT_ENABLE_PROXY_MULTIPART_UPLOAD = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_PROXY_MULTIPART_UPLOAD", False
)

#: Private environment variable that's set to ``True`` while running tests.
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_TESTING"`` — missing the ``_`` prefix
#:    required by the private-variable convention.  Corrected to ``"_SKPLT_TESTING"``.
_SKPLT_TESTING = BooleanEnvironmentVariable("_SKPLT_TESTING", False)

#: Specifies the username used to authenticate with a tracking server.
#: (default: ``None``)
SKPLT_TRACKING_USERNAME = EnvironmentVariable("SKPLT_TRACKING_USERNAME", str, None)

#: Specifies the password used to authenticate with a tracking server.
#: (default: ``None``)
SKPLT_TRACKING_PASSWORD = EnvironmentVariable("SKPLT_TRACKING_PASSWORD", str, None)

#: Specifies and takes precedence for setting the basic/bearer auth on http requests.
#: (default: ``None``)
SKPLT_TRACKING_TOKEN = EnvironmentVariable("SKPLT_TRACKING_TOKEN", str, None)

#: Specifies whether to verify TLS connection in ``requests.request`` function,
#: see https://requests.readthedocs.io/en/master/api/
#: (default: ``False``).
SKPLT_TRACKING_INSECURE_TLS = BooleanEnvironmentVariable(
    "SKPLT_TRACKING_INSECURE_TLS", False
)

#: Sets the ``verify`` param in ``requests.request`` function,
#: see https://requests.readthedocs.io/en/master/api/
#: (default: ``None``)
SKPLT_TRACKING_SERVER_CERT_PATH = EnvironmentVariable(
    "SKPLT_TRACKING_SERVER_CERT_PATH", str, None
)

#: Sets the ``cert`` param in ``requests.request`` function,
#: see https://requests.readthedocs.io/en/master/api/
#: (default: ``None``)
SKPLT_TRACKING_CLIENT_CERT_PATH = EnvironmentVariable(
    "SKPLT_TRACKING_CLIENT_CERT_PATH", str, None
)

#: Specified the ID of the run to log data to.
#: (default: ``None``)
SKPLT_RUN_ID = EnvironmentVariable("SKPLT_RUN_ID", str, None)

#: Specifies the default root directory for tracking `FileStore`.
#: (default: ``None``)
SKPLT_TRACKING_DIR = EnvironmentVariable("SKPLT_TRACKING_DIR", str, None)

#: Specifies the default root directory for registry `FileStore`.
#: (default: ``None``)
SKPLT_REGISTRY_DIR = EnvironmentVariable("SKPLT_REGISTRY_DIR", str, None)

#: Specifies the default experiment ID to create run to.
#: (default: ``None``)
SKPLT_EXPERIMENT_ID = EnvironmentVariable("SKPLT_EXPERIMENT_ID", str, None)

#: Specifies the default experiment name to create run to.
#: (default: ``None``)
SKPLT_EXPERIMENT_NAME = EnvironmentVariable("SKPLT_EXPERIMENT_NAME", str, None)

#: Specified the path to the configuration file for MLflow Authentication.
#: (default: ``None``)
SKPLT_AUTH_CONFIG_PATH = EnvironmentVariable("SKPLT_AUTH_CONFIG_PATH", str, None)

#: Specifies and takes precedence for setting the UC OSS basic/bearer auth on http requests.
#: (default: ``None``)
SKPLT_UC_OSS_TOKEN = EnvironmentVariable("SKPLT_UC_OSS_TOKEN", str, None)

#: Specifies the root directory to create Python virtual environments in.
#: (default: ``~/.scikitplot/envs``)
SKPLT_ENV_ROOT = EnvironmentVariable(
    "SKPLT_ENV_ROOT", str, str(_Path.home().joinpath(".scikitplot", "envs"))
)

#: Specifies whether or not to use DBFS FUSE mount to store artifacts on Databricks
#: (default: ``True``)
SKPLT_ENABLE_DBFS_FUSE_ARTIFACT_REPO = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_DBFS_FUSE_ARTIFACT_REPO", True
)

#: Specifies whether or not to use UC Volume FUSE mount to store artifacts on Databricks
#: (default: ``True``)
SKPLT_ENABLE_UC_VOLUME_FUSE_ARTIFACT_REPO = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_UC_VOLUME_FUSE_ARTIFACT_REPO", True
)

#: Private environment variable that should be set to ``True`` when running autologging tests.
#: (default: ``False``)
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_AUTOLOGGING_TESTING"`` — missing the ``_`` prefix.
_SKPLT_AUTOLOGGING_TESTING = BooleanEnvironmentVariable(
    "_SKPLT_AUTOLOGGING_TESTING", False
)

#: (Experimental, may be changed or removed)
#: Specifies the uri of a MLflow Gateway Server instance to be used with the Gateway Client APIs
#: (default: ``None``)
SKPLT_GATEWAY_URI = EnvironmentVariable("SKPLT_GATEWAY_URI", str, None)

#: (Experimental, may be changed or removed)
#: Specifies the uri of an MLflow AI Gateway instance to be used with the Deployments
#: Client APIs
#: (default: ``None``)
SKPLT_DEPLOYMENTS_TARGET = EnvironmentVariable("SKPLT_DEPLOYMENTS_TARGET", str, None)

#: Specifies the path of the config file for MLflow AI Gateway.
#: (default: ``None``)
SKPLT_GATEWAY_CONFIG = EnvironmentVariable("SKPLT_GATEWAY_CONFIG", str, None)

#: Specifies the path of the config file for MLflow AI Gateway.
#: (default: ``None``)
SKPLT_DEPLOYMENTS_CONFIG = EnvironmentVariable("SKPLT_DEPLOYMENTS_CONFIG", str, None)

#: Specifies whether to display the progress bar when uploading/downloading artifacts.
#: (default: ``True``)
SKPLT_ENABLE_ARTIFACTS_PROGRESS_BAR = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_ARTIFACTS_PROGRESS_BAR", True
)

#: Specifies the conda home directory to use.
#: (default: ``None``)
SKPLT_CONDA_HOME = EnvironmentVariable("SKPLT_CONDA_HOME", str, None)

#: Specifies the name of the command to use when creating the environments.
#: For example, let's say we want to use mamba (https://github.com/mamba-org/mamba)
#: instead of conda to create environments.
#: Then: > conda install mamba -n base -c conda-forge
#: If not set, use the same as conda_path
#: (default: ``conda``)
SKPLT_CONDA_CREATE_ENV_CMD = EnvironmentVariable(
    "SKPLT_CONDA_CREATE_ENV_CMD", str, "conda"
)

#: Specifies the flavor to serve in the scoring server.
#: (default ``None``)
SKPLT_DEPLOYMENT_FLAVOR_NAME = EnvironmentVariable(
    "SKPLT_DEPLOYMENT_FLAVOR_NAME", str, None
)

#: Specifies the MLflow Run context
#: (default: ``None``)
SKPLT_RUN_CONTEXT = EnvironmentVariable("SKPLT_RUN_CONTEXT", str, None)

#: Specifies the URL of the ECR-hosted Docker image a model is deployed into for SageMaker.
#: (default: ``None``)
SKPLT_SAGEMAKER_DEPLOY_IMG_URL = EnvironmentVariable(
    "SKPLT_SAGEMAKER_DEPLOY_IMG_URL", str, None
)

#: Specifies whether to disable creating a new conda environment for `mlflow models build-docker`.
#: (default: ``False``)
SKPLT_DISABLE_ENV_CREATION = BooleanEnvironmentVariable(
    "SKPLT_DISABLE_ENV_CREATION", False
)

#: Specifies the timeout value for downloading chunks of mlflow artifacts.
#: (default: ``300``)
SKPLT_DOWNLOAD_CHUNK_TIMEOUT = EnvironmentVariable(
    "SKPLT_DOWNLOAD_CHUNK_TIMEOUT", int, 300
)

#: Specifies if system metrics logging should be enabled.
SKPLT_ENABLE_SYSTEM_METRICS_LOGGING = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_SYSTEM_METRICS_LOGGING", False
)

#: Specifies the sampling interval for system metrics logging.
SKPLT_SYSTEM_METRICS_SAMPLING_INTERVAL = EnvironmentVariable(
    "SKPLT_SYSTEM_METRICS_SAMPLING_INTERVAL", float, None
)

#: Specifies the number of samples before logging system metrics.
SKPLT_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING = EnvironmentVariable(
    "SKPLT_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING", int, None
)

#: Specifies the node id of system metrics logging. This is useful in multi-node (distributed
#: training) setup.
SKPLT_SYSTEM_METRICS_NODE_ID = EnvironmentVariable(
    "SKPLT_SYSTEM_METRICS_NODE_ID", str, None
)


#: Private environment variable to specify the number of chunk download retries for multipart
#: download.
_SKPLT_MPD_NUM_RETRIES = EnvironmentVariable("_SKPLT_MPD_NUM_RETRIES", int, 3)

#: Private environment variable to specify the interval between chunk download retries for multipart
#: download.
_SKPLT_MPD_RETRY_INTERVAL_SECONDS = EnvironmentVariable(
    "_SKPLT_MPD_RETRY_INTERVAL_SECONDS", int, 1
)

#: Specifies the minimum file size in bytes to use multipart upload when logging artifacts
#: (default: ``524_288_000`` (500 MB))
SKPLT_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE = EnvironmentVariable(
    "SKPLT_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE", int, 500 * 1024**2
)

#: Specifies the minimum file size in bytes to use multipart download when downloading artifacts
#: (default: ``524_288_000`` (500 MB))
SKPLT_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE = EnvironmentVariable(
    "SKPLT_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE", int, 500 * 1024**2
)

#: Specifies the chunk size in bytes to use when performing multipart upload
#: (default: ``10_485_760`` (10 MB))
SKPLT_MULTIPART_UPLOAD_CHUNK_SIZE = EnvironmentVariable(
    "SKPLT_MULTIPART_UPLOAD_CHUNK_SIZE", int, 10 * 1024**2
)

#: Specifies the chunk size in bytes to use when performing multipart download
#: (default: ``104_857_600`` (100 MB))
SKPLT_MULTIPART_DOWNLOAD_CHUNK_SIZE = EnvironmentVariable(
    "SKPLT_MULTIPART_DOWNLOAD_CHUNK_SIZE", int, 100 * 1024**2
)

#: Specifies whether or not to allow the MLflow server to follow redirects when
#: making HTTP requests. If set to False, the server will throw an exception if it
#: encounters a redirect response.
#: (default: ``True``)
SKPLT_ALLOW_HTTP_REDIRECTS = BooleanEnvironmentVariable(
    "SKPLT_ALLOW_HTTP_REDIRECTS", True
)

#: Specifies the client-based timeout (in seconds) when making an HTTP request to a deployment
#: target. Used within the `predict` and `predict_stream` APIs.
#: (default: ``120``)
SKPLT_DEPLOYMENT_PREDICT_TIMEOUT = EnvironmentVariable(
    "SKPLT_DEPLOYMENT_PREDICT_TIMEOUT", int, 120
)

SKPLT_GATEWAY_RATE_LIMITS_STORAGE_URI = EnvironmentVariable(
    "SKPLT_GATEWAY_RATE_LIMITS_STORAGE_URI", str, None
)

#: If True, MLflow fluent logging APIs, e.g., `mlflow.log_metric` will log asynchronously.
SKPLT_ENABLE_ASYNC_LOGGING = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_ASYNC_LOGGING", False
)

#: Number of workers in the thread pool used for asynchronous logging, defaults to 10.
SKPLT_ASYNC_LOGGING_THREADPOOL_SIZE = EnvironmentVariable(
    "SKPLT_ASYNC_LOGGING_THREADPOOL_SIZE", int, 10
)

#: Specifies whether or not to have mlflow configure logging on import.
#: If set to True, mlflow will configure ``mlflow.<module_name>`` loggers with
#: logging handlers and formatters.
#: (default: ``True``)
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_LOGGING_CONFIGURE_LOGGING"`` (words swapped
#:    relative to Python name).  Corrected to ``"SKPLT_CONFIGURE_LOGGING"``.
SKPLT_CONFIGURE_LOGGING = BooleanEnvironmentVariable("SKPLT_CONFIGURE_LOGGING", True)

#: If set to True, the following entities will be truncated to their maximum length:
#: - Param value
#: - Tag value
#: If set to False, an exception will be raised if the length of the entity exceeds the maximum
#: length.
#: (default: ``True``)
SKPLT_TRUNCATE_LONG_VALUES = BooleanEnvironmentVariable(
    "SKPLT_TRUNCATE_LONG_VALUES", True
)

#: Whether to run slow tests with pytest. Default to False in normal runs,
#: but set to True in the weekly slow test jobs.
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_RUN_SLOW_TESTS"`` — missing the ``_`` prefix.
_SKPLT_RUN_SLOW_TESTS = BooleanEnvironmentVariable("_SKPLT_RUN_SLOW_TESTS", False)

#: The OpenJDK version to install in the Docker image used for MLflow models.
#: (default: ``11``)
SKPLT_DOCKER_OPENJDK_VERSION = EnvironmentVariable(
    "SKPLT_DOCKER_OPENJDK_VERSION", str, "11"
)


#: How long a trace can be "in-progress". When this is set to a positive value and a trace is
#: not completed within this time, it will be automatically halted and exported to the specified
#: backend destination with status "ERROR".
SKPLT_TRACE_TIMEOUT_SECONDS = EnvironmentVariable(
    "SKPLT_TRACE_TIMEOUT_SECONDS", int, None
)

#: How frequently to check for timed-out traces. For example, if this is set to 10, MLflow will
#: check for timed-out traces every 10 seconds (in a background worker) and halt any traces that
#: have exceeded the timeout. This is only effective if SKPLT_TRACE_TIMEOUT_SECONDS is set to a
#: positive value.
SKPLT_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS = EnvironmentVariable(
    "SKPLT_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS", int, 1
)

#: How long a trace can be buffered in-memory at client side before being abandoned.
SKPLT_TRACE_BUFFER_TTL_SECONDS = EnvironmentVariable(
    "SKPLT_TRACE_BUFFER_TTL_SECONDS", int, 3600
)

#: How many traces to be buffered in-memory at client side before being abandoned.
SKPLT_TRACE_BUFFER_MAX_SIZE = EnvironmentVariable(
    "SKPLT_TRACE_BUFFER_MAX_SIZE", int, 1000
)

#: Private configuration option.
#: Enables the ability to catch exceptions within MLflow evaluate for classification models
#: where a class imbalance due to a missing target class would raise an error in the
#: underlying metrology modules (scikit-learn). If set to True, specific exceptions will be
#: caught, alerted via the warnings module, and evaluation will resume.
#: (default: ``False``)
_SKPLT_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS = BooleanEnvironmentVariable(
    "_SKPLT_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS", False
)

#: Whether to warn (default) or raise (opt-in) for unresolvable requirements inference for
#: a model's dependency inference. If set to True, an exception will be raised if requirements
#: inference or the process of capturing imported modules encounters any errors.
SKPLT_REQUIREMENTS_INFERENCE_RAISE_ERRORS = BooleanEnvironmentVariable(
    "SKPLT_REQUIREMENTS_INFERENCE_RAISE_ERRORS", False
)

#: How many traces to display in Databricks Notebooks
SKPLT_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK = EnvironmentVariable(
    "SKPLT_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK", int, 10
)

#: Whether to write trace to the MLflow backend from a model running in a Databricks
#: model serving endpoint. If true, the trace will be written to both the MLflow backend
#: and the Inference Table.
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING"``
#:    — missing the ``_`` prefix required for private internal variables.
_SKPLT_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING = EnvironmentVariable(
    "_SKPLT_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING", bool, False
)

#: Default addressing style to use for boto client
SKPLT_BOTO_CLIENT_ADDRESSING_STYLE = EnvironmentVariable(
    "SKPLT_BOTO_CLIENT_ADDRESSING_STYLE", str, "auto"
)

#: Specify the timeout in seconds for Databricks endpoint HTTP request retries.
SKPLT_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT = EnvironmentVariable(
    "SKPLT_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT", int, 500
)

#: Specifies the number of connection pools to cache in urllib3. This environment variable sets the
#: `pool_connections` parameter in the `requests.adapters.HTTPAdapter` constructor. By adjusting
#: this variable, users can enhance the concurrency of HTTP requests made by MLflow.
SKPLT_HTTP_POOL_CONNECTIONS = EnvironmentVariable(
    "SKPLT_HTTP_POOL_CONNECTIONS", int, 10
)

#: Specifies the maximum number of connections to keep in the HTTP connection pool. This environment
#: variable sets the `pool_maxsize` parameter in the `requests.adapters.HTTPAdapter` constructor.
#: By adjusting this variable, users can enhance the concurrency of HTTP requests made by MLflow.
SKPLT_HTTP_POOL_MAXSIZE = EnvironmentVariable("SKPLT_HTTP_POOL_MAXSIZE", int, 10)

#: Enable Unity Catalog integration for MLflow AI Gateway.
#: (default: ``False``)
SKPLT_ENABLE_UC_FUNCTIONS = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_UC_FUNCTIONS", False
)

#: Specifies the length of time in seconds for the asynchronous logging thread to wait before
#: logging a batch.
SKPLT_ASYNC_LOGGING_BUFFERING_SECONDS = EnvironmentVariable(
    "SKPLT_ASYNC_LOGGING_BUFFERING_SECONDS", int, None
)

#: Whether to enable Databricks SDK. If true, MLflow uses databricks-sdk to send HTTP requests
#: to Databricks endpoint, otherwise MLflow uses ``requests`` library to send HTTP requests
#: to Databricks endpoint. Note that if you want to use OAuth authentication, you have to
#: set this environment variable to true.
#: (default: ``True``)
SKPLT_ENABLE_DB_SDK = BooleanEnvironmentVariable("SKPLT_ENABLE_DB_SDK", True)

#: A flag that's set to 'true' in the child process for capturing modules.
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_IN_CAPTURE_MODULE_PROCESS"`` — missing ``_`` prefix.
_SKPLT_IN_CAPTURE_MODULE_PROCESS = BooleanEnvironmentVariable(
    "_SKPLT_IN_CAPTURE_MODULE_PROCESS", False
)

#: Use DatabricksSDKModelsArtifactRepository when registering and loading models to and from
#: Databricks UC. This is required for SEG(Secure Egress Gateway) enabled workspaces and helps
#: eliminate models exfiltration risk associated with temporary scoped token generation used in
#: existing model artifact repo classes.
SKPLT_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC = BooleanEnvironmentVariable(
    "SKPLT_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", False
)

#: Specifies the model environment archive file downloading path when using
#: ``mlflow.pyfunc.spark_udf``. (default: ``None``)
SKPLT_MODEL_ENV_DOWNLOADING_TEMP_DIR = EnvironmentVariable(
    "SKPLT_MODEL_ENV_DOWNLOADING_TEMP_DIR", str, None
)

#: Specifies whether to log environment variable names used during model logging.
SKPLT_RECORD_ENV_VARS_IN_MODEL_LOGGING = BooleanEnvironmentVariable(
    "SKPLT_RECORD_ENV_VARS_IN_MODEL_LOGGING", True
)

#: Specifies the artifact compression method used when logging a model
#: allowed values are "lzma", "bzip2" and "gzip"
#: (default: ``None``, indicating no compression)
SKPLT_LOG_MODEL_COMPRESSION = EnvironmentVariable(
    "SKPLT_LOG_MODEL_COMPRESSION", str, None
)


#: Specifies whether to convert a {"messages": [{"role": "...", "content": "..."}]} input
#: to a List[BaseMessage] object when invoking a PyFunc model saved with langchain flavor.
#: This takes precedence over the default behavior of trying such conversion if the model
#: is not an AgentExecutor and the input schema doesn't contain a 'messages' field.
SKPLT_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN = BooleanEnvironmentVariable(
    "SKPLT_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN", None
)

#: A boolean flag which enables additional functionality in Python tests for GO backend.
#:
#: .. note::
#:    FIX: env var string was ``"SKPLT_GO_STORE_TESTING"`` — missing ``_`` prefix.
_SKPLT_GO_STORE_TESTING = BooleanEnvironmentVariable("_SKPLT_GO_STORE_TESTING", False)

#: Specifies whether the current environment is a serving environment.
#: This should only be used internally by MLflow to add some additional logic when running in a
#: serving environment.
_SKPLT_IS_IN_SERVING_ENVIRONMENT = BooleanEnvironmentVariable(
    "_SKPLT_IS_IN_SERVING_ENVIRONMENT", None
)

#: Secret key for the Flask app. This is necessary for enabling CSRF protection
#: in the UI signup page when running the app with basic authentication enabled
SKPLT_FLASK_SERVER_SECRET_KEY = EnvironmentVariable(
    "SKPLT_FLASK_SERVER_SECRET_KEY", str, None
)

#: Specifies the max length (in chars) of an experiment's artifact location.
#: The default is 2048.
SKPLT_ARTIFACT_LOCATION_MAX_LENGTH = EnvironmentVariable(
    "SKPLT_ARTIFACT_LOCATION_MAX_LENGTH", int, 2048
)

#: Path to SSL CA certificate file for MySQL connections
#: Used when creating a SQLAlchemy engine for MySQL
#: (default: ``None``)
SKPLT_MYSQL_SSL_CA = EnvironmentVariable("SKPLT_MYSQL_SSL_CA", str, None)

#: Path to SSL certificate file for MySQL connections
#: Used when creating a SQLAlchemy engine for MySQL
#: (default: ``None``)
SKPLT_MYSQL_SSL_CERT = EnvironmentVariable("SKPLT_MYSQL_SSL_CERT", str, None)

#: Path to SSL key file for MySQL connections
#: Used when creating a SQLAlchemy engine for MySQL
#: (default: ``None``)
SKPLT_MYSQL_SSL_KEY = EnvironmentVariable("SKPLT_MYSQL_SSL_KEY", str, None)


#: Specifies whether to enable async trace logging to Databricks Tracing Server.
#: Default: ``True``.
SKPLT_ENABLE_ASYNC_TRACE_LOGGING = BooleanEnvironmentVariable(
    "SKPLT_ENABLE_ASYNC_TRACE_LOGGING", True
)

#: Maximum number of worker threads to use for async trace logging.
#: (default: ``10``)
SKPLT_ASYNC_TRACE_LOGGING_MAX_WORKERS = EnvironmentVariable(
    "SKPLT_ASYNC_TRACE_LOGGING_MAX_WORKERS", int, 10
)

#: Maximum number of export tasks to queue for async trace logging.
#: When the queue is full, new export tasks will be dropped.
#: (default: ``1000``)
SKPLT_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE = EnvironmentVariable(
    "SKPLT_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE", int, 1000
)


#: Timeout seconds for retrying trace logging.
#: (default: ``500``)
SKPLT_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT = EnvironmentVariable(
    "SKPLT_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", int, 500
)


#: Default active LoggedModel ID.
#: This should only by used by MLflow internally, users should always use
#: `set_active_model` to set the active LoggedModel, and should not set
#: this environment variable directly.
#: (default: ``None``)
_SKPLT_ACTIVE_MODEL_ID = EnvironmentVariable("_SKPLT_ACTIVE_MODEL_ID", str, None)

#: Maximum number of parameters to include in the initial CreateLoggedModel request.
#: Additional parameters will be logged in separate requests.
#: (default: ``100``)
_SKPLT_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE = EnvironmentVariable(
    "_SKPLT_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE", int, 100
)


#: Maximum number of parameters to include in each batch when logging parameters
#: for a logged model.
#: (default: ``100``)
_SKPLT_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE = EnvironmentVariable(
    "_SKPLT_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE", int, 100
)

#: A boolean flag that enables printing URLs for logged and registered models when
#: they are created.
#: (default: ``True``)
SKPLT_PRINT_MODEL_URLS_ON_CREATION = BooleanEnvironmentVariable(
    "SKPLT_PRINT_MODEL_URLS_ON_CREATION", True
)

#: Maximum number of threads to use when downloading traces during search operations.
#: (default: ``max(32, (# of system CPUs * 4)``)
SKPLT_SEARCH_TRACES_MAX_THREADS = EnvironmentVariable(
    # Threads used to download traces during search are network IO-bound (waiting for downloads)
    # rather than CPU-bound, so we want more threads than CPU cores
    "SKPLT_SEARCH_TRACES_MAX_THREADS",
    int,
    max(32, (_os.cpu_count() or 1) * 4),
)


#: Specifies the logging level for MLflow. This can be set to any valid logging level
#: (e.g., "DEBUG", "INFO"). This environment must be set before importing mlflow to take
#: effect. To modify the logging level after importing mlflow, use `importlib.reload(mlflow)`.
#: (default: ``None``).
SKPLT_LOGGING_LEVEL = EnvironmentVariable("SKPLT_LOGGING_LEVEL", str, None)
