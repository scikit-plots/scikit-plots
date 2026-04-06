# scikitplot/mlflow/_security.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
_security.

Security policy framework for :py:mod:`scikitplot.mlflow`.

This module provides a declarative, composable security policy mechanism that
guards against common threat vectors in MLflow integration code:

- **Path traversal** in env files and local store URIs.
- **SSRF** via crafted tracking URIs pointing at cloud metadata endpoints
  (e.g., ``169.254.169.254``).
- **Command injection** via ``extra_args``, ``gunicorn_opts``, etc.
- **Header injection** via CORS/X-Frame-Options values containing ``\\r\\n``.
- **Environment variable poisoning** via ``extra_env`` setting linker hooks
  (``LD_PRELOAD``, ``LD_LIBRARY_PATH``).
- **Denial of service** via excessively large env values.

Two ready-made policies are provided:

:data:`DEFAULT_SECURITY_POLICY`
    Conservative policy for production and shared environments.
    Blocks the most dangerous vectors while preserving full MLflow functionality.

:data:`RELAXED_SECURITY_POLICY`
    Permissive policy for fully trusted, isolated local development.
    All restrictions lifted.

Notes
-----
By default no policy is active (``get_security_policy()`` returns ``None``),
so existing code is unaffected.  Policies are enforced only when explicitly set.

Examples
--------
Enable default security globally:

>>> from scikitplot.mlflow._security import DEFAULT_SECURITY_POLICY, set_security_policy
>>> set_security_policy(DEFAULT_SECURITY_POLICY)

Scope a policy to a single block:

>>> from scikitplot.mlflow._security import DEFAULT_SECURITY_POLICY, security_policy
>>> with security_policy(DEFAULT_SECURITY_POLICY):
...     pass  # session() / spawn_server() calls here are guarded

Disable enforcement (default; backwards-compatible):

>>> set_security_policy(None)
"""

from __future__ import annotations

import contextlib
import ipaddress
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._errors import SecurityPolicyViolationError

if TYPE_CHECKING:  # pragma: no cover
    from ._config import ServerConfig, SessionConfig  # noqa: F401

__all__ = [
    "DEFAULT_SECURITY_POLICY",
    "RELAXED_SECURITY_POLICY",
    "SecurityPolicy",
    "get_security_policy",
    "security_policy",
    "set_security_policy",
]

# ---------------------------------------------------------------------------
# Compiled patterns (module-level, evaluated once)
# ---------------------------------------------------------------------------

#: Shell metacharacters enabling command injection.
#:
#: CR (``\r``) and LF (``\n``) are intentionally excluded here; they are
#: covered exclusively by :data:`_HEADER_INJECT_RE` and the
#: ``block_header_injection`` flag.  Keeping the two checks orthogonal
#: means callers can disable header-injection checking without the shell-meta
#: check silently absorbing CRLF as a side-effect.
_SHELL_META_RE: re.Pattern[str] = re.compile(r"[;|&`$<>()\{\}]|\$\(|&&|\|\|")

#: CR / LF for HTTP header injection detection.
_HEADER_INJECT_RE: re.Pattern[str] = re.compile(r"[\r\n]")

#: IPv4 cloud-metadata / link-local CIDRs.
_CLOUD_METADATA_CIDRS_V4: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("169.254.0.0/16"),  # AWS/GCP/Azure metadata; also RFC 3927
    ipaddress.IPv4Network("100.64.0.0/10"),  # IANA shared address space
)

#: Provider-specific DNS names for metadata endpoints.
_CLOUD_METADATA_HOSTNAMES: frozenset[str] = frozenset(
    {
        "metadata.google.internal",
        "instance-data",
        "link-local",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cloud_metadata_host(hostname: str) -> bool:
    """
    Detect whether *hostname* refers to a cloud metadata endpoint.

    Parameters
    ----------
    hostname : str
        Raw hostname or IP string from a URI.

    Returns
    -------
    bool
        True if the host is a known metadata endpoint.

    Notes
    -----
    Checks both provider-specific DNS names and IANA-reserved IPv4 ranges.
    IPv6 cloud metadata is not standardised and is not checked.
    """
    lower = hostname.lower().strip()
    if lower in _CLOUD_METADATA_HOSTNAMES:
        return True
    try:
        addr = ipaddress.ip_address(lower)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in cidr for cidr in _CLOUD_METADATA_CIDRS_V4)
    return False


def _has_path_traversal(path: str) -> bool:
    """
    Detect ``..`` components in *path* that enable directory traversal.

    Parameters
    ----------
    path : str
        Filesystem path or URI path component.

    Returns
    -------
    bool
        True if a ``..`` traversal component is present.
    """
    normalised = path.replace("\\", "/")
    return bool(re.search(r"(^|/)\.\.(/|$)", normalised))


# ---------------------------------------------------------------------------
# SecurityPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityPolicy:
    """
    Declarative security policy for :py:mod:`scikitplot.mlflow` operations.

    Parameters
    ----------
    allowed_tracking_uri_schemes : frozenset[str]
        URI schemes accepted for tracking / registry URIs.
        Empty frozenset disables scheme enforcement.
        Default: ``{"http", "https", "file", "sqlite"}``.
    block_cloud_metadata_hosts : bool, default=True
        Reject HTTP(S) URIs whose hostname resolves to a cloud metadata endpoint
        (e.g., ``169.254.169.254``).  Prevents SSRF attacks.
    allow_spawn_server : bool, default=True
        Allow spawning a managed MLflow server subprocess.
        Set False in environments where process spawning is prohibited.
    allow_dev_mode : bool, default=False
        Allow ``ServerConfig(dev=True)``.
        Dev mode disables production hardening and must not be used in shared
        environments.
    allow_disable_security_middleware : bool, default=False
        Allow ``ServerConfig(disable_security_middleware=True)``.
    allow_cors_wildcard : bool, default=False
        Allow ``ServerConfig(cors_allowed_origins="*")``.
        Wildcard CORS grants any origin access to the server.
    blocked_env_key_prefixes : frozenset[str]
        Env key prefixes unconditionally rejected in ``extra_env``.
        Default: ``{"LD_"}`` blocks ``LD_PRELOAD``, ``LD_LIBRARY_PATH``, etc.
    max_env_value_length : int, default=65536
        Maximum byte-length of any single env value (64 KiB).
    max_env_pairs : int, default=256
        Maximum key-value pairs in ``extra_env``.
    block_path_traversal : bool, default=True
        Reject paths containing ``..`` traversal components.
    block_shell_metacharacters_in_args : bool, default=True
        Reject CLI option values containing shell metacharacters.
    block_header_injection : bool, default=True
        Reject header-like values containing CR or LF.

    Raises
    ------
    ValueError
        If ``max_env_value_length`` or ``max_env_pairs`` is not positive.
    TypeError
        If ``blocked_env_key_prefixes`` contains non-string elements.

    See Also
    --------
    DEFAULT_SECURITY_POLICY : Conservative production-grade preset.
    RELAXED_SECURITY_POLICY : Permissive preset for trusted local development.
    set_security_policy : Activate a policy globally.
    security_policy : Activate a policy for a context block.
    """

    allowed_tracking_uri_schemes: frozenset[str] = frozenset(
        {"http", "https", "file", "sqlite"}
    )
    block_cloud_metadata_hosts: bool = True

    allow_spawn_server: bool = True
    allow_dev_mode: bool = False
    allow_disable_security_middleware: bool = False
    allow_cors_wildcard: bool = False

    blocked_env_key_prefixes: frozenset[str] = frozenset({"LD_"})
    max_env_value_length: int = 65536
    max_env_pairs: int = 256

    block_path_traversal: bool = True
    block_shell_metacharacters_in_args: bool = True
    block_header_injection: bool = True

    def __post_init__(self) -> None:
        if self.max_env_value_length <= 0:
            raise ValueError(
                f"max_env_value_length must be a positive integer, "
                f"got {self.max_env_value_length!r}."
            )
        if self.max_env_pairs <= 0:
            raise ValueError(
                f"max_env_pairs must be a positive integer, got {self.max_env_pairs!r}."
            )
        for prefix in self.blocked_env_key_prefixes:
            if not isinstance(prefix, str):
                raise TypeError(
                    f"blocked_env_key_prefixes must contain str elements, "
                    f"got {type(prefix).__name__!r}."
                )

    # =========================================================================
    # Validation methods
    # =========================================================================

    def validate_tracking_uri(
        self,
        uri: str,
        *,
        context: str = "tracking_uri",
    ) -> None:
        """
        Validate a tracking or registry URI under this policy.

        Parameters
        ----------
        uri : str
            URI to validate.
        context : str, default="tracking_uri"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If the URI scheme is not allowed, the host is a cloud metadata
            endpoint, or the URI path contains a traversal sequence.
        """
        from urllib.parse import urlparse  # noqa: PLC0415

        parsed = urlparse(uri)
        scheme = (parsed.scheme or "").lower()

        if (
            self.allowed_tracking_uri_schemes
            and scheme not in self.allowed_tracking_uri_schemes
        ):
            raise SecurityPolicyViolationError(
                f"{context}: URI scheme {scheme!r} is not in the allowed set "
                f"{sorted(self.allowed_tracking_uri_schemes)!r}. "
                f"Rejected URI: {uri!r}."
            )

        if (
            self.block_cloud_metadata_hosts
            and scheme in {"http", "https"}
            and parsed.hostname
            and _is_cloud_metadata_host(parsed.hostname)
        ):
            raise SecurityPolicyViolationError(
                f"{context}: URI hostname {parsed.hostname!r} appears to be a "
                f"cloud metadata endpoint. Rejected URI: {uri!r}."
            )

        if (
            self.block_path_traversal
            and parsed.path
            and _has_path_traversal(parsed.path)
        ):
            raise SecurityPolicyViolationError(
                f"{context}: URI path contains a traversal sequence ('..') "
                f"which is not allowed. Rejected URI: {uri!r}."
            )

    def validate_server_config(
        self,
        cfg: Any,
        *,
        context: str = "server config",
    ) -> None:
        """
        Validate a :class:`~scikitplot.mlflow.ServerConfig` under this policy.

        Parameters
        ----------
        cfg : ServerConfig
            Server configuration to validate.
        context : str, default="server config"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If any field violates the policy.

        Notes
        -----
        Validated fields: ``dev``, ``disable_security_middleware``,
        ``cors_allowed_origins``, ``allowed_hosts``, ``x_frame_options``,
        ``gunicorn_opts``, ``uvicorn_opts``, ``waitress_opts``, ``extra_args``.
        """
        if not self.allow_dev_mode and getattr(cfg, "dev", False):
            raise SecurityPolicyViolationError(
                f"{context}: dev=True is not permitted by the active security "
                f"policy (allow_dev_mode=False). Dev mode disables production "
                f"hardening and must not be used in shared environments."
            )

        if not self.allow_disable_security_middleware and getattr(
            cfg, "disable_security_middleware", False
        ):
            raise SecurityPolicyViolationError(
                f"{context}: disable_security_middleware=True is not permitted "
                f"by the active security policy (allow_disable_security_middleware=False)."
            )

        cors = getattr(cfg, "cors_allowed_origins", None)
        if not self.allow_cors_wildcard and cors == "*":
            raise SecurityPolicyViolationError(
                f"{context}: cors_allowed_origins='*' (wildcard) is not permitted "
                f"by the active security policy (allow_cors_wildcard=False)."
            )

        if self.block_header_injection:
            for field_name in (
                "cors_allowed_origins",
                "x_frame_options",
                "allowed_hosts",
            ):
                val = getattr(cfg, field_name, None)
                if val is not None and _HEADER_INJECT_RE.search(val):
                    raise SecurityPolicyViolationError(
                        f"{context}: {field_name!r} contains CR or LF characters, "
                        f"enabling HTTP header injection. Rejected value: {val!r}."
                    )

        if self.block_shell_metacharacters_in_args:
            for field_name in ("gunicorn_opts", "uvicorn_opts", "waitress_opts"):
                val = getattr(cfg, field_name, None)
                if val is not None:
                    self.validate_cli_arg_value(val, context=f"{context}.{field_name}")

            extra_args = getattr(cfg, "extra_args", None)
            if extra_args:
                for idx, arg in enumerate(extra_args):
                    self.validate_cli_arg_value(
                        arg, context=f"{context}.extra_args[{idx}]"
                    )

    def validate_session_config(
        self,
        cfg: Any,
        *,
        context: str = "session config",
    ) -> None:
        """
        Validate a :class:`~scikitplot.mlflow.SessionConfig` under this policy.

        Parameters
        ----------
        cfg : SessionConfig
            Session configuration to validate.
        context : str, default="session config"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If any field violates the policy.

        Notes
        -----
        Validated fields: ``tracking_uri``, ``registry_uri``, ``env_file``,
        ``extra_env``.
        """
        tracking_uri = getattr(cfg, "tracking_uri", None)
        if tracking_uri is not None:
            self.validate_tracking_uri(tracking_uri, context=f"{context}.tracking_uri")

        registry_uri = getattr(cfg, "registry_uri", None)
        if registry_uri is not None:
            self.validate_tracking_uri(registry_uri, context=f"{context}.registry_uri")

        env_file = getattr(cfg, "env_file", None)
        if env_file is not None:
            self.validate_path(str(env_file), context=f"{context}.env_file")

        extra_env = getattr(cfg, "extra_env", None)
        if extra_env is not None:
            self.validate_env_mapping(extra_env, context=f"{context}.extra_env")

    def validate_env_item(
        self,
        key: str,
        value: str,
        *,
        context: str = "extra_env",
    ) -> None:
        """
        Validate a single environment variable key-value pair.

        Parameters
        ----------
        key : str
            Environment variable name.
        value : str
            Environment variable value.
        context : str, default="extra_env"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If the key matches a blocked prefix, the value exceeds
            ``max_env_value_length``, or the value contains CR/LF.
        """
        for prefix in self.blocked_env_key_prefixes:
            if key.startswith(prefix):
                raise SecurityPolicyViolationError(
                    f"{context}: environment key {key!r} starts with blocked "
                    f"prefix {prefix!r}. Keys matching this prefix can enable "
                    f"dynamic linker injection or similar attacks."
                )

        if len(value) > self.max_env_value_length:
            raise SecurityPolicyViolationError(
                f"{context}: value for key {key!r} has length {len(value)} "
                f"which exceeds max_env_value_length={self.max_env_value_length}."
            )

        if self.block_header_injection and _HEADER_INJECT_RE.search(value):
            raise SecurityPolicyViolationError(
                f"{context}: value for key {key!r} contains CR or LF characters "
                f"which can enable header injection attacks."
            )

    def validate_env_mapping(
        self,
        env: Mapping[str, str],
        *,
        context: str = "extra_env",
    ) -> None:
        """
        Validate an entire environment variable mapping.

        Parameters
        ----------
        env : Mapping[str, str]
            Environment variable mapping to validate.
        context : str, default="extra_env"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If the mapping exceeds ``max_env_pairs`` or any item fails
            :meth:`validate_env_item`.
        """
        n = len(env)
        if n > self.max_env_pairs:
            raise SecurityPolicyViolationError(
                f"{context}: mapping has {n} pairs which exceeds "
                f"max_env_pairs={self.max_env_pairs}."
            )
        for key, value in env.items():
            self.validate_env_item(key, value, context=context)

    def validate_path(
        self,
        path: str,
        *,
        context: str = "path",
    ) -> None:
        """
        Validate a filesystem path against path traversal.

        Parameters
        ----------
        path : str
            Filesystem path to validate.
        context : str, default="path"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If ``block_path_traversal=True`` and *path* contains ``..``.
        """
        if not self.block_path_traversal:
            return
        if _has_path_traversal(path):
            raise SecurityPolicyViolationError(
                f"{context}: path contains a traversal sequence ('..') which is "
                f"not allowed by the active security policy. "
                f"Rejected path: {path!r}."
            )

    def validate_cli_arg_value(
        self,
        value: str,
        *,
        context: str = "cli arg",
    ) -> None:
        """
        Validate a CLI argument value against shell metacharacter injection.

        Parameters
        ----------
        value : str
            CLI argument value to validate.
        context : str, default="cli arg"
            Label for error messages.

        Returns
        -------
        None

        Raises
        ------
        SecurityPolicyViolationError
            If ``block_shell_metacharacters_in_args=True`` and *value* contains
            shell metacharacters, or ``block_header_injection=True`` and *value*
            contains CR/LF.
        """
        # The two checks are orthogonal: CR/LF belongs solely to the header-
        # injection check; shell metacharacters belong solely to the shell-meta
        # check.  Disabling either flag must not cause the other to absorb its
        # characters as a side-effect.
        if self.block_header_injection and _HEADER_INJECT_RE.search(value):
            raise SecurityPolicyViolationError(
                f"{context}: value contains CR or LF characters which can "
                f"enable HTTP header injection. Rejected value: {value!r}."
            )
        if self.block_shell_metacharacters_in_args and _SHELL_META_RE.search(value):
            raise SecurityPolicyViolationError(
                f"{context}: value contains shell metacharacters which can "
                f"enable command injection. Rejected value: {value!r}."
            )


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

#: Conservative policy for production and shared environments.
#:
#: Enforces:
#: - Allowed URI schemes: ``http``, ``https``, ``file``, ``sqlite``.
#: - Cloud metadata SSRF protection (``169.254.0.0/16``).
#: - No dev mode on server.
#: - No disable of security middleware.
#: - No wildcard CORS.
#: - Blocks ``LD_*`` env key prefix (linker injection).
#: - Env value cap: 64 KiB.
#: - Max env pairs: 256.
#: - Path traversal protection.
#: - Shell metacharacter protection in CLI args.
#: - CR/LF header injection protection.
DEFAULT_SECURITY_POLICY: SecurityPolicy = SecurityPolicy()

#: Permissive policy for fully trusted, isolated local development.
#:
#: All restrictions lifted.  Do NOT use in shared or networked environments.
RELAXED_SECURITY_POLICY: SecurityPolicy = SecurityPolicy(
    allowed_tracking_uri_schemes=frozenset(
        {
            "http",
            "https",
            "file",
            "sqlite",
            "mysql",
            "mysql+pymysql",
            "postgresql",
            "postgresql+psycopg2",
            "mssql",
            "mssql+pyodbc",
        }
    ),
    block_cloud_metadata_hosts=False,
    allow_spawn_server=True,
    allow_dev_mode=True,
    allow_disable_security_middleware=True,
    allow_cors_wildcard=True,
    blocked_env_key_prefixes=frozenset(),
    max_env_value_length=1_048_576,  # 1 MiB
    max_env_pairs=1024,
    block_path_traversal=False,
    block_shell_metacharacters_in_args=False,
    block_header_injection=False,
)


# ---------------------------------------------------------------------------
# Global active policy state
# ---------------------------------------------------------------------------

_ACTIVE_POLICY: SecurityPolicy | None = None


def get_security_policy() -> SecurityPolicy | None:
    """
    Return the currently active :class:`SecurityPolicy`, or ``None``.

    Returns
    -------
    SecurityPolicy or None
        Active policy.  ``None`` means no enforcement (default; backwards-compatible).

    See Also
    --------
    set_security_policy : Activate a policy globally.
    security_policy : Activate a policy for a context block.
    """
    return _ACTIVE_POLICY


def set_security_policy(policy: SecurityPolicy | None) -> None:
    """
    Set the active :class:`SecurityPolicy` globally.

    Parameters
    ----------
    policy : SecurityPolicy or None
        Policy to activate.  Pass ``None`` to disable enforcement.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *policy* is not a :class:`SecurityPolicy` or ``None``.

    Notes
    -----
    Mutates module-level state.  For scope-limited use, prefer
    :func:`security_policy`.

    See Also
    --------
    security_policy : Activate a policy for a single context block.
    """
    global _ACTIVE_POLICY  # noqa: PLW0603
    if policy is not None and not isinstance(policy, SecurityPolicy):
        raise TypeError(
            f"policy must be a SecurityPolicy instance or None, "
            f"got {type(policy).__name__!r}."
        )
    _ACTIVE_POLICY = policy


@contextlib.contextmanager
def security_policy(policy: SecurityPolicy | None) -> Iterator[None]:
    """
    Temporarily activate a :class:`SecurityPolicy` for a context block.

    Parameters
    ----------
    policy : SecurityPolicy or None
        Policy to activate.  Pass ``None`` to disable enforcement within the block.

    Yields
    ------
    None

    Notes
    -----
    Exception-safe: previous policy is always restored on exit.

    Examples
    --------
    >>> from scikitplot.mlflow._security import DEFAULT_SECURITY_POLICY, security_policy
    >>> with security_policy(DEFAULT_SECURITY_POLICY):
    ...     pass  # session() calls here are guarded

    See Also
    --------
    set_security_policy : Set the active policy globally.
    """
    global _ACTIVE_POLICY  # noqa: PLW0603
    old = _ACTIVE_POLICY
    try:
        set_security_policy(policy)
        yield
    finally:
        _ACTIVE_POLICY = old
