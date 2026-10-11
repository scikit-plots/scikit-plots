# scikitplot/corpus/_downloader/_policy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Immutable downloader policy presets and validation helpers.

The policy groups transport/resource controls that otherwise appear as a long
list of loosely related keyword arguments.  It does not perform I/O and it does
not weaken the existing downloader defaults.  Callers may still override an
individual value explicitly after choosing a policy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from ._base import (
    _DEFAULT_MAX_BYTES,
    _DEFAULT_MAX_REDIRECTS,
    _DEFAULT_TIMEOUT,
    _DEFAULT_USER_AGENT,
)

__all__ = ["DownloadPolicy", "download_policy"]


@dataclasses.dataclass(frozen=True)
class DownloadPolicy:
    """
    Transport/resource policy shared by Corpus downloaders.

    This object intentionally contains no credentials, headers or destination
    paths, so it is safe to serialize in plans and diagnostics.
    """

    name: str = "secure"
    timeout: float = _DEFAULT_TIMEOUT
    max_bytes: int = _DEFAULT_MAX_BYTES
    verify_ssl: bool = True
    block_private_ips: bool = True
    max_redirects: int = _DEFAULT_MAX_REDIRECTS
    user_agent: str = _DEFAULT_USER_AGENT
    max_retries: int = 3
    retry_backoff: float = 1.0

    def __post_init__(  # ruff: ignore[too-many-branches]
        self,
    ) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DownloadPolicy.name must be a non-empty string")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("DownloadPolicy.timeout must be a number")
        if self.timeout <= 0:
            raise ValueError("DownloadPolicy.timeout must be > 0")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int):
            raise TypeError("DownloadPolicy.max_bytes must be an int")
        if self.max_bytes <= 0:
            raise ValueError("DownloadPolicy.max_bytes must be > 0")
        if not isinstance(self.verify_ssl, bool):
            raise TypeError("DownloadPolicy.verify_ssl must be bool")
        if not isinstance(self.block_private_ips, bool):
            raise TypeError("DownloadPolicy.block_private_ips must be bool")
        if isinstance(self.max_redirects, bool) or not isinstance(
            self.max_redirects, int
        ):
            raise TypeError("DownloadPolicy.max_redirects must be an int")
        if self.max_redirects < 0:
            raise ValueError("DownloadPolicy.max_redirects must be >= 0")
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("DownloadPolicy.user_agent must be a non-empty string")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("DownloadPolicy.max_retries must be an int")
        if self.max_retries < 0:
            raise ValueError("DownloadPolicy.max_retries must be >= 0")
        if isinstance(self.retry_backoff, bool) or not isinstance(
            self.retry_backoff, (int, float)
        ):
            raise TypeError("DownloadPolicy.retry_backoff must be a number")
        if self.retry_backoff <= 0:
            raise ValueError("DownloadPolicy.retry_backoff must be > 0")

    @classmethod
    def secure(cls) -> DownloadPolicy:
        """Return the security-first library defaults."""
        return cls()

    @classmethod
    def constrained(cls) -> DownloadPolicy:
        """Return a smaller CI/docs-friendly resource budget."""
        return cls(
            name="constrained",
            timeout=15.0,
            max_bytes=25 * 1024 * 1024,
            max_redirects=3,
            max_retries=1,
            retry_backoff=0.5,
        )

    @classmethod
    def large_files(cls) -> DownloadPolicy:
        """Allow large transfers without relaxing TLS or SSRF protections."""
        return cls(
            name="large-files",
            timeout=180.0,
            max_bytes=1024 * 1024 * 1024,
            max_redirects=5,
            max_retries=3,
            retry_backoff=1.0,
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Return constructor kwargs for downloader classes."""
        return {
            "timeout": float(self.timeout),
            "max_bytes": int(self.max_bytes),
            "verify_ssl": self.verify_ssl,
            "block_private_ips": self.block_private_ips,
            "max_redirects": int(self.max_redirects),
            "user_agent": self.user_agent,
            "max_retries": int(self.max_retries),
            "retry_backoff": float(self.retry_backoff),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy representation."""
        return {"name": self.name, **self.to_kwargs()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DownloadPolicy:
        """Build from JSON/YAML-shaped configuration with typo rejection."""
        if not isinstance(config, Mapping):
            raise TypeError("DownloadPolicy config must be a mapping")
        data = dict(config)
        preset = str(data.pop("preset", "secure")).strip().lower()
        base = download_policy(preset)
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"unknown DownloadPolicy config field(s) {unknown}; "
                f"expected subset of {sorted(allowed)} plus 'preset'"
            )
        return dataclasses.replace(base, **data)


_PRESETS = {
    "secure": DownloadPolicy.secure,
    "default": DownloadPolicy.secure,
    "constrained": DownloadPolicy.constrained,
    "ci": DownloadPolicy.constrained,
    "docs": DownloadPolicy.constrained,
    "large": DownloadPolicy.large_files,
    "large-files": DownloadPolicy.large_files,
}


def download_policy(
    value: DownloadPolicy | str | Mapping[str, Any] | None,
) -> DownloadPolicy:
    """Resolve an object, named preset or mapping without hidden global state."""
    if value is None:
        return DownloadPolicy.secure()
    if isinstance(value, DownloadPolicy):
        return value
    if isinstance(value, Mapping):
        return DownloadPolicy.from_config(value)
    key = str(value).strip().lower()
    try:
        return _PRESETS[key]()
    except KeyError as exc:
        raise ValueError(
            f"unknown download policy {value!r}; expected one of {sorted(_PRESETS)}"
        ) from exc
