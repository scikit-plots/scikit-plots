# SPDX-License-Identifier: BSD-3-Clause
"""Non-secret provenance helpers for optional build-time LLM augmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class AugmentationSettings:
    enabled: bool
    provider: str
    model: str
    base_url: str
    api_key_env: str
    allow_insecure_auth: bool
    max_input_chars: int
    timeout: int

    def public_record(self) -> dict[str, Any]:
        """
        Return non-secret provider provenance.

        URLs can accidentally carry credentials in userinfo/query/path fields.
        Persist only the origin (scheme + hostname + optional port), never the
        raw configured URL. The cache fingerprint below can still depend on the
        complete setting without exposing it.
        """

        parsed = urlparse(self.base_url)
        origin: str | None = None
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            try:
                port = f":{parsed.port}" if parsed.port is not None else ""
            except ValueError:
                port = ""
            origin = f"{parsed.scheme.lower()}://{host}{port}"
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "base_url_origin": origin,
            "api_key_env": self.api_key_env,
            "allow_insecure_auth": self.allow_insecure_auth,
            "max_input_chars": self.max_input_chars,
            "timeout": self.timeout,
        }

    def fingerprint(self, text: str, *, prompt_version: int) -> str:
        # The digest changes for every behavior-affecting setting, including the
        # complete endpoint, but only the digest is exposed to callers.
        payload = {
            "settings": {
                "enabled": self.enabled,
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "allow_insecure_auth": self.allow_insecure_auth,
                "max_input_chars": self.max_input_chars,
                "timeout": self.timeout,
            },
            "prompt_version": int(prompt_version),
            "text": text,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def endpoint_may_receive_secret(
    base_url: str, *, has_api_key: bool, allow_insecure_auth: bool
) -> bool:
    """Return whether credential transport is acceptable before provider I/O."""

    if not has_api_key:
        return True
    if not str(base_url).strip():
        # Empty means the provider client's HTTPS default endpoint.
        return True
    parsed = urlparse(str(base_url))
    scheme = parsed.scheme.lower()
    if scheme == "https":
        return True
    if scheme != "http":
        return False
    host = (parsed.hostname or "").lower()
    loopback = (
        host == "localhost"
        or host.endswith(".localhost")
        or host in {"127.0.0.1", "::1"}
    )
    return loopback or allow_insecure_auth


__all__ = ["AugmentationSettings", "endpoint_may_receive_secret"]
