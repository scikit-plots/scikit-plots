#!/usr/bin/env python3
"""
Collect sanitized Redis operational evidence for one control plane.

The Redis URL is read only from an environment variable named by the operator;
it is never accepted as a command-line value and never printed.  Output contains
no host, port, username, keys, values, replication offsets, or credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from _utils._redis_security import (  # noqa: E402
    RedisSecurityError,
    redis_connection_kwargs,
)

logger = logging.getLogger(__name__)


def _now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def collect(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    plane: str,
    url_env: str,
    client: Any | None = None,
) -> dict[str, Any]:
    if plane not in {"rateLimit", "share", "contribution"}:
        raise RuntimeError("PLANE_INVALID")
    if not url_env or any(
        ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in url_env
    ):
        raise RuntimeError("URL_ENV_NAME_INVALID")
    url = os.environ.get(url_env, "")
    try:
        policy, kwargs = redis_connection_kwargs(
            url, require_tls=True, socket_timeout_seconds=3.0
        )
    except RedisSecurityError as exc:
        raise RuntimeError(exc.code) from exc
    if client is None:
        try:
            import redis  # type: ignore[]  # ruff: ignore[import-outside-top-level]

            client = redis.Redis.from_url(url, **kwargs)
        except Exception as exc:
            raise RuntimeError("REDIS_CLIENT_UNAVAILABLE") from exc
    try:
        ping = bool(client.ping())
    except Exception as exc:
        raise RuntimeError("REDIS_PING_FAILED") from exc
    if not ping:
        raise RuntimeError("REDIS_PING_FAILED")

    non_default_identity = False
    acl_identity_observed = False
    try:
        who = client.execute_command("ACL", "WHOAMI")
        if isinstance(who, bytes):
            who = who.decode("utf-8", "replace")
        acl_identity_observed = bool(str(who or "").strip())
        non_default_identity = (
            acl_identity_observed and str(who).strip().lower() != "default"
        )
    except Exception:  # ruff: ignore[blind-except]
        # Managed providers may deny ACL inspection. Do not broaden permissions
        # just for this probe; leave the fact unproven for operator evidence.
        pass

    try:
        persistence = client.info("persistence") or {}
        replication = client.info("replication") or {}
    except Exception as exc:
        raise RuntimeError("REDIS_INFO_UNAVAILABLE") from exc
    aof_enabled = _int(persistence.get("aof_enabled")) == 1
    loading = _int(persistence.get("loading")) == 1
    role = str(replication.get("role") or "").lower()
    if role == "master":
        replication_observed = _int(replication.get("connected_slaves")) >= 1
    elif role in {"slave", "replica"}:
        replication_observed = (
            str(replication.get("master_link_status") or "").lower() == "up"
        )
    else:
        replication_observed = False

    return {
        "schemaVersion": 1,
        "plane": plane,
        "observedAt": _now(),
        "transport": policy.manifest(),
        "reachable": True,
        "aclIdentityObserved": acl_identity_observed,
        "nonDefaultIdentity": non_default_identity,
        "aofPersistenceObserved": aof_enabled,
        "loading": loading,
        "replicationObserved": replication_observed,
        "note": (
            "Observations only; least-privilege ACL review, provider durability, backups and restore tests require separate operator evidence."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plane", required=True, choices=("rateLimit", "share", "contribution")
    )
    parser.add_argument(
        "--url-env",
        required=True,
        help="Environment-variable name containing the Redis URL",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = collect(plane=args.plane, url_env=args.url_env)
    except RuntimeError as exc:
        logger.warning(
            json.dumps(
                {
                    "ok": False,
                    "code": str(exc),
                },
                sort_keys=True,
            ),
        )
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
