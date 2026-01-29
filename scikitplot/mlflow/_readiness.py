# scikitplot/mlflow/_readiness.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_readiness.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from ._server import SpawnedServer


def _build_url(base: str, path: str) -> str:
    """
    Build a REST URL from a base tracking URI.

    Parameters
    ----------
    base : str
        Tracking base URI, e.g. "http://127.0.0.1:5000".
    path : str
        REST path beginning with "/".

    Returns
    -------
    str
        Combined URL.

    Raises
    ------
    ValueError
        If base is not an HTTP(S) URI.
    """
    u = urlparse(base)
    if u.scheme not in {"http", "https"}:
        raise ValueError(f"tracking_uri must be http(s), got {base!r}.")
    return base.rstrip("/") + path


def wait_tracking_ready(
    tracking_uri: str,
    timeout_s: float,
    *,
    server: SpawnedServer | None = None,
) -> None:
    """
    Wait until the MLflow tracking REST API responds.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI, e.g., "http://127.0.0.1:5000".
    timeout_s : float
        Maximum seconds to wait.
    server : SpawnedServer or None, default=None
        If provided, and the server process exits while waiting, raise immediately
        with captured output.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `tracking_uri` is not an HTTP(S) URI.
    RuntimeError
        If the server exits before becoming ready.
    TimeoutError
        If readiness is not reached within `timeout_s`.

    Notes
    -----
    MLflow has evolved REST endpoints over time and can be deployed behind different
    server stacks. This function uses a deterministic two-step check:

    1) `POST /api/2.0/mlflow/experiments/search` (stable modern endpoint)
    2) If the server responds with 404/405, fallback to `GET /api/2.0/mlflow/experiments/list`

    Any 200 response from either endpoint is considered "ready".
    """
    search_url = _build_url(tracking_uri, "/api/2.0/mlflow/experiments/search")
    list_url = _build_url(tracking_uri, "/api/2.0/mlflow/experiments/list")

    payload = json.dumps({"max_results": 1}).encode("utf-8")
    req_search = urllib.request.Request(  # noqa: S310
        url=search_url,
        method="POST",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    req_list = urllib.request.Request(  # noqa: S310
        url=list_url,
        method="GET",
    )

    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None

    while time.monotonic() < deadline:
        if server is not None and server.process.poll() is not None:
            out = server.read_all_output()
            raise RuntimeError(
                "MLflow server exited before becoming ready. "
                f"returncode={server.process.returncode}. Command: {server.command!r}. Output:\n{out}"
            )

        try:
            with urllib.request.urlopen(req_search, timeout=2) as resp:  # noqa: S310
                if getattr(resp, "status", None) == 200:  # noqa: PLR2004
                    return
                last_err = RuntimeError(
                    f"MLflow not ready: HTTP {getattr(resp, 'status', None)!r} (search)"
                )
        except urllib.error.HTTPError as e:
            # Deterministic fallback when endpoint is not available on this server/version.
            if int(getattr(e, "code", 0)) in {404, 405}:
                try:
                    with urllib.request.urlopen(  # noqa: S310
                        req_list,
                        timeout=2,
                    ) as resp:
                        if getattr(resp, "status", None) == 200:  # noqa: PLR2004
                            return
                        last_err = RuntimeError(
                            f"MLflow not ready: HTTP {getattr(resp, 'status', None)!r} (list)"
                        )
                except Exception as e2:
                    last_err = e2
            else:
                last_err = e
        except Exception as e:
            last_err = e

        time.sleep(0.2)

    raise TimeoutError(
        f"MLflow tracking server not ready within {timeout_s:.1f}s. Last error: {last_err!r}"
    )
