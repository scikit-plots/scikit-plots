"""
Custom provider: an operator's own storage adapter, named in configuration.

Use it when neither the bundled providers nor a webhook fit: a database the
organization already runs, a message queue, an internal API client. The
adapter lives in the operator's code; the service still validates every
request, builds the event, enforces idempotency checks on the receipt and
coordinates mirrors.

Notes
-----
**Contract.** ``factory`` is ``"package.module:callable"``. At service start
the callable is imported and called once::

    adapter = factory(target=StorageTarget, env=Mapping[str, str])

``target.options`` carries the target's non-secret ``options`` object;
``env`` is the service environment, so the adapter reads its credential from
the names in ``target.token_env``. The adapter provides ``put``, plain or
``async``::

    def put(self, *, event: dict, request_hash: str) -> dict: ...

``put`` returns ``{"status": "accepted" | "replay", "provider": "custom",
"feedback_id": event["feedback"]["id"], "request_hash": request_hash}``,
raises :class:`FeedbackConflictError` when the feedback_id is stored with
different content, and must be idempotent (the browser retries with the same
feedback_id). A plain ``put`` runs in a worker thread.

**Security.** The factory is named by the operator in server configuration,
like an ASGI application path; it is never taken from a request. Import
failure stops the service at start rather than at the first submission.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from ._config import FeedbackServiceConfigError, StorageTarget


def load_custom_adapter(target: StorageTarget, env: Mapping[str, str]) -> Any:
    """
    Import ``target.factory`` and build the adapter.

    Parameters
    ----------
    target : StorageTarget
        A ``custom`` target.
    env : Mapping[str, str]
        Service environment handed to the factory.

    Returns
    -------
    object
        An object with a callable ``put``.

    Raises
    ------
    FeedbackServiceConfigError
        If the module or callable cannot be imported, the factory fails, or
        the result has no callable ``put``.
    """
    module_name, _, attribute = target.factory.partition(":")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise FeedbackServiceConfigError(
            f"custom storage target {target.id!r}: cannot import {target.factory!r}"
        ) from exc
    try:
        adapter = factory(target=target, env=env)
    except Exception as exc:  # ruff: ignore[blind-except]
        raise FeedbackServiceConfigError(
            f"custom storage target {target.id!r}: factory {target.factory!r} failed"
        ) from exc
    if not callable(getattr(adapter, "put", None)):
        raise FeedbackServiceConfigError(
            f"custom storage target {target.id!r}: adapter has no callable put()"
        )
    return adapter


__all__ = ["load_custom_adapter"]
