"""
Signed webhook delivery: hand each durable event to any HTTP receiver.

The webhook provider makes the service usable with storage it does not ship an
adapter for: a Cloudflare Worker writing to D1/KV/R2, a GitLab or Bitbucket
pipeline trigger, an intranet ticketing system, a queue gateway. The receiver
owns storage; this adapter owns signing, bounded transport and receipt checks.

Notes
-----
**Delivery contract** (``page.feedback-delivery.v1``).

Request::

    POST <url>
    Content-Type: application/json
    User-Agent: sphinx-feedback/<version>
    X-Feedback-Delivery: <feedback_id>
    X-Feedback-Signature-256: sha256=<hex HMAC-SHA256(secret, body)>

    {"contract":"page.feedback-delivery.v1","event":{...},
     "feedback_id":"feedback-...","request_hash":"<64 hex>"}

The body is canonical JSON (sorted keys, no spaces, UTF-8), so a receiver
verifies the signature over the exact bytes it received and never needs to
re-serialize. The secret is read from ``token_env`` and is never sent.

Response the adapter accepts::

    2xx  {"status": "accepted" | "replay",
          "feedback_id": <same>, "request_hash": <same>}
    409  the feedback_id is already stored with different content

Anything else (other statuses, a receipt for another event, a body over
256 KiB, a redirect) is a failure; the browser keeps the request pending and
retries with the same feedback_id, so a receiver **must** be idempotent:
store once per feedback_id, answer ``replay`` when the stored request_hash is
equal, ``409`` when it differs.

**Developer notes.** Redirects are never followed (a redirect could move the
signed body to another host). Errors carry stable codes and no response text,
so nothing a receiver returns reaches the public route.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from .. import __version__
from .._contracts import (
    FeedbackConflictError,
    feedback_event_request_hash,
    parse_feedback_event,
    validate_request_hash,
)
from ._config import StorageTarget
from ._github import ProviderWriteError, _bounded_json

DELIVERY_CONTRACT = "page.feedback-delivery.v1"
SIGNATURE_HEADER = "X-Feedback-Signature-256"
DELIVERY_HEADER = "X-Feedback-Delivery"


def delivery_body(event: dict[str, Any], request_hash: str) -> bytes:
    """
    Return the canonical bytes POSTed to a webhook receiver.

    Parameters
    ----------
    event : dict
        A durable feedback event (validated by the caller).
    request_hash : str
        The event's request commitment.

    Returns
    -------
    bytes
        Sorted-key, compact, UTF-8 JSON of the delivery envelope.
    """
    envelope = {
        "contract": DELIVERY_CONTRACT,
        "event": event,
        "feedback_id": event["feedback"]["id"],
        "request_hash": request_hash,
    }
    return json.dumps(
        envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sign(secret: str, body: bytes) -> str:
    """
    Return the ``X-Feedback-Signature-256`` value for *body*.

    Parameters
    ----------
    secret : str
        The shared secret from ``token_env``.
    body : bytes
        Exact request bytes.

    Returns
    -------
    str
        ``"sha256=" + hex HMAC-SHA256``.

    Examples
    --------
    >>> sign("s", b"{}")[:7]
    'sha256='
    """
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return "sha256=" + digest


def verify(secret: str, body: bytes, signature: str) -> bool:
    """
    Check a received signature in constant time (for Python receivers).

    Parameters
    ----------
    secret : str
        The shared secret.
    body : bytes
        Exact bytes received.
    signature : str
        The ``X-Feedback-Signature-256`` header value.

    Returns
    -------
    bool
        Whether the signature matches.
    """
    return hmac.compare_digest(sign(secret, body), str(signature or ""))


async def submit_webhook(
    *,
    client,
    target: StorageTarget,
    secret: str,
    event: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    """
    Deliver one event to ``target.url`` and return a verified receipt.

    Parameters
    ----------
    client : httpx.AsyncClient
        Transport; redirects are not followed.
    target : StorageTarget
        A ``webhook`` target.
    secret : str
        Signing secret resolved from ``target.token_env``.
    event : dict
        Durable feedback event.
    request_hash : str
        Its request commitment.

    Returns
    -------
    dict
        ``{"status", "provider": "webhook", "feedback_id", "request_hash"}``.

    Raises
    ------
    ProviderWriteError
        With codes ``credential_missing``, ``request_commitment_mismatch``,
        ``webhook_rejected`` (non-2xx) or ``webhook_receipt_invalid``.
    FeedbackConflictError
        When the receiver answers 409.
    """
    if not secret:
        raise ProviderWriteError(
            "credential_missing", "Webhook signing secret is not configured"
        )
    event = parse_feedback_event(event)
    validate_request_hash(request_hash)
    if request_hash != feedback_event_request_hash(event):
        raise ProviderWriteError(
            "request_commitment_mismatch",
            "Webhook request commitment does not match the durable event",
        )
    feedback_id = event["feedback"]["id"]
    body = delivery_body(event, request_hash)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"sphinx-feedback/{__version__}",
        DELIVERY_HEADER: feedback_id,
        SIGNATURE_HEADER: sign(secret, body),
    }
    status, payload = await _bounded_json(
        client, "POST", target.url, headers=headers, content=body, label="Webhook"
    )
    if status == 409:  # ruff: ignore[magic-value-comparison]
        raise FeedbackConflictError(
            "feedback_id was already used for different feedback content"
        )
    if not 200 <= status < 300:  # ruff: ignore[magic-value-comparison]
        raise ProviderWriteError(
            "webhook_rejected", f"Webhook receiver answered HTTP {int(status)}"
        )
    if (
        not isinstance(payload, dict)
        or payload.get("status") not in {"accepted", "replay"}
        or payload.get("feedback_id") != feedback_id
        or payload.get("request_hash") != request_hash
    ):
        raise ProviderWriteError(
            "webhook_receipt_invalid",
            "Webhook receiver returned a receipt for another or no event",
        )
    return {
        "status": payload["status"],
        "provider": "webhook",
        "feedback_id": feedback_id,
        "request_hash": request_hash,
    }


__all__ = [
    "DELIVERY_CONTRACT",
    "DELIVERY_HEADER",
    "SIGNATURE_HEADER",
    "delivery_body",
    "sign",
    "submit_webhook",
    "verify",
]
