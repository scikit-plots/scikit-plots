"""
Provider-neutral reviewed-feedback engine shared by GitHub, GitLab and Bitbucket.

Every forge provider runs one state machine: resolve the base branch, ensure a
deterministic ``feedback/<nonce>`` branch, commit exactly one immutable event
file, open (or find) one review, and prove the branch still carries only that
file. A broken connection can interrupt any step after the forge has acted, so
each step is reconciled from what the forge reports before a retry is asked
of the browser. The engine owns that logic once; a forge supplies eight small
operations (:class:`ForgeOps`) that speak its HTTP API.

Notes
-----
**Guarantees** (identical for every forge):

* the same feedback_id with the same event replays; with a different event
  it conflicts; nothing is ever overwritten;
* a review is reported only after the branch is shown to add exactly one
  file, the expected event, in exactly one commit (or, after a merge, the
  event is on the base branch and a review exists);
* the review branch may never equal the base branch;
* responses are bounded (256 KiB, 1024 frames), redirects are refused, and
  errors carry stable codes without provider text or credentials.

**Developer notes.** The engine was extracted from the GitHub adapter
unchanged in behaviour; the GitHub test-suite is its equivalence oracle. A
forge operation returns ``False``/``None`` for an explicit refusal so the
engine can reconcile, and raises :class:`ProviderWriteError` for an
ambiguous transport failure (code ``provider_transport_failed``) or a
response it cannot trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from .._contracts import (
    FeedbackConflictError,
    FeedbackValidationError,
    canonical_event_bytes,
    decode_feedback_event,
    feedback_event_request_hash,
    page_digest,
    parse_feedback_event,
    repository_event_bytes,
    validate_request_hash,
)

_MAX_RESPONSE = 256 * 1024
_MAX_RESPONSE_CHUNKS = 1024
_REQUEST_TIMEOUT_SECONDS = 20.0


class ProviderWriteError(RuntimeError):
    """Bounded provider failure safe to surface as a stable service code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def _bounded_json(  # ruff: ignore[too-many-branches]
    client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: Any = None,
    params: Any = None,
    content: bytes | None = None,
    data: dict[str, str] | None = None,
    raw: bool = False,
    label: str = "GitHub",
) -> tuple[int, Any]:
    """
    Perform one bounded provider request even with an unbounded shared client.

    Parameters
    ----------
    client : httpx.AsyncClient
        Shared transport; redirects are never followed.
    method, url : str
        Request line.
    headers : dict
        Request headers (credentials included by the caller).
    json_body : object, optional
        Serialized by the client when *content* and *data* are not given.
    params : mapping or sequence of pairs, optional
        Query parameters; a sequence allows repeated names.
    content : bytes, optional
        Exact body bytes (the webhook adapter signs these).
    data : dict, optional
        Form fields (``application/x-www-form-urlencoded``).
    raw : bool, default False
        Return the body bytes instead of parsed JSON.
    label : str, default "GitHub"
        Provider name for the stable, token-free error messages.

    Returns
    -------
    tuple of (int, object)
        Status code and parsed JSON (``{}`` for an empty body) or bytes.

    Raises
    ------
    ProviderWriteError
        ``provider_response_too_large``, ``provider_response_too_fragmented``,
        ``provider_response_invalid`` or ``provider_transport_failed``.
    """
    response = None
    try:
        request = client.build_request(
            method,
            url,
            headers=headers,
            json=json_body if content is None and data is None else None,
            content=content,
            data=data,
            params=params,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response = await client.send(request, stream=True, follow_redirects=False)
        length = response.headers.get("content-length")
        if length:
            try:
                if int(length) > _MAX_RESPONSE:
                    raise ProviderWriteError(
                        "provider_response_too_large",
                        f"{label} response exceeded the safety limit",
                    )
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        chunk_count = 0
        async for chunk in response.aiter_bytes():
            chunk_count += 1
            if chunk_count > _MAX_RESPONSE_CHUNKS:
                raise ProviderWriteError(
                    "provider_response_too_fragmented",
                    f"{label} response exceeded the response-frame safety limit",
                )
            total += len(chunk)
            if total > _MAX_RESPONSE:
                raise ProviderWriteError(
                    "provider_response_too_large",
                    f"{label} response exceeded the safety limit",
                )
            chunks.append(chunk)
        body = b"".join(chunks)
        if raw:
            return int(response.status_code), body
        if not body:
            payload: Any = {}
        else:
            try:
                payload = json.loads(body)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
                RecursionError,
            ) as exc:
                raise ProviderWriteError(
                    "provider_response_invalid",
                    f"{label} returned invalid JSON",
                ) from exc
        return int(response.status_code), payload
    except ProviderWriteError:
        raise
    except Exception as exc:
        # Do not leak low-level network details/tokens to the public route. The
        # exception remains chained for server-side diagnostics.
        raise ProviderWriteError(
            "provider_transport_failed",
            f"{label} feedback transport failed",
        ) from exc
    finally:
        if response is not None:
            try:  # ruff: ignore[suppressible-exception]
                await response.aclose()
            except Exception:  # ruff: ignore[blind-except]
                pass


def event_path(feedback_path: str, event: dict[str, Any]) -> str:
    """Return ``<feedback_path>/pages/<page digest>/<feedback_id>.json``."""
    bucket = page_digest(event["site_id"], event["page_id"])
    return f"{feedback_path}/pages/{bucket}/{event['feedback']['id']}.json"


def review_branch(event: dict[str, Any]) -> str:
    """Return the deterministic review branch ``feedback/<48 hex>``."""
    # Use the full 192-bit event nonce. Branch identity is event provenance, not
    # participant identity, and full entropy avoids unnecessary prefix collisions.
    return "feedback/" + event["feedback"]["id"].removeprefix("feedback-")


def review_page_text(page_id: str, *, maximum: int | None = None) -> str:
    """Return a provider-display-safe page label with no Markdown mention syntax."""
    text = quote(page_id, safe="/._-")
    if maximum is not None:
        text = text[:maximum]
    return text


def event_content_matches(existing_bytes: bytes | None, event: dict[str, Any]) -> bool:
    """Compare repository JSON by validated event semantics, never whitespace."""
    if existing_bytes is None:
        return False
    try:
        existing_event = decode_feedback_event(existing_bytes)
        expected_event = parse_feedback_event(event)
    except FeedbackValidationError:
        return False
    return canonical_event_bytes(existing_event) == canonical_event_bytes(
        expected_event,
    )


@dataclass(frozen=True)
class Comparison:
    """
    What a review branch adds over the base branch (merge-base semantics).

    Attributes
    ----------
    ahead_by : int
        Commits on the branch that the base does not have.
    files : tuple of (str, str)
        ``(path, status)`` per changed file; ``status`` is ``"added"`` for a
        new file and anything else otherwise.
    """

    ahead_by: int
    files: tuple[tuple[str, str], ...]


class ForgeOps:
    """
    The operations a forge must provide; see each method.

    Subclasses set ``provider`` (receipt value) and ``label`` (messages) and
    implement every method. ``ProviderWriteError`` with code
    ``provider_transport_failed`` means "the forge may or may not have acted".
    """

    provider = ""
    label = ""

    async def resolve_base(self) -> str:
        """Return the base branch head commit id (raise if unavailable)."""
        raise NotImplementedError

    async def create_branch(self, branch: str, commit: str) -> bool:
        """Create *branch* at *commit*; ``True`` created, ``False`` refused."""
        raise NotImplementedError

    async def branch_exists(self, branch: str) -> bool:
        """Return whether *branch* exists (raise if it cannot be told)."""
        raise NotImplementedError

    async def read_file(self, path: str, ref: str) -> bytes | None:
        """Return the file at *path* on branch *ref*, or ``None`` if absent."""
        raise NotImplementedError

    async def create_file(
        self, path: str, branch: str, data: bytes, message: str
    ) -> bool:
        """Commit a new file on *branch*; ``True`` done, ``False`` refused."""
        raise NotImplementedError

    async def compare(self, base: str, branch: str) -> Comparison:
        """Return what *branch* adds over *base* (raise if unverifiable)."""
        raise NotImplementedError

    async def find_review(self, branch: str, base: str) -> str | None:
        """Return the URL of a review from *branch* into *base* (any state)."""
        raise NotImplementedError

    async def open_review(
        self, branch: str, base: str, title: str, body: str
    ) -> str | None:
        """Open a review; return its URL, or ``None`` if the forge refused."""
        raise NotImplementedError


async def _has_no_unique_commits(ops: ForgeOps, base: str, branch: str) -> bool:
    return (await ops.compare(base, branch)).ahead_by == 0


async def verify_review_branch_scope(
    ops: ForgeOps,
    *,
    base: str,
    branch: str,
    path: str,
    event: dict[str, Any],
    review_exists: bool,
) -> None:
    """
    Require the review branch to contain exactly one immutable feedback file.

    A deterministic branch name is resumable, but it is also a security boundary:
    an existing branch must never be allowed to smuggle unrelated repository changes
    into the review merely because the expected feedback JSON is present.
    """
    existing_bytes = await ops.read_file(path, branch)
    if not event_content_matches(existing_bytes, event):
        if existing_bytes is None:
            raise FeedbackConflictError(
                "feedback review branch is missing the expected immutable event"
            )
        raise FeedbackConflictError(
            "feedback_id was already used for different feedback content"
        )
    comparison = await ops.compare(base, branch)
    if comparison.ahead_by == 0:
        # After a reviewed change is merged, the deterministic branch can compare
        # equal to the base. Preserve idempotent replay only when a review already
        # exists and the immutable event is now present on the base branch.
        if not review_exists:
            raise FeedbackConflictError(
                "feedback review branch contains an unexpected commit history"
            )
        base_bytes = await ops.read_file(path, base)
        if not event_content_matches(base_bytes, event):
            raise FeedbackConflictError(
                "feedback review branch no longer proves the expected reviewed event"
            )
        return
    if comparison.ahead_by != 1:
        raise FeedbackConflictError(
            "feedback review branch contains an unexpected commit history"
        )
    if comparison.files != ((path, "added"),):
        raise FeedbackConflictError(
            "feedback review branch contains unexpected repository changes"
        )


def _receipt(
    ops: ForgeOps,
    status: str,
    *,
    event: dict,
    url: str,
    request_hash: str,
    expose: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "provider": ops.provider,
        "feedback_id": event["feedback"]["id"],
        "review_url": url if expose else "",
        "request_hash": request_hash,
    }


def review_text(event: dict[str, Any], request_hash: str) -> tuple[str, str]:
    """Return the review title and body (no browser or participant data)."""
    rating = int(event["feedback"]["rating"])
    signed = f"{rating:+d}" if rating else "0"
    title = f"[Feedback] {signed} · {review_page_text(event['page_id'], maximum=100)}"
    body = (
        "Automated page-feedback review.\n\n"
        f"Request commitment: `{request_hash}`\n\n"
        f"Page: `{review_page_text(event['page_id'])}`\n\n"
        "The reviewed JSON contains no implicit browser, device, account, network, "
        "referrer, or telemetry fields."
    )
    return title, body


async def submit_forge_review(  # ruff: ignore[too-many-branches,too-many-statements]
    ops: ForgeOps,
    *,
    base_branch: str,
    feedback_path: str,
    expose_links: bool,
    event: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    """
    Idempotently ensure branch, immutable event and review exist on one forge.

    Parameters
    ----------
    ops : ForgeOps
        The forge's operations, already bound to repository and credential.
    base_branch : str
        Branch the review targets.
    feedback_path : str
        Repository directory for event files.
    expose_links : bool
        Whether the receipt carries the review URL.
    event : dict
        Durable feedback event.
    request_hash : str
        Its request commitment.

    Returns
    -------
    dict
        ``{"status": "accepted" | "replay", "provider", "feedback_id",
        "review_url", "request_hash"}``.

    Raises
    ------
    FeedbackConflictError
        Different content for this feedback_id, or a branch that does not
        prove exactly the expected event.
    ProviderWriteError
        Stable codes: ``request_commitment_mismatch``,
        ``branch_namespace_conflict``, ``base_ref_unavailable``,
        ``branch_create_failed``, ``content_write_failed``,
        ``review_open_failed`` and the forge's own read/transport codes.
    """
    event = parse_feedback_event(event)
    validate_request_hash(request_hash)
    label = ops.label
    if request_hash != feedback_event_request_hash(event):
        raise ProviderWriteError(
            "request_commitment_mismatch",
            f"{label} feedback request commitment does not match the durable event",
        )
    branch = review_branch(event)
    if branch == base_branch:
        raise ProviderWriteError(
            "branch_namespace_conflict",
            f"{label} feedback review branch collides with the configured base branch",
        )
    path = event_path(feedback_path, event)
    # Repository files are pretty-printed for human review; equality is semantic,
    # so older compact files and harmless formatting do not become conflicts.
    event_bytes = repository_event_bytes(event)
    base_commit = await ops.resolve_base()

    branch_created = False
    try:
        branch_created = await ops.create_branch(branch, base_commit)
    except ProviderWriteError as exc:
        if exc.code != "provider_transport_failed":
            raise
        # Broken pipe after the forge accepted branch creation: reconcile before
        # asking the browser to retry the whole workflow.
        try:
            exists = await ops.branch_exists(branch)
        except ProviderWriteError:
            raise exc  # ruff: ignore[raise-without-from-inside-except]
        if not exists:
            raise
    if not branch_created and not await ops.branch_exists(branch):
        raise ProviderWriteError(
            "branch_create_failed",
            f"{label} rejected the feedback branch and no resumable branch exists",
        )

    event_present = False
    if not branch_created:
        existing_bytes = await ops.read_file(path, branch)
        if existing_bytes is not None:
            if not event_content_matches(existing_bytes, event):
                raise FeedbackConflictError(
                    "feedback_id was already used for different feedback content"
                )
            event_present = True
        elif not await _has_no_unique_commits(ops, base_branch, branch):
            raise FeedbackConflictError(
                "feedback review branch exists without the expected immutable event"
            )

    if not event_present:
        message = f"Add page feedback {event['feedback']['id']}"
        try:
            written = await ops.create_file(path, branch, event_bytes, message)
        except ProviderWriteError as exc:
            if exc.code != "provider_transport_failed":
                raise
            # The commit may have succeeded before the connection failed. Verify
            # the immutable bytes once; otherwise preserve the ambiguous error so
            # a later retry can safely resume the same deterministic branch.
            try:
                existing_bytes = await ops.read_file(path, branch)
            except ProviderWriteError:
                raise exc  # ruff: ignore[raise-without-from-inside-except]
            if not event_content_matches(existing_bytes, event):
                raise
            written = True
        if not written:
            # Concurrent same-event submissions can race between the existence
            # check and the write. Re-read before treating a refusal as failure.
            existing_bytes = await ops.read_file(path, branch)
            if existing_bytes is not None and not event_content_matches(
                existing_bytes, event
            ):
                raise FeedbackConflictError(
                    "feedback_id was already used for different feedback content"
                )
            if existing_bytes is None:
                raise ProviderWriteError(
                    "content_write_failed",
                    f"{label} feedback event could not be committed",
                )

    existing_review = await ops.find_review(branch, base_branch)
    await verify_review_branch_scope(
        ops,
        base=base_branch,
        branch=branch,
        path=path,
        event=event,
        review_exists=existing_review is not None,
    )
    if existing_review is not None:
        return _receipt(
            ops,
            "replay",
            event=event,
            url=existing_review,
            request_hash=request_hash,
            expose=expose_links,
        )

    title, body = review_text(event, request_hash)
    try:
        review_url = await ops.open_review(branch, base_branch, title, body)
    except ProviderWriteError as exc:
        if exc.code != "provider_transport_failed":
            raise
        try:
            existing_review = await ops.find_review(branch, base_branch)
        except ProviderWriteError:
            raise exc  # ruff: ignore[raise-without-from-inside-except]
        if existing_review is None:
            raise
        await verify_review_branch_scope(
            ops,
            base=base_branch,
            branch=branch,
            path=path,
            event=event,
            review_exists=True,
        )
        return _receipt(
            ops,
            "replay",
            event=event,
            url=existing_review,
            request_hash=request_hash,
            expose=expose_links,
        )
    if review_url is None:
        # A provider race/non-success may still have opened the review. Reconcile
        # once for every refusal, not only the common "already exists" case.
        existing_review = await ops.find_review(branch, base_branch)
        if existing_review is None:
            raise ProviderWriteError(
                "review_open_failed", f"{label} feedback review could not be opened"
            )
        await verify_review_branch_scope(
            ops,
            base=base_branch,
            branch=branch,
            path=path,
            event=event,
            review_exists=True,
        )
        return _receipt(
            ops,
            "replay",
            event=event,
            url=existing_review,
            request_hash=request_hash,
            expose=expose_links,
        )
    await verify_review_branch_scope(
        ops,
        base=base_branch,
        branch=branch,
        path=path,
        event=event,
        review_exists=True,
    )
    return _receipt(
        ops,
        "accepted",
        event=event,
        url=review_url,
        request_hash=request_hash,
        expose=expose_links,
    )


__all__ = [
    "Comparison",
    "ForgeOps",
    "ProviderWriteError",
    "event_content_matches",
    "event_path",
    "review_branch",
    "review_page_text",
    "review_text",
    "submit_forge_review",
    "verify_review_branch_scope",
]
