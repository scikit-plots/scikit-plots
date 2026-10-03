"""
GitHub provider-review adapter for immutable feedback JSON events.

The adapter is deliberately retry-oriented. A network interruption may happen after
GitHub has created a branch, committed the immutable event, or opened a pull
request. Replaying the same feedback ID verifies each durable step and resumes at
the first missing one instead of treating a partially completed submission as a
permanent conflict.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import quote, urlsplit

from .. import __version__
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
from ._config import StorageTarget

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
    params: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """Perform one bounded GitHub API request even with an unbounded shared client."""
    response = None
    try:
        request = client.build_request(
            method,
            url,
            headers=headers,
            json=json_body,
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
                        "GitHub response exceeded the safety limit",
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
                    "GitHub response exceeded the response-frame safety limit",
                )
            total += len(chunk)
            if total > _MAX_RESPONSE:
                raise ProviderWriteError(
                    "provider_response_too_large",
                    "GitHub response exceeded the safety limit",
                )
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw:
            payload: Any = {}
        else:
            try:
                payload = json.loads(raw)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
                RecursionError,
            ) as exc:
                raise ProviderWriteError(
                    "provider_response_invalid",
                    "GitHub returned invalid JSON",
                ) from exc
        return int(response.status_code), payload
    except ProviderWriteError:
        raise
    except Exception as exc:
        # Do not leak low-level network details/tokens to the public route. The
        # exception remains chained for server-side diagnostics.
        raise ProviderWriteError(
            "provider_transport_failed",
            "GitHub feedback transport failed",
        ) from exc
    finally:
        if response is not None:
            try:  # ruff: ignore[suppressible-exception]
                await response.aclose()
            except Exception:  # ruff: ignore[blind-except]
                pass


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"sphinx-feedback/{__version__}",
    }


def _event_path(target: StorageTarget, event: dict[str, Any]) -> str:
    bucket = page_digest(event["site_id"], event["page_id"])
    return f"{target.feedback_path}/pages/{bucket}/{event['feedback']['id']}.json"


def _review_page_text(page_id: str, *, maximum: int | None = None) -> str:
    """Return a provider-display-safe page label with no Markdown mention syntax."""
    text = quote(page_id, safe="/._-")
    if maximum is not None:
        text = text[:maximum]
    return text


def _branch(event: dict[str, Any]) -> str:
    # Use the full 192-bit event nonce. Branch identity is event provenance, not
    # participant identity, and full entropy avoids unnecessary prefix collisions.
    return "feedback/" + event["feedback"]["id"].removeprefix("feedback-")


def _object_sha(payload: Any, *, code: str) -> str:
    try:
        sha = str(payload["object"]["sha"])
    except Exception as exc:
        raise ProviderWriteError(
            code,
            "GitHub branch response was incomplete",
        ) from exc
    _len = len(sha) != 40  # ruff: ignore[magic-value-comparison]
    if _len or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
        raise ProviderWriteError(
            code,
            "GitHub branch response contained an invalid object id",
        )
    return sha


def _review_url(value: Any, *, repo: str | None = None) -> str:
    text = str(value or "").strip()
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ProviderWriteError(
            "review_lookup_invalid", "GitHub feedback review URL was invalid"
        ) from exc
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "github.com"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port not in (None, 443)
    ):
        raise ProviderWriteError(
            "review_lookup_invalid", "GitHub feedback review URL was invalid"
        )
    if repo is not None:
        prefix = f"/{repo}/pull/"
        number = parsed.path.removeprefix(prefix)
        if not parsed.path.startswith(prefix) or not number.isdigit():
            raise ProviderWriteError(
                "review_lookup_invalid",
                "GitHub feedback review URL did not belong to the configured repository",
            )
    return text


async def _existing_review_url(
    client,
    *,
    api: str,
    headers: dict[str, str],
    repo: str,
    branch: str,
    base_branch: str,
) -> str | None:
    owner = repo.split("/", 1)[0]
    status, pulls = await _bounded_json(
        client,
        "GET",
        f"{api}/pulls",
        headers=headers,
        params={"state": "all", "head": f"{owner}:{branch}", "base": base_branch},
    )
    if status != 200:  # ruff: ignore[magic-value-comparison]
        raise ProviderWriteError(
            "review_lookup_failed",
            "Existing GitHub feedback review could not be verified",
        )
    if not isinstance(pulls, list):
        raise ProviderWriteError(
            "review_lookup_invalid",
            "GitHub feedback review lookup returned an invalid response",
        )
    if not pulls:
        return None
    first = pulls[0]
    if not isinstance(first, dict):
        raise ProviderWriteError(
            "review_lookup_invalid",
            "GitHub feedback review lookup returned an invalid entry",
        )
    return _review_url(first.get("html_url"), repo=repo)


async def _branch_comparison(
    client,
    *,
    api: str,
    headers: dict[str, str],
    base_branch: str,
    branch: str,
) -> dict[str, Any]:
    status, comparison = await _bounded_json(
        client,
        "GET",
        f"{api}/compare/{quote(base_branch, safe='')}...{quote(branch, safe='')}",
        headers=headers,
    )
    _status = status != 200  # ruff: ignore[magic-value-comparison]
    if _status or not isinstance(comparison, dict):
        raise ProviderWriteError(
            "existing_branch_unverifiable",
            "Existing GitHub feedback branch could not be compared safely",
        )
    ahead_by = comparison.get("ahead_by")
    if isinstance(ahead_by, bool) or not isinstance(ahead_by, int) or ahead_by < 0:
        raise ProviderWriteError(
            "existing_branch_unverifiable",
            "GitHub branch comparison was incomplete",
        )
    return comparison


async def _branch_has_no_unique_commits(
    client,
    *,
    api: str,
    headers: dict[str, str],
    base_branch: str,
    branch: str,
) -> bool:
    comparison = await _branch_comparison(
        client,
        api=api,
        headers=headers,
        base_branch=base_branch,
        branch=branch,
    )
    return comparison["ahead_by"] == 0


async def _verify_review_branch_scope(
    client,
    *,
    api: str,
    headers: dict[str, str],
    base_branch: str,
    branch: str,
    path: str,
    event: dict[str, Any],
    review_exists: bool,
) -> None:
    """
    Require the review branch to contain exactly one immutable feedback file.

    A deterministic branch name is resumable, but it is also a security boundary:
    an existing branch must never be allowed to smuggle unrelated repository changes
    into the pull request merely because the expected feedback JSON is present.
    """
    existing_bytes = await _existing_event_bytes(
        client, api=api, headers=headers, path=path, branch=branch
    )
    if not _event_content_matches(existing_bytes, event):
        if existing_bytes is None:
            raise FeedbackConflictError(
                "feedback review branch is missing the expected immutable event"
            )
        raise FeedbackConflictError(
            "feedback_id was already used for different feedback content"
        )

    comparison = await _branch_comparison(
        client,
        api=api,
        headers=headers,
        base_branch=base_branch,
        branch=branch,
    )
    if comparison["ahead_by"] == 0:
        # After a reviewed pull request is merged, the deterministic branch can
        # compare equal to the base. Preserve idempotent replay only when a review
        # already exists and the immutable event is now present on the base branch.
        if not review_exists:
            raise FeedbackConflictError(
                "feedback review branch contains an unexpected commit history"
            )
        base_bytes = await _existing_event_bytes(
            client, api=api, headers=headers, path=path, branch=base_branch
        )
        if not _event_content_matches(base_bytes, event):
            raise FeedbackConflictError(
                "feedback review branch no longer proves the expected reviewed event"
            )
        return
    if comparison["ahead_by"] != 1:
        raise FeedbackConflictError(
            "feedback review branch contains an unexpected commit history"
        )
    files = comparison.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise FeedbackConflictError(
            "feedback review branch contains unexpected repository changes"
        )
    changed = files[0]
    if changed.get("filename") != path or changed.get("status") != "added":
        raise FeedbackConflictError(
            "feedback review branch contains unexpected repository changes"
        )


async def _branch_exists(
    client, *, api: str, headers: dict[str, str], branch: str
) -> bool:
    status, payload = await _bounded_json(
        client,
        "GET",
        f"{api}/git/ref/heads/{quote(branch, safe='')}",
        headers=headers,
    )
    if status == 404:  # ruff: ignore[magic-value-comparison]
        return False
    if status != 200:  # ruff: ignore[magic-value-comparison]
        raise ProviderWriteError(
            "existing_branch_unverifiable",
            "Existing GitHub feedback branch could not be verified",
        )
    _object_sha(payload, code="existing_branch_invalid")
    return True


async def _existing_event_bytes(
    client,
    *,
    api: str,
    headers: dict[str, str],
    path: str,
    branch: str,
) -> bytes | None:
    content_status, existing = await _bounded_json(
        client,
        "GET",
        f"{api}/contents/{quote(path, safe='/')}",
        headers=headers,
        params={"ref": branch},
    )
    if content_status == 404:  # ruff: ignore[magic-value-comparison]
        return None
    _content_status = content_status != 200  # ruff: ignore[magic-value-comparison]
    if _content_status or not isinstance(existing, dict):
        raise ProviderWriteError(
            "existing_branch_unverifiable",
            "Existing GitHub feedback event could not be verified",
        )
    try:
        return base64.b64decode(
            str(existing.get("content", "")).replace("\n", ""), validate=True
        )
    except Exception as exc:
        raise ProviderWriteError(
            "existing_content_invalid",
            "Existing GitHub feedback event could not be verified",
        ) from exc


def _event_content_matches(existing_bytes: bytes | None, event: dict[str, Any]) -> bool:
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


async def submit_github_review(  # ruff: ignore[too-many-branches]
    *,
    client,
    target: StorageTarget,
    token: str,
    event: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    """Idempotently ensure branch, immutable event, and provider review exist."""
    if not token:
        raise ProviderWriteError(
            "credential_missing", "GitHub feedback credential is not configured"
        )
    event = parse_feedback_event(event)
    validate_request_hash(request_hash)
    if request_hash != feedback_event_request_hash(event):
        raise ProviderWriteError(
            "request_commitment_mismatch",
            "GitHub feedback request commitment does not match the durable event",
        )
    repo = target.repo
    api = "https://api.github.com/repos/" + repo
    headers = _headers(token)
    branch = _branch(event)
    if branch == target.branch:
        raise ProviderWriteError(
            "branch_namespace_conflict",
            "GitHub feedback review branch collides with the configured base branch",
        )
    path = _event_path(target, event)
    # Repository files are intentionally pretty-printed for human review.
    # Idempotency/equality remains semantic so pre-v0.6.1 compact files and
    # harmless formatting changes do not become false content conflicts.
    event_bytes = repository_event_bytes(event)

    status, base_ref = await _bounded_json(
        client,
        "GET",
        f"{api}/git/ref/heads/{quote(target.branch, safe='')}",
        headers=headers,
    )
    if status != 200:  # ruff: ignore[magic-value-comparison]
        raise ProviderWriteError(
            "base_ref_unavailable", "GitHub feedback base branch could not be resolved"
        )
    base_sha = _object_sha(base_ref, code="base_ref_invalid")

    branch_created = False
    try:
        status, _created = await _bounded_json(
            client,
            "POST",
            f"{api}/git/refs",
            headers=headers,
            json_body={"ref": "refs/heads/" + branch, "sha": base_sha},
        )
    except ProviderWriteError as exc:
        if exc.code != "provider_transport_failed":
            raise
        # Broken pipe after GitHub accepted branch creation: reconcile before
        # asking the browser to retry the whole workflow.
        try:
            if await _branch_exists(client, api=api, headers=headers, branch=branch):
                status = 422
            else:
                raise exc  # ruff: ignore[verbose-raise]
        except ProviderWriteError:
            raise exc  # ruff: ignore[raise-without-from-inside-except]
    if status == 201:  # ruff: ignore[magic-value-comparison]
        branch_created = True
    elif status == 422:  # ruff: ignore[magic-value-comparison]
        if not await _branch_exists(client, api=api, headers=headers, branch=branch):
            raise ProviderWriteError(
                "branch_create_failed",
                "GitHub rejected the feedback branch and no resumable branch exists",
            )
    else:
        raise ProviderWriteError(
            "branch_create_failed", "GitHub feedback review branch could not be created"
        )

    event_present = False
    if not branch_created:
        existing_bytes = await _existing_event_bytes(
            client, api=api, headers=headers, path=path, branch=branch
        )
        if existing_bytes is not None:
            if not _event_content_matches(existing_bytes, event):
                raise FeedbackConflictError(
                    "feedback_id was already used for different feedback content"
                )
            event_present = True
        elif not await _branch_has_no_unique_commits(
            client,
            api=api,
            headers=headers,
            base_branch=target.branch,
            branch=branch,
        ):
            raise FeedbackConflictError(
                "feedback review branch exists without the expected immutable event"
            )

    if not event_present:
        try:
            status, _content = await _bounded_json(
                client,
                "PUT",
                f"{api}/contents/{quote(path, safe='/')}",
                headers=headers,
                json_body={
                    "message": f"Add page feedback {event['feedback']['id']}",
                    "content": base64.b64encode(event_bytes).decode("ascii"),
                    "branch": branch,
                },
            )
        except ProviderWriteError as exc:
            if exc.code != "provider_transport_failed":
                raise
            # The commit may have succeeded before the connection failed. Verify
            # the immutable bytes once; otherwise preserve the ambiguous error so
            # a later retry can safely resume the same deterministic branch.
            try:
                existing_bytes = await _existing_event_bytes(
                    client, api=api, headers=headers, path=path, branch=branch
                )
            except ProviderWriteError:
                raise exc  # ruff: ignore[raise-without-from-inside-except]
            if _event_content_matches(existing_bytes, event):
                status = 201
            else:
                raise exc  # ruff: ignore[verbose-raise]
        if status not in {200, 201}:
            # Concurrent same-event submissions can race between the existence
            # check and PUT. Re-read immutable content before treating an
            # explicit provider rejection as failure.
            existing_bytes = await _existing_event_bytes(
                client, api=api, headers=headers, path=path, branch=branch
            )
            if _event_content_matches(existing_bytes, event):
                status = 201
            elif existing_bytes is not None:
                raise FeedbackConflictError(
                    "feedback_id was already used for different feedback content"
                )
            else:
                raise ProviderWriteError(
                    "content_write_failed",
                    "GitHub feedback event could not be committed",
                )

    existing_review = await _existing_review_url(
        client,
        api=api,
        headers=headers,
        repo=repo,
        branch=branch,
        base_branch=target.branch,
    )
    await _verify_review_branch_scope(
        client,
        api=api,
        headers=headers,
        base_branch=target.branch,
        branch=branch,
        path=path,
        event=event,
        review_exists=existing_review is not None,
    )
    if existing_review is not None:
        return {
            "status": "replay",
            "provider": "github",
            "feedback_id": event["feedback"]["id"],
            "review_url": existing_review if target.expose_links else "",
            "request_hash": request_hash,
        }

    rating = int(event["feedback"]["rating"])
    signed = f"{rating:+d}" if rating else "0"
    title_page = _review_page_text(event["page_id"], maximum=100)
    body_page = _review_page_text(event["page_id"])
    body = (
        "Automated page-feedback review.\n\n"
        f"Request commitment: `{request_hash}`\n\n"
        f"Page: `{body_page}`\n\n"
        "The reviewed JSON contains no implicit browser, device, account, network, "
        "referrer, or telemetry fields."
    )
    try:
        status, pull = await _bounded_json(
            client,
            "POST",
            f"{api}/pulls",
            headers=headers,
            json_body={
                "title": f"[Feedback] {signed} · {title_page}",
                "head": branch,
                "base": target.branch,
                "body": body,
            },
        )
    except ProviderWriteError as exc:
        if exc.code != "provider_transport_failed":
            raise
        try:
            existing_review = await _existing_review_url(
                client,
                api=api,
                headers=headers,
                repo=repo,
                branch=branch,
                base_branch=target.branch,
            )
        except ProviderWriteError:
            raise exc  # ruff: ignore[raise-without-from-inside-except]
        if existing_review is None:
            raise exc  # ruff: ignore[verbose-raise]
        await _verify_review_branch_scope(
            client,
            api=api,
            headers=headers,
            base_branch=target.branch,
            branch=branch,
            path=path,
            event=event,
            review_exists=True,
        )
        return {
            "status": "replay",
            "provider": "github",
            "feedback_id": event["feedback"]["id"],
            "review_url": existing_review if target.expose_links else "",
            "request_hash": request_hash,
        }
    if status not in {200, 201}:
        # A provider race/non-success may still have opened the review. Reconcile
        # once for every non-success status, not only the common 422 case.
        existing_review = await _existing_review_url(
            client,
            api=api,
            headers=headers,
            repo=repo,
            branch=branch,
            base_branch=target.branch,
        )
        if existing_review is not None:
            await _verify_review_branch_scope(
                client,
                api=api,
                headers=headers,
                base_branch=target.branch,
                branch=branch,
                path=path,
                event=event,
                review_exists=True,
            )
            return {
                "status": "replay",
                "provider": "github",
                "feedback_id": event["feedback"]["id"],
                "review_url": existing_review if target.expose_links else "",
                "request_hash": request_hash,
            }
        raise ProviderWriteError(
            "review_open_failed", "GitHub feedback pull request could not be opened"
        )
    if not isinstance(pull, dict):
        raise ProviderWriteError(
            "review_open_invalid", "GitHub feedback pull request response was invalid"
        )
    review_url = _review_url(pull.get("html_url"), repo=repo)
    await _verify_review_branch_scope(
        client,
        api=api,
        headers=headers,
        base_branch=target.branch,
        branch=branch,
        path=path,
        event=event,
        review_exists=True,
    )
    return {
        "status": "accepted",
        "provider": "github",
        "feedback_id": event["feedback"]["id"],
        "review_url": review_url if target.expose_links else "",
        "request_hash": request_hash,
    }
