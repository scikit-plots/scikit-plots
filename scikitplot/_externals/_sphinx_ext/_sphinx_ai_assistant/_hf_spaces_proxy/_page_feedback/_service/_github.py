"""
GitHub provider-review adapter for immutable feedback JSON events.

github.com and GitHub Enterprise Server (``api_url``). The reviewed-change state
machine (branch, immutable event, pull request, branch-scope proof, and the
reconciliation after a broken connection at every step) lives in
:mod:`._forge`; this module supplies GitHub's eight operations.

Notes
-----
**Developer notes.** The names ``ProviderWriteError``, ``_bounded_json``,
``_event_content_matches``, ``_review_page_text``, ``_branch`` and
``_event_path`` are kept importable here for the webhook and git providers,
the HF proxy mirror and the existing tests.
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import quote, urlsplit

from .. import __version__
from ._config import DEFAULT_GITHUB_API_URL, StorageTarget
from ._forge import (
    Comparison,
    ForgeOps,
    ProviderWriteError,
    _bounded_json,
    event_content_matches,
    event_path,
    review_branch,
    review_page_text,
    submit_forge_review,
)

# Backward-compatible private names (see module notes).
_event_content_matches = event_content_matches
_review_page_text = review_page_text
_branch = review_branch


def _event_path(target: StorageTarget, event: dict[str, Any]) -> str:
    return event_path(target.feedback_path, event)


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"sphinx-feedback/{__version__}",
    }


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


def _web_authority(api_url: str) -> tuple[str, int | None]:
    """
    Return the host and port that review (pull request) URLs must use.

    github.com serves its API from ``api.github.com``; GitHub Enterprise
    Server serves both from one host (``https://ghe.example/api/v3``).
    """
    parsed = urlsplit(api_url or DEFAULT_GITHUB_API_URL)
    host = (parsed.hostname or "").lower()
    if host == "api.github.com":
        return "github.com", None
    return host, parsed.port


def pinned_review_url(
    value: Any,
    *,
    label: str,
    web: tuple[str, int | None],
    path_prefix: str | None,
) -> str:
    """
    Return *value* if it is a review URL on the forge's own web host.

    Parameters
    ----------
    value : object
        URL reported by the forge.
    label : str
        Provider name for messages.
    web : tuple of (str, int or None)
        Expected host and port; HTTP only when an explicit port is configured
        (a loopback test server).
    path_prefix : str or None
        Required path prefix; the rest must be a decimal review number.

    Raises
    ------
    ProviderWriteError
        ``review_lookup_invalid``.
    """
    text = str(value or "").strip()
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ProviderWriteError(
            "review_lookup_invalid", f"{label} feedback review URL was invalid"
        ) from exc
    web_host, web_port = web
    if (
        parsed.scheme not in {"https", "http"}
        or (parsed.scheme == "http" and web_port is None)
        or (parsed.hostname or "").lower() != web_host
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port not in ({None, 443} if web_port in (None, 443) else {web_port})
    ):
        raise ProviderWriteError(
            "review_lookup_invalid", f"{label} feedback review URL was invalid"
        )
    if path_prefix is not None:
        number = parsed.path.removeprefix(path_prefix)
        if not parsed.path.startswith(path_prefix) or not number.isdigit():
            raise ProviderWriteError(
                "review_lookup_invalid",
                f"{label} feedback review URL did not belong to the configured repository",
            )
    return text


def _review_url(
    value: Any,
    *,
    repo: str | None = None,
    web: tuple[str, int | None] = ("github.com", None),
) -> str:
    return pinned_review_url(
        value,
        label="GitHub",
        web=web,
        path_prefix=None if repo is None else f"/{repo}/pull/",
    )


class GitHubOps(ForgeOps):
    """GitHub REST v3 operations for :func:`._forge.submit_forge_review`."""

    provider = "github"
    label = "GitHub"

    def __init__(self, *, client, target: StorageTarget, token: str) -> None:
        self.client = client
        self.target = target
        self.repo = target.repo
        self.api = (
            (target.api_url or DEFAULT_GITHUB_API_URL).rstrip("/")
            + "/repos/"
            + self.repo
        )
        self.web = _web_authority(target.api_url)
        self.headers = _headers(token)

    async def _call(self, method: str, url: str, **kwargs) -> tuple[int, Any]:
        return await _bounded_json(
            self.client, method, url, headers=self.headers, label=self.label, **kwargs
        )

    async def resolve_base(self) -> str:
        status, base_ref = await self._call(
            "GET", f"{self.api}/git/ref/heads/{quote(self.target.branch, safe='')}"
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise ProviderWriteError(
                "base_ref_unavailable",
                "GitHub feedback base branch could not be resolved",
            )
        return _object_sha(base_ref, code="base_ref_invalid")

    async def create_branch(self, branch: str, commit: str) -> bool:
        status, _created = await self._call(
            "POST",
            f"{self.api}/git/refs",
            json_body={"ref": "refs/heads/" + branch, "sha": commit},
        )
        if status == 201:  # ruff: ignore[magic-value-comparison]
            return True
        if status == 422:  # ruff: ignore[magic-value-comparison]
            return False
        raise ProviderWriteError(
            "branch_create_failed", "GitHub feedback review branch could not be created"
        )

    async def branch_exists(self, branch: str) -> bool:
        status, payload = await self._call(
            "GET", f"{self.api}/git/ref/heads/{quote(branch, safe='')}"
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

    async def read_file(self, path: str, ref: str) -> bytes | None:
        status, existing = await self._call(
            "GET", f"{self.api}/contents/{quote(path, safe='/')}", params={"ref": ref}
        )
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return None
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(existing, dict):
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

    async def create_file(
        self, path: str, branch: str, data: bytes, message: str
    ) -> bool:
        status, _content = await self._call(
            "PUT",
            f"{self.api}/contents/{quote(path, safe='/')}",
            json_body={
                "message": message,
                "content": base64.b64encode(data).decode("ascii"),
                "branch": branch,
            },
        )
        return status in {200, 201}

    async def compare(self, base: str, branch: str) -> Comparison:
        status, comparison = await self._call(
            "GET",
            f"{self.api}/compare/{quote(base, safe='')}...{quote(branch, safe='')}",
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
        files = comparison.get("files")
        entries: list[tuple[str, str]] = []
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    entries.append(("", "invalid"))
                    continue
                entries.append(
                    (str(item.get("filename", "")), str(item.get("status", "")))
                )
        elif ahead_by:
            entries.append(("", "invalid"))
        return Comparison(ahead_by=ahead_by, files=tuple(entries))

    async def find_review(self, branch: str, base: str) -> str | None:
        owner = self.repo.split("/", 1)[0]
        status, pulls = await self._call(
            "GET",
            f"{self.api}/pulls",
            params={"state": "all", "head": f"{owner}:{branch}", "base": base},
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
        return _review_url(first.get("html_url"), repo=self.repo, web=self.web)

    async def open_review(
        self, branch: str, base: str, title: str, body: str
    ) -> str | None:
        status, pull = await self._call(
            "POST",
            f"{self.api}/pulls",
            json_body={"title": title, "head": branch, "base": base, "body": body},
        )
        if status not in {200, 201}:
            return None
        if not isinstance(pull, dict):
            raise ProviderWriteError(
                "review_open_invalid",
                "GitHub feedback pull request response was invalid",
            )
        return _review_url(pull.get("html_url"), repo=self.repo, web=self.web)


async def submit_github_review(
    *,
    client,
    target: StorageTarget,
    token: str,
    event: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    """Idempotently ensure branch, immutable event, and pull request exist."""
    if not token:
        raise ProviderWriteError(
            "credential_missing", "GitHub feedback credential is not configured"
        )
    return await submit_forge_review(
        GitHubOps(client=client, target=target, token=token),
        base_branch=target.branch,
        feedback_path=target.feedback_path,
        expose_links=target.expose_links,
        event=event,
        request_hash=request_hash,
    )


__all__ = [
    "GitHubOps",
    "ProviderWriteError",
    "pinned_review_url",
    "submit_github_review",
]
