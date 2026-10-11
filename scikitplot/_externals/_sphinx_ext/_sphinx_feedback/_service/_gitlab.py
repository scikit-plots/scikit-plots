"""
GitLab provider: one reviewed merge request per feedback event.

gitlab.com and self-managed GitLab (``api_url``, default
``https://gitlab.com/api/v4``); projects may sit in nested groups. The state
machine and every reconciliation step are the shared ones in :mod:`._forge`;
this module maps them onto the GitLab REST API v4.

Notes
-----
**API calls** (all JSON, ``PRIVATE-TOKEN`` header):

=====================  ===============================================================
operation              request
=====================  ===============================================================
resolve base / exists  ``GET /projects/:id/repository/branches/:branch``
create branch          ``POST /projects/:id/repository/branches`` ``{branch, ref}``
read file              ``GET /projects/:id/repository/files/:path?ref=`` (base64)
create file            ``POST /projects/:id/repository/files/:path`` (base64)
compare                ``GET /projects/:id/repository/compare?from=&to=&straight=false``
find review            ``GET /projects/:id/merge_requests?source_branch=&target_branch=&state=all``
open review            ``POST /projects/:id/merge_requests`` (``remove_source_branch``)
=====================  ===============================================================

``:id`` is the URL-encoded project path and ``:path``/``:branch`` are URL-encoded,
as the API requires. ``straight=false`` compares against the merge base, the
same "what does the branch add" question GitHub's three-dot compare answers.
A compare that reports ``compare_timeout`` is treated as unverifiable, never as
a clean branch. A refused branch, file or merge request is reconciled by
reading the forge state, so the exact refusal status (undocumented for some
calls) never decides the outcome. Review URLs must be on the API's host under
``/<project>/-/merge_requests/<n>``.
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import quote, urlsplit

from .. import __version__
from ._config import DEFAULT_GITLAB_API_URL, StorageTarget
from ._forge import (
    Comparison,
    ForgeOps,
    ProviderWriteError,
    _bounded_json,
    submit_forge_review,
)
from ._github import pinned_review_url

_HEX = frozenset("0123456789abcdefABCDEF")


def commit_id(value: Any, *, code: str, label: str) -> str:
    """Return a 40- or 64-hex commit id, or raise ``ProviderWriteError(code)``."""
    text = str(value or "")
    if len(text) not in {40, 64} or any(ch not in _HEX for ch in text):
        raise ProviderWriteError(code, f"{label} returned an invalid commit id")
    return text


class GitLabOps(ForgeOps):
    """GitLab REST v4 operations for :func:`._forge.submit_forge_review`."""

    provider = "gitlab"
    label = "GitLab"

    def __init__(self, *, client, target: StorageTarget, token: str) -> None:
        self.client = client
        self.target = target
        self.repo = target.repo
        root = (target.api_url or DEFAULT_GITLAB_API_URL).rstrip("/")
        self.api = f"{root}/projects/{quote(self.repo, safe='')}"
        parsed = urlsplit(root)
        self.web = ((parsed.hostname or "").lower(), parsed.port)
        self.headers = {
            "Accept": "application/json",
            "PRIVATE-TOKEN": token,
            "User-Agent": f"sphinx-feedback/{__version__}",
        }

    async def _call(self, method: str, url: str, **kwargs) -> tuple[int, Any]:
        return await _bounded_json(
            self.client, method, url, headers=self.headers, label=self.label, **kwargs
        )

    def _branch_url(self, branch: str) -> str:
        return f"{self.api}/repository/branches/{quote(branch, safe='')}"

    def _file_url(self, path: str) -> str:
        return f"{self.api}/repository/files/{quote(path, safe='')}"

    async def resolve_base(self) -> str:
        status, payload = await self._call("GET", self._branch_url(self.target.branch))
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(payload, dict):
            raise ProviderWriteError(
                "base_ref_unavailable",
                "GitLab feedback base branch could not be resolved",
            )
        commit = payload.get("commit")
        return commit_id(
            commit.get("id") if isinstance(commit, dict) else None,
            code="base_ref_invalid",
            label=self.label,
        )

    async def create_branch(self, branch: str, commit: str) -> bool:
        status, _payload = await self._call(
            "POST",
            f"{self.api}/repository/branches",
            json_body={"branch": branch, "ref": commit},
        )
        return status == 201  # ruff: ignore[magic-value-comparison]

    async def branch_exists(self, branch: str) -> bool:
        status, payload = await self._call("GET", self._branch_url(branch))
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return False
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(payload, dict):
            raise ProviderWriteError(
                "existing_branch_unverifiable",
                "Existing GitLab feedback branch could not be verified",
            )
        return True

    async def read_file(self, path: str, ref: str) -> bytes | None:
        status, payload = await self._call(
            "GET", self._file_url(path), params={"ref": ref}
        )
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return None
        if (
            status != 200  # ruff: ignore[magic-value-comparison]
            or not isinstance(payload, dict)
            or payload.get("encoding") != "base64"
        ):
            raise ProviderWriteError(
                "existing_branch_unverifiable",
                "Existing GitLab feedback event could not be verified",
            )
        try:
            return base64.b64decode(str(payload.get("content", "")), validate=True)
        except Exception as exc:
            raise ProviderWriteError(
                "existing_content_invalid",
                "Existing GitLab feedback event could not be verified",
            ) from exc

    async def create_file(
        self, path: str, branch: str, data: bytes, message: str
    ) -> bool:
        status, _payload = await self._call(
            "POST",
            self._file_url(path),
            json_body={
                "branch": branch,
                "content": base64.b64encode(data).decode("ascii"),
                "encoding": "base64",
                "commit_message": message,
            },
        )
        return status == 201  # ruff: ignore[magic-value-comparison]

    async def compare(self, base: str, branch: str) -> Comparison:
        status, payload = await self._call(
            "GET",
            f"{self.api}/repository/compare",
            params={"from": base, "to": branch, "straight": "false"},
        )
        if (
            status != 200  # ruff: ignore[magic-value-comparison]
            or not isinstance(payload, dict)
            or payload.get("compare_timeout") is True
            or not isinstance(payload.get("commits"), list)
            or not isinstance(payload.get("diffs"), list)
        ):
            raise ProviderWriteError(
                "existing_branch_unverifiable",
                "Existing GitLab feedback branch could not be compared safely",
            )
        files: list[tuple[str, str]] = []
        for item in payload["diffs"]:
            if not isinstance(item, dict):
                files.append(("", "invalid"))
                continue
            added = (
                item.get("new_file") is True
                and item.get("renamed_file") is not True
                and item.get("deleted_file") is not True
            )
            files.append(
                (str(item.get("new_path", "")), "added" if added else "changed")
            )
        return Comparison(ahead_by=len(payload["commits"]), files=tuple(files))

    async def find_review(self, branch: str, base: str) -> str | None:
        status, payload = await self._call(
            "GET",
            f"{self.api}/merge_requests",
            params={"source_branch": branch, "target_branch": base, "state": "all"},
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise ProviderWriteError(
                "review_lookup_failed",
                "Existing GitLab feedback review could not be verified",
            )
        if not isinstance(payload, list):
            raise ProviderWriteError(
                "review_lookup_invalid",
                "GitLab feedback review lookup returned an invalid response",
            )
        if not payload:
            return None
        first = payload[0]
        if not isinstance(first, dict):
            raise ProviderWriteError(
                "review_lookup_invalid",
                "GitLab feedback review lookup returned an invalid entry",
            )
        return self._review_url(first.get("web_url"))

    def _review_url(self, value: Any) -> str:
        return pinned_review_url(
            value,
            label=self.label,
            web=self.web,
            path_prefix=f"/{self.repo}/-/merge_requests/",
        )

    async def open_review(
        self, branch: str, base: str, title: str, body: str
    ) -> str | None:
        status, payload = await self._call(
            "POST",
            f"{self.api}/merge_requests",
            json_body={
                "source_branch": branch,
                "target_branch": base,
                "title": title,
                "description": body,
                "remove_source_branch": True,
            },
        )
        if status != 201:  # ruff: ignore[magic-value-comparison]
            return None
        if not isinstance(payload, dict):
            raise ProviderWriteError(
                "review_open_invalid",
                "GitLab feedback merge request response was invalid",
            )
        return self._review_url(payload.get("web_url"))


async def submit_gitlab_review(
    *,
    client,
    target: StorageTarget,
    token: str,
    event: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    """Idempotently ensure branch, immutable event and merge request exist."""
    if not token:
        raise ProviderWriteError(
            "credential_missing", "GitLab feedback credential is not configured"
        )
    return await submit_forge_review(
        GitLabOps(client=client, target=target, token=token),
        base_branch=target.branch,
        feedback_path=target.feedback_path,
        expose_links=target.expose_links,
        event=event,
        request_hash=request_hash,
    )


__all__ = ["GitLabOps", "commit_id", "submit_gitlab_review"]
