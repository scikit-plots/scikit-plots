"""
Bitbucket Cloud provider: one reviewed pull request per feedback event.

``https://api.bitbucket.org/2.0`` only; Bitbucket Data Center has a different
REST API and is reached through the webhook provider. The state machine and
every reconciliation step are the shared ones in :mod:`._forge`; this module
maps them onto the Bitbucket Cloud REST API 2.0.

Notes
-----
**API calls** (``Authorization: Bearer`` with a repository, project or
workspace access token):

=====================  ==================================================================
operation              request
=====================  ==================================================================
branch head / exists   ``GET /repositories/{ws}/{repo}/refs/branches?q=name="<branch>"``
create branch          ``POST /repositories/{ws}/{repo}/refs/branches`` ``{name, target}``
read file              ``GET /repositories/{ws}/{repo}/src/{commit}/{path}`` (raw bytes)
create file            ``POST /repositories/{ws}/{repo}/src`` form ``{<path>, message, branch}``
compare                ``GET .../commits/{head}?exclude={base head}`` then
                       ``GET .../diffstat/{commit}`` for the single unique commit
find review            ``GET .../pullrequests?q=source.branch.name=... AND
                       destination.branch.name=...&state=OPEN&state=MERGED&...``
open review            ``POST .../pullrequests`` (``close_source_branch``)
=====================  ==================================================================

**Why these shapes.** Branches are looked up with a query rather than a path,
so ``feedback/<nonce>`` needs no path-encoding convention. Files are read at a
resolved commit hash, never a branch name, so the read cannot be redirected
(redirects are refused). "What the branch adds" is computed as git does it:
the commits reachable from the branch head but not from the base head, and,
when there is exactly one, that commit's own diffstat against its parent.
A paged answer (``next``) means more than fits the bound and fails closed.
Review URLs must be on ``bitbucket.org`` under ``/<ws>/<repo>/pull-requests/<n>``.

**Verification status.** The call shapes follow the Bitbucket Cloud REST
reference as known to the authors; they are exercised against a mock
transport here. Run ``python -m ..._sphinx_feedback check-provider`` against
a real repository (read-only) before enabling it in production.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .. import __version__
from ._config import BITBUCKET_CLOUD_API_URL, StorageTarget
from ._forge import (
    Comparison,
    ForgeOps,
    ProviderWriteError,
    _bounded_json,
    submit_forge_review,
)
from ._github import pinned_review_url
from ._gitlab import commit_id

_REVIEW_STATES = ("OPEN", "MERGED", "DECLINED", "SUPERSEDED")
_COMMIT_PAGE = 10


class BitbucketOps(ForgeOps):
    """Bitbucket Cloud REST 2.0 operations for :func:`._forge.submit_forge_review`."""

    provider = "bitbucket"
    label = "Bitbucket"

    def __init__(self, *, client, target: StorageTarget, token: str) -> None:
        self.client = client
        self.target = target
        self.repo = target.repo
        workspace, slug = target.repo.split("/", 1)
        self.api = (
            f"{BITBUCKET_CLOUD_API_URL}/repositories/"
            f"{quote(workspace, safe='')}/{quote(slug, safe='')}"
        )
        self.web = ("bitbucket.org", None)
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": f"sphinx-feedback/{__version__}",
        }

    async def _call(self, method: str, url: str, **kwargs) -> tuple[int, Any]:
        return await _bounded_json(
            self.client, method, url, headers=self.headers, label=self.label, **kwargs
        )

    async def _head(self, branch: str, *, code: str) -> str | None:
        """Return *branch*'s head commit, ``None`` when the branch is absent."""
        status, payload = await self._call(
            "GET", f"{self.api}/refs/branches", params={"q": f'name="{branch}"'}
        )
        values = payload.get("values") if isinstance(payload, dict) else None
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(values, list):
            raise ProviderWriteError(
                code, "Bitbucket feedback branch could not be verified"
            )
        for item in values:
            if isinstance(item, dict) and item.get("name") == branch:
                target = item.get("target")
                return commit_id(
                    target.get("hash") if isinstance(target, dict) else None,
                    code=code,
                    label=self.label,
                )
        return None

    async def resolve_base(self) -> str:
        head = await self._head(self.target.branch, code="base_ref_unavailable")
        if head is None:
            raise ProviderWriteError(
                "base_ref_unavailable",
                "Bitbucket feedback base branch could not be resolved",
            )
        return head

    async def create_branch(self, branch: str, commit: str) -> bool:
        status, _payload = await self._call(
            "POST",
            f"{self.api}/refs/branches",
            json_body={"name": branch, "target": {"hash": commit}},
        )
        return status == 201  # ruff: ignore[magic-value-comparison]

    async def branch_exists(self, branch: str) -> bool:
        return await self._head(branch, code="existing_branch_unverifiable") is not None

    async def read_file(self, path: str, ref: str) -> bytes | None:
        head = await self._head(ref, code="existing_branch_unverifiable")
        if head is None:
            return None
        status, body = await self._call(
            "GET", f"{self.api}/src/{head}/{quote(path, safe='/')}", raw=True
        )
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return None
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise ProviderWriteError(
                "existing_branch_unverifiable",
                "Existing Bitbucket feedback event could not be verified",
            )
        return body

    async def create_file(
        self, path: str, branch: str, data: bytes, message: str
    ) -> bool:
        status, _payload = await self._call(
            "POST",
            f"{self.api}/src",
            data={path: data.decode("utf-8"), "message": message, "branch": branch},
        )
        return status == 201  # ruff: ignore[magic-value-comparison]

    async def compare(self, base: str, branch: str) -> Comparison:
        code = "existing_branch_unverifiable"
        head = await self._head(branch, code=code)
        base_head = await self._head(base, code=code)
        if head is None or base_head is None:
            raise ProviderWriteError(
                code, "Bitbucket feedback branch could not be compared"
            )
        status, payload = await self._call(
            "GET",
            f"{self.api}/commits/{head}",
            params={"exclude": base_head, "pagelen": str(_COMMIT_PAGE)},
        )
        values = payload.get("values") if isinstance(payload, dict) else None
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(values, list):
            raise ProviderWriteError(
                code, "Bitbucket feedback branch could not be compared"
            )
        ahead_by = len(values) + (1 if payload.get("next") else 0)
        if ahead_by != 1:
            return Comparison(ahead_by=ahead_by, files=())
        only = values[0]
        commit = commit_id(
            only.get("hash") if isinstance(only, dict) else None,
            code=code,
            label=self.label,
        )
        status, stat = await self._call("GET", f"{self.api}/diffstat/{commit}")
        entries = stat.get("values") if isinstance(stat, dict) else None
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(entries, list):
            raise ProviderWriteError(
                code, "Bitbucket feedback branch could not be compared"
            )
        files: list[tuple[str, str]] = []
        for item in entries:
            new = item.get("new") if isinstance(item, dict) else None
            path = new.get("path") if isinstance(new, dict) else ""
            status_text = item.get("status") if isinstance(item, dict) else ""
            files.append(
                (str(path or ""), "added" if status_text == "added" else "changed")
            )
        if stat.get("next"):
            files.append(("", "truncated"))
        return Comparison(ahead_by=1, files=tuple(files))

    def _review_url(self, value: Any) -> str:
        return pinned_review_url(
            value,
            label=self.label,
            web=self.web,
            path_prefix=f"/{self.repo}/pull-requests/",
        )

    @staticmethod
    def _html_link(item: Any) -> Any:
        links = item.get("links") if isinstance(item, dict) else None
        html = links.get("html") if isinstance(links, dict) else None
        return html.get("href") if isinstance(html, dict) else None

    async def find_review(self, branch: str, base: str) -> str | None:
        query = f'source.branch.name="{branch}" AND destination.branch.name="{base}"'
        params = [("q", query)] + [("state", state) for state in _REVIEW_STATES]
        status, payload = await self._call(
            "GET", f"{self.api}/pullrequests", params=params
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise ProviderWriteError(
                "review_lookup_failed",
                "Existing Bitbucket feedback review could not be verified",
            )
        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            raise ProviderWriteError(
                "review_lookup_invalid",
                "Bitbucket feedback review lookup returned an invalid response",
            )
        if not values:
            return None
        return self._review_url(self._html_link(values[0]))

    async def open_review(
        self, branch: str, base: str, title: str, body: str
    ) -> str | None:
        status, payload = await self._call(
            "POST",
            f"{self.api}/pullrequests",
            json_body={
                "title": title,
                "description": body,
                "source": {"branch": {"name": branch}},
                "destination": {"branch": {"name": base}},
                "close_source_branch": True,
            },
        )
        if status != 201:  # ruff: ignore[magic-value-comparison]
            return None
        if not isinstance(payload, dict):
            raise ProviderWriteError(
                "review_open_invalid",
                "Bitbucket feedback pull request response was invalid",
            )
        return self._review_url(self._html_link(payload))


async def submit_bitbucket_review(
    *,
    client,
    target: StorageTarget,
    token: str,
    event: dict[str, Any],
    request_hash: str,
) -> dict[str, Any]:
    """Idempotently ensure branch, immutable event and pull request exist."""
    if not token:
        raise ProviderWriteError(
            "credential_missing", "Bitbucket feedback credential is not configured"
        )
    return await submit_forge_review(
        BitbucketOps(client=client, target=target, token=token),
        base_branch=target.branch,
        feedback_path=target.feedback_path,
        expose_links=target.expose_links,
        event=event,
        request_hash=request_hash,
    )


__all__ = ["BitbucketOps", "submit_bitbucket_review"]
