"""
Bitbucket Data Center provider: one reviewed pull request per feedback event.

Selected by a ``bitbucket`` target whose ``api_url`` is a Data Center REST
root, ``https://<host>[/<context path>]/rest/api/1.0`` (see
:func:`._config.bitbucket_flavor`). The state machine and every
reconciliation step are the shared ones in :mod:`._forge`; this module maps
them onto the Bitbucket Data Center REST API 1.0. ``repo`` is
``<project key>/<repository slug>``; personal repositories (``~user``) are
not supported.

Notes
-----
**API calls** (``Authorization: Bearer`` with an HTTP access token: personal,
project or repository; all under ``/projects/{key}/repos/{slug}``):

=====================  ======================================================================
operation              request
=====================  ======================================================================
branch head / exists   ``GET /branches?filterText=<branch>&orderBy=ALPHABETICAL&details=false``,
                       the entry whose ``id`` is ``refs/heads/<branch>`` (``latestCommit``)
create branch          ``POST /branches`` ``{name, startPoint: <base branch>}``
read file              ``GET /raw/{path}?at=<commit id>`` (raw bytes)
create file            ``PUT /browse/{path}`` multipart ``content`` (file), ``message``,
                       ``branch``; no ``sourceCommitId`` (a new file)
compare                ``GET /compare/commits?from=<branch head>&to=<base head>`` then
                       ``GET /commits/{commit}/changes`` for the single unique commit
find review            ``GET /pull-requests?at=refs/heads/<branch>&direction=OUTGOING&state=ALL``
open review            ``POST /pull-requests`` ``{title, description, fromRef, toRef}``
=====================  ======================================================================

**Why these shapes.** A branch is found by exact ``id`` among the
``filterText`` matches, and a page that is not the last one without that
``id`` fails closed (more matches than one page). Files are read at a resolved
commit id, never a branch name. "What the branch adds" is git's answer: the
commits reachable from the branch head but not from the base head (both are
always passed: an omitted side means the default branch) and, when there is
exactly one, that commit's changes against its first parent. Only change type
``ADD`` counts as an added file (``COPY`` creates a file from another, and is
not); a change path is the ``/``-joined ``components`` and must equal
``toString`` when that is present. A paged or capped answer fails closed. The
branch is created from the base *branch name*, as the documented examples do;
whatever it then holds, the shared scope check admits it only when it adds
exactly one commit with exactly the one event file over the base. The
multipart write sends ``X-Atlassian-Token: no-check``, the header Atlassian's
multipart resources require to pass their XSRF check. A pull request counts
only when its ``fromRef`` and ``toRef`` are this repository's
``refs/heads/<branch>`` and ``refs/heads/<base>``; its URL must be
``links.self[0].href`` on the API's host under
``<context>/projects/<key>/repos/<slug>/pull-requests/<id>`` (optionally
``/overview``), with ``<id>`` its own ``id``; the key compares
case-insensitively.

**Verification status.** Sources: Atlassian's REST reference index
(Bitbucket Server 5.16: ``branches`` POST/GET, ``browse/{path}`` GET/PUT,
``raw/{path}``, ``commits``, ``commits/{commitId}/changes``,
``compare/commits``, ``pull-requests`` POST/GET), the resource descriptions
generated from it (``raw`` ``at`` is "the commit ID or ref"; ``browse`` PUT
multipart fields, ``sourceCommitId`` "null if this is a new file";
``compare/commits`` "accessible from ``from`` but not in ``to``"; the
``pull-requests`` ``at``/``direction``/``state`` parameters; the changes
page cap), the ``ChangeType`` Javadoc (``ADD`` "a new file was added"), and
Atlassian's HTTP access token page (Bearer). The JSON key of a change's type
(``type``) is not quoted in those sources; a different key makes every
branch look changed (refused, never accepted). ``check-provider`` reads the
changes of the base head and fails when their shape differs, so an operator
can confirm it against the real server, read-only, before enabling.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlsplit

from .. import __version__
from ._config import BITBUCKET_DATA_CENTER_API_SUFFIX, StorageTarget
from ._forge import (
    Comparison,
    ForgeOps,
    ProviderWriteError,
    _bounded_json,
    submit_forge_review,
)
from ._github import pinned_review_url
from ._gitlab import commit_id

_HEADS = "refs/heads/"
_BRANCH_PAGE = 100
_COMMIT_PAGE = 10
_CHANGE_PAGE = 100
_REVIEW_PAGE = 25
#: ``com.atlassian.bitbucket.content.ChangeType``.
CHANGE_TYPES = frozenset({"ADD", "COPY", "DELETE", "MODIFY", "MOVE", "UNKNOWN"})
_OVERVIEW = "/overview"


def change_path(item: Any) -> str | None:
    """
    Return the path of one change entry, or ``None`` if its shape is not trusted.

    Parameters
    ----------
    item : object
        One entry of a changes page.

    Returns
    -------
    str or None
        The ``/``-joined ``path.components`` (each a non-empty string without
        ``/``), when ``path.toString`` is absent or equal to it.
    """
    path = item.get("path") if isinstance(item, dict) else None
    components = path.get("components") if isinstance(path, dict) else None
    if not isinstance(components, list) or not components:
        return None
    if not all(
        isinstance(part, str) and part and "/" not in part for part in components
    ):
        return None
    joined = "/".join(components)
    text = path.get("toString")
    if text is not None and text != joined:
        return None
    return joined


class BitbucketDataCenterOps(ForgeOps):
    """Bitbucket Data Center REST 1.0 operations for :func:`._forge.submit_forge_review`."""

    provider = "bitbucket"
    label = "Bitbucket Data Center"

    def __init__(self, *, client, target: StorageTarget, token: str) -> None:
        self.client = client
        self.target = target
        self.key, self.slug = target.repo.split("/", 1)
        root = target.api_url.rstrip("/")
        if not root.endswith(BITBUCKET_DATA_CENTER_API_SUFFIX):
            raise ProviderWriteError(
                "provider_config_invalid",
                "Bitbucket Data Center api_url must end in /rest/api/1.0",
            )
        self.api = (
            f"{root}/projects/{quote(self.key, safe='')}"
            f"/repos/{quote(self.slug, safe='')}"
        )
        parsed = urlsplit(root)
        self.web = ((parsed.hostname or "").lower(), parsed.port)
        context = parsed.path[: -len(BITBUCKET_DATA_CENTER_API_SUFFIX)]
        self.review_prefix = (
            f"{context}/projects/{self.key}/repos/{self.slug}/pull-requests/"
        )
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": f"sphinx-feedback/{__version__}",
        }

    async def _call(self, method: str, url: str, **kwargs) -> tuple[int, Any]:
        headers = dict(self.headers, **kwargs.pop("extra_headers", {}))
        return await _bounded_json(
            self.client, method, url, headers=headers, label=self.label, **kwargs
        )

    async def _head(self, branch: str, *, code: str) -> str | None:
        """Return *branch*'s head commit, ``None`` when the branch is absent."""
        status, payload = await self._call(
            "GET",
            f"{self.api}/branches",
            params={
                "filterText": branch,
                "orderBy": "ALPHABETICAL",
                "details": "false",
                "limit": str(_BRANCH_PAGE),
            },
        )
        values = payload.get("values") if isinstance(payload, dict) else None
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if _status or not isinstance(values, list):
            raise ProviderWriteError(
                code, f"{self.label} feedback branch could not be verified"
            )
        ref = _HEADS + branch
        for item in values:
            if isinstance(item, dict) and item.get("id") == ref:
                return commit_id(item.get("latestCommit"), code=code, label=self.label)
        if payload.get("isLastPage") is not True:
            raise ProviderWriteError(
                code,
                f"{self.label} feedback branch could not be verified: more matching "
                "branches than one page",
            )
        return None

    async def resolve_base(self) -> str:
        head = await self._head(self.target.branch, code="base_ref_unavailable")
        if head is None:
            raise ProviderWriteError(
                "base_ref_unavailable",
                f"{self.label} feedback base branch could not be resolved",
            )
        return head

    async def create_branch(self, branch: str, commit: str) -> bool:
        # ``commit`` is the base head the engine resolved; the request names the
        # base branch, as the documented examples do (see the module notes).
        status, _payload = await self._call(
            "POST",
            f"{self.api}/branches",
            json_body={"name": branch, "startPoint": self.target.branch},
        )
        return 200 <= status < 300  # ruff: ignore[magic-value-comparison]

    async def branch_exists(self, branch: str) -> bool:
        return await self._head(branch, code="existing_branch_unverifiable") is not None

    async def read_file(self, path: str, ref: str) -> bytes | None:
        head = await self._head(ref, code="existing_branch_unverifiable")
        if head is None:
            return None
        status, body = await self._call(
            "GET",
            f"{self.api}/raw/{quote(path, safe='/')}",
            params={"at": head},
            raw=True,
        )
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return None
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise ProviderWriteError(
                "existing_branch_unverifiable",
                f"Existing {self.label} feedback event could not be verified",
            )
        return body

    async def create_file(
        self, path: str, branch: str, data: bytes, message: str
    ) -> bool:
        status, _payload = await self._call(
            "PUT",
            f"{self.api}/browse/{quote(path, safe='/')}",
            data={"message": message, "branch": branch},
            files={"content": (path.rsplit("/", 1)[-1], data, "application/json")},
            extra_headers={"X-Atlassian-Token": "no-check"},
        )
        return 200 <= status < 300  # ruff: ignore[magic-value-comparison]

    async def _page(self, url: str, params: dict[str, str], *, code: str) -> dict:
        status, payload = await self._call("GET", url, params=params)
        _status = status != 200  # ruff: ignore[magic-value-comparison]
        if (
            _status
            or not isinstance(payload, dict)
            or not isinstance(payload.get("values"), list)
        ):
            raise ProviderWriteError(
                code, f"{self.label} feedback branch could not be compared"
            )
        return payload

    async def changes(self, commit: str, *, limit: int = _CHANGE_PAGE) -> dict:
        """Return the changes page of *commit* against its first parent."""
        return await self._page(
            f"{self.api}/commits/{commit}/changes",
            {"limit": str(limit), "withComments": "false"},
            code="existing_branch_unverifiable",
        )

    async def compare(self, base: str, branch: str) -> Comparison:
        code = "existing_branch_unverifiable"
        head = await self._head(branch, code=code)
        base_head = await self._head(base, code=code)
        if head is None or base_head is None:
            raise ProviderWriteError(
                code, f"{self.label} feedback branch could not be compared"
            )
        payload = await self._page(
            f"{self.api}/compare/commits",
            {"from": head, "to": base_head, "limit": str(_COMMIT_PAGE)},
            code=code,
        )
        values = payload["values"]
        ahead_by = len(values) + (0 if payload.get("isLastPage") is True else 1)
        if ahead_by != 1:
            return Comparison(ahead_by=ahead_by, files=())
        only = values[0]
        commit = commit_id(
            only.get("id") if isinstance(only, dict) else None,
            code=code,
            label=self.label,
        )
        page = await self.changes(commit)
        files: list[tuple[str, str]] = []
        for item in page["values"]:
            path = change_path(item)
            if path is None:
                files.append(("", "invalid"))
                continue
            added = isinstance(item, dict) and item.get("type") == "ADD"
            files.append((path, "added" if added else "changed"))
        if page.get("isLastPage") is not True:
            files.append(("", "truncated"))
        return Comparison(ahead_by=1, files=tuple(files))

    def _repository_is_ours(self, ref: Any) -> bool:
        repository = ref.get("repository") if isinstance(ref, dict) else None
        project = repository.get("project") if isinstance(repository, dict) else None
        key = project.get("key") if isinstance(project, dict) else None
        return (
            isinstance(key, str)
            and key.lower() == self.key.lower()
            and repository.get("slug") == self.slug
        )

    def _is_ours(self, item: Any, branch: str, base: str) -> bool:
        """Whether *item* is a pull request from *branch* into *base* of this repository."""
        if not isinstance(item, dict):
            return False
        source, target = item.get("fromRef"), item.get("toRef")
        return (
            isinstance(source, dict)
            and isinstance(target, dict)
            and source.get("id") == _HEADS + branch
            and target.get("id") == _HEADS + base
            and self._repository_is_ours(source)
            and self._repository_is_ours(target)
        )

    def _review_url(self, item: Any) -> str:
        number = item.get("id") if isinstance(item, dict) else None
        links = item.get("links") if isinstance(item, dict) else None
        own = links.get("self") if isinstance(links, dict) else None
        first = own[0] if isinstance(own, list) and own else None
        href = first.get("href") if isinstance(first, dict) else None
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or not isinstance(href, str)
        ):
            raise ProviderWriteError(
                "review_lookup_invalid", f"{self.label} feedback review URL was invalid"
            )
        url = href.strip()
        url = url[: -len(_OVERVIEW)] if url.endswith(_OVERVIEW) else url
        url = pinned_review_url(url, label=self.label, web=self.web, path_prefix=None)
        path = urlsplit(url).path
        prefix = self.review_prefix
        if not path.lower().startswith(prefix.lower()) or path[len(prefix) :] != str(
            number
        ):
            raise ProviderWriteError(
                "review_lookup_invalid",
                f"{self.label} feedback review URL did not belong to the configured repository",
            )
        return url

    async def find_review(self, branch: str, base: str) -> str | None:
        status, payload = await self._call(
            "GET",
            f"{self.api}/pull-requests",
            params={
                "at": _HEADS + branch,
                "direction": "OUTGOING",
                "state": "ALL",
                "order": "NEWEST",
                "limit": str(_REVIEW_PAGE),
                "withAttributes": "false",
                "withProperties": "false",
            },
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise ProviderWriteError(
                "review_lookup_failed",
                f"Existing {self.label} feedback review could not be verified",
            )
        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            raise ProviderWriteError(
                "review_lookup_invalid",
                f"{self.label} feedback review lookup returned an invalid response",
            )
        for item in values:
            if self._is_ours(item, branch, base):
                return self._review_url(item)
        if payload.get("isLastPage") is not True:
            raise ProviderWriteError(
                "review_lookup_invalid",
                f"{self.label} feedback review lookup returned more than one page",
            )
        return None

    def _ref(self, branch: str) -> dict[str, Any]:
        return {
            "id": _HEADS + branch,
            "repository": {"slug": self.slug, "project": {"key": self.key}},
        }

    async def open_review(
        self, branch: str, base: str, title: str, body: str
    ) -> str | None:
        status, payload = await self._call(
            "POST",
            f"{self.api}/pull-requests",
            json_body={
                "title": title,
                "description": body,
                "fromRef": self._ref(branch),
                "toRef": self._ref(base),
            },
        )
        if not 200 <= status < 300:  # ruff: ignore[magic-value-comparison]
            return None
        if not self._is_ours(payload, branch, base):
            raise ProviderWriteError(
                "review_open_invalid",
                f"{self.label} feedback pull request response was invalid",
            )
        return self._review_url(payload)

    async def check_read_shapes(self, base_head: str) -> str:
        """
        Confirm, read-only, that change entries have the shape :meth:`compare` reads.

        Parameters
        ----------
        base_head : str
            The base branch head commit (from :meth:`resolve_base`).

        Returns
        -------
        str
            A short description of what was checked.

        Raises
        ------
        ProviderWriteError
            ``provider_shape_unexpected`` when an entry has no trusted path or
            a ``type`` outside :data:`CHANGE_TYPES`.
        """
        page = await self.changes(base_head, limit=5)
        for item in page["values"]:
            kind = item.get("type") if isinstance(item, dict) else None
            if change_path(item) is None or kind not in CHANGE_TYPES:
                raise ProviderWriteError(
                    "provider_shape_unexpected",
                    f"{self.label} change entries do not have the expected shape",
                )
        count = len(page["values"])
        return (
            f"{count} change entr{'y' if count == 1 else 'ies'} of the base head read"
        )


async def submit_bitbucket_data_center_review(
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
            "credential_missing",
            "Bitbucket Data Center feedback credential is not configured",
        )
    return await submit_forge_review(
        BitbucketDataCenterOps(client=client, target=target, token=token),
        base_branch=target.branch,
        feedback_path=target.feedback_path,
        expose_links=target.expose_links,
        event=event,
        request_hash=request_hash,
    )


__all__ = [
    "CHANGE_TYPES",
    "BitbucketDataCenterOps",
    "change_path",
    "submit_bitbucket_data_center_review",
]
