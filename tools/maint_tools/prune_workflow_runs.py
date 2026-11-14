#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Delete old GitHub Actions workflow runs (and with them their logs).

GitHub has no bulk "delete old runs" action; this lists the runs of a
repository, newest first, and deletes those a policy selects. It is a
dry run unless ``--apply`` is given.

A run is deleted only when ALL of these hold:

1. it is not among the ``--keep-latest`` newest runs of the repository
   (25 is the first page of the Actions tab);
2. it has completed (a queued or running one cannot be deleted);
3. it is not the run executing this script (``GITHUB_RUN_ID``);
4. it was created before ``--before`` (when given);
5. it was created more than ``--older-than-days`` days ago (when given);
6. its workflow matches ``--workflow`` (when given).

Running it again deletes nothing more (idempotent). At most
``--max-delete`` runs are deleted in one go, to stay inside the API rate
limit of a workflow token (1,000 requests an hour): run it again for more.

Usage::

    python tools/maint_tools/prune_workflow_runs.py --repo OWNER/NAME
    python tools/maint_tools/prune_workflow_runs.py --repo OWNER/NAME \
        --older-than-days 1 --apply

The token is read from ``GH_TOKEN`` or ``GITHUB_TOKEN`` (``actions: read``
for a dry run, ``actions: write`` with ``--apply``). Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Iterator

API = "https://api.github.com"
PAGE_SIZE = 100
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")

#: ``request(method, url) -> (status, parsed JSON or None)``.
Request = Callable[[str, str], "tuple[int, Any]"]


class PruneError(Exception):
    """A setting is invalid or the API refused a request."""


def parse_time(value: str, what: str = "date") -> dt.datetime:
    """
    Return *value* (``YYYY-MM-DD`` or ISO 8601) as an aware UTC datetime.

    Raises
    ------
    PruneError
        If *value* is not a date; *what* names the setting.
    """
    text = value.strip()
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PruneError(
            f"{what} must be YYYY-MM-DD or ISO 8601; got {value!r}."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def cutoff(
    now: dt.datetime, before: str | None, older_than_days: float | None
) -> dt.datetime | None:
    """
    Return the instant a run must be created before, or ``None`` for no bound.

    With both settings the earlier instant applies (both must hold).
    """
    bounds = []
    if before:
        bounds.append(parse_time(before, "--before"))
    if older_than_days is not None:
        if not 0 <= older_than_days < 36_500:
            raise PruneError(
                f"--older-than-days must be >= 0; got {older_than_days!r}."
            )
        bounds.append(now - dt.timedelta(days=older_than_days))
    return min(bounds) if bounds else None


def _matches(run: dict, workflow: str | None) -> bool:
    if not workflow:
        return True
    wanted = workflow.casefold()
    path = str(run.get("path") or "")
    return wanted in (
        str(run.get("name") or "").casefold(),
        path.casefold(),
        path.rsplit("/", 1)[-1].casefold(),
        str(run.get("workflow_id") or ""),
    )


def select(  # noqa: PLR0913
    runs: Iterable[dict],
    *,
    keep_latest: int,
    created_before: dt.datetime | None = None,
    workflow: str | None = None,
    exclude_ids: Iterable[int] = (),
    max_delete: int,
) -> list[dict]:
    """
    Return the runs to delete, in listing order (newest first).

    Parameters
    ----------
    runs : iterable of dict
        Runs as the API lists them, newest first. Read lazily: listing
        stops as soon as ``max_delete`` runs are selected.
    keep_latest : int
        The newest runs of the listing that are always kept.
    created_before : datetime or None, optional
        Only runs created before this instant are deleted.
    workflow : str or None, optional
        Only runs of this workflow (its name, file name, path or id).
    exclude_ids : iterable of int, optional
        Run ids never deleted (the run executing this script).
    max_delete : int
        Most runs selected.

    Raises
    ------
    PruneError
        If ``keep_latest`` is negative or ``max_delete`` is not positive.
    """
    if keep_latest < 0:
        raise PruneError(f"--keep-latest must be >= 0; got {keep_latest}.")
    if max_delete <= 0:
        raise PruneError(f"--max-delete must be > 0; got {max_delete}.")
    excluded = set(exclude_ids)
    chosen: list[dict] = []
    for position, run in enumerate(runs):
        if position < keep_latest:
            continue
        if run.get("status") != "completed" or run.get("id") in excluded:
            continue
        if not _matches(run, workflow):
            continue
        if created_before is not None:
            created = run.get("created_at")
            # A run without a creation time is kept (safe default).
            if not created or parse_time(created) >= created_before:
                continue
        chosen.append(run)
        if len(chosen) >= max_delete:
            break
    return chosen


def list_runs(request: Request, repo: str) -> Iterator[dict]:
    """Yield every run of *repo*, newest first, a page at a time."""
    page = 1
    while True:
        status, body = request(
            "GET", f"{API}/repos/{repo}/actions/runs?per_page={PAGE_SIZE}&page={page}"
        )
        if status != 200 or not isinstance(body, dict):
            raise PruneError(f"listing runs failed (HTTP {status}): {body!r}")
        batch = body.get("workflow_runs") or []
        yield from batch
        if len(batch) < PAGE_SIZE:
            return
        page += 1


def http_request(token: str) -> Request:
    """Return a :data:`Request` that calls the GitHub API with *token*."""

    def request(method: str, url: str) -> tuple[int, Any]:
        call = urllib.request.Request(  # noqa: S310 - fixed https API host
            url,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "scikit-plots-prune-workflow-runs",
            },
        )
        try:
            with urllib.request.urlopen(call, timeout=60) as response:  # noqa: S310
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw) if raw else None
            except ValueError:
                return exc.code, raw.decode("utf-8", "replace")[:200]

    return request


def _row(run: dict) -> str:
    return (
        f"{run.get('id')}  {str(run.get('created_at'))[:19]}"
        f"  {str(run.get('conclusion') or '-'):<10}  {run.get('name')}"
    )


def prune(  # noqa: PLR0913
    request: Request,
    repo: str,
    *,
    apply: bool,
    keep_latest: int,
    created_before: dt.datetime | None,
    workflow: str | None,
    exclude_ids: Iterable[int],
    max_delete: int,
    out: Any = None,
) -> tuple[int, int]:
    """
    Select and, with ``apply``, delete runs.

    Returns
    -------
    tuple of int
        ``(selected, deleted)``; ``deleted`` is 0 for a dry run.

    Raises
    ------
    PruneError
        If listing fails. A refused deletion (rate limit, permission) stops
        the loop and is reported; what was deleted stays deleted.
    """
    out = sys.stdout if out is None else out
    if not _REPO.fullmatch(repo):
        raise PruneError(f"--repo must be OWNER/NAME; got {repo!r}.")
    # Listed in full before any deletion: deleting shifts the pages.
    chosen = select(
        list_runs(request, repo),
        keep_latest=keep_latest,
        created_before=created_before,
        workflow=workflow,
        exclude_ids=exclude_ids,
        max_delete=max_delete,
    )
    mode = "delete" if apply else "would delete (dry run)"
    print(f"{mode}: {len(chosen)} run(s) of {repo}", file=out)
    deleted = 0
    for run in chosen:
        if apply:
            status, body = request(
                "DELETE", f"{API}/repos/{repo}/actions/runs/{run['id']}"
            )
            if status == 404:  # already gone: the goal is met
                status = 204
            if status != 204:
                print(
                    f"stopped: deleting run {run['id']} failed (HTTP {status}):"
                    f" {body!r}. {deleted} deleted; run again to continue.",
                    file=out,
                )
                return len(chosen), deleted
            deleted += 1
        print(_row(run), file=out)
    if len(chosen) >= max_delete:
        print(f"reached --max-delete={max_delete}; run again for more.", file=out)
    return len(chosen), deleted


def _optional_number(kind: Callable[[str], Any]) -> Callable[[str], Any]:
    def convert(text: str) -> Any:
        return None if not text.strip() else kind(text)

    return convert


def main(argv: list[str] | None = None, request: Request | None = None) -> int:
    """Run the command line; return the exit status (0 ok, 1 API, 2 usage)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--keep-latest", type=int, default=25)
    parser.add_argument("--older-than-days", type=_optional_number(float), default=None)
    parser.add_argument("--before", default="")
    parser.add_argument("--workflow", default="")
    parser.add_argument("--max-delete", type=int, default=500)
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = parser.parse_args(argv)
    try:
        if request is None:
            token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
            if not token:
                raise PruneError("set GH_TOKEN or GITHUB_TOKEN.")
            request = http_request(token)
        current = os.environ.get("GITHUB_RUN_ID", "")
        selected, deleted = prune(
            request,
            args.repo,
            apply=args.apply,
            keep_latest=args.keep_latest,
            created_before=cutoff(
                dt.datetime.now(dt.timezone.utc),
                args.before.strip() or None,
                args.older_than_days,
            ),
            workflow=args.workflow.strip() or None,
            exclude_ids=[int(current)] if current.isdigit() else [],
            max_delete=args.max_delete,
        )
    except PruneError as exc:
        print(f"prune_workflow_runs: {exc}", file=sys.stderr)
        return 2 if str(exc).startswith("--") or "set GH_TOKEN" in str(exc) else 1
    return 1 if args.apply and deleted < selected else 0


if __name__ == "__main__":
    raise SystemExit(main())
