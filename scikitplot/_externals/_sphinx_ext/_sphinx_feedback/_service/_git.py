"""
Local git provider: commit each durable event to a working tree on the server.

For sites with no hosting service and no network egress (intranet, air-gapped,
a laptop): every accepted event becomes one reviewed-by-commit JSON file in a
git working tree the operator owns. Pushing, review and publishing stay with
the operator's own tooling (a cron ``git push``, a pull request from that
branch, ``python -m ..._sphinx_feedback aggregate`` over the directory).

Notes
-----
**Layout.** ``<repository_path>/<paths.feedback>/pages/<page digest>/<feedback_id>.json``,
the same layout and the same human-readable bytes the GitHub provider writes,
so one aggregate command reads either.

**Idempotency.** The file name is the event nonce. An existing file with the
same event is a replay (committed now if a crash left it uncommitted); with
different content it is a conflict. Nothing is ever overwritten.

**Developer notes.** ``repository_path`` must be the top of a working tree,
checked with ``git rev-parse --show-toplevel``, so a typo cannot commit into
an enclosing repository. The commit names only the event file
(``git commit -- <path>``), leaving anything else staged in that tree alone.
Commits use a fixed identity and ``commit.gpgsign=false`` so a server's
global signing configuration cannot block on a passphrase; repository hooks
still run. A per-repository thread lock serializes writers in one process,
and git's own ``index.lock`` (retried for a bounded time) serializes
processes.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 - fixed argument vectors, no shell
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .._contracts import (
    FeedbackConflictError,
    feedback_event_request_hash,
    page_digest,
    parse_feedback_event,
    repository_event_bytes,
    validate_request_hash,
)
from ._config import StorageTarget
from ._github import ProviderWriteError, _event_content_matches

GIT_AUTHOR = ("sphinx-feedback", "sphinx-feedback@localhost")
_COMMAND_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 10.0
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def event_relative_path(target: StorageTarget, event: dict[str, Any]) -> str:
    """Return the repository-relative event path (shared with the GitHub layout)."""
    bucket = page_digest(event["site_id"], event["page_id"])
    return f"{target.feedback_path}/pages/{bucket}/{event['feedback']['id']}.json"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    command = [
        "git",
        "-c",
        f"user.name={GIT_AUTHOR[0]}",
        "-c",
        f"user.email={GIT_AUTHOR[1]}",
        "-c",
        "commit.gpgsign=false",
        "-C",
        str(repo),
        *args,
    ]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    deadline = time.monotonic() + _LOCK_RETRY_SECONDS
    while True:
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv
                command,
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
                env=env,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ProviderWriteError(
                "git_unavailable", "The git executable is not installed"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderWriteError(
                "git_timeout", "A git command did not finish in time"
            ) from exc
        locked = "index.lock" in (result.stderr or "")
        if result.returncode == 0 or not locked or time.monotonic() >= deadline:
            return result
        time.sleep(0.05)


def _checked(repo: Path, code: str, *args: str) -> str:
    result = _git(repo, *args)
    if result.returncode != 0:
        raise ProviderWriteError(
            code, f"git {args[0]} failed in the feedback repository"
        )
    return result.stdout


def _lock_for(repo: Path) -> threading.Lock:
    key = str(repo)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".feedback-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def put_git_event(
    target: StorageTarget, *, event: dict[str, Any], request_hash: str
) -> dict[str, Any]:
    """
    Store and commit one event; return a verified receipt.

    Parameters
    ----------
    target : StorageTarget
        A ``git`` target.
    event : dict
        Durable feedback event.
    request_hash : str
        Its request commitment.

    Returns
    -------
    dict
        ``{"status": "accepted" | "replay", "provider": "git", ...}``.

    Raises
    ------
    FeedbackConflictError
        The file exists with a different event.
    ProviderWriteError
        ``git_unavailable``, ``git_timeout``, ``git_repository_unavailable``,
        ``git_path_unsafe``, ``git_add_failed`` or ``git_commit_failed``.
    """
    event = parse_feedback_event(event)
    validate_request_hash(request_hash)
    if request_hash != feedback_event_request_hash(event):
        raise ProviderWriteError(
            "request_commitment_mismatch",
            "Git request commitment does not match the durable event",
        )
    repo = Path(target.repository_path)
    try:
        resolved_repo = repo.resolve(strict=True)
    except OSError as exc:
        raise ProviderWriteError(
            "git_repository_unavailable", "The feedback repository does not exist"
        ) from exc
    top = _checked(
        resolved_repo, "git_repository_unavailable", "rev-parse", "--show-toplevel"
    )
    if Path(top.strip()).resolve() != resolved_repo:
        raise ProviderWriteError(
            "git_repository_unavailable",
            "repository_path must be the top of a git working tree",
        )
    relative = event_relative_path(target, event)
    path = (resolved_repo / relative).resolve()
    try:
        path.relative_to(resolved_repo)
    except ValueError as exc:
        raise ProviderWriteError(
            "git_path_unsafe", "The event path leaves the feedback repository"
        ) from exc
    with _lock_for(resolved_repo):
        if path.exists():
            if not _event_content_matches(path.read_bytes(), event):
                raise FeedbackConflictError(
                    "feedback_id was already used for different feedback content"
                )
            status = "replay"
        else:
            _write_atomically(path, repository_event_bytes(event))
            status = "accepted"
        pending = _checked(
            resolved_repo, "git_add_failed", "status", "--porcelain", "--", relative
        )
        if pending.strip():
            _checked(resolved_repo, "git_add_failed", "add", "--", relative)
            _checked(
                resolved_repo,
                "git_commit_failed",
                "commit",
                "--quiet",
                "-m",
                f"feedback: {event['feedback']['id']}",
                "--",
                relative,
            )
    return {
        "status": status,
        "provider": "git",
        "feedback_id": event["feedback"]["id"],
        "request_hash": request_hash,
    }


__all__ = ["GIT_AUTHOR", "event_relative_path", "put_git_event"]
