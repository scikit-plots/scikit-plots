#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Place approved comments on a pull request as a pending review, never submitted.

Usage (from any directory; ``gh`` must be logged in)::

    python3 post_pending_review.py --repo OWNER/NAME --pr N comments.json --dry-run
    python3 post_pending_review.py --repo OWNER/NAME --pr N comments.json

``comments.json`` is a list of objects with ``path``, ``line``, ``body`` and,
optionally, ``start_line``, ``side`` (``RIGHT``, the default, or ``LEFT``)
and ``start_side``. Line numbers are file line numbers, not diff positions.

Every comment is checked against the pull request's diff first; if one cannot
be placed, nothing is posted (exit 1). A comment identical to one already on
the pull request (same file, side, line and text), or already in your own
pending review (same file and text: GitHub reports no line for a pending
comment), is skipped, so a rerun after a partial failure posts nothing twice.
Without a pending review of yours, one is created through REST with no
``event``; otherwise the comments are added to it through GraphQL, one at a
time, and a failure stops with the list of what was added. The review is
never submitted.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

#: ``@@ -a,b +c,d @@`` (a count of 1 may be omitted).
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
SIDES = ("RIGHT", "LEFT")
#: (path, side, line) -> index of the hunk that holds it.
LineIndex = Dict[Tuple[str, str, int], int]


def hunk_lines(path: str, patch: str) -> LineIndex:
    """
    Return the commentable lines of one file's patch.

    Parameters
    ----------
    path : str
        File path in the repository.
    patch : str
        The file's unified diff, as GitHub returns it (``patch`` of a file).

    Returns
    -------
    dict
        ``(path, side, line) -> hunk index``. A context line is commentable
        on both sides, an added line on ``RIGHT``, a removed line on ``LEFT``.
    """
    index: LineIndex = {}
    old = new = 0
    hunk = -1
    for text in patch.split("\n"):
        header = HUNK.match(text)
        if header:
            hunk += 1
            old, new = int(header.group(1)), int(header.group(3))
            continue
        if hunk < 0 or text.startswith("\\"):
            continue
        if text.startswith("+"):
            index[(path, "RIGHT", new)] = hunk
            new += 1
        elif text.startswith("-"):
            index[(path, "LEFT", old)] = hunk
            old += 1
        else:
            index[(path, "RIGHT", new)] = hunk
            index[(path, "LEFT", old)] = hunk
            old += 1
            new += 1
    return index


def normalize(raw: dict) -> dict:
    """
    Return a comment with defaults applied and types checked.

    Raises
    ------
    ValueError
        When a field is missing or has the wrong type or value.
    """
    if not isinstance(raw, dict):
        raise ValueError("each comment is an object")
    path, line, body = raw.get("path"), raw.get("line"), raw.get("body")
    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise ValueError("'line' must be a positive integer")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("'body' must be non-empty text")
    side = raw.get("side", "RIGHT")
    if side not in SIDES:
        raise ValueError("'side' must be RIGHT or LEFT")
    comment = {"path": path, "line": line, "side": side, "body": body}
    if raw.get("start_line") is not None:
        start = raw["start_line"]
        if not isinstance(start, int) or isinstance(start, bool) or not 1 <= start <= line:
            raise ValueError("'start_line' must be an integer between 1 and 'line'")
        start_side = raw.get("start_side", side)
        if start_side not in SIDES:
            raise ValueError("'start_side' must be RIGHT or LEFT")
        if start != line or start_side != side:
            comment["start_line"] = start
            comment["start_side"] = start_side
    return comment


def check(comment: dict, index: LineIndex) -> Optional[str]:
    """Return why *comment* cannot be placed, or ``None`` when it can."""
    end = (comment["path"], comment["side"], comment["line"])
    if end not in index:
        return f"line {comment['line']} ({comment['side']}) is not in the diff of {comment['path']}"
    if "start_line" in comment:
        start = (comment["path"], comment["start_side"], comment["start_line"])
        if start not in index:
            return f"start line {comment['start_line']} is not in the diff of {comment['path']}"
        if index[start] != index[end]:
            return "a range must stay inside one diff hunk"
    return None


def key(comment: dict) -> Tuple[str, str, int, str]:
    """Return what makes two comments the same: file, side, line and text."""
    return (comment["path"], comment.get("side") or "RIGHT", int(comment["line"]), comment["body"].strip())


def _gh(args: List[str], payload: Optional[str] = None) -> str:
    result = subprocess.run(
        ["gh", *args], input=payload, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {result.stderr.strip()}")
    return result.stdout


def _json_lines(text: str) -> List[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def diff_index(repo: str, pr: int) -> LineIndex:
    """Return the commentable lines of every file of the pull request."""
    index: LineIndex = {}
    files = _gh(["api", "--paginate", f"repos/{repo}/pulls/{pr}/files",
                 "--jq", ".[] | {filename, patch}"])
    for item in _json_lines(files):
        index.update(hunk_lines(item["filename"], item.get("patch") or ""))
    return index


def existing(repo: str, pr: int) -> Set[Tuple[str, str, int, str]]:
    """Return the keys of the submitted review comments on the pull request."""
    text = _gh(["api", "--paginate", f"repos/{repo}/pulls/{pr}/comments",
                "--jq", ".[] | {path, line: (.line // .original_line), side, body}"])
    return {key(c) for c in _json_lines(text) if c.get("line")}


def pending_review(repo: str, pr: int) -> Optional[dict]:
    """
    Return the user's own pending review on the pull request, or ``None``.

    Returns
    -------
    dict or None
        ``{"id": <REST id>, "node_id": <GraphQL id>, "comments": [(path,
        body), ...]}``. Only the author sees a pending review, so at most one
        is returned.
    """
    login = _gh(["api", "user", "--jq", ".login"]).strip()
    text = _gh(["api", "--paginate", f"repos/{repo}/pulls/{pr}/reviews",
                "--jq", ".[] | {id, node_id, state, login: .user.login}"])
    for review in _json_lines(text):
        if review["state"] == "PENDING" and review["login"] == login:
            body = _gh(["api", "--paginate", f"repos/{repo}/pulls/{pr}/reviews/{review['id']}/comments",
                        "--jq", ".[] | {path, body}"])
            review["comments"] = [(c["path"], c["body"].strip()) for c in _json_lines(body)]
            return review
    return None


def to_post(comments: List[dict], submitted: Set[Tuple[str, str, int, str]],
            pending: Optional[dict]) -> List[dict]:
    """
    Return the comments that are on the pull request in neither form.

    A submitted comment matches on file, side, line and text (:func:`key`);
    a pending one on file and text, because GitHub reports no line for a
    comment while its review is pending.
    """
    drafts = set((pending or {}).get("comments", []))
    return [c for c in comments
            if key(c) not in submitted and (c["path"], c["body"].strip()) not in drafts]


def post(repo: str, pr: int, comments: List[dict], pending: Optional[dict]) -> str:
    """Post *comments* as, or into, the user's pending review; return a summary."""
    if pending is None:
        head = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "headRefOid",
                    "--jq", ".headRefOid"]).strip()
        payload = json.dumps({"commit_id": head, "comments": comments})
        out = json.loads(_gh(["api", "--method", "POST", f"repos/{repo}/pulls/{pr}/reviews",
                              "--input", "-"], payload))
        if out.get("state") != "PENDING":
            raise RuntimeError(f"expected a PENDING review, got {out.get('state')!r}")
        return f"created pending review {out.get('id')} with {len(comments)} comment(s)"
    mutation = (
        "mutation($review: ID!, $path: String!, $line: Int!, $side: DiffSide!,"
        " $startLine: Int, $startSide: DiffSide, $body: String!) {"
        " addPullRequestReviewThread(input: {pullRequestReviewId: $review, path: $path,"
        " line: $line, side: $side, startLine: $startLine, startSide: $startSide,"
        " body: $body}) { thread { id } } }"
    )
    review = pending["node_id"]
    added: List[str] = []
    for c in comments:
        args = ["api", "graphql", "-F", f"review={review}", "-F", f"path={c['path']}",
                "-F", f"line={c['line']}", "-F", f"side={c['side']}",
                "-F", f"body={c['body']}", "-f", f"query={mutation}"]
        if "start_line" in c:
            args[2:2] = ["-F", f"startLine={c['start_line']}", "-F", f"startSide={c['start_side']}"]
        try:
            _gh(args)
        except RuntimeError as exc:
            done = "\n  ".join(added) or "none"
            raise RuntimeError(f"{exc}\nadded before the failure:\n  {done}\n"
                               "rerun to post the rest; added comments are skipped") from None
        added.append(f"{c['path']}:{c['line']}")
    return f"added {len(comments)} comment(s) to your pending review {review}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("comments", help="JSON file of comments, or - for standard input")
    parser.add_argument("--dry-run", action="store_true", help="check and report; post nothing")
    args = parser.parse_args(argv)
    if "/" not in args.repo:
        parser.error("--repo must be OWNER/NAME")
    text = sys.stdin.read() if args.comments == "-" else open(args.comments, encoding="utf-8").read()
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        parser.error("the comments file must hold a non-empty list")
    problems: List[str] = []
    numbered: List[Tuple[int, dict]] = []
    for number, item in enumerate(raw, 1):
        try:
            numbered.append((number, normalize(item)))
        except ValueError as exc:
            problems.append(f"#{number}: {exc}")
    comments = [c for _, c in numbered]
    index = diff_index(args.repo, args.pr)
    for number, comment in numbered:
        reason = check(comment, index)
        status = f"INVALID: {reason}" if reason else "ok"
        print(f"#{number} {comment['path']}:{comment['line']} {comment['side']}: {status}")
        if reason:
            problems.append(f"#{number}: {reason}")
    if problems:
        print("nothing posted:", *sorted(problems, key=lambda s: int(s[1:].split(":", 1)[0])), sep="\n  ", file=sys.stderr)
        return 1
    pending = pending_review(args.repo, args.pr)
    new = to_post(comments, existing(args.repo, args.pr), pending)
    where = f"; you have pending review {pending['id']}" if pending else ""
    print(f"{len(comments) - len(new)} already on the pull request, {len(new)} to post{where}")
    if args.dry_run or not new:
        return 0
    try:
        print(post(args.repo, args.pr, new, pending))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Open the pull request's 'Files changed' tab to edit and submit it yourself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
