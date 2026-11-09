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
the pull request (same file, side, line and text) is skipped, so a rerun
posts nothing twice. Without an existing pending review of yours, one is
created through REST with no ``event``; otherwise the comments are added to
it through GraphQL. The review is never submitted.
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
    """Return the keys of the review comments already on the pull request."""
    text = _gh(["api", "--paginate", f"repos/{repo}/pulls/{pr}/comments",
                "--jq", ".[] | {path, line: (.line // .original_line), side, body}"])
    return {key(c) for c in _json_lines(text) if c.get("line")}


def pending_review_id(repo: str, pr: int) -> Optional[str]:
    """Return the node id of the user's pending review on the pull request."""
    owner, name = repo.split("/", 1)
    query = (
        "query($owner: String!, $name: String!, $number: Int!) {"
        " viewer { login }"
        " repository(owner: $owner, name: $name) { pullRequest(number: $number) {"
        " reviews(states: PENDING, first: 50) { nodes { id author { login } } } } } }"
    )
    data = json.loads(_gh(["api", "graphql", "-F", f"owner={owner}", "-F", f"name={name}",
                           "-F", f"number={pr}", "-f", f"query={query}"]))["data"]
    login = data["viewer"]["login"]
    for node in data["repository"]["pullRequest"]["reviews"]["nodes"]:
        if (node.get("author") or {}).get("login") == login:
            return node["id"]
    return None


def post(repo: str, pr: int, comments: List[dict]) -> str:
    """Post *comments* as, or into, the user's pending review; return a summary."""
    review = pending_review_id(repo, pr)
    if review is None:
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
    for c in comments:
        args = ["api", "graphql", "-F", f"review={review}", "-F", f"path={c['path']}",
                "-F", f"line={c['line']}", "-F", f"side={c['side']}",
                "-F", f"body={c['body']}", "-f", f"query={mutation}"]
        if "start_line" in c:
            args[2:2] = ["-F", f"startLine={c['start_line']}", "-F", f"startSide={c['start_side']}"]
        _gh(args)
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
    problems, comments = [], []
    for number, item in enumerate(raw, 1):
        try:
            comments.append(normalize(item))
        except ValueError as exc:
            problems.append(f"#{number}: {exc}")
    index = diff_index(args.repo, args.pr)
    for number, comment in enumerate(comments, 1):
        reason = check(comment, index)
        status = f"INVALID: {reason}" if reason else "ok"
        print(f"#{number} {comment['path']}:{comment['line']} {comment['side']}: {status}")
        if reason:
            problems.append(f"#{number}: {reason}")
    if problems:
        print("nothing posted:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    seen = existing(args.repo, args.pr)
    new = [c for c in comments if key(c) not in seen]
    print(f"{len(comments) - len(new)} already on the pull request, {len(new)} to post")
    if args.dry_run or not new:
        return 0
    print(post(args.repo, args.pr, new))
    print("Open the pull request's 'Files changed' tab to edit and submit it yourself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
