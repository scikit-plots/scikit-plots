#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Keep a cache of a repository's open pull requests and draw the next one to brief.

Usage (``gh`` must be logged in)::

    python3 worklist.py --repo OWNER/NAME fetch            # refresh when older than a day
    python3 worklist.py --repo OWNER/NAME list --counts    # how the filter narrows the list
    python3 worklist.py --repo OWNER/NAME pick --briefs DIR --seed 7

The worklist is the open, non-draft pull requests that touch ``--paths`` and
that no maintainer other than the author has commented on or reviewed: the
ones whose worth nobody has weighed yet. A maintainer is an author
association of ``OWNER``, ``MEMBER`` or ``COLLABORATOR``. A pull request with
a ``Needs ...`` label is left out: its question is already recorded as open.
A brief is a file in ``--briefs`` whose name starts with ``<number>-``;
briefed pull requests are skipped unless ``--include-briefed`` is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import subprocess
import sys
from typing import Iterable, List, Optional

MAINTAINERS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
#: 25 per page: larger pages with nested comments and files exceed GitHub's
#: node limit, which answers with an empty body rather than an error.
PAGE = 25
QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: %d, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url isDraft createdAt updatedAt
        author { login } authorAssociation
        labels(first: 30) { nodes { name } }
        files(first: 100) { nodes { path } }
        comments(first: 100) { nodes { author { login } authorAssociation } }
        reviews(first: 100) { nodes { author { login } authorAssociation } }
      }
    }
  }
}
""" % PAGE


def default_cache(repo: str) -> pathlib.Path:
    """Return the cache path for *repo*, outside any checkout."""
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache")
    return base / "scikitplot-pr-impact-brief" / (repo.replace("/", "-") + ".json")


def _gh(args: List[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh failed: {result.stderr.strip()}")
    return result.stdout


def _names(connection: Optional[dict], field: str) -> List[str]:
    return [node[field] for node in (connection or {}).get("nodes", []) if node]


def flatten(node: dict) -> dict:
    """Return one cached pull request from a GraphQL node."""
    author = (node.get("author") or {}).get("login", "")
    engaged = [
        n for kind in ("comments", "reviews")
        for n in (node.get(kind) or {}).get("nodes", [])
        if n and n.get("authorAssociation") in MAINTAINERS
        and (n.get("author") or {}).get("login") != author
    ]
    return {
        "number": node["number"], "title": node["title"], "url": node["url"],
        "draft": bool(node["isDraft"]), "created": node["createdAt"], "updated": node["updatedAt"],
        "author": author, "association": node.get("authorAssociation", ""),
        "labels": _names(node.get("labels"), "name"),
        "files": _names(node.get("files"), "path"),
        "maintainer_engaged": bool(engaged),
    }


def fetch(repo: str, cache: pathlib.Path, max_age_hours: float, force: bool) -> List[dict]:
    """Return the cached open pull requests, refetching when stale."""
    now = dt.datetime.now(dt.timezone.utc)
    if cache.is_file() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        age = now - dt.datetime.fromisoformat(data["fetched"])
        if age.total_seconds() < max_age_hours * 3600:
            print(f"cache fresh ({age.total_seconds() / 3600:.1f} h old): {cache}", file=sys.stderr)
            return data["pulls"]
    owner, name = repo.split("/", 1)
    pulls, after = [], None
    while True:
        args = ["api", "graphql", "-F", f"owner={owner}", "-F", f"name={name}", "-f", f"query={QUERY}"]
        if after:
            args[2:2] = ["-F", f"after={after}"]
        page = json.loads(_gh(args))["data"]["repository"]["pullRequests"]
        pulls.extend(flatten(n) for n in page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"repo": repo, "fetched": now.isoformat(), "pulls": pulls}, indent=1),
                     encoding="utf-8")
    print(f"fetched {len(pulls)} open pull requests into {cache}", file=sys.stderr)
    return pulls


def briefed(folder: Optional[pathlib.Path]) -> set:
    """Return the numbers that already have a brief in *folder*."""
    if folder is None or not folder.is_dir():
        return set()
    return {int(p.name.split("-", 1)[0]) for p in folder.iterdir()
            if p.name.split("-", 1)[0].isdigit()}


def worklist(pulls: Iterable[dict], paths: List[str], done: set, include_briefed: bool) -> dict:
    """
    Apply the filter in stages and return each stage's pull requests.

    Returns
    -------
    dict
        Stage name -> list of pull requests, in order: ``open``,
        ``not draft``, ``touches paths``, ``no open decision``,
        ``no maintainer yet``, ``not briefed``. The last is the worklist.
    """
    stages = {"open": list(pulls)}
    stages["not draft"] = [p for p in stages["open"] if not p["draft"]]
    stages["touches paths"] = [p for p in stages["not draft"]
                               if any(f.startswith(tuple(paths)) for f in p["files"])]
    stages["no open decision"] = [p for p in stages["touches paths"]
                                  if not any(l.startswith("Needs") for l in p["labels"])]
    stages["no maintainer yet"] = [p for p in stages["no open decision"] if not p["maintainer_engaged"]]
    stages["not briefed"] = [p for p in stages["no maintainer yet"]
                             if include_briefed or p["number"] not in done]
    return stages


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--cache", type=pathlib.Path, help="cache file (default under ~/.cache)")
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--paths", nargs="+", default=["scikitplot/"],
                        help="path prefixes that count as the project's source")
    parser.add_argument("--briefs", type=pathlib.Path, help="folder of written briefs")
    parser.add_argument("--include-briefed", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--force", action="store_true")
    l = sub.add_parser("list")
    l.add_argument("--counts", action="store_true", help="print how each stage narrows the list")
    l.add_argument("--json", action="store_true")
    p = sub.add_parser("pick")
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--seed", type=int, help="make the draw reproducible")
    args = parser.parse_args(argv)
    if "/" not in args.repo:
        parser.error("--repo must be OWNER/NAME")
    cache = args.cache or default_cache(args.repo)
    pulls = fetch(args.repo, cache, args.max_age_hours, getattr(args, "force", False))
    if args.command == "fetch":
        return 0
    stages = worklist(pulls, args.paths, briefed(args.briefs), args.include_briefed)
    work = stages["not briefed"]
    if args.command == "list":
        if args.counts:
            for stage, items in stages.items():
                print(f"{len(items):6d}  {stage}")
        elif args.json:
            print(json.dumps(work, indent=1))
        else:
            for item in work:
                print(f"#{item['number']:<6d} {item['created'][:10]}  {item['title']}")
        return 0
    chosen = random.Random(args.seed).sample(work, min(args.n, len(work)))
    for item in chosen:
        print(f"#{item['number']} {item['title']}\n  {item['url']}")
    if not chosen:
        print("the worklist is empty", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
