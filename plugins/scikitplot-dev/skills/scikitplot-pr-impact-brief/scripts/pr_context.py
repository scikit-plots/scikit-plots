#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Assemble the sourced, mechanical context of one pull request for an impact brief.

Usage (from the repository's checkout; ``gh`` must be logged in)::

    python3 pr_context.py 123
    python3 pr_context.py 123 --repo OWNER/NAME --no-prior-art --out context.md

It gathers what needs no judgment: the pull request's metadata, body and
(capped) diff; the issues it links with their labels and discussion; the
change to the public Python API, from parsing the base and head versions of
each changed ``.py`` file with :mod:`ast`; and prior art, from a search of
closed pull requests and issues with the title's words. Reach, severity, the
counterfactual and the open questions stay the reviewer's work.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

#: "Fixes #12", "closes #3", "resolves owner/name#4" and issue links.
LINKED = re.compile(
    r"(?i)\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?|see|refs?)\s*:?\s+"
    r"(?:[\w.-]+/[\w.-]+)?#(\d+)|/issues/(\d+)\b"
)
STOP = frozenset("a an and the of to in for on with by from is are be fix add use when".split())


def _run(args: List[str], check: bool = True) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {result.stderr.strip()}")
    return result.stdout if result.returncode == 0 else ""


def linked_issues(text: str) -> List[int]:
    """Return the issue numbers *text* links, in order, without repeats."""
    seen: List[int] = []
    for match in LINKED.finditer(text or ""):
        number = int(match.group(1) or match.group(2))
        if number not in seen:
            seen.append(number)
    return seen


def _signature(node: ast.AST) -> str:
    """Return a stable text for a function's parameters (Python 3.8 has no ast.unparse)."""
    a = node.args  # type: ignore[attr-defined]
    parts = [p.arg for p in getattr(a, "posonlyargs", [])] + ["/"] if getattr(a, "posonlyargs", []) else []
    parts += [p.arg for p in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def public_api(source: str) -> Dict[str, str]:
    """
    Return the public names a module defines, with their parameter lists.

    Parameters
    ----------
    source : str
        Python source of one module; empty for a file that does not exist.

    Returns
    -------
    dict
        Name -> parameters for functions, ``"class"`` for classes (and
        ``"Class.method"`` -> parameters for their public methods), ``"name"``
        for other assignments. Public is ``__all__`` when the module declares
        it as a literal, otherwise every name without a leading underscore.
    """
    if not source.strip():
        return {}
    tree = ast.parse(source)
    declared: Optional[set] = None
    found: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            found[node.name] = "class"
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not item.name.startswith("_") or item.name == "__init__"
                ):
                    found[f"{node.name}.{item.name}"] = _signature(item)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        declared = {e.value for e in node.value.elts
                                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
                    else:
                        found[target.id] = "name"
    def public(name: str) -> bool:
        head = name.split(".", 1)[0]
        return head in declared if declared is not None else not head.startswith("_")
    return {k: v for k, v in found.items() if public(k)}


def api_delta(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, List[str]]:
    """Return the names added, removed and changed between two public APIs."""
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(n for n in set(before) & set(after) if before[n] != after[n]),
    }


def _show(ref: str, path: str) -> str:
    return _run(["git", "show", f"{ref}:{path}"], check=False)


def context(number: int, repo: Optional[str], diff_lines: int, prior_art: bool) -> dict:
    """Return the context bundle of pull request *number*."""
    repo_args = ["--repo", repo] if repo else []
    fields = "number,title,url,author,state,isDraft,createdAt,baseRefName,headRefOid,labels,body,files,additions,deletions"
    pr = json.loads(_run(["gh", "pr", "view", str(number), *repo_args, "--json", fields]))
    diff = _run(["gh", "pr", "diff", str(number), *repo_args]).splitlines()
    issues = []
    for issue in linked_issues(pr.get("body") or ""):
        text = _run(["gh", "issue", "view", str(issue), *repo_args, "--json",
                     "number,title,url,state,labels,body,comments"], check=False)
        if text:
            issues.append(json.loads(text))
    # Base and head sources, from the checkout (the head is fetched read-only).
    _run(["git", "fetch", "--quiet", "origin", f"pull/{number}/head"], check=False)
    base = f"origin/{pr['baseRefName']}"
    deltas: Dict[str, Dict[str, List[str]]] = {}
    for f in pr.get("files", []):
        path = f["path"]
        if path.endswith(".py") and "/tests/" not in path and not pathlib.PurePath(path).name.startswith("test_"):
            try:
                delta = api_delta(public_api(_show(base, path)), public_api(_show(pr["headRefOid"], path)))
            except SyntaxError as exc:
                delta = {"added": [], "removed": [], "changed": [f"(could not parse: {exc.msg})"]}
            if any(delta.values()):
                deltas[path] = delta
    prior: List[dict] = []
    if prior_art:
        words = [w for w in re.findall(r"[a-z][a-z0-9_]+", pr["title"].lower()) if w not in STOP][:5]
        if words:
            query = " ".join(words)
            for kind in ("prs", "issues"):
                text = _run(["gh", "search", kind, query, *(["--repo", repo] if repo else []),
                             "--limit", "10", "--json", "number,title,url,state"], check=False)
                prior += [dict(item, kind=kind) for item in json.loads(text or "[]")
                          if item.get("number") != number]
    return {"pr": pr, "diff": diff[:diff_lines], "diff_truncated": len(diff) > diff_lines,
            "issues": issues, "api_delta": deltas, "prior_art": prior}


def render(bundle: dict) -> str:
    """Return the bundle as Markdown, every item with its source."""
    pr = bundle["pr"]
    out = [f"# Context: PR #{pr['number']}, {pr['title']}", "", pr["url"], "",
           f"Author {pr['author']['login']}, opened {pr['createdAt'][:10]}, {pr['state'].lower()}"
           f"{', draft' if pr.get('isDraft') else ''}; +{pr['additions']} -{pr['deletions']} in"
           f" {len(pr.get('files', []))} files; labels: {', '.join(l['name'] for l in pr['labels']) or 'none'}.",
           "", "## Description (the author's claims, to check)", "", pr.get("body") or "(empty)", "",
           "## Linked issues", ""]
    for issue in bundle["issues"]:
        labels = ", ".join(l["name"] for l in issue.get("labels", [])) or "none"
        out += [f"- #{issue['number']} {issue['title']} ({issue['state'].lower()}; labels: {labels};"
                f" {len(issue.get('comments', []))} comments) {issue['url']}"]
    if not bundle["issues"]:
        out.append("None linked.")
    out += ["", "## Public API change (from the code)", ""]
    for path, delta in bundle["api_delta"].items():
        for kind in ("added", "removed", "changed"):
            if delta[kind]:
                out.append(f"- `{path}` {kind}: {', '.join(delta[kind])}")
    if not bundle["api_delta"]:
        out.append("No public name added, removed or changed in non-test Python files.")
    out += ["", "## Prior art (search by title words; read before citing)", ""]
    out += [f"- {p['kind'][:-1]} #{p['number']} {p['title']} ({p['state'].lower()}) {p['url']}"
            for p in bundle["prior_art"]] or ["None found, or the search was skipped."]
    out += ["", "## Diff" + (" (truncated)" if bundle["diff_truncated"] else ""), "", "```diff",
            *bundle["diff"], "```", ""]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("number", type=int)
    parser.add_argument("--repo", help="OWNER/NAME (default: the checkout's repository)")
    parser.add_argument("--diff-lines", type=int, default=2000)
    parser.add_argument("--no-prior-art", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)
    bundle = context(args.number, args.repo, args.diff_lines, not args.no_prior_art)
    text = json.dumps(bundle, indent=1) if args.json else render(bundle)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
