#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Assemble the sourced, mechanical context of one pull request for an impact brief.

Usage (from any directory; ``gh`` must be logged in)::

    python3 pr_context.py 123 --repo OWNER/NAME
    python3 pr_context.py 123 --repo OWNER/NAME --no-prior-art --out context.md

It gathers what needs no judgment: the pull request's metadata, body and
(capped) diff; the issues of the same repository it links, with their labels
and discussion; the change to the public Python API, from parsing each
changed ``.py`` file with :mod:`ast` at the merge base and at the head; and
prior art, from a search of pull requests and issues with the title's words.
Reach, severity, the counterfactual and the open questions stay the
reviewer's work.

Everything is read through GitHub's REST API, so the script works from a
fork's checkout or from no checkout, and changes no file, branch or ref. A
part that cannot be read is reported as such, never as "no change".
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional
from urllib.parse import quote

#: "Fixes #12", "closes #3", "see owner/name#4", and issue links.
LINKED = re.compile(
    r"(?i)\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?|see|refs?)\s*:?\s+"
    r"(?:([\w.-]+/[\w.-]+))?#(\d+)"
    r"|github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)\b"
)
STOP = frozenset("a an and the of to in for on with by from is are be fix add use when".split())


def _gh(args: List[str], check: bool = True) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if check:
            raise RuntimeError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()}")
        return ""
    return result.stdout


def linked_issues(text: str, repo: Optional[str] = None) -> List[int]:
    """
    Return the issue numbers of *repo* that *text* links, in order, without repeats.

    A reference to another repository (``other/name#4`` or its link) is left
    out: its number means a different issue here. Without *repo*, only
    unqualified references (``#4``) count.
    """
    seen: List[int] = []
    for match in LINKED.finditer(text or ""):
        where = match.group(1) or match.group(3)
        number = int(match.group(2) or match.group(4))
        if where is not None and (repo is None or where.lower() != repo.lower()):
            continue
        if number not in seen:
            seen.append(number)
    return seen


def _default(node: Optional[ast.AST]) -> str:
    """Return a short, stable text for a default value."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        return "=" + repr(node.value)
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return "=" + ".".join([node.id, *reversed(parts)])
    return "=..."


def _signature(node: ast.AST) -> str:
    """
    Return a stable text for a function's parameters, defaults included.

    Python 3.8 has no :func:`ast.unparse`; a default that is not a literal or
    a dotted name shows as ``=...``, so a change inside such an expression is
    not seen.
    """
    a = node.args  # type: ignore[attr-defined]
    positional = list(getattr(a, "posonlyargs", [])) + list(a.args)
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    parts = [p.arg + _default(d) for p, d in zip(positional, defaults)]
    if getattr(a, "posonlyargs", []):
        parts.insert(len(a.posonlyargs), "/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [p.arg + _default(d) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
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


def _source(repo: str, path: str, ref: str) -> str:
    """Return *path* at *ref*; raise ``RuntimeError`` when it cannot be read."""
    return _gh(["api", "-H", "Accept: application/vnd.github.raw",
                f"repos/{repo}/contents/{quote(path)}?ref={ref}"])


def _is_test(path: str) -> bool:
    name = pathlib.PurePosixPath(path).name
    return "/tests/" in f"/{path}" or name.startswith("test_") or name == "conftest.py"


def file_delta(repo: str, item: dict, base: str, head: str) -> Dict[str, List[str]]:
    """
    Return the public-API change of one changed file of the pull request.

    *item* is a REST file entry (``filename``, ``status``,
    ``previous_filename``). An added file has no base and a removed one no
    head; a renamed file is read under its old name at the base. A file
    that cannot be read or parsed gives ``{"error": [reason]}``.
    """
    status = item.get("status", "modified")
    path = item["filename"]
    old = item.get("previous_filename") or path
    try:
        before = "" if status == "added" else _source(repo, old, base)
        after = "" if status == "removed" else _source(repo, path, head)
        return api_delta(public_api(before), public_api(after))
    except (RuntimeError, SyntaxError, ValueError) as exc:
        return {"error": [str(getattr(exc, "msg", exc)).splitlines()[0][:200]]}


def context(number: int, repo: str, diff_lines: int, prior_art: bool) -> dict:
    """Return the context bundle of pull request *number* of *repo*."""
    pr = json.loads(_gh(["api", f"repos/{repo}/pulls/{number}"]))
    files = [json.loads(line) for line in _gh([
        "api", "--paginate", f"repos/{repo}/pulls/{number}/files",
        "--jq", ".[] | {filename, status, previous_filename, additions, deletions, patch}",
    ]).splitlines() if line.strip()]
    diff: List[str] = []
    for f in files:
        diff += [f"--- {f.get('previous_filename') or f['filename']}", f"+++ {f['filename']}"]
        diff += (f.get("patch") or "(no patch: binary or too large)").splitlines()
    issues = []
    for issue in linked_issues(pr.get("body") or "", repo):
        text = _gh(["api", f"repos/{repo}/issues/{issue}"], check=False)
        if text:
            data = json.loads(text)
            issues.append(dict({k: data.get(k) for k in ("number", "title", "html_url", "state", "comments")},
                               labels=[l["name"] for l in data.get("labels", [])],
                               pull_request="pull_request" in data))
        else:
            issues.append({"number": issue, "error": "could not be read"})
    head = pr["head"]["sha"]
    compare = _gh(["api", f"repos/{repo}/compare/{pr['base']['sha']}...{head}",
                   "--jq", ".merge_base_commit.sha"], check=False).strip()
    deltas: Dict[str, Dict[str, List[str]]] = {}
    for f in files:
        if not f["filename"].endswith(".py") or _is_test(f["filename"]):
            continue
        if not compare:
            deltas[f["filename"]] = {"error": ["merge base not found"]}
            continue
        delta = file_delta(repo, f, compare, head)
        if any(delta.values()):
            deltas[f["filename"]] = delta
    prior: List[dict] = []
    if prior_art:
        words = [w for w in re.findall(r"[a-z][a-z0-9_]+", pr["title"].lower()) if w not in STOP][:5]
        if words:
            for kind in ("is:pr", "is:issue"):
                try:
                    text = _gh(["api", "-X", "GET", "search/issues", "-f",
                                f"q={' '.join(words)} repo:{repo} {kind}", "-f", "per_page=10", "--jq",
                                ".items[] | {number, title, html_url,"
                                " state: (if .pull_request.merged_at then \"merged\" else .state end),"
                                " pull_request: (.pull_request != null)}"])
                except RuntimeError as exc:
                    prior.append({"error": f"search {kind} failed: {str(exc).splitlines()[0][:200]}"})
                    continue
                prior += [item for item in (json.loads(line) for line in text.splitlines() if line.strip())
                          if item["number"] != number]
    return {"pr": pr, "files": len(files), "merge_base": compare or None,
            "diff": diff[:diff_lines], "diff_truncated": len(diff) > diff_lines,
            "issues": issues, "api_delta": deltas, "prior_art": prior,
            "prior_art_searched": bool(prior_art)}


def render(bundle: dict) -> str:
    """Return the bundle as Markdown, every item with its source."""
    pr = bundle["pr"]
    labels = ", ".join(l["name"] for l in pr.get("labels", [])) or "none"
    out = [f"# Context: PR #{pr['number']}, {pr['title']}", "", pr["html_url"], "",
           f"Author {pr['user']['login']} ({pr.get('author_association', '').lower()}),"
           f" opened {pr['created_at'][:10]}, {pr['state']}{', draft' if pr.get('draft') else ''};"
           f" +{pr['additions']} -{pr['deletions']} in {bundle['files']} files; labels: {labels}.",
           "", "## Description (the author's claims, to check)", "", pr.get("body") or "(empty)", "",
           "## Linked issues", ""]
    for issue in bundle["issues"]:
        if "error" in issue:
            out.append(f"- #{issue['number']}: {issue['error']}")
            continue
        kind = "pull request" if issue["pull_request"] else "issue"
        out.append(f"- {kind} #{issue['number']} {issue['title']} ({issue['state']};"
                   f" labels: {', '.join(issue['labels']) or 'none'}; {issue['comments']} comments)"
                   f" {issue['html_url']}")
    if not bundle["issues"]:
        out.append("None linked in this repository.")
    base = bundle["merge_base"] or "unknown"
    out += ["", f"## Public API change (merge base {base[:12]} to head {pr['head']['sha'][:12]})", ""]
    for path, delta in bundle["api_delta"].items():
        for kind in ("error", "added", "removed", "changed"):
            if delta.get(kind):
                label = "could not compute" if kind == "error" else kind
                out.append(f"- `{path}` {label}: {', '.join(delta[kind])}")
    if not bundle["api_delta"]:
        out.append("No public name added, removed or changed in non-test Python files.")
    out += ["", "## Prior art (search by title words; read before citing)", ""]
    out += [f"- {p['error']}" if "error" in p else
            f"- {'pull request' if p['pull_request'] else 'issue'} #{p['number']} {p['title']}"
            f" ({p['state']}) {p['html_url']}" for p in bundle["prior_art"]] or [
        "None found." if bundle["prior_art_searched"] else "Search skipped."]
    out += ["", "## Diff" + (" (truncated)" if bundle["diff_truncated"] else ""), "", "```diff",
            *bundle["diff"], "```", ""]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("number", type=int)
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--diff-lines", type=int, default=2000)
    parser.add_argument("--no-prior-art", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)
    if "/" not in args.repo:
        parser.error("--repo must be OWNER/NAME")
    if args.diff_lines < 0:
        parser.error("--diff-lines must be 0 or more")
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
