#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Security review of a change: the S1 to S8 table, ready to paste.

Reads a unified diff (``git diff``), looks at the lines it adds and the
files it touches, and prints one Markdown table row per check with each
hit as ``file:line [kind]``:

S1  secrets, tokens, private keys, session links, per-user noreply addresses
S2  dependency files changed; exact ``==`` pins added
S3  code that runs other code (subprocess, eval/exec, pickle, dynamic import)
S4  network access in code
S5  file system writes and archive extraction (path handling to check)
S6  untrusted-input parsers (XML, archives, YAML, pickle-like formats)
S7  CI, release and build files; unpinned actions, write permissions,
    ``pull_request_target``, ``${{ github.event... }}`` in a workflow
S8  weakened tests: test files deleted, test functions removed, skip/xfail

A hit is a place to look, not a verdict: the table says where, the
reviewer decides. A matched secret is never printed, only its place and
kind. A hit in documentation (``.md``, ``.rst``, ``.txt``) is labelled
``doc``. Running it twice on the same diff prints the same table.

Exit status: 0, or 1 when S1 finds a credential pattern (a private key or
a token format) anywhere, so a leaked secret fails the job; 2 on bad
usage. Session links and noreply addresses are reported, never failed.

Usage::

    python tools/maint_tools/security_review.py --base origin/main
    git diff -U0 origin/main | python tools/maint_tools/security_review.py -

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator

#: (check, title, needs the maintainer's explicit approval when not none).
CHECKS = (
    ("S1", "Secrets, keys, personal data, session links", False),
    ("S2", "Dependencies", False),
    ("S3", "Code that runs other code", True),
    ("S4", "Network access", True),
    ("S5", "File system writes, archive extraction", False),
    ("S6", "Untrusted input parsing", False),
    ("S7", "CI, release and build files", True),
    ("S8", "Tests weakened", True),
)

_DOC = re.compile(r"\.(md|rst|txt)$|(^|/)(docs?|LICENSES?)/", re.IGNORECASE)
_CODE = re.compile(r"\.(py|pyx|pxd|pyi|sh|bash|js|ts|c|cc|cpp|h|hpp)$")
_TEST = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
_WORKFLOW = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")
_CI_PATHS = re.compile(
    r"^(\.github/|\.circleci/|tools/|scikitplot/_build_utils/|plugins/|"
    r"(.*/)?meson\.build$|meson\.options$|pyproject\.toml$|setup\.(py|cfg)$|"
    r"(.*/)?Dockerfile[^/]*$|\.pre-commit-config\.yaml$)"
)
_DEPS = re.compile(
    r"(^|/)(requirements[^/]*\.txt|requirements/[^/]+\.txt|pyproject\.toml|setup\.cfg|setup\.py|"
    r"environment[^/]*\.ya?ml|package(-lock)?\.json|Pipfile(\.lock)?|"
    r"poetry\.lock|uv\.lock)$"
)

#: S1 credential formats: a hit fails the job (exit 1).
_SECRETS = (
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{40,})")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("API key", re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{32,}")),
    (
        "assigned secret",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api_?key|access_?token|auth_?token)"
            r"\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)
#: S1 identity formats: reported, never failed.
_IDENTITY = (
    ("session link", re.compile(r"claude\.ai/code/session_|\bsession_[0-9A-Za-z]{20,}")),
    ("per-user noreply address", re.compile(r"[\w.+-]+@users\.noreply\.github\.com")),
)
_EXACT_PIN = re.compile(r"^\s*['\"]?[A-Za-z0-9_.\[\],-]+\s*==\s*[0-9]")

_PATTERNS = {
    "S3": re.compile(
        r"\b(?:subprocess\.|os\.(?:system|popen|exec\w*|spawn\w*)\(|eval\(|exec\(|"
        r"pickle\.loads?\(|marshal\.loads?\(|__import__\(|importlib\.import_module\(|"
        r"shell\s*=\s*True)"
    ),
    "S4": re.compile(
        r"\b(?:urllib\.request|urlopen\(|requests\.\w+\(|httpx\.|aiohttp\.|"
        r"socket\.(?:socket|create_connection)\(|http\.client|ftplib|smtplib)"
    ),
    "S5": re.compile(
        r"\b(?:\.extractall\(|\.extract\(|shutil\.(?:rmtree|move|copy\w*)\(|"
        r"os\.(?:remove|unlink|rename|replace|symlink|chmod)\(|"
        r"\.write_(?:text|bytes)\(|open\([^)]*['\"][wax]b?\+?['\"])"
    ),
    "S6": re.compile(
        r"\b(?:xml\.etree|ElementTree\.(?:parse|fromstring)|minidom|xml\.sax|lxml\.|"
        r"zipfile\.ZipFile\(|tarfile\.open\(|yaml\.(?:load|unsafe_load)\(|"
        r"shelve\.open\(|np\.load\([^)]*allow_pickle\s*=\s*True|torch\.load\()"
    ),
}
_WEAKER_TEST = re.compile(
    r"(?:pytest\.mark\.(?:skip|skipif|xfail)\b|pytest\.(?:skip|xfail)\(|"
    r"unittest\.skip|@skip\w*\(|#\s*pragma:\s*no cover)"
)
_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FULL_SHA = re.compile(r"@[0-9a-f]{40}\b")


@dataclass(frozen=True, order=True)
class Hit:
    """One place a check points at."""

    check: str
    path: str
    line: int
    kind: str
    snippet: str = ""
    fails: bool = False

    def render(self) -> str:
        """``path:line [kind] snippet`` (no snippet for S1)."""
        where = f"`{self.path}:{self.line}`" if self.line else f"`{self.path}`"
        text = f" `{self.snippet}`" if self.snippet else ""
        return f"{where} [{self.kind}]{text}"


@dataclass
class FileChange:
    """The lines a diff adds and removes in one file."""

    path: str
    status: str = "modified"  # added, deleted, renamed, modified
    added: list[tuple[int, str]] | None = None
    removed: list[tuple[int, str]] | None = None

    def __post_init__(self) -> None:
        self.added = self.added or []
        self.removed = self.removed or []


def _unquote(name: str) -> str:
    return name[2:] if name[:2] in ("a/", "b/") else name


def parse_diff(text: str) -> list[FileChange]:
    """
    Parse a unified diff (``git diff``, any context size) into file changes.

    Binary files appear with no lines; ``/dev/null`` marks an added or
    deleted file.
    """
    files: list[FileChange] = []
    current: FileChange | None = None
    old_line = new_line = 0
    in_hunk = False  # past the first @@: "---"/"+++" are then content
    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            parts = raw.split(" ")
            current = FileChange(_unquote(parts[-1]))
            files.append(current)
            in_hunk = False
            continue
        if current is None:
            continue
        if in_hunk and raw[:1] in ("+", "-", " "):
            pass
        elif raw.startswith("new file mode"):
            current.status = "added"
        elif raw.startswith("deleted file mode"):
            current.status = "deleted"
        elif raw.startswith("rename to "):
            current.status = "renamed"
            current.path = raw[len("rename to ") :]
        elif raw.startswith("--- ") and current.status == "deleted":
            current.path = _unquote(raw[4:].strip())
        elif raw.startswith("+++ ") or raw.startswith("--- "):
            continue
        elif raw.startswith("@@"):
            match = _HUNK.match(raw)
            if match:
                old_line, new_line = int(match.group(1)), int(match.group(3))
                in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("+"):
            current.added.append((new_line, raw[1:]))
            new_line += 1
        elif raw.startswith("-"):
            current.removed.append((old_line, raw[1:]))
            old_line += 1
        elif raw.startswith(" "):
            old_line += 1
            new_line += 1
    return files


def _label(path: str) -> str:
    if _DOC.search(path):
        return "doc"
    if _TEST.search(path):
        return "test"
    return "code" if _CODE.search(path) else "config"


def _short(line: str, width: int = 80) -> str:
    line = line.strip().replace("`", "'")
    return line if len(line) <= width else line[: width - 3] + "..."


def review(files: Iterable[FileChange]) -> list[Hit]:
    """Return every hit of S1 to S8 for *files*, sorted (deterministic)."""
    hits: set[Hit] = set()
    for change in files:
        path, label = change.path, _label(change.path)
        workflow = bool(_WORKFLOW.match(path))
        for number, line in change.added:
            for kind, pattern in _SECRETS:
                if pattern.search(line):
                    hits.add(Hit("S1", path, number, f"{kind}, {label}", fails=True))
            for kind, pattern in _IDENTITY:
                if pattern.search(line):
                    hits.add(Hit("S1", path, number, f"{kind}, {label}"))
            if _DEPS.search(path) and _EXACT_PIN.search(line):
                hits.add(Hit("S2", path, number, "exact pin", _short(line)))
            if label in ("code", "test"):
                for check, pattern in _PATTERNS.items():
                    if pattern.search(line):
                        hits.add(Hit(check, path, number, label, _short(line)))
            if workflow:
                stripped = line.strip()
                if "uses:" in stripped and "@" in stripped and not stripped.startswith(
                    "#"
                ):
                    if "uses: ./" not in stripped and not _FULL_SHA.search(stripped):
                        hits.add(Hit("S7", path, number, "action not pinned to a commit", _short(line)))
                if re.search(r"\b(?:contents|actions|packages|id-token|pull-requests|issues|deployments|checks|statuses|security-events|attestations|pages)\s*:\s*write\b|write-all", stripped):
                    hits.add(Hit("S7", path, number, "write permission", _short(line)))
                if "pull_request_target" in stripped:
                    hits.add(Hit("S7", path, number, "pull_request_target", _short(line)))
                if "${{" in stripped and "github.event." in stripped:
                    hits.add(Hit("S7", path, number, "event data in an expression", _short(line)))
            if label == "test" and _WEAKER_TEST.search(line):
                hits.add(Hit("S8", path, number, "skip or xfail added", _short(line)))
        if _DEPS.search(path) and change.status != "deleted":
            hits.add(Hit("S2", path, 0, f"dependency file {change.status}"))
        if _CI_PATHS.match(path):
            hits.add(Hit("S7", path, 0, f"{change.status}"))
        if label == "test":
            if change.status == "deleted":
                hits.add(Hit("S8", path, 0, "test file deleted"))
            else:
                kept = {m.group(1) for _, line in change.added if (m := _TEST_DEF.match(line))}
                for number, line in change.removed:
                    match = _TEST_DEF.match(line)
                    if match and match.group(1) not in kept:
                        hits.add(Hit("S8", path, number, f"{match.group(1)} removed (old line)"))
    return sorted(hits)


def render(hits: list[Hit], max_per_check: int = 20) -> str:
    """The Markdown table: one row per check, ``none`` when it has no hit."""
    rows = ["| # | Check | Result |", "|---|---|---|"]
    approval: list[str] = []
    for check, title, needs_approval in CHECKS:
        found = [hit for hit in hits if hit.check == check]
        if not found:
            rows.append(f"| {check} | {title} | none |")
            continue
        shown = "<br>".join(hit.render() for hit in found[:max_per_check])
        more = len(found) - max_per_check
        if more > 0:
            shown += f"<br>... and {more} more"
        rows.append(f"| {check} | {title} | {len(found)}: {shown} |")
        if needs_approval and any(hit.kind.split(",")[-1].strip() != "doc" for hit in found):
            approval.append(check)
    out = "## Security review (generated)\n\n" + "\n".join(rows) + "\n"
    if approval:
        out += (
            f"\n**Needs the maintainer's explicit approval:** {', '.join(approval)}"
            " (each hit above is a place to look, not a verdict).\n"
        )
    if any(hit.fails for hit in hits):
        out += "\n**S1: a credential pattern was added. Remove it and rotate the secret.**\n"
    return out


def _git_diff(base: str, head: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        # base...head: the change since the merge base, so commits that
        # reached the base after the branch was cut do not read as removed.
        ["git", "diff", "-U0", "--no-color", "--no-ext-diff", "-M", f"{base}...{head}", "--"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(f"security_review: git diff failed: {done.stderr.strip()}")
    return done.stdout


def main(argv: list[str] | None = None) -> int:
    """Print the table for a diff; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("diff", nargs="?", help="a diff file, or - for stdin")
    parser.add_argument("--base", help="git revision to diff against (with --head)")
    parser.add_argument("--head", default="HEAD", help="default: HEAD")
    args = parser.parse_args(argv)
    if bool(args.diff) == bool(args.base):
        parser.error("give a diff file (or -) or --base, not both")
    if args.base:
        text = _git_diff(args.base, args.head)
    elif args.diff == "-":
        text = sys.stdin.read()
    else:
        with open(args.diff, encoding="utf-8", errors="replace") as fh:  # noqa: PTH123
            text = fh.read()
    hits = review(parse_diff(text))
    sys.stdout.write(render(hits))
    return 1 if any(hit.fails for hit in hits) else 0


if __name__ == "__main__":
    sys.exit(main())
