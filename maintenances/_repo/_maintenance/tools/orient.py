# maintenances/_repo/_maintenance/tools/orient.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Print where the work stands, read from the checkout's own records.

Usage::

    python -B maintenances/_repo/_maintenance/tools/orient.py [--pr N] [--json]

Notes
-----
**User notes.** The first command of a new chat, whoever or whatever runs
it. It lists, from the files in this checkout and nothing else:

* the open ledger: every note under ``upcoming_changes/`` whose front matter
  says ``status: open``, marked ``deferred`` when the start template defers it;
* the release-note fragments already filed for the pull request;
* per maintenance area, how many findings ``REVIEW.json`` holds and the next
  free id of each id series, and the first lines of ``RESUME.md``'s
  ``Next action`` section where that file exists.

The pull request number and the deferred notes come from the ``FILL IN``
block of ``.claude/new-chat-template.md``, so they are written in one place;
``--pr`` overrides the number for one run.

**Developer notes.** Read-only and deterministic: every list is sorted and
nothing is written, so running it twice gives the same output. It uses the
standard library only and never imports ``scikitplot``, so it runs on a tree
whose compiled extensions are not built. A malformed record is reported in
``problems`` rather than raised, because the point is to see the whole state
at once; the exit code is 1 when there is any problem.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: The start template whose FILL IN block names the PR and the deferred notes.
TEMPLATE = Path(".claude/new-chat-template.md")

#: The reading list a new chat starts from.
INIT = Path(".claude/init")

#: The documents that tell a new chat where to read; every path they name
#: must exist (:func:`dead_paths`).
READ_ORDER = (
    INIT,
    TEMPLATE,
    Path("AGENTS.md"),
    Path("maintenances/MAINTAINING.md"),
    Path("maintenances/_repo/_maintenance/FRESH_CHAT_HANDOFF.md"),
)

#: A word that starts at one of the checkout's top-level folders.
_TOP = re.compile(
    r"^(?:\.claude|maintenances|plugins|docs|upcoming_changes|scikitplot|tools"
    r"|libs|galleries)/"
)

#: A file at the checkout root, named bare (``AGENTS.md``).
_ROOT_FILE = re.compile(r"^[A-Z][A-Z_]*\.md$")

_PR_LINE = re.compile(
    r"^- Pull request number for new fragments:\s*`(\d+)`", re.MULTILINE
)
_PATH = re.compile(r"`(upcoming_changes/[^`]+\.md)`")
_STATUS = re.compile(r"^status:\s*([A-Za-z_-]+)\s*$", re.MULTILINE)
_TITLE = re.compile(r"^title:\s*\"?(.*?)\"?\s*$", re.MULTILINE)
_ID = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")
_SKIP_NOTES = frozenset({"README.md", "TEMPLATE.md"})


def repository_root(start: Path) -> Path:
    """
    Return the checkout root above ``start``.

    Parameters
    ----------
    start : Path
        Any path inside the checkout.

    Returns
    -------
    Path
        The first parent holding both ``upcoming_changes`` and
        ``maintenances``.

    Raises
    ------
    SystemExit
        If no such directory exists above ``start``.
    """
    for candidate in (start, *start.parents):
        if (candidate / "upcoming_changes").is_dir() and (
            candidate / "maintenances"
        ).is_dir():
            return candidate
    msg = f"no checkout root above {start}"
    raise SystemExit(msg)


def template_settings(root: Path) -> tuple[str | None, list[str]]:
    """
    Return the pull request number and deferred notes from the start template.

    Parameters
    ----------
    root : Path
        The checkout root.

    Returns
    -------
    pr : str or None
        The number in the FILL IN block, or ``None`` when absent.
    deferred : list of str
        Note paths listed under "Notes I am deferring", sorted.
    """
    path = root / TEMPLATE
    if not path.is_file():
        return None, []
    text = path.read_text(encoding="utf-8")
    block = text.split("## 1.", 1)[0]
    match = _PR_LINE.search(block)
    deferred = []
    if "Notes I am deferring" in block:
        part = block.split("Notes I am deferring", 1)[1].split("\n- ", 1)[0]
        deferred = sorted(set(_PATH.findall(part)))
    return (match.group(1) if match else None), deferred


def open_notes(root: Path, deferred: list[str]) -> list[dict]:
    """
    Return every open note under ``upcoming_changes/``, sorted by path.

    Parameters
    ----------
    root : Path
        The checkout root.
    deferred : list of str
        Paths to mark as deferred.

    Returns
    -------
    list of dict
        ``path``, ``title`` and ``deferred`` for each note whose front matter
        has ``status: open``.
    """
    notes = []
    for path in sorted((root / "upcoming_changes").rglob("*.md")):
        if path.name in _SKIP_NOTES:
            continue
        head = path.read_text(encoding="utf-8").split("\n---", 1)[0]
        status = _STATUS.search(head)
        if not status or status.group(1) != "open":
            continue
        rel = path.relative_to(root).as_posix()
        title = _TITLE.search(head)
        notes.append(
            {
                "path": rel,
                "title": title.group(1) if title else "",
                "deferred": rel in deferred,
            }
        )
    return notes


def fragments(root: Path, pr: str | None) -> list[str]:
    """Return the release-note fragments filed for ``pr``, sorted."""
    if pr is None:
        return []
    base = root / "docs/source/whats_new/upcoming_changes"
    return sorted(
        path.relative_to(base).as_posix() for path in base.glob(f"*/{pr}.*.rst")
    )


def other_fragment_numbers(root: Path, pr: str | None) -> dict[str, int]:
    """
    Return how many fragments each other pull request number has.

    Parameters
    ----------
    root : Path
        The checkout root.
    pr : str or None
        The pull request this chat works on.

    Returns
    -------
    dict of str to int
        Number to fragment count, sorted, ``pr`` itself left out.

    Notes
    -----
    **Developer notes.** A pull request's fragments are named after its
    number, which is known only once it is opened; a renumbered PR leaves
    every fragment under the old number. ``state`` reports that as a problem
    when ``pr`` has none, instead of printing an empty list that looks fine.
    """
    base = root / "docs/source/whats_new/upcoming_changes"
    counts: dict[str, int] = {}
    for path in base.glob("*/*.rst"):
        number = path.name.split(".", 1)[0]
        if number.isdigit() and number != pr:
            counts[number] = counts.get(number, 0) + 1
    return dict(sorted(counts.items()))


def next_ids(ids: list[str]) -> dict[str, str]:
    """
    Return the next free id of each id series.

    Parameters
    ----------
    ids : list of str
        Finding ids such as ``CP-119`` or ``LOG-FMT-001``.

    Returns
    -------
    dict of str to str
        Series prefix to its next id, zero-padded as the series is.

    Examples
    --------
    >>> next_ids(["CP-9", "CP-119", "LOG-FMT-001"])
    {'CP-': 'CP-120', 'LOG-FMT-': 'LOG-FMT-002'}
    """
    best: dict[str, tuple[int, int]] = {}
    for one in ids:
        match = _ID.match(one)
        if not match:
            continue
        number = int(match.group("number"))
        width = len(match.group("number"))
        prefix = match.group("prefix")
        old = best.get(prefix, (-1, 0))
        best[prefix] = (max(old[0], number), max(old[1], width))
    return {
        prefix: f"{prefix}{number + 1:0{width}d}"
        for prefix, (number, width) in sorted(best.items())
    }


def areas(root: Path, problems: list[str]) -> list[dict]:
    """Return each maintenance area's findings count, next ids and next action."""
    out = []
    for review in sorted((root / "maintenances").glob("*/REVIEW.json")):
        area = review.parent.name
        try:
            document = json.loads(review.read_text(encoding="utf-8"))
        except ValueError as error:
            problems.append(f"{review.relative_to(root)}: {error}")
            continue
        findings = document.get("findings", [])
        ids = [f.get("id", "") for f in findings if isinstance(f, dict)]
        resume = review.parent / "_maintenance" / "RESUME.md"
        action = ""
        if resume.is_file():
            text = resume.read_text(encoding="utf-8")
            if "\n## Next action" in text:
                section = text.split("\n## Next action", 1)[1].split("\n## ", 1)[0]
                lines = [line for line in section.strip().splitlines() if line]
                action = " ".join(lines[:2])
        out.append(
            {
                "area": area,
                "findings": len(ids),
                "next_ids": next_ids(ids),
                "next_action": action,
            }
        )
    return out


def dead_paths(root: Path) -> list[str]:
    """
    Return every repository path a read-order document names that does not exist.

    Parameters
    ----------
    root : Path
        The checkout root.

    Returns
    -------
    list of str
        Sorted ``"<document>: <path>"`` entries; empty when every path
        resolves. A missing document is reported as itself.

    Notes
    -----
    **User notes.** The documents a new chat follows to find its way
    (:data:`READ_ORDER`) must never send it to a file that moved. A path is
    any word inside backticks that starts at a top-level folder
    (:data:`_TOP`), whether alone or inside a command line, plus in
    ``.claude/init`` a bare root file such as ``AGENTS.md``. Words with
    ``<placeholders>``, ``*`` or ``{`` are patterns, not paths, and are
    skipped.

    **Developer notes.** Only the read-order documents are checked, not every
    record: ``maintenances/README.md`` names retired locations on purpose,
    and history quotes old paths verbatim. ``check_records.py`` already
    checks Markdown links; this catches backticked paths, which it does not.
    """
    out = []
    for document in READ_ORDER:
        path = root / document
        if not path.is_file():
            out.append(f"{document.as_posix()}: (the document itself)")
            continue
        words = set()
        for token in re.findall(r"`([^`\n]+)`", path.read_text(encoding="utf-8")):
            for word in token.split():
                if set("<*{") & set(word):
                    continue
                if _TOP.match(word) or (document == INIT and _ROOT_FILE.match(word)):
                    words.add(word.rstrip(".,;:"))
        out.extend(
            f"{document.as_posix()}: {word}"
            for word in sorted(words)
            if not (root / word).exists()
        )
    return out


def state(root: Path, pr: str | None = None) -> dict:
    """
    Return the whole state as one mapping.

    Parameters
    ----------
    root : Path
        The checkout root.
    pr : str, optional
        Pull request number; defaults to the start template's.

    Returns
    -------
    dict
        ``pr``, ``open_notes``, ``fragments``, ``areas`` and ``problems``.
    """
    problems: list[str] = []
    template_pr, deferred = template_settings(root)
    pr = pr or template_pr
    if pr is None:
        problems.append(f"no pull request number: fill in {TEMPLATE} or pass --pr")
    missing = [p for p in deferred if not (root / p).is_file()]
    problems.extend(f"deferred note not found: {p}" for p in missing)
    problems.extend(f"dead path in {entry}" for entry in dead_paths(root))
    filed = fragments(root, pr)
    others = other_fragment_numbers(root, pr)
    if pr is not None and not filed and others:
        found = ", ".join(f"{number} ({count})" for number, count in others.items())
        problems.append(
            f"no fragments are filed under {pr}, but some are under {found}: "
            "rename them, or fix the number in the template"
        )
    return {
        "pr": pr,
        "open_notes": open_notes(root, deferred),
        "fragments": filed,
        "areas": areas(root, problems),
        "problems": problems,
    }


def render(report: dict) -> str:
    """Return the report as plain text."""
    lines = [f"pull request: {report['pr']}", "", "open ledger:"]
    for note in report["open_notes"]:
        mark = " (deferred)" if note["deferred"] else ""
        lines.append(f"  {note['path']}{mark}")
    lines += ["", f"fragments for {report['pr']}:"]
    lines += [f"  {one}" for one in report["fragments"]] or ["  (none)"]
    lines += ["", "areas with findings:"]
    for area in report["areas"]:
        if not area["findings"] and not area["next_action"]:
            continue
        ids = ", ".join(area["next_ids"].values())
        lines.append(f"  {area['area']}: {area['findings']} findings; next {ids}")
        if area["next_action"]:
            lines.append(f"    next action: {area['next_action'][:200]}")
    if report["problems"]:
        lines += ["", "problems:"] + [f"  {p}" for p in report["problems"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print the state; exit 1 when any record could not be read."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--pr", help="pull request number (default: the template's)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="checkout")
    args = parser.parse_args(argv)
    report = state(repository_root(args.root.resolve()), args.pr)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render(report))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
