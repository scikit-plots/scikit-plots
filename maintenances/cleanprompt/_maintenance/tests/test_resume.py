"""
The working log a fresh chat starts from must agree with the follow-up ledger.

Notes
-----
**User notes.** ``RESUME.md`` is where the round in progress is recorded step
by step, so that work can continue in a session that has no history. Its
**Open ledger** section lists every note under
``upcoming_changes/scikitplot/cleanprompt/``. These tests fail when the two
disagree.

**Developer notes — why a test and not a habit.** Continuity fails silently.
A note added to ``upcoming_changes`` and never mentioned in the log is work a
fresh session will not know exists; a note deleted from the ledger but still
listed in the log is work a fresh session will try to do twice. Both are
cheap to detect and expensive to discover later, so they are detected here, on
every run of the maintenance plane.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAINTENANCE = Path(__file__).resolve().parents[1]
CHECKOUT = Path(__file__).resolve().parents[4]
RESUME = MAINTENANCE / "RESUME.md"
LEDGER = CHECKOUT / "upcoming_changes" / "scikitplot" / "cleanprompt"

#: The sections a fresh session needs, in the order it needs them.
REQUIRED_SECTIONS = (
    "## How to use this file",
    "## Baseline commands",
    "## Current round",
    "## Step log",
    "## Last verified numbers",
    "## Next action",
    "## Open ledger",
)

_LEDGER_LINE = re.compile(
    r"^- `(upcoming_changes/scikitplot/cleanprompt/[^`]+\.md)` — (\S+)", re.MULTILINE
)
_STATUSES = frozenset({"open", "planned", "in-progress", "blocked", "promoted"})


def _resume_text():
    """Return the log, or fail: a maintenance plane without one cannot resume."""
    assert RESUME.is_file(), "RESUME.md is missing from the maintenance plane"
    return RESUME.read_text(encoding="utf-8")


def _section(text, heading):
    """Return the body of one ``##`` section."""
    start = text.index(heading) + len(heading)
    following = text.find("\n## ", start)
    return text[start:] if following == -1 else text[start:following]


def _listed():
    """Return ``{path: status}`` from the Open ledger section."""
    body = _section(_resume_text(), "## Open ledger")
    return {match.group(1): match.group(2) for match in _LEDGER_LINE.finditer(body)}


def _on_disk():
    """Return the ledger's note paths, relative to the checkout."""
    if not LEDGER.parent.parent.is_dir():
        pytest.skip(f"upcoming_changes not present in this checkout: {LEDGER}")
    if not LEDGER.is_dir():
        return set()
    return {
        path.relative_to(CHECKOUT).as_posix()
        for path in LEDGER.rglob("*.md")
        if path.is_file()
    }


def test_every_required_section_is_present_in_order():
    text = _resume_text()
    positions = [text.find(heading) for heading in REQUIRED_SECTIONS]
    missing = [h for h, p in zip(REQUIRED_SECTIONS, positions) if p == -1]
    assert not missing, f"RESUME.md lacks sections: {missing}"
    assert positions == sorted(positions), "RESUME.md sections are out of order"


def test_every_ledger_note_is_listed():
    missing = sorted(_on_disk() - set(_listed()))
    assert not missing, (
        f"notes in upcoming_changes that RESUME.md does not list: {missing}; "
        "add them under '## Open ledger'"
    )


def test_every_listed_note_exists():
    stale = sorted(set(_listed()) - _on_disk())
    assert not stale, (
        f"RESUME.md lists notes that no longer exist: {stale}; "
        "remove them from '## Open ledger'"
    )


def test_every_listed_status_is_a_ledger_status():
    wrong = {path: status for path, status in _listed().items() if status not in _STATUSES}
    assert not wrong, f"unknown statuses in RESUME.md: {wrong}"


def test_listed_status_matches_the_note_front_matter():
    disagree = {}
    for path, status in _listed().items():
        note = CHECKOUT / path
        if not note.is_file():
            continue
        match = re.search(r"^status:\s*(\S+)", note.read_text(encoding="utf-8"), re.MULTILINE)
        declared = match.group(1) if match else None
        if declared != status:
            disagree[path] = (status, declared)
    assert not disagree, f"RESUME.md status differs from the note: {disagree}"


def test_next_action_is_not_empty():
    body = _section(_resume_text(), "## Next action").strip()
    assert body, "RESUME.md must say what to do next"
