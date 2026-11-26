# maintenances/_repo/_maintenance/tools/next_challenges.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Choose the next challenges so that many contributors can work at once without conflicts.

Usage::

    python -B maintenances/_repo/_maintenance/tools/next_challenges.py \\
        [--count 10] [--slot K] [--area PREFIX] [--open-prs FILE | --github] [--json]

Notes
-----
**User notes.** When ten or twenty people (or chats) each ask "what next?"
at the same time, they must not pick the same ledger note, nor two notes
that edit the same files. This tool answers for all of them at once:

1. It reads every open ledger note under ``upcoming_changes/``, as
   ``orient.py`` does, and the paths each note says it touches (its
   ``Affected paths and ownership`` section; the note's ``area`` when the
   section names none).
2. It reads the open pull requests (``--open-prs FILE``, a JSON list of
   ``{"number", "title", "body", "files"}``, or ``--github`` to fetch them
   with ``gh``). A note an open pull request names (its path, or its file
   stem when no other note has that stem) is *claimed*; a note whose paths overlap the files an open pull request
   changes is *busy*. Neither is offered.
3. It ranks the rest by a fixed order (``status``, then ``kind``, then
   path) and takes notes greedily, skipping any whose paths overlap a note
   already taken. The result: up to ``--count`` *slots*, pairwise
   disjoint, none touching open work.

Everyone running it on the same checkout and the same open pull requests
gets the same slots in the same order, and the same ``fingerprint`` to
prove it. Contributor *K* takes slot *K* (``--slot K``).

**Claiming.** Open a draft pull request at once, naming the note's path in
its body. GitHub is the lock: the next run sees the claim and moves on.
Name the branch as given (``chal/<area>/<note>``). Name the release-note
fragment only after the pull request has its number. Several people
deriving "the next number" at the same time would get the same one.

**Shared files.** Some files are edited by almost every change: generated
listings and skill mirrors. They are left out of the overlap test and
listed under ``shared``. Regenerate them after rebasing instead of merging
them by hand.

**Developer notes.** Standard library only. It never imports
``scikitplot``, and nothing is written. The ranking and overlap test are
pure functions of their inputs (:func:`plan`), so the output is
deterministic and idempotent. Paths overlap when one is a path-segment
prefix of the other (``a/b`` overlaps ``a/b/c.py``, not ``a/bc``).
Greedy selection is O(n·k) for n notes and k slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Ledger statuses that can be started now. ``in-progress`` is someone's;
#: ``blocked`` waits on named evidence.
STARTABLE = ("planned", "open")

#: The order kinds are offered in: what can hurt users first.
KIND_ORDER = (
    "security",
    "bug",
    "reliability",
    "compatibility",
    "packaging",
    "build",
    "api",
    "docs-contract",
    "other",
)

#: Files almost every change regenerates; overlap here is resolved by
#: regenerating after a rebase, so it does not make two notes conflict.
SHARED = (
    "docs/source/apis_reference.py",
    "scikitplot/_build_utils/generate_apis_reference/apis_reference.py.in",
    "plugins/scikitplot/skills/",
    "docs/source/whats_new/upcoming_changes/",
)

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_FIELD = re.compile(r'^(\w+):\s*"?(.*?)"?\s*$', re.MULTILINE)
_SECTION = re.compile(
    r"^## Affected paths and ownership\s*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL
)
_TICKED = re.compile(r"`([^`\s]+)`")
#: A backticked word is a path when it starts at a top-level folder.
_TOP = re.compile(
    r"^(?:scikitplot|libs|docs|galleries|tools|plugins|upcoming_changes|build"
    r"|requirements|\.github|\.circleci)(?:/|$)"
)
_SKIP_NOTES = frozenset({"README.md", "TEMPLATE.md"})


def _clean(path: str) -> str:
    return path.strip().rstrip("/").rstrip(".,;:")


def overlaps(a: str, b: str) -> bool:
    """Return whether one path is a path-segment prefix of the other."""
    a, b = _clean(a), _clean(b)
    return a == b or b.startswith(a + "/") or a.startswith(b + "/")


def _shared(path: str) -> bool:
    return any(overlaps(path, s) or path.startswith(s) for s in SHARED)


def read_note(root: Path, path: Path) -> dict[str, Any] | None:
    """
    Return one ledger note's fields and footprint, or ``None`` if it has no front matter.

    The footprint is the backticked top-level paths of its ``Affected paths
    and ownership`` section, without the maintenance owners (``maintenances/``)
    and shared files; when none are left, the note's ``area``.
    """
    text = path.read_text(encoding="utf-8")
    front = _FRONT.match(text)
    if front is None:
        return None
    fields = dict(_FIELD.findall(front.group(1)))
    section = _SECTION.search(text)
    found = _TICKED.findall(section.group(1)) if section else []
    paths = sorted(
        {_clean(p) for p in found if _TOP.match(p) and not _shared(_clean(p))}
    )
    area = _clean(fields.get("area", ""))
    return {
        "path": path.relative_to(root).as_posix(),
        "title": fields.get("title", ""),
        "status": fields.get("status", ""),
        "kind": fields.get("kind", "other"),
        "area": area,
        "footprint": paths or ([area] if area else []),
    }


def read_notes(root: Path) -> list[dict[str, Any]]:
    """Return every ledger note, sorted by path."""
    notes = []
    for path in sorted((root / "upcoming_changes").rglob("*.md")):
        if path.name in _SKIP_NOTES:
            continue
        note = read_note(root, path)
        if note is not None:
            notes.append(note)
    return notes


def deferred_notes(root: Path) -> set[str]:
    """Return the note paths the start template defers."""
    template = root / ".claude/new-chat-template.md"
    if not template.is_file():
        return set()
    text = template.read_text(encoding="utf-8")
    block = text.split("Notes I am deferring", 1)
    if len(block) < 2:
        return set()
    lines = block[1].split("\n- ", 1)[0]
    return set(re.findall(r"`(upcoming_changes/[^`]+\.md)`", lines))


def fetch_open_prs(repo: str) -> list[dict[str, Any]]:
    """
    Return the repository's open pull requests with their changed files, through ``gh``.

    Raises
    ------
    RuntimeError
        If ``gh`` is not installed or a call fails.
    """
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError("--github needs the gh command; pass --open-prs FILE instead")

    def call(endpoint: str) -> Any:
        done = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [gh, "api", "--paginate", "--slurp", endpoint],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if done.returncode:
            raise RuntimeError(f"gh api {endpoint} failed: {done.stderr.strip()}")
        pages = json.loads(done.stdout)
        return [item for page in pages for item in page]

    prs = []
    for pr in call(f"repos/{repo}/pulls?state=open&per_page=100"):
        files = call(f"repos/{repo}/pulls/{pr['number']}/files?per_page=100")
        prs.append(
            {
                "number": pr["number"],
                "title": pr.get("title") or "",
                "body": pr.get("body") or "",
                "files": sorted(f["filename"] for f in files),
            }
        )
    return sorted(prs, key=lambda pr: pr["number"])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def plan(
    notes: Sequence[dict[str, Any]],
    open_prs: Sequence[dict[str, Any]],
    *,
    count: int = 10,
    area: str | None = None,
    deferred: Iterable[str] = (),
) -> dict[str, Any]:
    """
    Choose up to *count* notes that can be worked on at once, from the inputs alone.

    Parameters
    ----------
    notes : sequence of dict
        As :func:`read_notes` returns.
    open_prs : sequence of dict
        ``number``, ``title``, ``body``, ``files`` per open pull request.
    count : int, optional
        Most slots. Default: 10.
    area : str or None, optional
        Offer only notes whose ``area`` overlaps this path.
    deferred : iterable of str
        Note paths not to offer.

    Returns
    -------
    dict
        ``slots`` (the chosen notes, in order, numbered from 1), ``claimed``
        (note path to the pull requests naming it), ``busy`` (note path to
        the pull requests whose files it overlaps), ``waiting`` (startable
        notes left out because they overlap a slot), ``shared`` and
        ``fingerprint``, a digest of everything the choice depends on.

    Raises
    ------
    ValueError
        If *count* is not a positive integer.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"count must be a positive integer, not {count!r}")
    deferred = set(deferred)
    claimed: dict[str, list[int]] = {}
    busy: dict[str, list[int]] = {}
    candidates = []
    stems: dict[str, int] = {}
    for note in notes:
        stems[Path(note["path"]).stem] = stems.get(Path(note["path"]).stem, 0) + 1
    for note in notes:
        if note["status"] not in STARTABLE or note["path"] in deferred:
            continue
        if area is not None and not overlaps(note["area"], area):
            continue
        stem = Path(note["path"]).stem
        names = [
            pr["number"]
            for pr in open_prs
            if note["path"] in (pr["body"] + pr["title"])
            or note["path"] in pr["files"]
            or (
                stems[stem] == 1  # a stem two notes share names neither
                and re.search(rf"(?<![\w/-]){re.escape(stem)}(?![\w-])", pr["body"] + pr["title"])
            )
        ]
        if names:
            claimed[note["path"]] = names
            continue
        touching = [
            pr["number"]
            for pr in open_prs
            if any(
                overlaps(f, p) for f in pr["files"] if not _shared(f) for p in note["footprint"]
            )
        ]
        if touching:
            busy[note["path"]] = touching
            continue
        candidates.append(note)

    def rank(note: dict[str, Any]) -> tuple[int, int, str]:
        kind = KIND_ORDER.index(note["kind"]) if note["kind"] in KIND_ORDER else len(KIND_ORDER)
        return (STARTABLE.index(note["status"]), kind, note["path"])

    slots: list[dict[str, Any]] = []
    waiting: dict[str, str] = {}
    for note in sorted(candidates, key=rank):
        clash = next(
            (
                slot["path"]
                for slot in slots
                if any(overlaps(a, b) for a in note["footprint"] for b in slot["footprint"])
            ),
            None,
        )
        if clash is not None:
            waiting[note["path"]] = clash
            continue
        if len(slots) < count:
            area_slug = _slug(note["area"].split("/", 1)[-1] or "repo")
            slots.append(
                {
                    **note,
                    "slot": len(slots) + 1,
                    "branch": f"chal/{area_slug}/{Path(note['path']).stem}",
                }
            )
        else:
            waiting[note["path"]] = "(no slot left)"
    digest = hashlib.sha256(
        json.dumps(
            {
                "notes": [[n["path"], n["status"], n["kind"], n["footprint"]] for n in notes],
                "prs": [[p["number"], sorted(p["files"])] for p in open_prs],
                "count": count,
                "area": area,
                "deferred": sorted(deferred),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "fingerprint": digest,
        "open_prs": [pr["number"] for pr in open_prs],
        "slots": slots,
        "claimed": dict(sorted(claimed.items())),
        "busy": dict(sorted(busy.items())),
        "waiting": dict(sorted(waiting.items())),
        "shared": list(SHARED),
    }


def render(report: dict[str, Any], slot: int | None = None) -> str:
    """Render a :func:`plan` report as Markdown; one slot only when *slot* is given."""
    slots = report["slots"]
    if slot is not None:
        slots = [s for s in slots if s["slot"] == slot]
    out = [
        f"# Next challenges (fingerprint {report['fingerprint']})",
        "",
        f"Open pull requests considered: {', '.join(f'#{n}' for n in report['open_prs']) or 'none'}.",
        "",
    ]
    if not report["open_prs"]:
        out += ["**No open pull requests given**: claims and busy paths are unknown;"
                " pass --github or --open-prs FILE.", ""]
    for s in slots:
        out += [
            f"## Slot {s['slot']}: {s['title']}",
            f"- note: `{s['path']}` ({s['kind']}, {s['status']})",
            f"- touches: {', '.join(f'`{p}`' for p in s['footprint']) or '(unknown)'}",
            f"- branch: `{s['branch']}`",
            "- claim: open a draft pull request now, naming the note's path in its body.",
            "",
        ]
    if slot is None:
        out += [
            f"Claimed by open pull requests: {len(report['claimed'])};"
            f" busy (paths in open pull requests): {len(report['busy'])};"
            f" waiting on a slot's paths: {len(report['waiting'])}.",
            "",
            "Shared files (regenerate after rebase, never merge by hand): "
            + ", ".join(f"`{s}`" for s in report["shared"]),
        ]
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the slots; exit 2 on bad input, 1 when ``--slot`` names no slot."""
    parser = argparse.ArgumentParser(
        prog="next_challenges.py",
        description="Choose the next ledger notes that many people can work on at once.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--slot", type=int, default=None)
    parser.add_argument("--area", default=None, help="only notes in this area")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--open-prs", type=Path, metavar="FILE")
    source.add_argument("--github", action="store_true")
    parser.add_argument("--repo", default="scikit-plots/scikit-plots")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.open_prs is not None:
            open_prs = json.loads(args.open_prs.read_text(encoding="utf-8"))
        elif args.github:
            open_prs = fetch_open_prs(args.repo)
        else:
            open_prs = []
        report = plan(
            read_notes(args.root),
            open_prs,
            count=args.count,
            area=args.area,
            deferred=deferred_notes(args.root),
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"next_challenges: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))  # noqa: T201
    else:
        print(render(report, args.slot), end="")  # noqa: T201
    if args.slot is not None and not any(s["slot"] == args.slot for s in report["slots"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
