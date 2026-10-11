#!/usr/bin/env python3
r"""
Keep every repository path short enough to check out on Windows.

Why this exists
---------------
``pip install "scikit-plots-skinny @ git+https://...#subdirectory=libs/skinny"``
clones the **whole** repository, not only ``libs/skinny``. On Windows without
``core.longpaths`` Git refuses any file whose full path reaches the classic
``MAX_PATH`` limit of 260 characters (259 usable, plus the terminating NUL),
and the install fails before anything is built::

    error: unable to create file maintenances/.../FRESH_CHAT_..._HANDOFF.md:
    Filename too long

The full path is *the clone directory* plus *the repository path*. pip clones
into a directory it names itself, so the repository path is the only part this
project controls. This tool computes the longest clone directory pip can
create, derives the length every repository path must stay within, and
reports, explains and (on request) fixes the paths that do not.

The budget, derived rather than guessed
---------------------------------------
pip's Windows clone directory is::

    C:\Users\<user>\AppData\Local\Temp\pip-install-<8>\<dist>_<32 hex>\

``<user>`` is at most 20 characters (the Windows account-name limit) and
``<dist>`` is the longest distribution name a user can install from this
repository by URL: ``scikit-plots`` itself and every ``libs/*/pyproject.toml``
project. The observed failure (user ``devel``, ``scikit-plots-skinny``) had a
108-character prefix: paths of 152 and 166 characters failed, 149 did not —
exactly ``108 + len(path) <= 259``. The worst case is reported by ``budget``.

Rules
-----
``long-path``
    A tracked file whose repository path is longer than the budget.
``long-name``
    A Markdown file under ``maintenances/`` or ``docs/maintenances/`` whose
    file name is longer than :data:`NAME_LIMIT` characters. Maintenance notes
    are written by people and agents; a name is a label, the title belongs in
    the file's first heading.

Known debt outside the maintenance planes is listed, with a reason, in
``path_length_baseline.txt`` next to this file. A listed path is reported as
known and does not fail the check; a listed path that no longer violates is a
*stale* entry and fails it, so the list can only shrink.

Commands
--------
``budget``
    Print the prefix model and the resulting budget.
``check``
    Report violations; exit 1 if any is not in the baseline, or a baseline
    entry is stale.
``suggest``
    Print a rename plan for violating maintenance Markdown files. Read-only.
``fix --apply``
    Carry the plan out: rename each file (``git mv`` in a Git checkout) and
    rewrite references to its old name in every text file of the repository.
    Without ``--apply`` it prints what it would do.

How a name is shortened (deterministic)
---------------------------------------
1. In a directory with at least one violation, a token prefix or suffix
   shared by *every* Markdown file there is dropped from all of them, and the
   directory's own name as a leading token is dropped
   (``history/fresh_chat/FRESH_CHAT_X_HANDOFF.md`` becomes ``X.md``), so the
   directory stays consistent.
2. If still too long: filler words (``AND``, ``OF``, ``THE``, ...) go.
3. If still too long: trailing words go, one at a time. A leading identifier
   (a token with a digit, ``B44``, ``R173T8``, ``YTG-M002``) and one word
   after it are always kept, so ``<ID>_*.md`` lookups keep working.
4. A name that would collide with another file in the directory is refused.

Examples
--------
.. code-block:: sh

    python tools/maint_tools/check_path_lengths.py budget
    python tools/maint_tools/check_path_lengths.py check
    python tools/maint_tools/check_path_lengths.py suggest
    python tools/maint_tools/check_path_lengths.py fix --apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Iterable, Sequence

#: Windows ``MAX_PATH``: 260 characters including the terminating NUL.
MAX_PATH = 260

#: Longest Windows account name (``USERNAME``).
USERNAME_MAX = 20

#: Longest maintenance Markdown file name, extension included.
NAME_LIMIT = 64

#: Where the long-name rule applies (repository-relative directory prefixes).
NAME_SCOPES = ("maintenances/", "docs/maintenances/")

#: Words dropped first when a name must shrink.
FILLER = frozenset(
    {"A", "AN", "AND", "OR", "THE", "OF", "TO", "FOR", "WITH", "IN", "ON", "BY"}
)

#: Directories never part of a checkout.
_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)

#: Documents whose references are rewritten after a rename, anywhere.
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".json", ".toml", ".yml", ".yaml"})

#: Code and configuration, rewritten only inside the maintenance planes: a
#: test or tool elsewhere may name an old file on purpose (this tool's own
#: tests do), and code outside the planes does not read maintenance notes.
_CODE_SUFFIXES = frozenset({".py", ".cfg", ".ini", ".in"})


def _rewritable(path: str) -> bool:
    """Whether references in ``path`` are rewritten after a rename."""
    suffix = Path(path).suffix
    return suffix in _DOC_SUFFIXES or (
        suffix in _CODE_SUFFIXES and path.startswith(NAME_SCOPES)
    )


_ID_TOKEN = re.compile(r"^[A-Za-z]*\d[\w-]*$")
_PROJECT_NAME = re.compile(r'^\s*name\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)

BASELINE = Path(__file__).with_name("path_length_baseline.txt")


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """The clone-prefix model and the path length it leaves."""

    longest_distribution: str
    prefix_example: str
    prefix_length: int
    budget: int


def distribution_names(root: Path) -> list[str]:
    """Return ``scikit-plots`` and every ``libs/*/pyproject.toml`` project name."""
    names = ["scikit-plots"]
    for pyproject in sorted((root / "libs").glob("*/pyproject.toml")):
        text = pyproject.read_text(encoding="utf-8")
        section = text.split("[project]", 1)
        if len(section) == 2:  # noqa: PLR2004 - a split in two
            match = _PROJECT_NAME.search(section[1].split("\n[", 1)[0])
            if match:
                names.append(match.group(1))
    return names


def compute_budget(root: Path, prefix_length: int | None = None) -> Budget:
    """
    Return the longest repository path that checks out under every pip prefix.

    Parameters
    ----------
    root : Path
        Repository root.
    prefix_length : int, optional
        Use this clone-prefix length instead of the computed worst case
        (for example ``108`` to reproduce one user's report).
    """
    longest = max(distribution_names(root), key=len)
    example = (
        "C:\\Users\\"
        + "u" * USERNAME_MAX
        + "\\AppData\\Local\\Temp\\pip-install-"
        + "x" * 8
        + "\\"
        + longest
        + "_"
        + "0" * 32
        + "\\"
    )
    length = len(example) if prefix_length is None else prefix_length
    return Budget(longest, example, length, MAX_PATH - 1 - length)


# ---------------------------------------------------------------------------
# finding violations
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """One path that breaks a rule."""

    path: str
    rule: str
    length: int
    limit: int
    known: bool = False


@dataclass
class Report:
    """What ``check`` found."""

    budget: Budget
    violations: list[Violation] = field(default_factory=list)
    stale_baseline: list[str] = field(default_factory=list)

    @property
    def failing(self) -> list[Violation]:
        """Violations the baseline does not accept."""
        return [item for item in self.violations if not item.known]

    @property
    def ok(self) -> bool:
        """Whether ``check`` passes: nothing failing, nothing stale."""
        return not self.failing and not self.stale_baseline


def tracked_files(root: Path) -> list[str]:
    """
    Return repository-relative file paths, with ``/`` separators, sorted.

    Notes
    -----
    In a Git checkout this is ``git ls-files`` — exactly what a clone creates.
    Elsewhere (an exported archive) every file except caches is listed.
    """
    if (root / ".git").exists():
        try:
            done = subprocess.run(  # noqa: S603 - fixed argument list
                ["git", "-C", str(root), "ls-files", "-z"],  # noqa: S607 - git from PATH
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            return sorted(p for p in done.stdout.decode("utf-8").split("\0") if p)
    found = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        rel = Path(current).relative_to(root)
        found.extend((rel / name).as_posix() for name in files)
    return sorted(found)


def read_baseline(path: Path = BASELINE) -> dict[str, str]:
    """Return ``{path: reason}`` from the baseline file (``path  # reason``)."""
    if not path.is_file():
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        entry, _, reason = text.partition("#")
        entries[entry.strip()] = reason.strip()
    return entries


def _in_name_scope(path: str) -> bool:
    return path.endswith(".md") and path.startswith(NAME_SCOPES)


def check(
    root: Path,
    files: Sequence[str] | None = None,
    baseline: dict[str, str] | None = None,
    prefix_length: int | None = None,
) -> Report:
    """Return every violation, marking those the baseline accepts."""
    budget = compute_budget(root, prefix_length)
    files = tracked_files(root) if files is None else list(files)
    baseline = read_baseline() if baseline is None else baseline
    report = Report(budget)
    for path in files:
        if len(path) > budget.budget:
            report.violations.append(
                Violation(path, "long-path", len(path), budget.budget, path in baseline)
            )
        name = path.rsplit("/", 1)[-1]
        if _in_name_scope(path) and len(name) > NAME_LIMIT:
            report.violations.append(
                Violation(path, "long-name", len(name), NAME_LIMIT)
            )
    if prefix_length is not None:
        # The baseline describes the worst case; under another prefix an entry
        # that fits is expected, not stale.
        return report
    violating = {item.path for item in report.violations}
    present = set(files)
    report.stale_baseline = sorted(
        entry for entry in baseline if entry not in violating or entry not in present
    )
    return report


# ---------------------------------------------------------------------------
# shortening names
# ---------------------------------------------------------------------------


def _split(stem: str) -> tuple[str, list[str]]:
    sep = "_" if "_" in stem else "-"
    return sep, [token for token in stem.split(sep) if token]


def _common_affixes(stems: list[list[str]]) -> tuple[int, int]:
    """Return how many leading and trailing tokens every stem shares."""
    if len(stems) < 2:  # noqa: PLR2004 - "shared" needs two names
        return 0, 0
    shortest = min(len(tokens) for tokens in stems)
    lead = 0
    while lead < shortest - 1 and len({tokens[lead].upper() for tokens in stems}) == 1:
        lead += 1
    trail = 0
    while (
        trail < shortest - 1 - lead
        and len({tokens[-1 - trail].upper() for tokens in stems}) == 1
    ):
        trail += 1
    return lead, trail


def _fits(directory: str, name: str, budget: int) -> bool:
    return (
        len(f"{directory}/{name}" if directory else name) <= budget
        and len(name) <= NAME_LIMIT
    )


def shorten(
    directory: str, name: str, budget: int, lead: int = 0, trail: int = 0
) -> str | None:
    """
    Return a shorter name for ``directory/name`` that fits, or ``None``.

    ``lead``/``trail`` tokens are dropped first (the directory's shared
    affixes); then filler words; then trailing words, keeping a leading
    identifier and one word after it.
    """
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    suffix = f".{ext}" if ext else ""
    sep, tokens = _split(stem)
    parent = directory.rsplit("/", 1)[-1]
    _psep, parent_tokens = _split(parent.strip("_"))
    upper_parent = [token.upper() for token in parent_tokens]
    if [t.upper() for t in tokens[: len(upper_parent)]] == upper_parent and len(
        tokens
    ) > len(upper_parent):
        tokens = tokens[len(upper_parent) :]
        lead = max(0, lead - len(upper_parent))
    if lead or trail:
        kept = tokens[lead : len(tokens) - trail if trail else None]
        tokens = kept or tokens

    def render(parts: list[str]) -> str:
        return sep.join(parts) + suffix

    if _fits(directory, render(tokens), budget):
        return render(tokens)
    keep_first = bool(tokens) and bool(_ID_TOKEN.match(tokens[0]))
    tokens = [
        t
        for i, t in enumerate(tokens)
        if (i == 0 and keep_first) or t.upper() not in FILLER
    ] or tokens
    floor = 2 if keep_first else 1
    while len(tokens) > floor and not _fits(directory, render(tokens), budget):
        tokens = tokens[:-1]
    candidate = render(tokens)
    return candidate if _fits(directory, candidate, budget) else None


@dataclass
class Rename:
    """One planned rename."""

    old: str
    new: str


def plan(
    root: Path, report: Report, files: Sequence[str] | None = None
) -> tuple[list[Rename], list[str]]:
    """
    Return the renames that fix every maintenance-Markdown violation.

    Returns
    -------
    renames : list of Rename
        Planned renames, directory by directory.
    refused : list of str
        Paths that cannot be shortened automatically.
    """
    files = tracked_files(root) if files is None else list(files)
    targets = {
        item.path
        for item in report.violations
        if not item.known and _in_name_scope(item.path)
    }
    by_dir: dict[str, list[str]] = {}
    for path in files:
        directory, _, name = path.rpartition("/")
        by_dir.setdefault(directory, []).append(name)
    renames: list[Rename] = []
    refused: list[str] = []
    for directory in sorted({path.rpartition("/")[0] for path in targets}):
        siblings = sorted(by_dir.get(directory, []))
        markdown = [name for name in siblings if name.endswith(".md")]
        lead, trail = _common_affixes([_split(name[:-3])[1] for name in markdown])
        normalise = (
            markdown
            if (lead or trail)
            else [name for name in markdown if f"{directory}/{name}" in targets]
        )
        taken = set(siblings) - set(normalise)
        for name in normalise:
            new = shorten(directory, name, report.budget.budget, lead, trail)
            if new is None or new in taken:
                refused.append(f"{directory}/{name}")
                continue
            taken.add(new)
            if new != name:
                renames.append(Rename(f"{directory}/{name}", f"{directory}/{new}"))
    return renames, refused


def apply_renames(
    root: Path, renames: Iterable[Rename], files: Sequence[str]
) -> dict[str, int]:
    """
    Rename each file and rewrite references to it; return edits per old path.

    References are rewritten as the full repository path, and as the bare
    file name and stem when that name is unique in the repository (a name
    shared by two files cannot be rewritten safely and is left alone).
    """
    renames = list(renames)
    names = [path.rsplit("/", 1)[-1] for path in files]
    counts: dict[str, int] = {}
    replacements: list[tuple[str, str, str]] = []
    for item in renames:
        old_name, new_name = item.old.rsplit("/", 1)[-1], item.new.rsplit("/", 1)[-1]
        replacements.append((item.old, item.new, item.old))
        if names.count(old_name) == 1:
            replacements.append((old_name, new_name, item.old))
            old_stem, new_stem = old_name.rsplit(".", 1)[0], new_name.rsplit(".", 1)[0]
            replacements.append((old_stem, new_stem, item.old))
        counts[item.old] = 0
    use_git = (root / ".git").exists()
    for item in renames:
        if use_git:
            subprocess.run(  # noqa: S603 - fixed argument list, paths validated by plan()
                ["git", "-C", str(root), "mv", item.old, item.new],  # noqa: S607 - git from PATH
                check=True,
            )
        else:
            (root / item.old).rename(root / item.new)
    moved = {item.old: item.new for item in renames}
    replacements.sort(key=lambda rep: len(rep[0]), reverse=True)  # longest first
    for path in files:
        current = moved.get(path, path)
        target = root / current
        if not _rewritable(current) or not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new, owner in replacements:
            if old in updated:
                counts[owner] += updated.count(old)
                updated = updated.replace(old, new)
        if updated != text:
            target.write_text(updated, encoding="utf-8")
    return counts


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_path_lengths.py",
        description="Keep repository paths short enough to check out on Windows.",
    )
    parser.add_argument("command", choices=("budget", "check", "suggest", "fix"))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=None,
        help="Clone-prefix length to use instead of the computed worst case.",
    )
    parser.add_argument("--apply", action="store_true", help="With fix: really rename.")
    return parser


def _emit_json(data: object, out: IO[str]) -> None:
    json.dump(data, out, indent=2)
    out.write("\n")


def _cmd_budget(budget: Budget, fmt: str, out: IO[str]) -> int:
    if fmt == "json":
        _emit_json(asdict(budget), out)
        return 0
    out.write(
        f"longest distribution : {budget.longest_distribution}\n"
        f"worst clone prefix   : {budget.prefix_example} ({budget.prefix_length} chars)\n"
        f"path budget          : {budget.budget} characters "
        f"(MAX_PATH {MAX_PATH} - 1 - {budget.prefix_length})\n"
    )
    return 0


def _cmd_check(report: Report, fmt: str, out: IO[str]) -> int:
    budget = report.budget
    if fmt == "json":
        _emit_json(
            {
                "budget": asdict(budget),
                "violations": [asdict(item) for item in report.violations],
                "stale_baseline": report.stale_baseline,
                "ok": report.ok,
            },
            out,
        )
        return 0 if report.ok else 1
    out.write(f"path budget {budget.budget} (clone prefix {budget.prefix_length})\n")
    for item in report.violations:
        status = "known" if item.known else "FAIL "
        out.write(
            f"{status} {item.rule:9} {item.length:4} > {item.limit:<4} {item.path}\n"
        )
    for entry in report.stale_baseline:
        out.write(f"FAIL  stale baseline entry (no longer violates): {entry}\n")
    out.write(
        f"{len(report.failing)} failing, "
        f"{len(report.violations) - len(report.failing)} known, "
        f"{len(report.stale_baseline)} stale\n"
    )
    if report.failing:
        out.write("hint: python tools/maint_tools/check_path_lengths.py suggest\n")
    return 0 if report.ok else 1


def _cmd_fix(
    root: Path, report: Report, files: list, args: argparse.Namespace, out: IO[str]
) -> int:
    renames, refused = plan(root, report, files)
    if args.format == "json":
        _emit_json(
            {
                "renames": [asdict(r) for r in renames],
                "refused": refused,
                "applied": False,
            },
            out,
        )
    else:
        for item in renames:
            out.write(f"{item.old}\n  -> {item.new.rsplit('/', 1)[-1]}\n")
        for path in refused:
            out.write(f"cannot shorten automatically: {path}\n")
    if args.command == "suggest" or not args.apply:
        if args.command == "fix":
            out.write("dry run: add --apply to rename and rewrite references\n")
        return 0 if not refused else 1
    counts = apply_renames(root, renames, files)
    for old, count in counts.items():
        out.write(f"renamed {old} ({count} reference(s) rewritten)\n")
    return 0 if not refused else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Run the tool; return the process exit status."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.prefix_length is not None and not 0 < args.prefix_length < MAX_PATH:
        err.write(f"error: --prefix-length must be between 1 and {MAX_PATH - 1}\n")
        return 2
    files = tracked_files(root)
    report = check(root, files, prefix_length=args.prefix_length)
    if args.command == "budget":
        return _cmd_budget(report.budget, args.format, out)
    if args.command == "check":
        return _cmd_check(report, args.format, out)
    return _cmd_fix(root, report, files, args, out)


if __name__ == "__main__":
    raise SystemExit(main())
