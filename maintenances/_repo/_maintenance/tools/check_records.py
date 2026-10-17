# maintenances/_repo/_maintenance/tools/check_records.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Check the repository's work records: where they live, and that they hold.

Since increment 4R the records live by plane instead of in two files under
``tasks/``:

* ``maintenances/<area>/todo/todo.md`` - active tasks of one area;
* ``maintenances/<area>/_maintenance/history/`` - completed tasks, verbatim;
* ``maintenances/_repo/`` - repository-wide work (CI, ``libs/``, labeling);
* ``docs/tasks/`` - the documentation site;
* ``skills/_engineering/`` - the numbered engineering rules, by category;
* ``tasks/README.md`` - a router, nothing else.

Usage::

    python -B maintenances/_repo/_maintenance/tools/check_records.py [--json]
    python -B maintenances/_repo/_maintenance/tools/check_records.py --write-index
    python -B maintenances/_repo/_maintenance/tools/check_records.py --prove <commit>

Notes
-----
**User.** Each check prints what is wrong and where; fix the record, or for
a generated index run ``--write-index``. Exit status 0 means every check
passed.

**Developer.** The checks are pure functions of the files under the
repository root, so a test can run them on a copy with one file broken.

* Archived parts are found by the sha256 recorded when they were moved
  (``records_manifest.json``): a part cannot change or disappear unnoticed.
  Text may be added around a part (a note, another part); active files are
  not hash-checked, because they are meant to change.
* Fenced code blocks are skipped when rule headings and links are read, so a
  rule quoted in an example is not counted and a link in a shell snippet is
  not resolved.
* ``HISTORY_INDEX.md`` lists, per archive, the areas whose packages the
  archived text names as paths (:func:`areas_named`). It is a finding aid,
  not a claim about what a task changed; a grep of the archives is the
  complete search.
* ``--prove`` rebuilds the sources of a move from ``git show`` (Rule 124).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

#: Repository root: this file is maintenances/_repo/_maintenance/tools/.
DEFAULT_ROOT = Path(__file__).resolve().parents[4]

ROUTER = "tasks/README.md"
SKILL_DIR = "skills/_engineering"
RULES_DIR = f"{SKILL_DIR}/rules"
INDEX = f"{SKILL_DIR}/INDEX.md"
SKILL = f"{SKILL_DIR}/SKILL.md"
MANIFEST = "maintenances/_repo/_maintenance/records_manifest.json"
HISTORY_INDEX = "maintenances/_repo/_maintenance/HISTORY_INDEX.md"

#: Line budgets. A session reads the router, the skill and its active todo
#: whole at the start; past these sizes they stop being an overview, and
#: completed work belongs in ``_maintenance/history/``.
ROUTER_MAX_LINES = 80
SKILL_MAX_LINES = 150
ACTIVE_TODO_MAX_LINES = 400

#: Files whose relative links must resolve. Archives are left out: they are
#: kept verbatim, links included.
LINKED = (
    ROUTER,
    f"{SKILL_DIR}/**/*.md",
    "docs/tasks/**/*.md",
    "docs/maintenances/*.md",
    "docs/skills/*.md",
    "maintenances/_repo/*.md",
    "maintenances/**/todo/*.md",
    "maintenances/**/MAINTAINING.md",
    "skills/**/SKILL.md",
)

_RULE = re.compile(r"^### Rule (\d+): (.+?) - (.+)$", re.MULTILINE)
#: Anything that looks like a rule heading (up to three spaces of indent,
#: "Rule" or "Rules", an optional "#"); each must also match ``_RULE``.
_RULE_LIKE = re.compile(
    r"^[ \t]{0,3}#{1,6}[ \t]*Rules?[ \t]*#?[ \t]*\d+", re.MULTILINE | re.IGNORECASE
)
_CATEGORIES = re.compile(r"^Categories: (.+)$", re.MULTILINE)
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
#: Inline link targets: ``<...>``, or text with one level of balanced
#: parentheses, optionally followed by a title.
_INLINE_LINK = re.compile(
    r"\]\(\s*(?:<([^<>\n]*)>|((?:[^()\s]|\([^()\s]*\))+))(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
#: Reference definitions; ``[^1]:`` footnotes are not links.
_REFERENCE_LINK = re.compile(
    r"^[ \t]{0,3}\[(?!\^)[^\]]+\]:[ \t]*<?([^\s>]+)>?", re.MULTILINE
)
_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*:")
_OTHER_AREAS = re.compile(r"(?<![\w./-])(libs|docs|\.github)/")
_PREFIXED = re.compile(
    r"(?<![\w./-])scikitplot[/.]([A-Za-z0-9_]+(?:[/.][A-Za-z0-9_]+)*)"
)
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./{}-]*")
#: Package folders that are never areas.
_NOT_AREAS = ("tests", "test", "__pycache__")


class Rule(NamedTuple):
    """One numbered rule: its id, category, title and the file holding it."""

    number: int
    category: str
    title: str
    file: str


def read(root: Path, rel: str) -> str:
    """Return the UTF-8 text of *rel* under *root*."""
    return (root / rel).read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    """Write *text* to *path* as UTF-8 with LF line ends (works on Python 3.8+)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def blank_fences(text: str) -> tuple[str, int | None]:
    """Blank every fenced code block of *text*, keeping its lines.

    Parameters
    ----------
    text : str
        Markdown.

    Returns
    -------
    text : str
        The same number of lines; each line inside a fence (the fence lines
        included) is empty, so line numbers are kept.
    unclosed : int or None
        The line of a fence still open at the end of *text* (everything
        after it would be read as code), else ``None``.
    """
    out = []
    fence, opened = "", 0
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        match = _FENCE.match(line)
        if fence:
            if (
                match
                and match.group(1)[0] == fence[0]
                and len(match.group(1)) >= len(fence)
                and not line[match.end() :].strip()
            ):
                fence = ""
            out.append("\n" if line.endswith("\n") else "")
        elif match:
            fence, opened = match.group(1), number
            out.append("\n" if line.endswith("\n") else "")
        else:
            out.append(line)
    return "".join(out), (opened if fence else None)


def outside_fences(text: str) -> str:
    """Return *text* with every fenced code block blanked (see :func:`blank_fences`)."""
    return blank_fences(text)[0]


def is_archive(rel: str) -> bool:
    """Whether the repository path *rel* is inside an archive folder.

    Archives are ``<area>/_maintenance/history/`` and ``docs/tasks/history/``;
    a live folder that happens to be called ``history`` is not one.
    """
    parts = rel.split("/")
    return any(
        a == "_maintenance" and b == "history" for a, b in zip(parts, parts[1:])
    ) or rel.startswith("docs/tasks/history/")


def parse_rules(root: Path) -> tuple[list[Rule], list[str]]:
    """Parse every rules file.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    rules : list of Rule
        Files in name order; within a file, in the order the rules appear.
    errors : list of str
        A rules file below a subdirectory, a file without a ``Categories:``
        line, a category declared in two files, a rule heading in any other
        form than ``### Rule N: Category - Title``, a rule whose category its
        file does not declare, a duplicate or missing number.
    """
    rules: list[Rule] = []
    errors: list[str] = []
    declared: dict[str, str] = {}
    base = root / RULES_DIR
    errors.extend(
        f"{p.relative_to(root).as_posix()}: rules files live directly in {RULES_DIR}/"
        for p in sorted(base.rglob("*.md"))
        if p.parent != base
    )
    files = sorted(base.glob("*.md"))
    if not files:
        return [], [*errors, f"no rules files in {RULES_DIR}/"]
    for path in files:
        rel = path.relative_to(root).as_posix()
        text, unclosed = blank_fences(path.read_text(encoding="utf-8"))
        if unclosed is not None:
            errors.append(f"{rel}:{unclosed}: code fence never closed")
        line = _CATEGORIES.search(text)
        if line is None:
            errors.append(f"{rel}: no 'Categories:' line")
            continue
        cats = [c.strip() for c in line.group(1).split(",")]
        for cat in cats:
            if cat in declared:
                errors.append(
                    f"{rel}: category {cat!r} is also declared in {declared[cat]}"
                )
            declared[cat] = rel
        strict = {m.start() for m in _RULE.finditer(text)}
        errors.extend(
            f"{rel}: malformed rule heading {text[m.start():text.find(chr(10), m.start())]!r}"
            " (write '### Rule N: Category - Title')"
            for m in _RULE_LIKE.finditer(text)
            if m.start() not in strict
        )
        for m in _RULE.finditer(text):
            rule = Rule(int(m.group(1)), m.group(2), m.group(3).strip(), rel)
            if rule.category not in cats:
                errors.append(
                    f"{rel}: Rule {rule.number} has category {rule.category!r},"
                    f" which this file does not declare ({', '.join(cats)})"
                )
            rules.append(rule)
    numbers = sorted(r.number for r in rules)
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        errors.append(f"duplicate rule numbers: {duplicates}")
    if numbers:
        missing = sorted(set(range(1, numbers[-1] + 1)) - set(numbers))
        if missing:
            errors.append(f"missing rule numbers: {missing}")
    return rules, errors


def render_index(rules: list[Rule]) -> str:
    """Return the text of ``INDEX.md`` for *rules* (one row per rule)."""
    lines = [
        "# Rule index",
        "",
        "Generated by `maintenances/_repo/_maintenance/tools/check_records.py",
        "--write-index`; do not edit. One row per rule, by number: read the",
        "titles, then open the file that holds the rules you need.",
        "",
        "| Rule | Category | Title | File |",
        "|---|---|---|---|",
    ]
    for rule in sorted(rules, key=lambda r: r.number):
        name = rule.file.rsplit("/", 1)[1]
        title = rule.title.replace("|", "\\|")
        lines.append(
            f"| {rule.number} | {rule.category} | {title} | [{name}](rules/{name}) |"
        )
    return "\n".join(lines) + "\n"


def _archived_parts(root: Path) -> list[dict]:
    """Return the manifest parts whose destination is an archive."""
    manifest = json.loads(read(root, MANIFEST))
    return [
        dict(part, source=source)
        for source, record in manifest["sources"].items()
        for part in record["parts"]
        if is_archive(part["dest"])
    ]


def _find_part(text: str, heading: str, chars: int, digest: str) -> str | None:
    """Return the part of *text* that starts a line with *heading* and has *digest*."""
    start = 0
    while True:
        i = text.find(heading, start)
        if i < 0:
            return None
        if i == 0 or text[i - 1] == "\n":
            part = text[i : i + chars]
            if hashlib.sha256(part.encode("utf-8")).hexdigest() == digest:
                return part
        start = i + 1


def areas(root: Path) -> tuple[frozenset[str], dict[str, str]]:
    """Return the packages of ``scikitplot/`` and the unique underscore leaves.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    packages : frozenset of str
        Paths below ``scikitplot/`` of folders holding an ``__init__.py``,
        test packages left out. Empty or stray local folders do not count.
    leaves : dict
        ``_name`` -> package path, for underscore names that occur once
        (``_sphinx_feedback`` -> ``_externals/_sphinx_ext/_sphinx_feedback``).
    """
    base = root / "scikitplot"
    found = set()
    if base.is_dir():
        for init in base.rglob("__init__.py"):
            rel = init.parent.relative_to(base).as_posix()
            if rel != "." and not set(rel.split("/")) & set(_NOT_AREAS):
                found.add(rel)
    names: dict[str, list[str]] = {}
    for rel in found:
        names.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    leaves = {
        name: paths[0]
        for name, paths in names.items()
        if name.startswith("_") and len(paths) == 1
    }
    return frozenset(found), leaves


def area_of(root: Path, package: str) -> str:
    """Return the maintenance area that owns *package*.

    The area is the longest prefix of *package* whose folder under
    ``maintenances/`` holds a file; a package with none is its own top-level
    folder.
    """
    segments = package.split("/")
    for k in range(len(segments), 0, -1):
        candidate = "/".join(segments[:k])
        folder = root / "maintenances" / candidate
        if folder.is_dir() and any(p.is_file() for p in folder.rglob("*")):
            return candidate
    return segments[0]


def areas_named(
    text: str, root: Path, known: tuple[frozenset[str], dict[str, str]]
) -> list[str]:
    """Return the sorted areas whose packages *text* names.

    Counted: ``scikitplot/<path>`` or ``scikitplot.<dotted>`` anywhere; and,
    inside backticks, a relative path (with ``/``) whose first segment is a
    top-level package or a unique underscore leaf, or such a leaf alone. A
    name maps to its longest package, then to the area that owns it
    (:func:`area_of`). Also ``libs/``, ``docs/`` and ``.github/``.
    """
    packages, leaves = known
    top = {p for p in packages if "/" not in p}

    def longest(segments: list[str]) -> str | None:
        for k in range(len(segments), 0, -1):
            candidate = "/".join(segments[:k])
            if candidate in packages:
                return candidate
        return None

    named: set[str] = set()
    for match in _PREFIXED.finditer(text):
        package = longest(re.split(r"[/.]", match.group(1)))
        if package:
            named.add(package)
    for span in _CODE_SPAN.finditer(text):
        for raw in _TOKEN.findall(span.group(2)):
            token = raw.strip("./-")
            if "/" in token:
                segments = token.split("/")
                if segments[0] in leaves:
                    segments = leaves[segments[0]].split("/") + segments[1:]
                elif segments[0] not in top:
                    continue
                package = longest(segments)
            else:
                package = leaves.get(token)
            if package:
                named.add(package)
    found = {area_of(root, package) for package in named}
    found.update(m.group(1) for m in _OTHER_AREAS.finditer(text))
    return sorted(found)


def render_history_index(root: Path) -> str:
    """Return the text of ``HISTORY_INDEX.md``: each archive and the areas it names."""
    known = areas(root)
    by_archive: dict[str, list[dict]] = {}
    for part in _archived_parts(root):
        by_archive.setdefault(part["dest"], []).append(part)
    lines = [
        "# History index",
        "",
        "Generated by `maintenances/_repo/_maintenance/tools/check_records.py",
        "--write-index`; do not edit. Every archived record, with the areas whose",
        "packages its text names as paths (a finding aid, see `areas_named`).",
        "For a complete search, grep the archives for the area's name.",
        "",
        "| Archive | Records | Areas named |",
        "|---|---|---|",
    ]
    for dest in sorted(by_archive):
        text = (
            (root / dest).read_text(encoding="utf-8") if (root / dest).is_file() else ""
        )
        named: set[str] = set()
        titles = []
        for part in sorted(by_archive[dest], key=lambda p: (p["source"], p["line"])):
            found = (
                _find_part(text, part["heading"], part["chars"], part["sha256"]) or ""
            )
            named.update(areas_named(found, root, known))
            titles.append(part["heading"].lstrip("# ").replace("|", "\\|"))
        link = (
            Path(dest).relative_to("maintenances/_repo/_maintenance")
            if dest.startswith("maintenances/_repo/_maintenance/")
            else Path("../../..", dest)
        )
        lines.append(
            f"| [`{dest}`]({link.as_posix()}) | {'<br>'.join(titles)} |"
            f" {', '.join(f'`{a}`' for a in sorted(named))} |"
        )
    return "\n".join(lines) + "\n"


def check_router(root: Path) -> list[str]:
    """Check that ``tasks/`` holds only the router."""
    errors = []
    folder = root / "tasks"
    others = (
        sorted(
            p.relative_to(root).as_posix()
            for p in folder.rglob("*")
            if (p.is_file() or p.is_symlink())
            and p.relative_to(root).as_posix() != ROUTER
        )
        if folder.is_dir()
        else []
    )
    if not (root / ROUTER).is_file():
        errors.append(f"{ROUTER} is missing")
    errors.extend(
        f"{rel}: tasks/ holds only the router; move this record to its"
        " plane (see tasks/README.md)"
        for rel in others
    )
    return errors


def check_rules(root: Path) -> list[str]:
    """Check rule numbering, categories, the skill's links and the index."""
    rules, errors = parse_rules(root)
    skill = read(root, SKILL) if (root / SKILL).is_file() else ""
    if not skill:
        errors.append(f"{SKILL} is missing")
    links = {target for target, _ in _links(skill)}
    errors.extend(
        f"{SKILL} does not link rules/{path.name}"
        for path in sorted((root / RULES_DIR).glob("*.md"))
        if f"rules/{path.name}" not in links
    )
    actual = read(root, INDEX) if (root / INDEX).is_file() else ""
    if actual != render_index(rules):
        errors.append(f"{INDEX} is out of date (run with --write-index)")
    return errors


def check_archives(root: Path) -> list[str]:
    """Check that every archived part is present, unchanged, in its destination."""
    if not (root / MANIFEST).is_file():
        return [f"{MANIFEST} is missing"]
    errors: list[str] = []
    for part in _archived_parts(root):
        dest, where = part["dest"], f"{part['source']} line {part['line']}"
        path = root / dest
        if not path.is_file():
            errors.append(f"{dest}: missing (archived from {where})")
            continue
        text = path.read_text(encoding="utf-8")
        if _find_part(text, part["heading"], part["chars"], part["sha256"]) is None:
            errors.append(
                f"{dest}: the part archived from {where}"
                f" ({part['heading'][:60]!r}) was changed or removed"
            )
    actual = read(root, HISTORY_INDEX) if (root / HISTORY_INDEX).is_file() else ""
    if not errors and actual != render_history_index(root):
        errors.append(f"{HISTORY_INDEX} is out of date (run with --write-index)")
    return errors


def _links(text: str) -> list[tuple[str, int]]:
    """Return ``(target, line)`` for each link outside fences and code spans."""
    text = _CODE_SPAN.sub(lambda m: " " * len(m.group(0)), outside_fences(text))

    def line_of(match: re.Match[str]) -> int:
        return text.count("\n", 0, match.start()) + 1

    found = [
        (match.group(1) or match.group(2), line_of(match))
        for match in _INLINE_LINK.finditer(text)
    ]
    found.extend(
        (match.group(1), line_of(match)) for match in _REFERENCE_LINK.finditer(text)
    )
    return found


def check_links(root: Path) -> list[str]:
    """Check that relative links in the live records resolve inside the repository."""
    errors = []
    files: set[Path] = set()
    for pattern in LINKED:
        files.update(
            p
            for p in root.glob(pattern)
            if p.is_file() and not is_archive(p.relative_to(root).as_posix())
        )
    # Lexical normalisation, not resolve(): a symlinked folder of the checkout
    # is inside the repository wherever it points.
    top = os.path.normpath(root)
    for path in sorted(files):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        unclosed = blank_fences(text)[1]
        if unclosed is not None:
            errors.append(f"{rel}:{unclosed}: code fence never closed")
        for target, line in _links(text):
            if _URL.match(target) or target.startswith("#"):
                continue
            where = f"{rel}:{line}: link target {target!r}"
            if target.startswith("/"):
                errors.append(f"{where} is absolute; write it relative to the file")
                continue
            resolved = os.path.normpath(path.parent / unquote(target.split("#", 1)[0]))
            if os.path.commonpath([top, resolved]) != top:
                errors.append(f"{where} points outside the repository")
            elif not os.path.exists(resolved):
                errors.append(f"{where} does not exist")
    return errors


def check_budgets(root: Path) -> list[str]:
    """Keep the router, the skill and every active ``todo.md`` within budget."""
    errors = []
    active = [
        p
        for p in (
            *root.glob("maintenances/**/todo/todo.md"),
            *root.glob("docs/tasks/todo.md"),
        )
        if not is_archive(p.relative_to(root).as_posix())
    ]
    budgets = [(root / ROUTER, ROUTER_MAX_LINES), (root / SKILL, SKILL_MAX_LINES)]
    budgets.extend((path, ACTIVE_TODO_MAX_LINES) for path in active)
    for path, limit in budgets:
        if not path.is_file():
            continue
        n = len(path.read_text(encoding="utf-8").splitlines())
        if n > limit:
            errors.append(
                f"{path.relative_to(root).as_posix()}: {n} lines, budget {limit};"
                " archive completed tasks to its _maintenance/history/"
            )
    return errors


CHECKS = {
    "router": check_router,
    "rules": check_rules,
    "archives": check_archives,
    "links": check_links,
    "budgets": check_budgets,
}


def run(root: Path) -> dict[str, list[str]]:
    """Run every check on *root*; return the errors by check name."""
    return {name: check(root) for name, check in CHECKS.items()}


def prove(root: Path, commit: str) -> list[str]:
    """Check that the manifest's parts at *commit* rebuild their sources.

    Parameters
    ----------
    root : Path
        A git checkout.
    commit : str
        The commit of the move (``4R-1`` for increment 4R). Its manifest,
        archives and active files are read with ``git show``; each source is
        read at the manifest's ``source_commit``.

    Returns
    -------
    list of str
        Empty when every source is rebuilt byte for byte (Rule 124).
    """

    git = shutil.which("git")
    if git is None:
        return ["--prove needs git on PATH"]

    def show(revision: str, rel: str) -> str:
        # A fixed argument list, no shell: revision and path are data.
        return subprocess.run(  # noqa: S603
            [git, "-C", str(root), "show", f"{revision}:{rel}"],
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")

    manifest = json.loads(show(commit, MANIFEST))
    errors = []
    for source, record in manifest["sources"].items():
        texts: dict[str, str] = {}
        rebuilt = []
        for part in record["parts"]:
            dest = part["dest"]
            if dest not in texts:
                texts[dest] = (
                    record["preamble"] if dest == "preamble" else show(commit, dest)
                )
            found = _find_part(
                texts[dest], part["heading"], part["chars"], part["sha256"]
            )
            if found is None:
                errors.append(
                    f"{source} line {part['line']}: not found in {dest} at {commit}"
                )
            else:
                rebuilt.append(found)
        if "".join(rebuilt) != show(manifest["source_commit"], source):
            errors.append(f"{source}: the parts at {commit} do not rebuild it")
    return errors


def write_indexes(root: Path) -> None:
    """Regenerate ``INDEX.md`` and ``HISTORY_INDEX.md`` under *root*."""
    rules, _ = parse_rules(root)
    write(root / INDEX, render_index(rules))
    if (root / MANIFEST).is_file():
        write(root / HISTORY_INDEX, render_history_index(root))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument(
        "--write-index",
        action="store_true",
        help=f"regenerate {INDEX} and {HISTORY_INDEX}, then check",
    )
    parser.add_argument(
        "--prove",
        metavar="COMMIT",
        help="check that the move at COMMIT rebuilds its sources byte for byte, then exit",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.prove:
        problems = prove(root, args.prove)
        for problem in problems:
            sys.stdout.write(f"     - {problem}\n")
        sys.stdout.write(("FAIL" if problems else "ok  ") + f" prove {args.prove}\n")
        return 1 if problems else 0
    if args.write_index:
        write_indexes(root)
    errors = run(root)
    failed = any(errors.values())
    if args.json:
        report = {"status": "FAIL" if failed else "PASS", "errors": errors}
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        for name, found in errors.items():
            sys.stdout.write(f"{'FAIL' if found else 'ok  '} {name}\n")
            for error in found:
                sys.stdout.write(f"     - {error}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
