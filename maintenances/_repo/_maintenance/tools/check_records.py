# maintenances/_repo/_maintenance/tools/check_records.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Check the repository's work records: where they live, and that they hold.

Since CI-6 there are two homes, each split into groups: ``maintenances/``
for how the project is worked on (records), ``plugins/`` for what ships to
Claude (skills, agents, MCP):

* ``maintenances/<area>/todo/todo.md`` - active tasks of one area;
* ``maintenances/<area>/_maintenance/history/`` - completed tasks, verbatim;
* ``maintenances/_repo/`` - repository-wide work (CI, ``libs/``, labeling);
* ``maintenances/_docs/`` - the documentation site;
* ``plugins/scikitplot-dev/skills/scikitplot-engineering-rules/`` - the numbered engineering rules, by category;
* ``plugins/scikitplot-dev/skills/<name>/SKILL.md`` and ``plugins/scikitplot-dev/agents/<name>.md`` - development skills
  and their agents, ``plugins/scikitplot/`` - usage skills and agents for any
  project (``plugins/INDEX.md`` lists both by package; Rules 164 and 165);
* ``maintenances/README.md`` - the router; the retired locations
  (:data:`RETIRED`) must not come back.
* ``plugins/_shared/skills/`` - skills every plugin carries, copied byte for
  byte into each plugin by ``--write-index``.

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

ROUTER = "maintenances/README.md"
PLUGINS_DIR = "plugins"
DEV_PLUGIN_DIR = f"{PLUGINS_DIR}/scikitplot-dev"
SKILLS_DIR = f"{DEV_PLUGIN_DIR}/skills"
SKILL_DIR = f"{SKILLS_DIR}/scikitplot-engineering-rules"
RULES_DIR = f"{SKILL_DIR}/rules"
INDEX = f"{SKILL_DIR}/INDEX.md"
SKILL = f"{SKILL_DIR}/SKILL.md"
MANIFEST = "maintenances/_repo/_maintenance/records_manifest.json"
HISTORY_INDEX = "maintenances/_repo/_maintenance/HISTORY_INDEX.md"
#: Locations retired by CI-6: records live under ``maintenances/``, Claude
#: material under ``plugins/``. A file in any of them fails ``router``.
RETIRED = (
    "tasks",
    "docs/tasks",
    "docs/maintenances",
    "docs/skills",
    "skills",
    "agents",
    "plugins/maintenances",
    "scikitplot/mcp/_plugins/.claude-plugin",
)

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
    "maintenances/_docs/**/*.md",
    "maintenances/*.md",
    "plugins/scikitplot-dev/*.md",
    "maintenances/_repo/*.md",
    "maintenances/**/todo/*.md",
    "maintenances/**/MAINTAINING.md",
    "plugins/scikitplot-dev/skills/**/SKILL.md",
    "plugins/INDEX.md",
    "plugins/README.md",
    "maintenances/_plugins/*.md",
    "plugins/scikitplot/README.md",
    "plugins/scikitplot/skills/**/SKILL.md",
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

    Archives are ``<area>/_maintenance/history/`` and ``maintenances/_docs/_maintenance/history/``;
    a live folder that happens to be called ``history`` is not one.
    """
    parts = rel.split("/")
    return any(
        a == "_maintenance" and b == "history" for a, b in zip(parts, parts[1:])
    ) or rel.startswith("maintenances/_docs/_maintenance/history/")


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
            # A part that ends the file may have lost trailing newlines to an
            # end-of-file fixer (pre-commit end-of-file-fixer keeps exactly
            # one). Accept it only when the missing tail is newlines and the
            # file ends right there.
            missing = i + chars - len(text)
            if 0 < missing <= chars and text.endswith("\n"):
                padded = text[i:] + "\n" * missing
                if hashlib.sha256(padded.encode("utf-8")).hexdigest() == digest:
                    return padded
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
    """Check that the router exists and that no retired location holds a file.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list of str
        One problem per missing router or per file found under a path of
        :data:`RETIRED` (each such file belongs under ``maintenances/`` or
        ``plugins/``; see the router).
    """
    errors = []
    if not (root / ROUTER).is_file():
        errors.append(f"{ROUTER} is missing")
    for base in RETIRED:
        folder = root / base
        if not folder.is_dir():
            continue
        errors.extend(
            f"{p.relative_to(root).as_posix()}: {base}/ is retired; records live"
            f" under maintenances/, skills and agents under plugins/ (see {ROUTER})"
            for p in sorted(folder.rglob("*"))
            if p.is_file() or p.is_symlink()
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
    active = sorted(
        p
        for p in root.glob("maintenances/**/todo/todo.md")
        if not is_archive(p.relative_to(root).as_posix())
    )
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


# --------------------------------------------------------------------------- skills
#
# Skills follow the Agent Skills specification (https://agentskills.io/specification):
# a skill is ``<skills dir>/<name>/SKILL.md`` and ``name`` (1-64 characters of
# ``a-z``, ``0-9`` and single hyphens) equals its folder. There are two sets
# (Rules 164 and 165):
#
# * development skills, ``plugins/scikitplot-dev/skills/`` - working on scikit-plots in this
#   checkout. One per maintained path below ``scikitplot/`` and
#   ``maintenances/``, recorded as ``metadata.scikitplot-mirror`` and named
#   ``scikitplot-<derived>-dev``; plus the cross-cutting
#   ``scikitplot-engineering-rules``.
# * usage skills, ``plugins/scikitplot/skills/`` - using scikit-plots in any
#   project, one per public module, recorded as ``metadata.scikitplot-module``
#   and named ``scikitplot`` (root) or ``scikitplot-<derived>``. Generated by
#   ``tools/maint_tools/generate_usage_skills.py``.
#
# ``<derived>`` comes from the package path (:func:`derived_skill_names`), so
# two paths that differ only by a leading underscore (``externals``,
# ``_externals``) never share a name. Every skill of either set except the
# cross-cutting ones has one agent of the same name, generated here:
# ``plugins/scikitplot-dev/agents/<name>.md`` (development) and ``plugins/scikitplot/agents/<name>.md``.

#: The keys the specification defines for the frontmatter.
SKILL_KEYS = ("name", "description", "license", "compatibility", "metadata", "allowed-tools")
#: Where a development skill names the path of the checkout it maintains.
MIRROR_KEY = "scikitplot-mirror"
#: Where a usage skill names the module it explains.
MODULE_KEY = "scikitplot-module"
PREFIX = "scikitplot"
DEV_SUFFIX = "-dev"
#: Development skills that maintain no single path (no agent of their own).
CROSS_CUTTING_SKILLS = ("scikitplot-engineering-rules",)
#: Third-party skills kept as they were published.
VENDORED_SKILLS = ("skill-creator",)
SKILL_NAME_MAX = 64
SKILL_DESCRIPTION_MAX = 1024
SKILL_COMPATIBILITY_MAX = 500
SKILL_INDEX = f"{PLUGINS_DIR}/INDEX.md"
AGENTS_DIR = f"{DEV_PLUGIN_DIR}/agents"
PLUGIN_DIR = f"{PLUGINS_DIR}/scikitplot"
#: Skills every plugin carries: one source here, a generated copy in each of
#: :data:`SHARED_TARGETS` (plugins cannot reference files outside their own
#: folder; https://code.claude.com/docs/en/plugins/marketplace-reference.md).
SHARED_SKILLS_DIR = f"{PLUGINS_DIR}/_shared/skills"
MARKETPLACE = ".claude-plugin/marketplace.json"
USAGE_SKILLS_DIR = f"{PLUGIN_DIR}/skills"
USAGE_AGENTS_DIR = f"{PLUGIN_DIR}/agents"
SHARED_TARGETS = (SKILLS_DIR, f"{PLUGIN_DIR}/skills")
PLUGIN_README = f"{PLUGIN_DIR}/README.md"
#: The repository that serves released versions of the plugin (Rule 166).
STABLE_REPO = "scikit-plots/skills"
STABLE_MARKETPLACE = "scikit-plots-skills"
GUIDE_URL = "https://github.com/scikit-plots/scikit-plots/blob/main/plugins/README.md"
#: Files allowed directly in a skills folder (every folder there is one skill).
SKILLS_ROOT_FILES = (".gitignore",)
_SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_YAML_KEY = re.compile(r"([A-Za-z0-9][A-Za-z0-9_-]*):(?: (.*))?")
#: A plain YAML scalar may not start with an indicator character.
_YAML_INDICATORS = tuple("-?:,[]{}#&*!|>'\"%@`")


def _yaml_scalar(text: str) -> str:
    """
    Return the string a one-line YAML scalar denotes; raise for anything else.

    Parameters
    ----------
    text : str
        The value after ``key:``.

    Returns
    -------
    str
        The value.

    Raises
    ------
    ValueError
        For an empty value, a block scalar, an unterminated quote, or a plain
        scalar YAML would read differently (``: `` or `` #`` inside it, an
        indicator first): quote it.

    Notes
    -----
    **Developer notes.** The subset every skill here uses, read without a YAML
    library so the result does not depend on one being installed: plain,
    single-quoted (``''`` is a quote) and double-quoted (JSON escapes)
    scalars on one line. It is stricter than YAML, never looser: whatever it
    accepts, YAML reads as the same string.
    """
    value = text.strip()
    if not value:
        raise ValueError("an empty value")
    if value[0] == '"':
        if len(value) < 2 or value[-1] != '"':
            raise ValueError("an unterminated double-quoted string")
        return json.loads(value)
    if value[0] == "'":
        if len(value) < 2 or value[-1] != "'" or "'" in value[1:-1].replace("''", ""):
            raise ValueError("an unterminated single-quoted string")
        return value[1:-1].replace("''", "'")
    if value.startswith(_YAML_INDICATORS) or ": " in value or " #" in value or value.endswith(":"):
        raise ValueError(
            "a plain value YAML reads differently (it holds ': ' or ' #', or starts"
            " with an indicator); put it in double quotes"
        )
    return value


def parse_skill_frontmatter(text: str) -> tuple[dict[str, object], list[str]]:
    """
    Return a ``SKILL.md`` file's frontmatter and the problems found in it.

    Parameters
    ----------
    text : str
        The whole file.

    Returns
    -------
    fields : dict
        Top-level keys to strings; ``metadata`` to a dict of strings.
    problems : list of str
        Empty when the frontmatter is well formed.
    """
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}, ["does not start with a '---' frontmatter line"]
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, ["has no closing '---' frontmatter line"]
    fields: dict[str, object] = {}
    problems: list[str] = []
    block: dict[str, str] | None = None
    for number, line in enumerate(lines[1:end], start=2):
        indented = line[:1] in (" ", "\t")
        match = _YAML_KEY.fullmatch(line.strip())
        if not line.strip():
            continue
        if match is None:
            problems.append(f"line {number}: not 'key: value'")
            continue
        key, value = match.groups()
        target = block if indented else fields
        if indented and block is None:
            problems.append(f"line {number}: indented outside 'metadata:'")
            continue
        if key in target:
            problems.append(f"line {number}: {key!r} given twice")
            continue
        if not indented:
            block = None
            if value is None:
                if key != "metadata":
                    problems.append(f"line {number}: {key!r} has no value")
                    continue
                block = {}
                fields[key] = block
                continue
        try:
            target[key] = _yaml_scalar(value or "")
        except ValueError as exc:
            problems.append(f"line {number}: {key!r} is {exc}")
    return fields, problems


def mirror_keys(root: Path) -> set[str]:
    """
    Return every path a skill may mirror.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    set of str
        Each package below ``scikitplot/`` (a folder with ``__init__.py``,
        tests and caches left out) as a ``/`` path, and each folder directly
        under ``maintenances/`` (``_repo`` has no package).
    """
    keys: set[str] = set()
    package = root / "scikitplot"
    for folder, names, files in os.walk(package):
        names[:] = [
            n for n in names if n not in _NOT_AREAS and not n.startswith((".", "__"))
        ]
        if "__init__.py" in files and Path(folder) != package:
            keys.add(Path(folder).relative_to(package).as_posix())
    maintenances = root / "maintenances"
    if maintenances.is_dir():
        keys.update(
            p.name
            for p in maintenances.iterdir()
            if p.is_dir() and not p.name.startswith((".", "__"))
        )
    return keys


def _slug(part: str) -> str:
    return part.strip("_").replace("_", "-")


def derived_skill_names(keys: set[str]) -> dict[str, str]:
    """
    Return the name part each package path contributes to a skill name.

    Parameters
    ----------
    keys : set of str
        :func:`mirror_keys`.

    Returns
    -------
    dict
        Path to name part. It is the path's last part, with leading and
        trailing underscores dropped and ``_`` spelled ``-``; when a path
        nearer the root already has that, the parent's part is put in front,
        then the grandparent's, and so on. A path with no part left to add
        (``_externals`` after ``externals``) gets ``internal-`` in front.

    Notes
    -----
    **Developer notes.** Paths are taken nearest the root first, a public
    part before a private one of the same depth, then by spelling, so a name
    depends only on the package tree: ``annoy`` keeps ``annoy``, and
    ``cexternals/_annoy`` becomes ``cexternals-annoy``. Adding a package
    nearer the root can take a name a deeper skill holds; the check then
    names the skill to rename.
    """
    order = sorted(
        keys, key=lambda k: (k.count("/"), k.rsplit("/", 1)[-1].startswith("_"), k)
    )
    taken: dict[str, str] = {}
    names: dict[str, str] = {}
    for key in order:
        parts = key.split("/")
        for n in range(1, len(parts) + 1):
            name = "-".join(_slug(p) for p in parts[-n:])
            if name not in taken:
                break
        else:
            name = "internal-" + "-".join(_slug(p) for p in parts)
            if name in taken:  # cannot happen for paths that differ
                raise ValueError(f"{key} and {taken[name]} both derive {name!r}")
        taken[name] = key
        names[key] = name
    return names


def dev_skill_name(path: str, derived: dict[str, str]) -> str:
    """Return the development skill name of a mirrored path (Rule 164)."""
    return f"{PREFIX}-{derived[path]}{DEV_SUFFIX}"


def usage_skill_name(module: str, derived: dict[str, str]) -> str:
    """Return the usage skill name of a dotted module (Rule 164)."""
    if module == PREFIX:
        return PREFIX
    return f"{PREFIX}-{derived[module[len(PREFIX) + 1 :].replace('.', '/')]}"


class Skill(NamedTuple):
    base: str
    folder: str
    name: str
    description: str
    mirror: str


def read_skills(root: Path, base: str, key: str) -> tuple[list[Skill], list[str]]:
    """
    Read every ``SKILL.md`` under *base*; return the skills and problems.

    Parameters
    ----------
    root : Path
        Repository root.
    base : str
        A skills folder: :data:`SKILLS_DIR` or :data:`USAGE_SKILLS_DIR`.
    key : str
        The metadata key holding the skill's path or module.

    Returns
    -------
    skills : list of Skill
        One per readable ``<base>/<folder>/SKILL.md``; ``mirror`` is the
        value of *key*, ``""`` when absent.
    problems : list of str
        Specification and layout problems, each naming its file.
    """
    skills: list[Skill] = []
    problems: list[str] = []
    folder_path = root / base
    if not folder_path.is_dir():
        return skills, problems
    for path in sorted(folder_path.rglob("SKILL.md")):
        rel = path.relative_to(root).as_posix()
        if len(path.relative_to(folder_path).parts) != 2:
            problems.append(
                f"{rel}: a skill is {base}/<name>/SKILL.md; move it up and record"
                f" its path in metadata.{key}"
            )
            continue
        fields, found = parse_skill_frontmatter(path.read_text(encoding="utf-8"))
        problems.extend(f"{rel}: frontmatter {p}" for p in found)
        unknown = sorted(set(fields) - set(SKILL_KEYS))
        if unknown:
            problems.append(f"{rel}: frontmatter keys outside the specification: {unknown}")
        folder = path.parent.name
        name = fields.get("name", "")
        description = fields.get("description", "")
        metadata = fields.get("metadata", {})
        if not isinstance(name, str) or not name:
            problems.append(f"{rel}: no 'name'")
            name = ""
        elif len(name) > SKILL_NAME_MAX or not _SKILL_NAME.fullmatch(name):
            problems.append(
                f"{rel}: name {name!r} must be 1-{SKILL_NAME_MAX} characters of a-z,"
                " 0-9 and single hyphens, not first or last"
            )
        elif name != folder:
            problems.append(f"{rel}: name {name!r} must equal its folder {folder!r}")
        if not isinstance(description, str) or not description:
            problems.append(f"{rel}: no 'description'")
            description = ""
        elif len(description) > SKILL_DESCRIPTION_MAX:
            problems.append(
                f"{rel}: description is {len(description)} characters, at most"
                f" {SKILL_DESCRIPTION_MAX}"
            )
        compatibility = fields.get("compatibility")
        if isinstance(compatibility, str) and len(compatibility) > SKILL_COMPATIBILITY_MAX:
            problems.append(f"{rel}: compatibility is longer than {SKILL_COMPATIBILITY_MAX}")
        if not isinstance(metadata, dict):
            problems.append(f"{rel}: metadata must be a mapping")
            metadata = {}
        skills.append(Skill(base, folder, name, description, metadata.get(key, "")))
    for entry in sorted(folder_path.iterdir()):
        if entry.is_file() and entry.name not in SKILLS_ROOT_FILES:
            problems.append(f"{base}/{entry.name}: only skill folders belong here")
        if entry.is_dir() and not (entry / "SKILL.md").is_file():
            problems.append(f"{base}/{entry.name}/: a folder here is a skill; no SKILL.md")
    return skills, problems


def shared_skill_names(root: Path) -> tuple[str, ...]:
    """Return the folder names of the shared skills (``plugins/_shared/skills/``)."""
    base = root / SHARED_SKILLS_DIR
    if not base.is_dir():
        return ()
    return tuple(sorted(p.parent.name for p in base.glob("*/SKILL.md")))


def shared_copies(root: Path) -> dict[str, bytes]:
    """
    Return every generated copy of a shared skill, by repository path.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict
        ``<target>/<skill>/<file>`` -> the bytes of
        ``plugins/_shared/skills/<skill>/<file>``, for each target of
        :data:`SHARED_TARGETS`. Files are copied as bytes, so a copy is
        identical to its source.
    """
    out: dict[str, bytes] = {}
    base = root / SHARED_SKILLS_DIR
    for name in shared_skill_names(root):
        for path in sorted((base / name).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(base).as_posix()
            for target in SHARED_TARGETS:
                out[f"{target}/{rel}"] = path.read_bytes()
    return out


def dev_skills(root: Path) -> tuple[list[Skill], list[str]]:
    """Read the development skills (``plugins/scikitplot-dev/skills/``)."""
    return read_skills(root, SKILLS_DIR, MIRROR_KEY)


def usage_skills(root: Path) -> tuple[list[Skill], list[str]]:
    """Read the usage skills (``plugins/scikitplot/skills/``)."""
    return read_skills(root, USAGE_SKILLS_DIR, MODULE_KEY)


def skill_mirrors(root: Path) -> dict[str, str]:
    """
    Return mirrored path to development skill folder.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict
        ``{"_cli": "scikitplot-cli-dev", ...}``; the PR labeler maps a
        package's development skill and agent with it.
    """
    skills, _ = dev_skills(root)
    return {s.mirror: s.folder for s in skills if s.mirror}


def usage_mirrors(root: Path) -> dict[str, str]:
    """
    Return package path to usage skill folder (``""`` for the root package).

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict
        ``{"annoy": "scikitplot-annoy", "": "scikitplot", ...}``.
    """
    skills, _ = usage_skills(root)
    out = {}
    for skill in skills:
        if skill.mirror == PREFIX:
            out[""] = skill.folder
        elif skill.mirror.startswith(PREFIX + "."):
            out[skill.mirror[len(PREFIX) + 1 :].replace(".", "/")] = skill.folder
    return out


def _yaml_list(items: list[str]) -> list[str]:
    return [f"  - {item}" for item in items]


def render_dev_agent(skill: Skill) -> str:
    """
    Render ``plugins/scikitplot-dev/agents/<name>.md`` for a development skill.

    Parameters
    ----------
    skill : Skill
        A development skill with a mirror.

    Returns
    -------
    str
        A Claude Code subagent: same name as the skill, preloading it and
        the engineering rules, inheriting every tool.
    """
    path = skill.mirror
    description = (
        f"Develop and maintain scikit-plots' {path} in a scikit-plots checkout: change, test"
        f" and verify it under the project's numbered engineering rules. Use for work on"
        f" scikitplot/{path}/ or maintenances/{path}/."
    )
    lines = [
        "---",
        f"name: {skill.name}",
        f"description: {json.dumps(description)}",
        "skills:",
        *_yaml_list([skill.name, *CROSS_CUTTING_SKILLS]),
        "---",
        "",
        f"<!-- Generated by check_records.py --write-index from {SKILLS_DIR}/{skill.name}/SKILL.md (Rule 165). -->",
        "",
        f"You develop and maintain `{path}` of scikit-plots, working in a scikit-plots",
        "checkout (the folder holding `scikitplot/`, `maintenances/` and `plugins/`).",
        "",
        f"- Follow the preloaded `{skill.name}` skill: read what it lists first and run its checks.",
        "- Follow the numbered rules of `scikitplot-engineering-rules` for every change.",
        "- Find the root cause before changing code; keep the change to what the task needs.",
        "- Prove each claim with a command you ran and its output; say what you could not verify.",
        "- Previous chat history is not an authority; the checkout and its records are.",
    ]
    return "\n".join(lines) + "\n"


def render_usage_agent(skill: Skill) -> str:
    """
    Render ``plugins/scikitplot/agents/<name>.md`` for a usage skill.

    Parameters
    ----------
    skill : Skill
        A usage skill.

    Returns
    -------
    str
        A Claude Code subagent: same name as the skill, preloading it,
        inheriting every tool.
    """
    module = skill.mirror
    description = (
        f"Write, explain and fix code that uses {module} (scikit-plots) in the user's"
        f" project. Use when a task calls {module} or asks how to do something with it."
    )
    lines = [
        "---",
        f"name: {skill.name}",
        f"description: {json.dumps(description)}",
        "skills:",
        *_yaml_list([skill.name]),
        "---",
        "",
        f"<!-- Generated by check_records.py --write-index from {USAGE_SKILLS_DIR}/{skill.name}/SKILL.md (Rule 165). -->",
        "",
        f"You help people use `{module}` from scikit-plots in their own project.",
        "",
        f"- Start from the preloaded `{skill.name}` skill.",
        "- Check the installed version, and read a name's signature and docstring before",
        "  calling it; the skill was generated from one version.",
        "- Use only names the module declares in `__all__`; do not rely on private modules.",
        "- Run the code you write, or say plainly that it was not run.",
    ]
    return "\n".join(lines) + "\n"


def wanted_agents(root: Path) -> dict[str, str]:
    """Return every generated agent file, by repository path."""
    out = {}
    for skill in dev_skills(root)[0]:
        if skill.mirror:
            out[f"{AGENTS_DIR}/{skill.name}.md"] = render_dev_agent(skill)
    for skill in usage_skills(root)[0]:
        if skill.mirror:
            out[f"{USAGE_AGENTS_DIR}/{skill.name}.md"] = render_usage_agent(skill)
    return out


def _rel_to_plugins(path: str) -> str:
    """Return the repository path *path* relative to :data:`PLUGINS_DIR`."""
    return path[len(PLUGINS_DIR) + 1 :]


def render_skill_index(root: Path) -> str:
    """Render ``plugins/INDEX.md``: every plugin's skills, by path, with their agents."""
    dev, _ = dev_skills(root)
    usage, _ = usage_skills(root)
    shared = set(shared_skill_names(root))
    skills, agents = _rel_to_plugins(SKILLS_DIR), _rel_to_plugins(AGENTS_DIR)
    uskills, uagents = _rel_to_plugins(USAGE_SKILLS_DIR), _rel_to_plugins(USAGE_AGENTS_DIR)
    mirrored = sorted((s for s in dev if s.mirror), key=lambda s: s.mirror)
    other = sorted(
        (s for s in dev if not s.mirror and s.folder not in shared), key=lambda s: s.folder
    )
    lines = [
        "# Skills and agents by package",
        "",
        "Generated by `python -B maintenances/_repo/_maintenance/tools/check_records.py"
        " --write-index`; do not edit. Skills follow the Agent Skills specification (the"
        " `name` equals the folder); names are derived from the package path (Rule 164);"
        " each agent has its skill's name and preloads it (Rule 165).",
        "",
        "## Developing scikit-plots (this checkout: the `scikitplot-dev` plugin)",
        "",
        "| `scikitplot/` and `maintenances/` path | Skill | Agent |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{s.mirror}` | [`{s.folder}`]({skills}/{s.folder}/SKILL.md)"
        f" | [`{s.folder}`]({agents}/{s.folder}.md) |"
        for s in mirrored
    ]
    if other:
        lines += ["", "For the whole checkout:", ""]
        lines += [f"- [`{s.folder}`]({skills}/{s.folder}/SKILL.md)" for s in other]
    lines += [
        "",
        "## Using scikit-plots (any project: the `scikitplot` plugin)",
        "",
        "| Module | Skill | Agent |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{s.mirror}` | [`{s.folder}`]({uskills}/{s.folder}/SKILL.md)"
        f" | [`{s.folder}`]({uagents}/{s.folder}.md) |"
        for s in sorted((s for s in usage if s.mirror), key=lambda s: s.mirror)
    ]
    if shared:
        lines += [
            "",
            "## Shared by every plugin",
            "",
            f"Source in `{_rel_to_plugins(SHARED_SKILLS_DIR)}/`; each plugin carries a"
            " byte-identical copy written by `--write-index`.",
            "",
        ]
        lines += [
            f"- [`{name}`]({_rel_to_plugins(SHARED_SKILLS_DIR)}/{name}/SKILL.md)"
            for name in sorted(shared)
        ]
    return "\n".join(lines) + "\n"


def render_plugin_readme(root: Path) -> str:
    """Render ``plugins/scikitplot/README.md``."""
    usage, _ = usage_skills(root)
    lines = [
        "# The `scikitplot` plugin",
        "",
        "<!-- Generated by check_records.py --write-index; do not edit. -->",
        "",
        "Skills and agents for using scikit-plots in any project. Each skill follows the",
        "[Agent Skills specification](https://agentskills.io/specification), so a skill",
        "folder can also be copied into any tool that reads that format.",
        "",
        "## What it runs and sends",
        "",
        "Nothing. The plugin is Markdown only: skills and agents. It has no hooks, no",
        "MCP servers and no scripts, runs no code of its own and sends no data anywhere.",
        "The documentation search server is a separate, opt-in plugin, `scikitplot-mcp`,",
        "in the development marketplace.",
        "",
        "## Data handling",
        "",
        "| Question | Answer |",
        "|---|---|",
        "| Does it read or store personal data (names, emails, addresses)? | No. It has no connectors and no code; it only tells Claude how to use the scikit-plots library. |",
        "| Does any skill send data to a service? | No. No skill or agent makes a network call. The Python examples in the skills run only when you run them, in your own environment, and do only what the library call in them does. |",
        "| How long is data received from Claude kept? | It is not kept: there is no service, so the plugin receives nothing from Claude. |",
        "| Is it intended for users under 18? | No. It is a developer tool for a Python library. |",
        "| What does it change on my computer? | Nothing by itself. Installing it adds Markdown files to Claude's plugin folder; any code Claude writes or runs with its help follows your Claude permission settings. |",
        "| Does it cost anything or use my plan? | Installing does not. While it is enabled, its skill descriptions add a small amount of context to each session. |",
        "",
        "What Claude itself does with your conversations is set by your Claude",
        "account, not by this plugin.",
        "",
        "## Install",
        "",
        "Released versions, whose skills describe that release of scikit-plots (the",
        "plugin `version` is the library version), come from",
        f"[`{STABLE_REPO}`](https://github.com/{STABLE_REPO}): its default branch is the",
        "newest release, `@vX.Y.Z` one exact release, `@maintenance/X.Y.x` a release line",
        "and `@main` the development build. In Claude Code, from a shell:",
        "",
        "```sh",
        f"claude plugin marketplace add {STABLE_REPO}            # or {STABLE_REPO}@v0.5.0",
        f"claude plugin install scikitplot@{STABLE_MARKETPLACE}",
        "```",
        "",
        "The development version, whose skills describe the `main` branch, also comes",
        "from the scikit-plots repository itself:",
        "",
        "```sh",
        "claude plugin marketplace add scikit-plots/scikit-plots --sparse .claude-plugin plugins",
        "claude plugin install scikitplot@scikit-plots",
        "```",
        "",
        "Or copy one skill folder, for example into a project:",
        "`cp -r skills/scikitplot-annoy <project>/.claude/skills/`.",
        "The full guide (claude.ai, Cowork, versions, updates, troubleshooting) is",
        f"[`plugins/README.md`]({GUIDE_URL}).",
        "",
        "## What you need",
        "",
        "- Claude Code (any paid Claude plan, or an Anthropic Console account), or",
        "  claude.ai and Cowork on a paid plan. In claude.ai chat the skills work and",
        "  the agents are not used; in Claude Code and Cowork both work.",
        "- scikit-plots installed in the Python environment of your project",
        "  (`pip install scikit-plots`), when Claude is to run the code it writes.",
        "",
        "## Quick start",
        "",
        "After installing, open your project and ask in plain words, naming",
        "scikit-plots and what you want. For example:",
        "",
        "- *\"Using scikitplot, plot a confusion matrix and an ROC curve for my",
        "  LogisticRegression on the test set.\"* (`scikitplot.api`:",
        "  `plot_confusion_matrix`, `plot_roc`)",
        "- *\"Show a learning curve and feature importances for this random forest",
        "  with scikitplot.\"* (`plot_learning_curve`, `plot_feature_importances`)",
        "- *\"Make cumulative gains and lift charts for this churn model with",
        "  scikitplot.decile.\"* (`ModelPlotPy`, `plot_cumgains`, `plot_cumlift`)",
        "- *\"Which scikitplot modules do I have, and what is my installed version?\"*",
        "",
        "The matching skill loads by itself; you do not need to name it. In Claude",
        "Code you can also call a skill directly, such as `/scikitplot:scikitplot-api`,",
        "or hand a task to its agent, such as `@scikitplot:scikitplot-api`.",
        "",
        "## How it works",
        "",
        "- **One skill per public module.** Each is generated from that module's own",
        "  `__all__` and docstrings: what the module is for, how to install it, its",
        "  public names with one-line summaries, and a first example. The full list",
        "  of names with signatures is in the skill's `references/API.md`, which",
        "  Claude opens only when it needs it.",
        "- **One agent per skill**, with the same name, that starts with that skill",
        "  loaded and inherits the tools Claude Code allows it.",
        "- **Shared skills**, such as `scholarly-inquiry-cycle`, for structured",
        "  investigation of any question.",
        "- **The skills ask Claude to check before writing code**: your installed",
        "  version, and the signature and docstring of each name it calls.",
        "",
        "## Match the plugin to your library",
        "",
        "The skills describe one version of scikit-plots. Compare it with yours:",
        "",
        "```sh",
        "python -c \"import scikitplot; print(scikitplot.__version__)\"",
        "claude plugin list        # the plugin's Version",
        "```",
        "",
        "A release plugin has the library version (`0.5.0`); a development build has",
        "a commit. When they differ, either install the plugin of your release",
        f"(`claude plugin marketplace add {STABLE_REPO}@v<your version>`) or upgrade",
        "the library. Claude still checks each name in your environment before using it.",
        "",
        "## Update, disable, remove",
        "",
        "```sh",
        f"claude plugin marketplace update {STABLE_MARKETPLACE}",
        f"claude plugin update scikitplot@{STABLE_MARKETPLACE}",
        f"claude plugin disable scikitplot@{STABLE_MARKETPLACE}     # keep it, stop loading it",
        f"claude plugin uninstall scikitplot@{STABLE_MARKETPLACE}",
        "```",
        "",
        "Nothing updates by itself unless you turn on auto-update for the",
        "marketplace in `/plugin`. On claude.ai, manage it under **Customize →",
        "Plugins**. After an update, run `/reload-plugins` or start a new session.",
        "",
        "## Troubleshooting",
        "",
        "| You see | What to do |",
        "|---|---|",
        "| `Plugin \"scikitplot\" not found in marketplace` | Add the marketplace first (see Install), or run `claude plugin marketplace update`. |",
        "| The skills name something your library lacks | Your library and the plugin are different versions; see *Match the plugin to your library*. |",
        "| `ModuleNotFoundError` for a scikitplot module | Upgrade scikit-plots, or install the extra the skill names. |",
        "| Agents do nothing in claude.ai chat | Expected: chat uses skills, not agents. Use Claude Code or Cowork. |",
        "| `update` says it is already the latest | Update the marketplace first, then the plugin. |",
        "",
        "## Support and license",
        "",
        "- Questions and bug reports: <https://github.com/scikit-plots/scikit-plots/issues>",
        "- Documentation: <https://scikit-plots.github.io>",
        "- License: BSD-3-Clause; third-party licenses are in `LICENSES/`.",
        "",
        "## Skills and agents",
        "",
        "| Module | Skill | Agent |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{s.mirror}` | [`{s.folder}`](skills/{s.folder}/SKILL.md) | [`{s.folder}`](agents/{s.folder}.md) |"
        for s in sorted((s for s in usage if s.mirror), key=lambda s: s.mirror)
    ]
    lines += [
        "",
        "The skills are generated from the package's own `__all__` and docstrings by",
        "`tools/maint_tools/generate_usage_skills.py`; the agents and this file by",
        "`maintenances/_repo/_maintenance/tools/check_records.py --write-index`.",
    ]
    return "\n".join(lines) + "\n"


def generated_files(root: Path) -> dict[str, str]:
    """Return every file ``--write-index`` owns for skills, by repository path."""
    out = {SKILL_INDEX: render_skill_index(root), PLUGIN_README: render_plugin_readme(root)}
    out.update(wanted_agents(root))
    return out


def _agent_files(root: Path) -> set[str]:
    found = set()
    for base in (AGENTS_DIR, USAGE_AGENTS_DIR):
        folder = root / base
        if folder.is_dir():
            found.update(p.relative_to(root).as_posix() for p in folder.rglob("*") if p.is_file())
    return found


def write_skill_files(root: Path) -> None:
    """Write the generated skill files and remove agents no skill has any more."""
    wanted = generated_files(root)
    for rel in sorted(_agent_files(root) - set(wanted)):
        (root / rel).unlink()
    for rel, text in wanted.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        write(root / rel, text)
    copies = shared_copies(root)
    shared = shared_skill_names(root)
    for target in SHARED_TARGETS:
        for name in shared:
            folder = root / target / name
            if folder.is_dir():
                for path in sorted(folder.rglob("*"), reverse=True):
                    if path.is_file() and path.relative_to(root).as_posix() not in copies:
                        path.unlink()
    for rel, data in copies.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


def check_skills(root: Path) -> list[str]:
    """
    Check both skill sets, their names, agents and generated files.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list of str
        Empty when every skill is ``<skills dir>/<name>/SKILL.md`` with a
        valid frontmatter and the name Rule 164 derives, each maintained path
        and each module has one skill, every agent is the one generated from
        its skill (Rule 165), and the generated indexes are current.
    """
    if not (root / SKILLS_DIR).is_dir():
        return []
    dev, errors = dev_skills(root)
    usage, usage_errors = usage_skills(root)
    errors += usage_errors
    shared = shared_skill_names(root)
    _, shared_errors = read_skills(root, SHARED_SKILLS_DIR, MODULE_KEY)
    errors += shared_errors
    keys = mirror_keys(root)
    derived = derived_skill_names(keys)
    seen: dict[str, str] = {}
    for skill in dev:
        rel = f"{SKILLS_DIR}/{skill.folder}/SKILL.md"
        if not skill.mirror:
            if skill.name.endswith(DEV_SUFFIX):
                errors.append(
                    f"{rel}: a {DEV_SUFFIX} skill names the path it maintains in"
                    f" metadata.{MIRROR_KEY}"
                )
            elif skill.name not in CROSS_CUTTING_SKILLS + VENDORED_SKILLS + shared:
                errors.append(
                    f"{rel}: a skill without metadata.{MIRROR_KEY} is one of"
                    f" {list(CROSS_CUTTING_SKILLS + VENDORED_SKILLS)}"
                )
            continue
        if skill.mirror not in keys:
            errors.append(
                f"{rel}: metadata.{MIRROR_KEY} {skill.mirror!r} is neither a package"
                " below scikitplot/ nor a folder of maintenances/"
            )
            continue
        if skill.mirror in seen:
            errors.append(f"{rel}: {skill.mirror!r} already has skill {seen[skill.mirror]!r}")
            continue
        seen[skill.mirror] = skill.folder
        expected = dev_skill_name(skill.mirror, derived)
        if skill.name != expected:
            errors.append(
                f"{rel}: the development skill for {skill.mirror!r} is named"
                f" {expected!r} (Rule 164); rename the folder and 'name' from {skill.name!r}"
            )
    seen = {}
    for skill in usage:
        rel = f"{USAGE_SKILLS_DIR}/{skill.folder}/SKILL.md"
        module = skill.mirror
        if not module and skill.folder in shared:
            continue
        path = module[len(PREFIX) + 1 :].replace(".", "/") if module.startswith(PREFIX + ".") else ""
        if module != PREFIX and path not in keys:
            errors.append(f"{rel}: metadata.{MODULE_KEY} {module!r} is not a package of scikitplot")
            continue
        if module in seen:
            errors.append(f"{rel}: {module!r} already has skill {seen[module]!r}")
            continue
        seen[module] = skill.folder
        expected = usage_skill_name(module, derived)
        if skill.name != expected:
            errors.append(
                f"{rel}: the usage skill for {module!r} is named {expected!r} (Rule 164);"
                f" rename the folder and 'name' from {skill.name!r}"
            )
    names = [s.name for s in dev + usage if s.folder not in shared]
    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"skill name {name!r} is used twice; skill names are global")
    wanted = generated_files(root)
    for rel in sorted(_agent_files(root) - set(wanted)):
        errors.append(f"{rel}: no skill has this agent (run with --write-index)")
    for rel, text in sorted(wanted.items()):
        if not (root / rel).is_file() or read(root, rel) != text:
            errors.append(f"{rel} is out of date (run with --write-index)")
    copies = shared_copies(root)
    for rel, data in sorted(copies.items()):
        if not (root / rel).is_file() or (root / rel).read_bytes() != data:
            errors.append(
                f"{rel} differs from its source in {SHARED_SKILLS_DIR}/ (run with --write-index)"
            )
    for target in SHARED_TARGETS:
        for name in shared:
            folder = root / target / name
            errors.extend(
                f"{p.relative_to(root).as_posix()}: not in {SHARED_SKILLS_DIR}/{name}/"
                " (run with --write-index)"
                for p in sorted(folder.rglob("*")) if folder.is_dir()
                if p.is_file() and p.relative_to(root).as_posix() not in copies
            )
    return errors


def check_plugins(root: Path) -> list[str]:
    """
    Check the development marketplace and every plugin it lists.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    list of str
        Empty when ``.claude-plugin/marketplace.json`` is valid JSON with a
        ``plugins`` list, each entry's relative ``source`` is
        ``./plugins/<name>`` and holds ``.claude-plugin/plugin.json`` whose
        ``name`` equals the entry's, every folder of ``plugins/`` holding a
        manifest is listed, and no manifest sets ``version`` (the development
        channel is unversioned, Rule 166; releases are stamped by
        ``tools/maint_tools/build_skills_repo.py``).

    Notes
    -----
    **Developer notes.** The rules come from the marketplace reference
    (https://code.claude.com/docs/en/plugins/marketplace-reference.md): a
    relative source is resolved from the marketplace root and may not use
    ``..``; the entry name and the manifest name must agree. ``claude plugin
    validate`` checks more; this check needs no Claude Code install.
    """
    path = root / MARKETPLACE
    if not path.is_file():
        return [f"{MARKETPLACE} is missing"]
    try:
        market = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"{MARKETPLACE}: invalid JSON ({exc})"]
    entries = market.get("plugins")
    if not isinstance(entries, list) or not entries:
        return [f"{MARKETPLACE}: 'plugins' must be a non-empty list"]
    errors: list[str] = []
    listed: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"{MARKETPLACE}: plugins[{i}]"
        name, source = entry.get("name"), entry.get("source")
        if not isinstance(name, str) or not _SKILL_NAME.fullmatch(name or ""):
            errors.append(f"{where}: 'name' {name!r} is not kebab-case")
            continue
        if source != f"./{PLUGINS_DIR}/{name}":
            errors.append(f"{where}: 'source' is {source!r}; write './{PLUGINS_DIR}/{name}'")
            continue
        listed.add(name)
        manifest = root / PLUGINS_DIR / name / ".claude-plugin" / "plugin.json"
        if not manifest.is_file():
            errors.append(f"{where}: {manifest.relative_to(root).as_posix()} is missing")
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError as exc:
            errors.append(f"{manifest.relative_to(root).as_posix()}: invalid JSON ({exc})")
            continue
        if data.get("name") != name:
            errors.append(
                f"{manifest.relative_to(root).as_posix()}: 'name' {data.get('name')!r}"
                f" must equal the marketplace entry {name!r}"
            )
        if "version" in data or "version" in entry:
            errors.append(
                f"{manifest.relative_to(root).as_posix()}: the development channel sets no"
                " 'version' (Rule 166)"
            )
    for manifest in sorted((root / PLUGINS_DIR).glob("*/.claude-plugin/plugin.json")):
        name = manifest.parent.parent.name
        if name not in listed and not name.startswith("_"):
            errors.append(f"{PLUGINS_DIR}/{name}/: a plugin {MARKETPLACE} does not list")
    return errors


CHECKS = {
    "router": check_router,
    "rules": check_rules,
    "archives": check_archives,
    "links": check_links,
    "budgets": check_budgets,
    "skills": check_skills,
    "plugins": check_plugins,
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
    """Regenerate ``INDEX.md``, ``plugins/INDEX.md`` and ``HISTORY_INDEX.md``."""
    rules, _ = parse_rules(root)
    write(root / INDEX, render_index(rules))
    if (root / SKILLS_DIR).is_dir():
        write_skill_files(root)
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
        help=f"regenerate {INDEX}, {SKILL_INDEX}, the agents and {HISTORY_INDEX}, then check",
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
