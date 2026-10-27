#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Generate, check and explain the Scikit-Plots pull-request and issue labelers.

Three configuration files drive labeling (``.github/workflows/pr_labeler.yml``
and ``issue_labeler.yml``):

``.github/labeler-by-module.yml`` (generated)
    One label per release-note owner, derived from the same policy and tree as
    the Towncrier sections (``[tool.scikitplot.maintenance.towncrier]``), so a
    new package gets its label the moment it gets its release-note section.
    Module labels also fire for the module's satellites (maintenance plane,
    user guide, gallery, follow-up ledger, release-note fragments); library
    labels (``libs/<dist>``) fire for the library directory *and* for every
    path the distribution ships (``scikitplot/_distributions.py``).
``.github/labeler-issue.yml`` (generated)
    Title rules for issues and pull requests, from one prefix/keyword registry
    (``[tool.scikitplot.maintenance.labeler.title_prefixes]`` and
    ``title_keywords``): ``BUG: ...``, ``[Bug]: ...`` and ``BUG, TST: ...`` all
    give ``00 - Bug``; a prefix never gives two different labels.
``.github/labeler.yml`` (curated)
    Cross-cutting file rules (CI, documentation, languages, tests...). Edited by
    hand; ``check`` validates it against the labeler's real semantics.

Commands
--------
``check``    read-only: regenerate in memory and compare; lint every rule.
``sync``     plan, or with ``--apply`` write, the generated files.
``explain``  which labels a set of changed paths and a title would get.
``list``     the module/library labels and where they come from.

Notes
-----
**Semantics reproduced.** File rules follow ``actions/labeler`` v7
(``src/api/get-label-configs.ts``, ``src/labeler.ts``, ``src/changedFiles.ts``):
a label matches when every match object passes; bare keys of all list items
join one ``any`` group; inside one ``changed-files`` list every option is a
separate check, OR-ed in ``any`` and AND-ed in ``all``; globs are minimatch with
``dot: true``. Title rules follow ``github/issue-labeler`` v3.5: each value is a
JavaScript regular expression (``/body/flags`` form allowed), several values
for one label are AND-ed, and the text is ``title + "\\n\\n"``.

**Glob subset.** Every glob must use literal characters ``[A-Za-z0-9._-]``,
``*`` and ``?`` inside a segment, whole ``**`` segments, and at most one
leading ``!``. On that subset this module's matcher equals minimatch (the
test-suite cross-checks it against minimatch 10 when Node is available);
anything else (braces, classes, extglobs) is refused instead of guessed.

**Design.** Same shape as ``generate_towncrier_sections.py``: importing has no
side effects, read-only by default, ``sync --apply`` is the only writer,
``--json`` for automation, ``main(argv, stdout=, stderr=)`` returns a status.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess  # noqa: S404 - fixed argument vector (git ls-files)
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

try:  # Python >= 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.8-3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Python < 3.11 requires 'tomli' to run the labeler maintenance helper."
        ) from exc

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_TABLE = ("tool", "scikitplot", "maintenance", "labeler")
TOWNCRIER_SCRIPT = Path(__file__).resolve().parent / "generate_towncrier_sections.py"
#: Reads each skill's ``metadata.scikitplot-mirror``: ``plugins/scikitplot-dev/skills/`` is flat, so a
#: package's skill folder is looked up, not spelled from its path (Rule 164).
RECORDS_SCRIPT = DEFAULT_REPO_ROOT / "maintenances/_repo/_maintenance/tools/check_records.py"
DISTRIBUTIONS_FILE = "scikitplot/_distributions.py"
PLACEHOLDER = "@GENERATED_RULES@"
SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", "build", ".tox", ".nox"})

#: labeler v7 schema.
CHANGED_FILES_OPTIONS = (
    "any-glob-to-any-file",
    "any-glob-to-all-files",
    "all-globs-to-any-file",
    "all-globs-to-all-files",
)
BASE_KEYS = ("changed-files", "head-branch", "base-branch")
TOP_LEVEL_OPTIONS = ("changed-files-labels-limit", "max-files-changed")

_SEGMENT_RE = re.compile(r"[A-Za-z0-9._\-*?]+\Z")
_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9]*\Z")
_KEYWORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 \-]*[A-Za-z0-9]\Z|[A-Za-z0-9]\Z")
_CONTAINS_LABEL_RE = re.compile(
    r"contains\(\s*github\.event\.pull_request\.labels\.\*\.name\s*,\s*'([^']+)'\s*\)"
)


class LabelerError(ValueError):
    """Invalid policy, template or configuration (message is user-facing)."""


# --------------------------------------------------------------------------- globs


def validate_glob(glob: str) -> None:
    """
    Refuse a glob outside the subset whose minimatch semantics are reproduced.

    Parameters
    ----------
    glob : str
        One pattern from a labeler configuration.

    Raises
    ------
    LabelerError
        For empty segments, absolute or ``./`` paths, ``**`` mixed into a
        segment, a trailing ``**`` (use ``/**/*``), repeated negation, or any
        character other than ``[A-Za-z0-9._-*?]`` and ``/``.
    """
    if not isinstance(glob, str) or not glob:
        raise LabelerError("glob must be a non-empty string")
    body = glob[1:] if glob.startswith("!") else glob
    if body.startswith("!"):
        raise LabelerError(f"glob {glob!r}: only one leading '!' is supported")
    if body.startswith(("/", "./")) or body.endswith("/"):
        raise LabelerError(f"glob {glob!r} must be a relative path without a trailing '/'")
    segments = body.split("/")
    for index, segment in enumerate(segments):
        if segment == "**":
            if index == len(segments) - 1:
                raise LabelerError(f"glob {glob!r}: end with '/**/*', not '/**'")
            if index and segments[index - 1] == "**":
                raise LabelerError(f"glob {glob!r}: repeated '**' segments")
            continue
        if not segment or segment in {".", ".."} or "**" in segment:
            raise LabelerError(f"glob {glob!r}: invalid segment {segment!r}")
        if not _SEGMENT_RE.fullmatch(segment):
            raise LabelerError(
                f"glob {glob!r}: segment {segment!r} uses characters outside "
                "[A-Za-z0-9._-*?]"
            )


def _segment_regex(segment: str) -> str:
    out = []
    for char in segment:
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(char))
    return "".join(out)


def compile_glob(glob: str) -> Callable[[str], bool]:
    """
    Return a matcher equal to ``minimatch(path, glob, {dot: true})``.

    Parameters
    ----------
    glob : str
        A pattern accepted by :func:`validate_glob`.

    Returns
    -------
    callable
        ``matcher(path) -> bool`` for POSIX paths relative to the repository.

    Examples
    --------
    >>> compile_glob("**/conftest.py")("conftest.py")
    True
    >>> compile_glob("!docs/**/*")("src/a.py")
    True
    """
    validate_glob(glob)
    negated = glob.startswith("!")
    segments = (glob[1:] if negated else glob).split("/")
    parts: List[str] = []
    for index, segment in enumerate(segments):
        follows_globstar = index > 0 and segments[index - 1] == "**"
        if segment == "**":
            parts.append("(?:[^/]+/)*" if index == 0 else "/(?:[^/]+/)*")
            continue
        parts.append(("" if index == 0 or follows_globstar else "/") + _segment_regex(segment))
    pattern = re.compile("".join(parts) + r"\Z")

    def matcher(path: str) -> bool:
        return bool(pattern.match(path)) != negated

    return matcher


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class TitlePrefix:
    token: str
    label: str


@dataclass(frozen=True)
class TitleKeyword:
    label: str
    words: Tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    curated_config: str
    module_config: str
    title_config: str
    module_template: str
    title_template: str
    granular_owner_roots: Tuple[str, ...]
    rollup_owner_roots: Tuple[str, ...]
    satellites: Mapping[str, Tuple[str, ...]]
    label_aliases: Mapping[str, str]
    max_label_length: int
    manual_labels: Tuple[str, ...]
    title_prefixes: Tuple[TitlePrefix, ...]
    title_keywords: Tuple[TitleKeyword, ...]


def _table(data: Mapping[str, object], keys: Sequence[str]) -> Mapping[str, object]:
    current: object = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise LabelerError(f"missing pyproject table [{'.'.join(keys)}]")
        current = current[key]
    if not isinstance(current, Mapping):
        raise LabelerError(f"invalid pyproject table [{'.'.join(keys)}]")
    return current


def _strings(table: Mapping[str, object], key: str) -> Tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise LabelerError(f"[{'.'.join(POLICY_TABLE)}].{key} must be an array of strings")
    if len(set(value)) != len(value):
        raise LabelerError(f"[{'.'.join(POLICY_TABLE)}].{key} contains duplicates")
    return tuple(value)


def _string(table: Mapping[str, object], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise LabelerError(f"[{'.'.join(POLICY_TABLE)}].{key} must be a non-empty string")
    return value


def load_policy(data: Mapping[str, object]) -> Policy:
    """
    Read and validate ``[tool.scikitplot.maintenance.labeler]``.

    Parameters
    ----------
    data : Mapping
        Parsed ``pyproject.toml``.

    Returns
    -------
    Policy

    Raises
    ------
    LabelerError
        On a missing or malformed key, a duplicate, an unknown satellite
        placeholder, a prefix token that is not upper-case ASCII, or two
        labels for one token.
    """
    table = _table(data, POLICY_TABLE)
    satellites_raw = table.get("satellites", {})
    if not isinstance(satellites_raw, Mapping):
        raise LabelerError("labeler.satellites must be a table of arrays")
    satellites: Dict[str, Tuple[str, ...]] = {}
    for root, templates in satellites_raw.items():
        if not isinstance(templates, list) or not all(isinstance(t, str) for t in templates):
            raise LabelerError(f"labeler.satellites.{root} must be an array of strings")
        for template in templates:
            unknown = set(re.findall(r"\{(\w+)\}", template)) - {"rel", "section", "skill", "usage"}
            if unknown:
                raise LabelerError(
                    f"satellite {template!r} uses unknown placeholder(s) {sorted(unknown)}"
                )
        satellites[root] = tuple(templates)
    aliases = table.get("label_aliases", {})
    if not isinstance(aliases, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in aliases.items()
    ):
        raise LabelerError("labeler.label_aliases must map owner paths to label strings")
    max_length = table.get("max_label_length", 50)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise LabelerError("labeler.max_label_length must be a positive integer")
    prefixes_raw = table.get("title_prefixes", {})
    if not isinstance(prefixes_raw, Mapping) or not prefixes_raw:
        raise LabelerError("labeler.title_prefixes must be a non-empty table")
    prefixes = []
    for token, label in prefixes_raw.items():
        if not _TOKEN_RE.fullmatch(token):
            raise LabelerError(f"title prefix {token!r} must be upper-case ASCII letters/digits")
        if not isinstance(label, str) or not label:
            raise LabelerError(f"title prefix {token!r} needs a label string")
        prefixes.append(TitlePrefix(token, label))
    keywords = []
    for index, raw in enumerate(table.get("title_keywords", [])):
        if not isinstance(raw, Mapping) or set(raw) != {"label", "words"}:
            raise LabelerError(f"title_keywords[{index}] needs exactly 'label' and 'words'")
        words = raw["words"]
        if (
            not isinstance(raw["label"], str)
            or not raw["label"]
            or not isinstance(words, list)
            or not words
            or not all(isinstance(w, str) and _KEYWORD_RE.fullmatch(w) for w in words)
        ):
            raise LabelerError(
                f"title_keywords[{index}]: words must be letters, digits, spaces or hyphens"
            )
        keywords.append(TitleKeyword(raw["label"], tuple(words)))
    return Policy(
        curated_config=_string(table, "curated_config"),
        module_config=_string(table, "module_config"),
        title_config=_string(table, "title_config"),
        module_template=_string(table, "module_template"),
        title_template=_string(table, "title_template"),
        granular_owner_roots=_strings(table, "granular_owner_roots"),
        rollup_owner_roots=_strings(table, "rollup_owner_roots"),
        satellites=satellites,
        label_aliases=dict(aliases),
        max_label_length=max_length,
        manual_labels=_strings(table, "manual_labels"),
        title_prefixes=tuple(prefixes),
        title_keywords=tuple(keywords),
    )


def _load_pyproject(repo_root: Path) -> Mapping[str, object]:
    with (repo_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LabelerError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- module labels


@dataclass(frozen=True)
class ModuleRule:
    """One generated label and the globs (with their origin) that select it."""

    label: str
    owner: str
    globs: Tuple[str, ...]
    origins: Tuple[str, ...] = field(default=())


def _label_for(owner: str, policy: Policy) -> str:
    if owner in policy.label_aliases:
        return policy.label_aliases[owner]
    root, _, rest = owner.partition(".")
    if not rest:
        return f"{root} (top-level)"
    return root + "/" + rest.replace(".", "/")


def _skill_mirrors(repo_root: Path) -> Dict[str, Dict[str, str]]:
    """
    Return ``{"skill": {path: folder}, "usage": {path: folder}}`` from frontmatter.

    ``skill`` maps a package path to its development skill (``plugins/scikitplot-dev/skills/``),
    ``usage`` to its usage skill (``plugins/scikitplot/skills/``); both are
    read by ``check_records.py`` (Rules 164 and 165).
    """
    if not (repo_root/"plugins").is_dir():
        return {"skill": {}, "usage": {}}
    records = _load_module("_records_for_labeler", RECORDS_SCRIPT)
    return {
        "skill": dict(records.skill_mirrors(repo_root)),
        "usage": dict(records.usage_mirrors(repo_root)),
    }


def _expand_satellite(
    template: str, rel: str, owner: str, mirrors: Mapping[str, Mapping[str, str]]
) -> List[str]:
    """
    Return the globs one satellite template gives for one owner.

    Parameters
    ----------
    template : str
        A satellite with ``{rel}``, ``{section}`` and at most one of
        ``{skill}`` and ``{usage}``.
    rel, owner : str
        The owner's path below its root, and its Towncrier section.
    mirrors : Mapping
        :func:`_skill_mirrors`.

    Returns
    -------
    list of str
        One glob; or, for ``{skill}`` (development skill) and ``{usage}``
        (usage skill), one per skill whose package path is ``rel`` or below
        it (a parent owner also covers its children's skills), sorted.
    """
    kinds = [kind for kind in ("skill", "usage") if "{" + kind + "}" in template]
    if not kinds:
        return [template.format(rel=rel, section=owner)]
    if len(kinds) > 1:
        raise LabelerError(f"satellite {template!r} uses both {{skill}} and {{usage}}")
    kind = kinds[0]
    folders = sorted(
        folder
        for path, folder in mirrors[kind].items()
        if path and (path == rel or path.startswith(rel + "/"))
    )
    return [template.format(rel=rel, section=owner, **{kind: folder}) for folder in folders]


def _owner_rel(owner: str) -> str:
    return owner.partition(".")[2].replace(".", "/")


def _distribution_paths(repo_root: Path) -> Dict[str, Tuple[str, ...]]:
    """Return ``libs`` directory -> repository paths the distribution ships."""
    path = repo_root / DISTRIBUTIONS_FILE
    if not path.is_file():
        return {}
    module = _load_module("_scikitplot_distributions_for_labeler", path)
    prefix = "scikit-plots-"
    result: Dict[str, Tuple[str, ...]] = {}
    for dist in module.DISTRIBUTIONS:
        if not dist.name.startswith(prefix):
            raise LabelerError(f"distribution {dist.name!r} lacks the {prefix!r} prefix")
        globs = [f"scikitplot/{tree}/**/*" for tree in dist.trees]
        globs += [f"scikitplot/{name}" for name in dist.files]
        result[dist.name[len(prefix):]] = tuple(globs)
    return result


def module_rules(repo_root: Path, data: Mapping[str, object], policy: Policy, files: Sequence[str]) -> List[ModuleRule]:
    """
    Derive every generated module/library label from policy and tree.

    Parameters
    ----------
    repo_root : pathlib.Path
        Checkout root.
    data : Mapping
        Parsed ``pyproject.toml`` (Towncrier owner policy is read from it).
    policy : Policy
        Labeler policy.
    files : sequence of str
        Repository files; a satellite glob is emitted only if it matches one.

    Returns
    -------
    list of ModuleRule
        Sorted by owner path.

    Raises
    ------
    LabelerError
        If an owner root is unknown, a label is too long or duplicated, an
        alias names no owner, or a ``libs`` directory has no distribution.
    """
    towncrier = _load_module("_towncrier_sections_for_labeler", TOWNCRIER_SCRIPT)
    try:
        owners = towncrier._discover_owner_sections(repo_root, towncrier._load_policy(data))
    except SystemExit as exc:  # the towncrier helper reports policy errors this way
        raise LabelerError(f"towncrier owner policy: {exc}") from exc
    known_roots = set(policy.granular_owner_roots) | set(policy.rollup_owner_roots)
    distributions = _distribution_paths(repo_root)
    mirrors = _skill_mirrors(repo_root)
    rules: List[ModuleRule] = []
    owner_paths = {section.path for section in owners}
    stale_aliases = sorted(set(policy.label_aliases) - owner_paths)
    if stale_aliases:
        raise LabelerError("label_aliases name no owner: " + ", ".join(stale_aliases))
    for root in policy.rollup_owner_roots:
        rules.append(ModuleRule(label=root, owner=root, globs=(f"{root}/**/*",), origins=("tree",)))
    for section in sorted(owners, key=lambda s: s.path):
        owner = section.path
        root = owner.partition(".")[0]
        if root not in known_roots:
            raise LabelerError(f"owner {owner} is under a root the labeler policy does not name")
        if root in policy.rollup_owner_roots:
            continue
        rel = _owner_rel(owner)
        if not rel:
            globs: List[Tuple[str, str]] = [(f"{root}/*", "tree")]
        else:
            globs = [(f"{root}/{rel}/**/*", "tree")]
            for template in policy.satellites.get(root, ()):
                for glob in _expand_satellite(template, rel, owner, mirrors):
                    matcher = compile_glob(glob)
                    if any(matcher(path) for path in files):
                        globs.append((glob, "satellite"))
            if root == "libs" and rel != "_tools":
                if rel not in distributions:
                    raise LabelerError(f"libs/{rel} has no entry in {DISTRIBUTIONS_FILE}")
                globs += [(glob, "ships") for glob in distributions[rel]]
        label = _label_for(owner, policy)
        rules.append(
            ModuleRule(
                label=label,
                owner=owner,
                globs=tuple(g for g, _ in globs),
                origins=tuple(o for _, o in globs),
            )
        )
    seen: Dict[str, str] = {}
    for rule in rules:
        if len(rule.label) > policy.max_label_length:
            raise LabelerError(
                f"label {rule.label!r} for {rule.owner} has {len(rule.label)} characters "
                f"(GitHub allows {policy.max_label_length}); add it to label_aliases"
            )
        if rule.label in seen:
            raise LabelerError(f"label {rule.label!r} is generated for {seen[rule.label]} and {rule.owner}")
        seen[rule.label] = rule.owner
        for glob in rule.globs:
            validate_glob(glob)
    return rules


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_module_rules(rules: Sequence[ModuleRule]) -> str:
    """Return the labeler v7 YAML for *rules* (deterministic text)."""
    blocks = []
    for rule in rules:
        lines = [f"{_yaml_string(rule.label)}:", "- changed-files:", "  - any-glob-to-any-file:"]
        lines += [f"    - {_yaml_string(glob)}" for glob in rule.globs]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- title labels


def _prefix_regex(tokens: Sequence[str], all_tokens: Sequence[str]) -> str:
    """
    Return the issue-labeler regex for a title whose prefix list names a token.

    Grammar (case-insensitive, at the start of the title)::

        [ "[" ] TOKEN { ("," | "/" | "+") TOKEN } ( "]" [":"] | "(" scope ")" ":" | ":" )

    Every TOKEN is a registry token; the label applies when one of *tokens* is
    in the list. The explicit terminator is what makes case-insensitivity
    safe: ``[Bug]: ...``, ``fix: ...`` and ``BUG, TST: ...`` match, while
    ``Fix typo``, ``Bug in reader`` and ``DOCKER image`` do not.
    """
    known = "|".join(sorted(all_tokens, key=lambda t: (-len(t), t)))
    wanted = "|".join(sorted(tokens, key=lambda t: (-len(t), t)))
    separator = r"\s*[,/+]\s*"
    return (
        rf"/^\s*(?:\[\s*)?(?:(?:{known}){separator})*(?:{wanted})"
        rf"(?:{separator}(?:{known}))*\s*(?:\]\s*:?|\([^)]*\)\s*:|:)/i"
    )


def _keyword_regex(words: Sequence[str]) -> str:
    alternatives = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in words)
    return rf"/\b(?:{alternatives})\b/i"


def title_rules(policy: Policy) -> List[Tuple[str, str]]:
    """
    Return ``(label, regex)`` pairs in registry order.

    Raises
    ------
    LabelerError
        If a label appears both as a prefix label and a keyword label, or a
        keyword label is repeated.
    """
    by_label: Dict[str, List[str]] = {}
    for prefix in policy.title_prefixes:
        by_label.setdefault(prefix.label, []).append(prefix.token)
    tokens = [p.token for p in policy.title_prefixes]
    rules = [(label, _prefix_regex(toks, tokens)) for label, toks in by_label.items()]
    keyword_labels = [k.label for k in policy.title_keywords]
    if len(set(keyword_labels)) != len(keyword_labels):
        raise LabelerError("a title keyword label is listed twice")
    clash = sorted(set(keyword_labels) & set(by_label))
    if clash:
        raise LabelerError("labels used by both prefixes and keywords: " + ", ".join(clash))
    rules += [(k.label, _keyword_regex(k.words)) for k in policy.title_keywords]
    return rules


def _yaml_single(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_title_rules(rules: Sequence[Tuple[str, str]]) -> str:
    """Return the issue-labeler YAML for *rules* (one regex per label)."""
    return "\n\n".join(f"{_yaml_string(label)}:\n  - {_yaml_single(regex)}" for label, regex in rules)


def js_regex(value: str) -> re.Pattern[str]:
    """
    Compile an issue-labeler value the way the action does, in Python.

    ``/body/flags`` uses the flags (only ``i`` is accepted here); anything
    else is a case-sensitive pattern. The generated patterns use only syntax
    that JavaScript and Python read alike.
    """
    match = re.match(r"^/(.+)/(.*)$", value, re.DOTALL)
    if not match:
        return re.compile(value)
    body, flags = match.groups()
    if set(flags) - {"i"}:
        raise LabelerError(f"regex {value!r}: only the 'i' flag is supported here")
    return re.compile(body, re.IGNORECASE if "i" in flags else 0)


# --------------------------------------------------------------------------- templates


def render_file(repo_root: Path, template: str, body: str) -> str:
    """Return *template* with :data:`PLACEHOLDER` replaced by *body*."""
    path = repo_root / template
    if not path.is_file():
        raise LabelerError(f"template {template} does not exist")
    text = path.read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise LabelerError(f"template {template} must contain {PLACEHOLDER} exactly once")
    return text.replace(PLACEHOLDER, body).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- files


def repository_files(repo_root: Path, files_from: Optional[TextIO] = None) -> List[str]:
    """
    Return the repository's files as sorted POSIX paths.

    Parameters
    ----------
    repo_root : pathlib.Path
        Checkout root.
    files_from : file-like, optional
        One path per line (for example ``git ls-files``); overrides discovery.

    Notes
    -----
    Without *files_from*: ``git ls-files`` when ``repo_root`` is a git
    checkout, otherwise a walk that skips :data:`SKIP_DIRS`.
    """
    if files_from is not None:
        return sorted({line.strip() for line in files_from if line.strip()})
    if (repo_root / ".git").exists():
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "ls-files", "-z"],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return sorted(p for p in result.stdout.decode("utf-8").split("\0") if p)
    found = []
    for current, dirs, names in os.walk(repo_root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in names:
            found.append((Path(current) / name).relative_to(repo_root).as_posix())
    return sorted(found)


# --------------------------------------------------------------------------- labeler semantics


@dataclass
class Group:
    kind: str  # "any" | "all"
    bases: List[Mapping[str, object]]


def normalise_label_config(label: str, raw: object) -> List[Group]:
    """
    Convert one label's YAML value into match groups exactly as labeler v7 does.

    Raises
    ------
    LabelerError
        Where labeler v7 throws, and also for an unknown key (labeler only
        logs it, so a typo would silently disable the rule).
    """
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise LabelerError(f"label {label!r}: value must be a list of match objects")
    groups: List[Group] = []
    for item in raw:
        for key, value in item.items():
            if key in ("any", "all"):
                if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
                    raise LabelerError(f"label {label!r}: '{key}' must be a list of objects")
                for base in value:
                    _check_base(label, base)
                groups.append(Group(key, list(value)))
            elif key in BASE_KEYS:
                _check_base(label, {key: value})
                existing = next((g for g in groups if g.kind == "any"), None)
                if existing is None:
                    groups.append(Group("any", [{key: value}]))
                else:
                    existing.bases.append({key: value})
            else:
                raise LabelerError(f"label {label!r}: unknown key {key!r}")
    if not groups:
        raise LabelerError(f"label {label!r}: no match objects")
    return groups


def _check_base(label: str, base: Mapping[str, object]) -> None:
    unknown = sorted(set(base) - set(BASE_KEYS))
    if unknown:
        raise LabelerError(f"label {label!r}: unknown key(s) {unknown}")
    for key in ("head-branch", "base-branch"):
        if key in base:
            values = base[key] if isinstance(base[key], list) else [base[key]]
            for pattern in values:
                if not isinstance(pattern, str):
                    raise LabelerError(f"label {label!r}: {key} values must be strings")
                re.compile(pattern)
    if "changed-files" in base:
        entries = base["changed-files"]
        entries = entries if isinstance(entries, list) else [entries]
        for entry in entries:
            if not isinstance(entry, dict) or not entry:
                raise LabelerError(f"label {label!r}: changed-files entries must be objects")
            for option, globs in entry.items():
                if option not in CHANGED_FILES_OPTIONS:
                    raise LabelerError(f"label {label!r}: unknown changed-files option {option!r}")
                for glob in globs if isinstance(globs, list) else [globs]:
                    try:
                        validate_glob(glob)
                    except LabelerError as exc:
                        raise LabelerError(f"label {label!r}: {exc}") from exc


def _changed_files_checks(base: Mapping[str, object]) -> List[Tuple[str, List[str]]]:
    entries = base.get("changed-files")
    if not entries:
        return []
    entries = entries if isinstance(entries, list) else [entries]
    checks = []
    for entry in entries:
        for option, globs in entry.items():
            checks.append((option, list(globs) if isinstance(globs, list) else [globs]))
    return checks


def _option_matches(option: str, globs: Sequence[str], files: Sequence[str]) -> bool:
    matchers = [compile_glob(g) for g in globs]
    if option == "any-glob-to-any-file":
        return any(m(f) for m in matchers for f in files)
    if option == "all-globs-to-any-file":
        return any(all(m(f) for m in matchers) for f in files)
    if option == "any-glob-to-all-files":
        return any(all(m(f) for f in files) for m in matchers)
    return all(all(m(f) for m in matchers) for f in files)  # all-globs-to-all-files


def _branch_matches(patterns: object, branch: Optional[str]) -> bool:
    if branch is None:
        return False
    values = patterns if isinstance(patterns, list) else [patterns]
    return any(re.search(p, branch) for p in values)


def label_matches(
    groups: Sequence[Group],
    files: Sequence[str],
    *,
    head_branch: Optional[str] = None,
    base_branch: Optional[str] = None,
) -> bool:
    """Return whether a label with *groups* applies (labeler v7 ``checkMatchConfigs``)."""
    for group in groups:
        if not group.bases or not any(group.bases):
            return False
        if group.kind == "any":
            # Each key of a base is its own check (one object may hold several).
            if not any(
                ("base-branch" in base and _branch_matches(base["base-branch"], base_branch))
                or ("head-branch" in base and _branch_matches(base["head-branch"], head_branch))
                or any(_option_matches(o, g, files) for o, g in _changed_files_checks(base))
                for base in group.bases
            ):
                return False
        else:
            for base in group.bases:
                if "base-branch" in base and not _branch_matches(base["base-branch"], base_branch):
                    return False
                if "head-branch" in base and not _branch_matches(base["head-branch"], head_branch):
                    return False
                checks = _changed_files_checks(base)
                if checks:
                    if not files:
                        return False
                    if not all(_option_matches(o, g, files) for o, g in checks):
                        return False
    return True


def lint_label(label: str, groups: Sequence[Group]) -> List[str]:
    """
    Return semantic errors that make a rule match (almost) every change.

    A negated glob is true for every path it does not name. In
    ``any-glob-to-any-file`` / ``any-glob-to-all-files`` one such glob makes
    the option true for nearly any change; an option list made only of
    negations is meaningful only as ``all-globs-to-all-files`` inside an
    ``all`` group (the documented "except" idiom).
    """
    errors = []
    for group in groups:
        for base in group.bases:
            for option, globs in _changed_files_checks(base):
                negated = [g for g in globs if g.startswith("!")]
                if not negated:
                    continue
                if option.startswith("any-glob"):
                    errors.append(
                        f"label {label!r}: negated glob(s) {negated} in {option} match "
                        "almost every change; use all-globs-to-any-file with a positive glob"
                    )
                elif len(negated) == len(globs) and not (
                    option == "all-globs-to-all-files" and group.kind == "all"
                ):
                    errors.append(
                        f"label {label!r}: {option} {negated} has no positive glob and "
                        f"sits in an '{group.kind}' group, so it matches almost every change"
                    )
    return errors


def load_yaml(path: Path) -> object:
    try:
        import yaml  # noqa: PLC0415 - optional at import time
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise LabelerError("PyYAML is required to read labeler configurations") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_labeler_config(path: Path) -> Dict[str, List[Group]]:
    """Parse a labeler v7 file into ``{label: groups}`` (top-level options skipped)."""
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise LabelerError(f"{path.name} must be a mapping of labels")
    return {
        label: normalise_label_config(label, raw)
        for label, raw in data.items()
        if label not in TOP_LEVEL_OPTIONS
    }


# --------------------------------------------------------------------------- vocabulary


def referenced_labels(repo_root: Path) -> Dict[str, List[str]]:
    """
    Return labels other repository files rely on, keyed by where they appear.

    Sources: issue-template ``labels:``, workflow ``if:`` expressions of the
    form ``contains(github.event.pull_request.labels.*.name, '<label>')``, and
    the gilesbot changelog labels in ``pyproject.toml``.
    """
    found: Dict[str, List[str]] = {}
    templates = repo_root / ".github" / "ISSUE_TEMPLATE"
    if templates.is_dir():
        for path in sorted(templates.glob("*.yml")):
            data = load_yaml(path)
            labels = data.get("labels", []) if isinstance(data, dict) else []
            for label in labels if isinstance(labels, list) else [labels]:
                found.setdefault(str(label), []).append(path.relative_to(repo_root).as_posix())
    workflows = repo_root / ".github" / "workflows"
    if workflows.is_dir():
        for path in sorted(workflows.glob("*.yml")):
            for expression in _if_expressions(load_yaml(path)):
                for label in _CONTAINS_LABEL_RE.findall(expression):
                    found.setdefault(label, []).append(path.relative_to(repo_root).as_posix())
    giles = _load_pyproject(repo_root).get("tool", {}).get("gilesbot", {}).get("towncrier_changelog", {})
    for key in ("whatsnew_label", "changelog_noop_label", "changelog_skip_label"):
        if isinstance(giles.get(key), str):
            found.setdefault(giles[key], []).append(f"pyproject.toml [tool.gilesbot] {key}")
    return found


def _if_expressions(node: object) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "if" and isinstance(value, str):
                yield value
            else:
                yield from _if_expressions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _if_expressions(item)


# --------------------------------------------------------------------------- check / sync


@dataclass
class Report:
    errors: List[str]
    warnings: List[str]
    stale: List[str]
    counts: Dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors


def _generated(repo_root: Path, files: Sequence[str]) -> Tuple[Policy, List[ModuleRule], Dict[str, str]]:
    data = _load_pyproject(repo_root)
    policy = load_policy(data)
    rules = module_rules(repo_root, data, policy, files)
    outputs = {
        policy.module_config: render_file(repo_root, policy.module_template, render_module_rules(rules)),
        policy.title_config: render_file(repo_root, policy.title_template, render_title_rules(title_rules(policy))),
    }
    return policy, rules, outputs


def check(repo_root: Path, files: Sequence[str], *, strict: bool = False) -> Report:
    """
    Validate policy, generated files and the curated configuration.

    Errors: stale generated files, invalid schema or glob, negation misuse,
    label collisions, labels over the length limit, labels relied on elsewhere
    that no rule or policy defines. Warnings (errors with ``strict``): curated
    globs that match no repository file.
    """
    errors: List[str] = []
    warnings: List[str] = []
    stale: List[str] = []
    try:
        policy, rules, outputs = _generated(repo_root, files)
    except LabelerError as exc:
        return Report([str(exc)], [], [], {})
    for relative, text in outputs.items():
        path = repo_root / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(relative)
    if stale:
        errors.append(
            "generated file(s) out of date: " + ", ".join(stale)
            + " (run: python tools/maint_tools/generate_pr_labeler.py sync --apply)"
        )
    curated_path = repo_root / policy.curated_config
    try:
        curated = parse_labeler_config(curated_path)
    except LabelerError as exc:
        return Report(errors + [f"{policy.curated_config}: {exc}"], warnings, stale, {})
    module_labels = {rule.label for rule in rules}
    title_labels = {label for label, _ in title_rules(policy)}
    for label, groups in curated.items():
        errors.extend(lint_label(label, groups))
        if len(label) > policy.max_label_length:
            errors.append(f"curated label {label!r} is longer than {policy.max_label_length}")
        if label in module_labels:
            errors.append(f"curated label {label!r} is also a generated module label")
        for group in groups:
            for base in group.bases:
                for _option, globs in _changed_files_checks(base):
                    for glob in globs:
                        if glob.startswith("!"):
                            continue
                        matcher = compile_glob(glob)
                        if not any(matcher(path) for path in files):
                            (errors if strict else warnings).append(
                                f"{policy.curated_config}: label {label!r} glob {glob!r} "
                                "matches no repository file"
                            )
    for label in sorted(title_labels):
        if len(label) > policy.max_label_length:
            errors.append(f"title label {label!r} is longer than {policy.max_label_length}")
    vocabulary = module_labels | title_labels | set(curated) | set(policy.manual_labels)
    for label, places in sorted(referenced_labels(repo_root).items()):
        if label not in vocabulary:
            errors.append(
                f"label {label!r} used by {', '.join(sorted(set(places)))} is defined "
                "by no labeler rule and not listed in manual_labels"
            )
    counts = {
        "module_labels": len(rules),
        "title_labels": len(title_labels),
        "curated_labels": len(curated),
        "files": len(files),
    }
    return Report(errors, warnings, stale, counts)


def sync(repo_root: Path, files: Sequence[str], *, apply: bool) -> List[str]:
    """Return the generated files that differ; write them when *apply*."""
    _policy, _rules, outputs = _generated(repo_root, files)
    changed = []
    for relative, text in outputs.items():
        path = repo_root / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            changed.append(relative)
            if apply:
                path.write_text(text, encoding="utf-8")
    return changed


def explain(
    repo_root: Path,
    paths: Sequence[str],
    title: Optional[str],
    *,
    head_branch: Optional[str] = None,
    base_branch: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Return the labels each configuration file would apply.

    Parameters
    ----------
    repo_root : pathlib.Path
        Checkout whose *current* configuration files are evaluated.
    paths : sequence of str
        Changed files (POSIX, repository-relative).
    title : str or None
        Pull-request or issue title.
    head_branch, base_branch : str, optional
        Branch names for ``head-branch``/``base-branch`` rules.
    """
    policy = load_policy(_load_pyproject(repo_root))
    result: Dict[str, List[str]] = {}
    for relative in (policy.curated_config, policy.module_config):
        configs = parse_labeler_config(repo_root / relative)
        result[relative] = sorted(
            label
            for label, groups in configs.items()
            if label_matches(groups, paths, head_branch=head_branch, base_branch=base_branch)
        )
    if title is not None:
        data = load_yaml(repo_root / policy.title_config) or {}
        text = f"{title}\n\n"
        labels = []
        for label, patterns in data.items():
            patterns = patterns if isinstance(patterns, list) else [patterns]
            if all(js_regex(p).search(text) for p in patterns):
                labels.append(label)
        result[policy.title_config] = labels
    return result


# --------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, check and explain the PR/issue labeler configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="checkout root")
    parser.add_argument(
        "--files-from",
        type=argparse.FileType("r"),
        default=None,
        help="file listing repository paths, one per line ('-' for stdin); "
        "default: git ls-files, else a directory walk",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check_p = sub.add_parser("check", help="validate everything (read-only)")
    check_p.add_argument("--json", action="store_true")
    check_p.add_argument("--strict", action="store_true", help="unmatched curated globs are errors")
    sync_p = sub.add_parser("sync", help="regenerate the generated files")
    sync_p.add_argument("--apply", action="store_true", help="write; without it, only report")
    sync_p.add_argument("--json", action="store_true")
    explain_p = sub.add_parser("explain", help="labels for changed paths and a title")
    explain_p.add_argument("paths", nargs="*", help="changed paths")
    explain_p.add_argument("--paths-from", type=argparse.FileType("r"), default=None)
    explain_p.add_argument("--title", default=None)
    explain_p.add_argument("--head-branch", default=None)
    explain_p.add_argument("--base-branch", default=None)
    explain_p.add_argument("--json", action="store_true")
    list_p = sub.add_parser("list", help="generated module/library labels")
    list_p.add_argument("--json", action="store_true")
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Run the CLI; return 0 (ok), 1 (check failed / stale) or 2 (usage or policy error)."""
    args = _parser().parse_args(argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    repo_root = (DEFAULT_REPO_ROOT if args.repo_root is None else args.repo_root).resolve()
    if not (repo_root / "pyproject.toml").is_file():
        err.write(f"error: {repo_root} has no pyproject.toml\n")
        return 2
    try:
        if args.command == "explain":
            paths = list(args.paths)
            if args.paths_from is not None:
                paths += [line.strip() for line in args.paths_from if line.strip()]
            result = explain(
                repo_root, paths, args.title, head_branch=args.head_branch, base_branch=args.base_branch
            )
            if args.json:
                json.dump(result, out, indent=2, sort_keys=True)
                out.write("\n")
            else:
                for source, labels in result.items():
                    out.write(f"{source}: {', '.join(labels) or '(none)'}\n")
            return 0
        files = repository_files(repo_root, args.files_from)
        if args.command == "check":
            report = check(repo_root, files, strict=args.strict)
            if args.json:
                json.dump(
                    {"ok": report.ok, "errors": report.errors, "warnings": report.warnings,
                     "stale": report.stale, "counts": report.counts},
                    out, indent=2, sort_keys=True,
                )
                out.write("\n")
            else:
                for warning in report.warnings:
                    out.write(f"warning: {warning}\n")
                if report.ok:
                    out.write(
                        "Labeler check: PASS ("
                        + ", ".join(f"{k}={v}" for k, v in sorted(report.counts.items()))
                        + ")\n"
                    )
                else:
                    err.write("Labeler check: FAIL\n")
                    for error in report.errors:
                        err.write(f"- {error}\n")
            return 0 if report.ok else 1
        if args.command == "sync":
            changed = sync(repo_root, files, apply=args.apply)
            if args.json:
                json.dump({"applied": args.apply, "changed": changed}, out, indent=2)
                out.write("\n")
            else:
                mode = "APPLIED" if args.apply else "DRY-RUN"
                out.write(f"Labeler sync: {mode}; changed: {', '.join(changed) or 'none'}\n")
                if changed and not args.apply:
                    out.write("- rerun with --apply to write these files\n")
            return 0
        if args.command == "list":
            data = _load_pyproject(repo_root)
            policy = load_policy(data)
            rules = module_rules(repo_root, data, policy, files)
            if args.json:
                json.dump(
                    [{"label": r.label, "owner": r.owner, "globs": list(r.globs), "origins": list(r.origins)} for r in rules],
                    out, indent=2,
                )
                out.write("\n")
            else:
                for rule in rules:
                    out.write(f"{rule.label:50} {len(rule.globs)} glob(s)\n")
            return 0
    except LabelerError as exc:
        err.write(f"error: {exc}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
