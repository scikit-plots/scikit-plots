#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Maintain Scikit-Plots Towncrier section ownership and fragment layout.

This helper is intentionally repository-local and network-free.  It derives
release-note owners from policy stored in ``pyproject.toml`` and the current
repository tree, then validates or synchronizes the Towncrier configuration and
fragment directories.

CLI design follows the same safety principles as ``scikitplot.mcp``:

* importing the module has no CLI side effects;
* ``_parser()`` owns argument construction;
* ``main(argv, ...)`` returns an explicit process status;
* read-only behavior is the default;
* filesystem mutations require the explicit ``sync --apply`` acknowledgement;
* machine-readable JSON output is available for automation/AI callers.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, TextIO, Tuple

try:  # Python >= 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.8-3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Python < 3.11 requires 'tomli' to run the Towncrier maintenance helper. "
            "Install the repository's development/legacy dependencies first."
        ) from exc


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_NAME = "pyproject.toml"
DEFAULT_FRAGMENT_ROOT = "docs/source/whats_new/upcoming_changes"
DEFAULT_TEMPLATE = (
    "docs/source/whats_new/upcoming_changes/towncrier_template.rst.jinja2"
)
POLICY_TABLE = ("tool", "scikitplot", "maintenance", "towncrier")
OWNER_BLOCK_BEGIN = "# BEGIN AUTO-GENERATED TOWNCRIER OWNER SECTIONS"
OWNER_BLOCK_END = "# END AUTO-GENERATED TOWNCRIER OWNER SECTIONS"
ROOT_ALLOWED_FILES = {"README.md", "towncrier_template.rst.jinja2"}
FRAGMENT_RE = re.compile(r"^(?P<issue>\d+)\.(?P<kind>[^.]+)\.rst$")


@dataclass(frozen=True)
class Section:
    """One Towncrier section entry."""

    name: str
    path: str


@dataclass(frozen=True)
class Policy:
    """Repository policy used to derive Towncrier owner sections."""

    python_package_roots: Tuple[str, ...]
    directory_roots: Tuple[str, ...]
    root_owner_sections: Tuple[str, ...]
    exclude_owner_sections: Tuple[str, ...]
    cross_cutting_sections: Tuple[str, ...]
    nested_owner_sections: Tuple[Tuple[str, Tuple[str, ...]], ...]

    @property
    def owner_roots(self) -> Tuple[str, ...]:
        return self.python_package_roots + self.directory_roots


@dataclass
class ValidationReport:
    """Structured validation result for human or machine output."""

    errors: List[str]
    expected_sections: List[Section]
    configured_sections: List[Section]
    fragment_types: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class SyncPlan:
    """Planned synchronization actions."""

    rewrite_pyproject: bool
    create_directories: List[str]
    prune_directories: List[str]
    blocked_prune_directories: List[str]

    @property
    def changed(self) -> bool:
        return bool(
            self.rewrite_pyproject
            or self.create_directories
            or self.prune_directories
            or self.blocked_prune_directories
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or synchronize Scikit-Plots Towncrier ownership",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing pyproject.toml; defaults to this checkout",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="Validate policy, configuration, directories, and fragments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report",
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List the expected Towncrier sections derived from current policy/tree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    list_parser.add_argument(
        "--kind",
        choices=("all", "cross-cutting", "owners"),
        default="all",
        help="Subset of expected sections to print",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    sync = subparsers.add_parser(
        "sync",
        help="Plan or apply deterministic Towncrier owner/config directory sync",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned changes; without this flag sync is read-only",
    )
    sync.add_argument(
        "--prune-empty",
        action="store_true",
        help=(
            "Allow removal of stale section directories only when they are empty or "
            "contain only .gitkeep"
        ),
    )
    sync.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    return parser


def _resolve_repo_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit("--repo-root does not exist or is not a directory: %s" % root)
    if not (root / PYPROJECT_NAME).is_file():
        raise SystemExit("--repo-root does not contain pyproject.toml: %s" % root)
    return root


def _load_pyproject(path: Path) -> Mapping[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _nested_table(data: Mapping[str, object], keys: Sequence[str]) -> Mapping[str, object]:
    current: object = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise SystemExit(
                "missing required pyproject table: [%s]" % ".".join(keys)
            )
        current = current[key]
    if not isinstance(current, Mapping):
        raise SystemExit("invalid pyproject table: [%s]" % ".".join(keys))
    return current


def _string_list(table: Mapping[str, object], key: str) -> Tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(
            "[%s].%s must be an array of strings"
            % (".".join(POLICY_TABLE), key)
        )
    return tuple(value)


def _nested_owner_mapping(table: Mapping[str, object]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    raw = table.get("nested_owner_sections", {})
    if not isinstance(raw, Mapping):
        raise SystemExit(
            "[%s].nested_owner_sections must be a TOML table"
            % ".".join(POLICY_TABLE)
        )

    items: List[Tuple[str, Tuple[str, ...]]] = []
    for parent, children in sorted(raw.items()):
        if not isinstance(parent, str):
            raise SystemExit("nested_owner_sections keys must be strings")
        if not isinstance(children, list) or not all(
            isinstance(child, str) for child in children
        ):
            raise SystemExit(
                "nested_owner_sections.%s must be an array of strings" % parent
            )
        if len(children) != len(set(children)):
            raise SystemExit(
                "nested_owner_sections.%s contains duplicate child paths" % parent
            )
        items.append((parent, tuple(children)))
    return tuple(items)


def _load_policy(data: Mapping[str, object]) -> Policy:
    table = _nested_table(data, POLICY_TABLE)
    policy = Policy(
        python_package_roots=_string_list(table, "python_package_roots"),
        directory_roots=_string_list(table, "directory_roots"),
        root_owner_sections=_string_list(table, "root_owner_sections"),
        exclude_owner_sections=_string_list(table, "exclude_owner_sections"),
        cross_cutting_sections=_string_list(table, "cross_cutting_sections"),
        nested_owner_sections=_nested_owner_mapping(table),
    )

    all_roots = set(policy.owner_roots)
    if len(all_roots) != len(policy.owner_roots):
        raise SystemExit("Towncrier maintenance policy contains duplicate owner roots")
    unknown_root_sections = sorted(set(policy.root_owner_sections) - all_roots)
    if unknown_root_sections:
        raise SystemExit(
            "root_owner_sections must name configured roots: %s"
            % ", ".join(unknown_root_sections)
        )
    return policy




def _package_path(repo_root: Path, dotted: str) -> Path:
    return repo_root.joinpath(*dotted.split("."))


def _validate_dotted_package_name(value: str, *, label: str) -> None:
    if not value or value.startswith(".") or value.endswith("."):
        raise SystemExit("%s must be a non-empty dotted package path: %r" % (label, value))
    parts = value.split(".")
    if any(not part.isidentifier() for part in parts):
        raise SystemExit("%s contains a non-identifier package component: %r" % (label, value))


def _discover_nested_owner_paths(repo_root: Path, policy: Policy) -> Set[str]:
    """Return explicitly requested nested package owners after strict validation."""

    excluded = set(policy.exclude_owner_sections)
    python_roots = set(policy.python_package_roots)
    discovered: Set[str] = set()

    for parent, children in policy.nested_owner_sections:
        _validate_dotted_package_name(parent, label="nested owner parent")
        parent_root = parent.split(".", 1)[0]
        if parent_root not in python_roots:
            raise SystemExit(
                "nested owner parent %s is outside configured Python package roots" % parent
            )
        parent_path = _package_path(repo_root, parent)
        if not parent_path.is_dir() or not (parent_path / "__init__.py").is_file():
            raise SystemExit(
                "nested owner parent is not an importable package in this checkout: %s"
                % parent
            )

        for relative in children:
            _validate_dotted_package_name(relative, label="nested owner child")
            full = "%s.%s" % (parent, relative)
            if full in excluded:
                raise SystemExit(
                    "nested owner %s is also listed in exclude_owner_sections" % full
                )
            if full in discovered:
                raise SystemExit("nested owner is configured more than once: %s" % full)
            path = _package_path(repo_root, full)
            if not path.is_dir() or not (path / "__init__.py").is_file():
                raise SystemExit(
                    "configured nested owner is not an importable package: %s" % full
                )
            discovered.add(full)

    return discovered

def _section_name(path: str) -> str:
    if path in {"scikitplot", "tools", "libs"}:
        return "%s (top-level)" % path
    if path.startswith("libs."):
        return path.replace(".", "/", 1)
    if path.startswith("tools."):
        return path.replace(".", "/", 1)
    return path


def _discover_owner_sections(repo_root: Path, policy: Policy) -> List[Section]:
    excluded = set(policy.exclude_owner_sections)
    discovered: Set[str] = set()

    for root_name in policy.root_owner_sections:
        if root_name not in excluded:
            discovered.add(root_name)

    for root_name in policy.python_package_roots:
        root = repo_root / root_name
        if not root.is_dir():
            raise SystemExit("configured Python package root does not exist: %s" % root_name)
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            owner = "%s.%s" % (root_name, child.name)
            if owner in excluded:
                continue
            if (child / "__init__.py").is_file():
                discovered.add(owner)

    for root_name in policy.directory_roots:
        root = repo_root / root_name
        if not root.is_dir():
            raise SystemExit("configured directory owner root does not exist: %s" % root_name)
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            owner = "%s.%s" % (root_name, child.name)
            if owner not in excluded:
                discovered.add(owner)

    nested = _discover_nested_owner_paths(repo_root, policy)
    redundant = sorted(discovered & nested)
    if redundant:
        raise SystemExit(
            "nested_owner_sections redundantly configures default owner(s): %s"
            % ", ".join(redundant)
        )
    discovered.update(nested)

    return [Section(name=_section_name(path), path=path) for path in sorted(discovered)]


def _configured_sections(data: Mapping[str, object]) -> List[Section]:
    towncrier = _nested_table(data, ("tool", "towncrier"))
    raw_sections = towncrier.get("section", [])
    if not isinstance(raw_sections, list):
        raise SystemExit("[tool.towncrier].section must be an array of tables")

    sections: List[Section] = []
    for index, raw in enumerate(raw_sections):
        if not isinstance(raw, Mapping):
            raise SystemExit("tool.towncrier.section[%d] is not a table" % index)
        name = raw.get("name")
        path = raw.get("path")
        if not isinstance(name, str) or not isinstance(path, str):
            raise SystemExit(
                "tool.towncrier.section[%d] must contain string name/path" % index
            )
        sections.append(Section(name=name, path=path))
    return sections


def _fragment_types(data: Mapping[str, object]) -> List[str]:
    towncrier = _nested_table(data, ("tool", "towncrier"))
    raw_types = towncrier.get("type", [])
    if not isinstance(raw_types, list):
        raise SystemExit("[tool.towncrier].type must be an array of tables")
    values: List[str] = []
    for index, raw in enumerate(raw_types):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("directory"), str):
            raise SystemExit(
                "tool.towncrier.type[%d] must contain string directory" % index
            )
        values.append(raw["directory"])
    return values


def _expected_sections(repo_root: Path, data: Mapping[str, object]) -> List[Section]:
    policy = _load_policy(data)
    configured = _configured_sections(data)
    configured_by_path = {section.path: section for section in configured}

    cross: List[Section] = []
    for path in policy.cross_cutting_sections:
        section = configured_by_path.get(path)
        if section is None:
            # Keep the expected path visible in diagnostics even when the configured
            # display name is missing. Cross-cutting labels remain curated by humans.
            section = Section(name=path, path=path)
        cross.append(section)

    return cross + _discover_owner_sections(repo_root, policy)


def _validate(repo_root: Path) -> ValidationReport:
    data = _load_pyproject(repo_root / PYPROJECT_NAME)
    policy = _load_policy(data)
    towncrier = _nested_table(data, ("tool", "towncrier"))
    configured = _configured_sections(data)
    expected = _expected_sections(repo_root, data)
    types = _fragment_types(data)
    errors: List[str] = []

    configured_paths = [section.path for section in configured]
    configured_names = [section.name for section in configured]
    if len(configured_paths) != len(set(configured_paths)):
        errors.append("duplicate [[tool.towncrier.section]].path values")
    if len(configured_names) != len(set(configured_names)):
        errors.append("duplicate [[tool.towncrier.section]].name values")

    expected_paths = {section.path for section in expected}
    configured_path_set = set(configured_paths)
    missing = sorted(expected_paths - configured_path_set)
    unexpected = sorted(configured_path_set - expected_paths)
    if missing:
        errors.append("missing configured sections: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected/stale configured sections: " + ", ".join(unexpected))

    expected_owner_names = {
        section.path: section.name
        for section in _discover_owner_sections(repo_root, policy)
    }
    configured_by_path = {section.path: section for section in configured}
    for path, expected_name in sorted(expected_owner_names.items()):
        current = configured_by_path.get(path)
        if current is not None and current.name != expected_name:
            errors.append(
                "owner section %s has name %r; expected %r"
                % (path, current.name, expected_name)
            )

    directory = towncrier.get("directory")
    if directory != DEFAULT_FRAGMENT_ROOT:
        errors.append(
            "tool.towncrier.directory is %r; expected %r"
            % (directory, DEFAULT_FRAGMENT_ROOT)
        )
    template = towncrier.get("template")
    if template != DEFAULT_TEMPLATE:
        errors.append(
            "tool.towncrier.template is %r; expected %r"
            % (template, DEFAULT_TEMPLATE)
        )

    fragment_root = repo_root / DEFAULT_FRAGMENT_ROOT
    if not fragment_root.is_dir():
        errors.append("fragment root does not exist: %s" % DEFAULT_FRAGMENT_ROOT)
        return ValidationReport(errors, expected, configured, types)

    actual_dirs = {
        child.name
        for child in fragment_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    }
    missing_dirs = sorted(configured_path_set - actual_dirs)
    stale_dirs = sorted(actual_dirs - configured_path_set)
    if missing_dirs:
        errors.append("configured sections without directories: " + ", ".join(missing_dirs))
    if stale_dirs:
        errors.append("unconfigured section directories: " + ", ".join(stale_dirs))

    if len(types) != len(set(types)):
        errors.append("duplicate [[tool.towncrier.type]].directory values")
    type_set = set(types)

    for section in sorted(actual_dirs):
        section_dir = fragment_root / section
        if section_dir.is_symlink():
            errors.append("section directory must not be a symlink: %s" % section)
            continue
        for fragment in sorted(section_dir.iterdir()):
            if fragment.name == ".gitkeep":
                continue
            relative = fragment.relative_to(repo_root)
            if fragment.is_symlink():
                errors.append("fragment must not be a symlink: %s" % relative)
                continue
            if not fragment.is_file():
                errors.append("unexpected non-file in section %s: %s" % (section, fragment.name))
                continue
            match = FRAGMENT_RE.fullmatch(fragment.name)
            if match is None:
                errors.append("invalid fragment filename: %s" % relative)
                continue
            if match.group("kind") not in type_set:
                errors.append(
                    "unknown fragment type in %s: %s"
                    % (relative, match.group("kind"))
                )
            text = fragment.read_text(encoding="utf-8").strip()
            bullet_lines = [line for line in text.splitlines() if line.startswith("- ")]
            if len(bullet_lines) != 1:
                errors.append(
                    "fragment must contain exactly one top-level '- ' bullet: %s"
                    % relative
                )

    root_files = {
        child.name
        for child in fragment_root.iterdir()
        if child.is_file() and not child.name.startswith(".")
    }
    unexpected_root_files = sorted(root_files - ROOT_ALLOWED_FILES)
    if unexpected_root_files:
        errors.append(
            "unexpected files at fragment root: " + ", ".join(unexpected_root_files)
        )

    return ValidationReport(errors, expected, configured, types)


def _render_owner_block(repo_root: Path, data: Mapping[str, object]) -> str:
    policy = _load_policy(data)
    owners = _discover_owner_sections(repo_root, policy)
    by_root: Dict[str, List[Section]] = {root: [] for root in policy.owner_roots}
    for section in owners:
        root = section.path.split(".", 1)[0]
        by_root[root].append(section)

    lines: List[str] = ["  " + OWNER_BLOCK_BEGIN]
    for root in policy.owner_roots:
        group = sorted(by_root[root], key=lambda section: section.path)
        if not group:
            continue
        lines.extend(["", "  # --- %s ownership ---" % root, ""])
        for section in group:
            lines.extend(
                [
                    "  [[tool.towncrier.section]]",
                    '    name = "%s"' % section.name,
                    '    path = "%s"' % section.path,
                    "",
                ]
            )
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("  " + OWNER_BLOCK_END)
    return "\n".join(lines)


def _replace_managed_owner_block(text: str, rendered_block: str) -> str:
    begin_index = text.find(OWNER_BLOCK_BEGIN)
    end_index = text.find(OWNER_BLOCK_END)
    if begin_index < 0 or end_index < 0 or end_index <= begin_index:
        raise SystemExit(
            "pyproject.toml is missing the managed Towncrier owner block markers"
        )
    line_start = text.rfind("\n", 0, begin_index) + 1
    line_end = text.find("\n", end_index)
    if line_end < 0:
        line_end = len(text)
    else:
        line_end += 1
    replacement = rendered_block + "\n"
    return text[:line_start] + replacement + text[line_end:]


def _is_prunable_directory(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    entries = list(path.iterdir())
    return not entries or all(entry.is_file() and entry.name == ".gitkeep" for entry in entries)


def _sync_plan(repo_root: Path, prune_empty: bool) -> SyncPlan:
    pyproject = repo_root / PYPROJECT_NAME
    data = _load_pyproject(pyproject)
    configured = _configured_sections(data)
    configured_paths = {section.path for section in configured}
    expected_paths = {section.path for section in _expected_sections(repo_root, data)}

    current_text = pyproject.read_text(encoding="utf-8")
    rendered = _render_owner_block(repo_root, data)
    desired_text = _replace_managed_owner_block(current_text, rendered)
    rewrite_pyproject = desired_text != current_text

    fragment_root = repo_root / DEFAULT_FRAGMENT_ROOT
    actual_dirs = {
        child.name
        for child in fragment_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    }

    # Directories are synchronized against the *expected* set so a dry run can
    # show the full desired state even before pyproject.toml is rewritten.
    create = sorted(expected_paths - actual_dirs)
    stale = sorted(actual_dirs - expected_paths)
    prune: List[str] = []
    blocked: List[str] = []
    if prune_empty:
        for path in stale:
            if _is_prunable_directory(fragment_root / path):
                prune.append(path)
            else:
                blocked.append(path)
    else:
        # Do not allow an apply run to leave a known stale directory behind and
        # then fail only after partially updating pyproject/directories. The user
        # must explicitly opt into safe empty-directory pruning or review it manually.
        blocked.extend(stale)

    # A configured-but-unexpected path with content must never be hidden by an
    # automatic pyproject rewrite. Surface it as a blocked prune until a human
    # decides where its fragment belongs.
    for path in sorted(configured_paths - expected_paths):
        directory = fragment_root / path
        if directory.exists() and path not in prune and path not in blocked:
            if prune_empty and _is_prunable_directory(directory):
                prune.append(path)
            else:
                blocked.append(path)

    return SyncPlan(
        rewrite_pyproject=rewrite_pyproject,
        create_directories=create,
        prune_directories=sorted(set(prune)),
        blocked_prune_directories=sorted(set(blocked)),
    )


def _apply_sync(repo_root: Path, plan: SyncPlan) -> None:
    pyproject = repo_root / PYPROJECT_NAME
    data = _load_pyproject(pyproject)
    fragment_root = repo_root / DEFAULT_FRAGMENT_ROOT

    if plan.rewrite_pyproject:
        current_text = pyproject.read_text(encoding="utf-8")
        desired_text = _replace_managed_owner_block(
            current_text, _render_owner_block(repo_root, data)
        )
        pyproject.write_text(desired_text, encoding="utf-8")

    for section in plan.create_directories:
        path = fragment_root / section
        path.mkdir(parents=False, exist_ok=False)
        (path / ".gitkeep").write_text("", encoding="utf-8")

    for section in plan.prune_directories:
        path = fragment_root / section
        if not _is_prunable_directory(path):
            raise SystemExit(
                "refusing to remove non-empty or unsafe stale directory: %s" % path
            )
        shutil.rmtree(str(path))


def _report_payload(report: ValidationReport) -> Dict[str, object]:
    expected_paths = [section.path for section in report.expected_sections]
    return {
        "ok": report.ok,
        "errors": report.errors,
        "counts": {
            "expected_sections": len(report.expected_sections),
            "configured_sections": len(report.configured_sections),
            "fragment_types": len(report.fragment_types),
        },
        "expected_sections": expected_paths,
    }


def _sync_payload(plan: SyncPlan, applied: bool) -> Dict[str, object]:
    return {
        "applied": applied,
        "changed": plan.changed,
        "rewrite_pyproject": plan.rewrite_pyproject,
        "create_directories": plan.create_directories,
        "prune_directories": plan.prune_directories,
        "blocked_prune_directories": plan.blocked_prune_directories,
    }


def _write_json(payload: Mapping[str, object], stream: TextIO) -> None:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def _run_check(repo_root: Path, as_json: bool, stdout: TextIO, stderr: TextIO) -> int:
    report = _validate(repo_root)
    if as_json:
        _write_json(_report_payload(report), stdout)
        return 0 if report.ok else 1

    if report.ok:
        print("Towncrier section validation: PASS", file=stdout)
        print("- configured sections: %d" % len(report.configured_sections), file=stdout)
        print("- configured fragment types: %d" % len(report.fragment_types), file=stdout)
        return 0

    print("Towncrier section validation: FAIL", file=stderr)
    for error in report.errors:
        print("- %s" % error, file=stderr)
    return 1


def _run_list(
    repo_root: Path,
    kind: str,
    as_json: bool,
    stdout: TextIO,
) -> int:
    data = _load_pyproject(repo_root / PYPROJECT_NAME)
    policy = _load_policy(data)
    configured_by_path = {
        section.path: section for section in _configured_sections(data)
    }
    cross = [
        configured_by_path.get(path, Section(name=path, path=path))
        for path in policy.cross_cutting_sections
    ]
    owners = _discover_owner_sections(repo_root, policy)
    sections = cross + owners
    if kind == "cross-cutting":
        sections = cross
    elif kind == "owners":
        sections = owners

    if as_json:
        _write_json(
            {"sections": [{"name": section.name, "path": section.path} for section in sections]},
            stdout,
        )
    else:
        for section in sections:
            print("%-36s %s" % (section.path, section.name), file=stdout)
    return 0


def _run_sync(
    repo_root: Path,
    apply: bool,
    prune_empty: bool,
    as_json: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    plan = _sync_plan(repo_root, prune_empty=prune_empty)

    if apply and plan.blocked_prune_directories:
        payload = _sync_payload(plan, applied=False)
        if as_json:
            _write_json(payload, stdout)
        else:
            print("Towncrier sync: REFUSED", file=stderr)
            print(
                "- stale directories contain files and require manual review: %s"
                % ", ".join(plan.blocked_prune_directories),
                file=stderr,
            )
        return 2

    if apply:
        _apply_sync(repo_root, plan)
        post = _validate(repo_root)
        if not post.ok:
            if as_json:
                payload = _sync_payload(plan, applied=True)
                payload["post_check"] = _report_payload(post)
                _write_json(payload, stdout)
            else:
                print("Towncrier sync applied, but post-check failed:", file=stderr)
                for error in post.errors:
                    print("- %s" % error, file=stderr)
            return 1

    if as_json:
        _write_json(_sync_payload(plan, applied=apply), stdout)
    else:
        mode = "APPLIED" if apply else "DRY-RUN"
        print("Towncrier sync: %s" % mode, file=stdout)
        print("- rewrite pyproject.toml: %s" % plan.rewrite_pyproject, file=stdout)
        print(
            "- create directories: %s"
            % (", ".join(plan.create_directories) or "none"),
            file=stdout,
        )
        print(
            "- prune empty stale directories: %s"
            % (", ".join(plan.prune_directories) or "none"),
            file=stdout,
        )
        if plan.blocked_prune_directories:
            print(
                "- manual-review stale directories: %s"
                % ", ".join(plan.blocked_prune_directories),
                file=stdout,
            )
        if not apply and plan.changed:
            print("- rerun with --apply to make these changes", file=stdout)
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Run the Towncrier maintenance CLI and return a process status."""
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = _resolve_repo_root(
        DEFAULT_REPO_ROOT if args.repo_root is None else args.repo_root
    )
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if args.command == "check":
        return _run_check(repo_root, args.json, out, err)
    if args.command == "list":
        return _run_list(repo_root, args.kind, args.json, out)
    if args.command == "sync":
        return _run_sync(
            repo_root,
            apply=args.apply,
            prune_empty=args.prune_empty,
            as_json=args.json,
            stdout=out,
            stderr=err,
        )
    parser.error("unknown command: %s" % args.command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
