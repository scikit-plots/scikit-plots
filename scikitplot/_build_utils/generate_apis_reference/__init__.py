#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Inspect, validate, and safely synchronize ``docs/source/apis_reference.py``.

The API reference mixes two kinds of information:

* editorial structure (titles, guide links, descriptions, ordering, inheritance
  diagrams), whose canonical source is the checked-in ``apis_reference.py.in`` template;
  and
* public symbol inventories, which can be checked against an installed package.

``docs/source/apis_reference.py`` is a disposable generated artifact.  ``rebuild``
can recreate the complete file from the adjacent ``apis_reference.py.in`` template
without importing Scikit-Plots, so a deleted or stale generated file is recoverable
from the installed/source maintenance package.  Runtime
``check``/``plan``/``inventory`` verification remains separate and imports the
installed package only when explicitly requested.

This helper deliberately does **not** execute ``apis_reference.py`` while
inspecting it.  It parses the file with :mod:`ast`, keeping optional/compiled
module imports from becoming hidden side effects of maintenance.

The command line is read-only by default.  ``rebuild`` and ``generate`` show
reviewable diffs; canonical writes require ``--apply``.  ``build_reference_source``
and ``reference_blueprint_copy`` also provide an importable customization API for
advanced build tooling without invoking argparse.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TextIO,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT_NAME = "pyproject.toml"
POLICY_TABLE = ("tool", "scikitplot", "maintenance", "api_reference")
DEFAULT_REFERENCE_FILE = "docs/source/apis_reference.py"
DEFAULT_DISTRIBUTION = "scikit-plots"
DEFAULT_PACKAGE_ROOT = "scikitplot"
PUBLIC_SOURCES = ("auto", "all", "local")


class APIReferenceError(RuntimeError):
    """Raised for malformed reference configuration or unsafe generation."""


class ResolutionUnavailable(RuntimeError):
    """A reference could not be verified because an optional import failed."""


@dataclass(frozen=True)
class Policy:
    """Repository policy for API-reference maintenance."""

    reference_file: str = DEFAULT_REFERENCE_FILE
    distribution: str = DEFAULT_DISTRIBUTION
    package_root: str = DEFAULT_PACKAGE_ROOT
    public_source: str = "auto"
    require_distribution: bool = True
    include_module_objects: bool = False


@dataclass
class SectionReference:
    """One curated section in an ``APIS_REFERENCE`` module entry."""

    module: str
    index: int
    title: Optional[str]
    autosummary: list[str]
    classes: list[str]
    sources: list[str]
    exclude: set[str]
    autosummary_node: ast.list
    classes_node: Optional[ast.list]

    @property
    def label(self) -> str:
        return self.title or "<untitled section>"


@dataclass
class ModuleReference:
    """Static representation of one ``APIS_REFERENCE`` module entry."""

    name: str
    sections: list[SectionReference]
    public_source: Optional[str] = None
    ignore: set[str] = field(default_factory=set)
    sync: bool = True

    @property
    def documented(self) -> list[str]:
        values: list[str] = []
        for section in self.sections:
            values.extend(section.autosummary)
        return values


@dataclass
class ReferenceModel:
    """AST-backed representation of the canonical API-reference file."""

    path: Path
    source: str
    modules: dict[str, ModuleReference]


@dataclass
class ModuleInspection:
    """Installed-package comparison for one documented module."""

    module: str
    public_source: str = ""
    public_names: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    stale_classes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    assigned: dict[str, int] = field(default_factory=dict)
    ambiguous: dict[str, list[int]] = field(default_factory=dict)
    unassigned: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    import_error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class InspectionReport:
    """Structured result for human, CI, or AI callers."""

    modules: dict[str, ModuleInspection]
    structural_errors: list[str] = field(default_factory=list)
    environment_errors: list[str] = field(default_factory=list)

    @property
    def stale_count(self) -> int:
        return sum(
            (
                len(item.stale) + len(item.stale_classes)
                for item in self.modules.values()
            ),
        )

    @property
    def missing_count(self) -> int:
        return sum(len(item.missing) for item in self.modules.values())

    @property
    def import_error_count(self) -> int:
        return sum(item.import_error is not None for item in self.modules.values())

    def has_errors(self, *, strict_coverage: bool = False) -> bool:
        if self.structural_errors or self.environment_errors or self.import_error_count:
            return True
        if self.stale_count:
            return True
        if strict_coverage and self.missing_count:
            return True
        return False


@dataclass
class RenderResult:
    """Candidate source plus actions that were applied or blocked."""

    source: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    blocked_removals: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


# ---------------------------------------------------------------------------
# Canonical template and full-file rebuild model
# ---------------------------------------------------------------------------


DEFAULT_TEMPLATE_NAME = "apis_reference.py.in"


@dataclass(frozen=True)
class PythonExpression:
    """A trusted Python expression preserved from the canonical template."""

    source: str


def _expr(source: str) -> PythonExpression:
    """Create a validated expression for programmatic blueprint customization."""

    if not isinstance(source, str) or not source.strip():
        raise APIReferenceError("PythonExpression source must be a non-empty string")
    try:
        ast.parse("(" + source + ")", mode="eval")
    except SyntaxError as exc:
        raise APIReferenceError("invalid API-reference expression: %s" % exc) from exc
    return PythonExpression(source.strip())


def default_template_path() -> Path:
    """Return the installed canonical ``apis_reference.py.in`` template path."""

    return Path(__file__).resolve().with_name(DEFAULT_TEMPLATE_NAME)


def _read_template(
    template_path: Optional[Path] = None,
    template_source: Optional[str] = None,
) -> tuple[Path, str]:
    if template_path is not None and template_source is not None:
        raise APIReferenceError(
            "template_path and template_source are mutually exclusive",
        )
    path = (template_path or default_template_path()).expanduser().resolve()
    if template_source is not None:
        return path, template_source
    if not path.is_file():
        raise APIReferenceError(
            "API-reference template does not exist: %s" % path,
        )
    return path, path.read_text(encoding="utf-8")


def _assignment_value_nodes(
    source: str,
    *,
    filename: str,
) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise APIReferenceError(
            "cannot parse API-reference template %s: %s" % (filename, exc),
        ) from exc

    wanted = {"APIS_REFERENCE", "DEPRECATED_APIS_REFERENCE"}
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        value: Optional[ast.AST] = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                if target.id in found:
                    raise APIReferenceError(
                        "duplicate %s assignment in %s" % (target.id, filename),
                    )
                found[target.id] = value

    missing = sorted(wanted - set(found))
    if missing:
        raise APIReferenceError(
            "template %s is missing assignment(s): %s" % (filename, ", ".join(missing)),
        )
    return found


def _node_to_blueprint(
    source: str,
    node: ast.AST,
    *,
    context: str,
) -> object:
    """Convert template AST into editable data while preserving expressions."""

    if isinstance(node, ast.dict):
        result: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise APIReferenceError(
                    "%s cannot use dictionary unpacking" % context,
                )
            key = _literal(key_node)
            if not isinstance(key, str):
                raise APIReferenceError(
                    "%s mapping keys must be literal strings" % context,
                )
            result[key] = _node_to_blueprint(
                source,
                value_node,
                context="%s.%s" % (context, key),
            )
        return result
    if isinstance(node, ast.list):
        return [
            _node_to_blueprint(
                source,
                item,
                context="%s[%d]" % (context, index),
            )
            for index, item in enumerate(node.elts)
        ]
    if isinstance(node, ast.tuple):
        return [
            _node_to_blueprint(
                source,
                item,
                context="%s[%d]" % (context, index),
            )
            for index, item in enumerate(node.elts)
        ]
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        segment = ast.get_source_segment(source, node)
        if not segment or not segment.strip():
            raise APIReferenceError(
                "cannot preserve Python expression at %s" % context,
            )
        return _expr(segment)


def _blueprint_from_source(
    path: Path,
    source: str,
) -> dict[str, object]:
    assignments = _assignment_value_nodes(source, filename=str(path))
    blueprint = {
        "api_reference": _node_to_blueprint(
            source,
            assignments["APIS_REFERENCE"],
            context="api_reference",
        ),
        "deprecated_api_reference": _node_to_blueprint(
            source,
            assignments["DEPRECATED_APIS_REFERENCE"],
            context="deprecated_api_reference",
        ),
    }
    validate_reference_blueprint(blueprint)
    return blueprint


def load_reference_blueprint(
    template_path: Optional[Path] = None,
    *,
    template_source: Optional[str] = None,
) -> dict[str, object]:
    """
    Parse the canonical template into a mutable declarative blueprint.

    The template remains the durable source of truth.  This conversion exists
    for advanced tooling that wants structured customization without editing the
    generated ``docs/source/apis_reference.py`` artifact directly.
    """

    path, source = _read_template(template_path, template_source)
    return _blueprint_from_source(path, source)


def reference_blueprint_copy(
    template_path: Optional[Path] = None,
    *,
    template_source: Optional[str] = None,
) -> dict[str, object]:
    """Return an isolated mutable blueprint derived from the canonical template."""

    return copy.deepcopy(
        load_reference_blueprint(
            template_path,
            template_source=template_source,
        ),
    )


def _validate_python_expression(
    value: PythonExpression,
    *,
    context: str,
) -> None:
    try:
        ast.parse("(" + value.source + ")", mode="eval")
    except SyntaxError as exc:
        raise APIReferenceError(
            "%s contains invalid Python expression: %s" % (context, exc),
        ) from exc


def validate_reference_blueprint(
    blueprint: Mapping[str, object],
) -> None:
    """Validate a structured API-reference blueprint without importing Scikit-Plots."""

    allowed_top = {
        "api_reference",
        "deprecated_api_reference",
    }
    unknown_top = sorted(set(blueprint) - allowed_top)
    if unknown_top:
        raise APIReferenceError(
            "unknown blueprint key(s): %s" % ", ".join(unknown_top),
        )

    api_reference = blueprint.get("api_reference")
    if not isinstance(api_reference, Mapping) or not api_reference:
        raise APIReferenceError(
            "blueprint.api_reference must be a non-empty mapping",
        )

    def _validate_value(
        value: object,
        *,
        context: str,
    ) -> None:
        if isinstance(value, PythonExpression):
            _validate_python_expression(value, context=context)
            return
        if value is None or isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
            ),
        ):
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                _validate_value(
                    item,
                    context="%s[%d]" % (context, index),
                )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise APIReferenceError(
                        "%s has non-string mapping key %r" % (context, key),
                    )
                _validate_value(item, context="%s.%s" % (context, key))
            return
        raise APIReferenceError(
            "%s contains unsupported value %r" % (context, type(value).__name__),
        )

    for module_name, module_info in api_reference.items():
        if not isinstance(module_name, str) or not module_name.strip():
            raise APIReferenceError(
                "api_reference module names must be non-empty strings",
            )
        if not isinstance(module_info, Mapping):
            raise APIReferenceError(
                "%s blueprint entry must be a mapping" % module_name,
            )
        if "short_summary" not in module_info:
            raise APIReferenceError(
                "%s blueprint entry is missing short_summary" % module_name,
            )
        sections = module_info.get("sections")
        if not isinstance(sections, list) or not sections:
            raise APIReferenceError(
                "%s.sections must be a non-empty list" % module_name,
            )
        for index, section in enumerate(sections):
            context = "%s.sections[%d]" % (module_name, index)
            if not isinstance(section, Mapping):
                raise APIReferenceError("%s must be a mapping" % context)
            if "autosummary" not in section:
                raise APIReferenceError("%s is missing autosummary" % context)
            autosummary = section.get("autosummary")
            if not isinstance(autosummary, list) or not all(
                (isinstance(item, str) for item in autosummary),
            ):
                raise APIReferenceError(
                    "%s.autosummary must be a list of strings" % context,
                )
            classes = section.get("classes", [])
            if not isinstance(classes, list) or not all(
                (isinstance(item, str) for item in classes),
            ):
                raise APIReferenceError(
                    "%s.classes must be a list of strings" % context,
                )
            outside = sorted(set(classes) - set(autosummary))
            if outside:
                raise APIReferenceError(
                    "%s.classes contains names outside autosummary: %s"
                    % (context, ", ".join(outside))
                )
            sources = section.get("sources", [])
            if not isinstance(sources, list) or not all(
                (isinstance(item, str) for item in sources),
            ):
                raise APIReferenceError(
                    "%s.sources must be a list of strings" % context,
                )
            exclude = section.get("exclude", [])
            if not isinstance(exclude, list) or not all(
                (isinstance(item, str) for item in exclude),
            ):
                raise APIReferenceError(
                    "%s.exclude must be a list of strings" % context,
                )
            _validate_value(section, context=context)
        _validate_value(module_info, context=module_name)

    deprecated = blueprint.get("deprecated_api_reference", {})
    if not isinstance(deprecated, Mapping):
        raise APIReferenceError(
            "blueprint.deprecated_api_reference must be a mapping",
        )
    for version, names in deprecated.items():
        if not isinstance(version, str) or not version.strip():
            raise APIReferenceError(
                "deprecated API version keys must be non-empty strings",
            )
        if not isinstance(names, list) or not all(
            (isinstance(item, str) for item in names),
        ):
            raise APIReferenceError(
                "deprecated API entries for %s must be a list of strings" % version,
            )


def _render_python_value(
    value: object,
    *,
    indent: int = 0,
) -> str:
    """Render blueprint data as deterministic readable Python source."""

    pad = " " * indent
    child_pad = " " * (indent + 4)
    if isinstance(value, PythonExpression):
        lines = textwrap.dedent(value.source).strip().splitlines()
        if len(lines) == 1:
            return lines[0]
        rendered = ["("]
        rendered.extend(child_pad + line.lstrip() for line in lines)
        rendered.append(pad + ")")
        return "\n".join(rendered)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        rendered: list[str] = ["["]
        for item in value:
            text = _render_python_value(item, indent=indent + 4)
            item_lines = text.splitlines() or [""]
            rendered.append(child_pad + item_lines[0])
            rendered.extend(child_pad + line for line in item_lines[1:])
            rendered[-1] += ","
        rendered.append(pad + "]")
        return "\n".join(rendered)
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        rendered = ["{"]
        for key, item in value.items():
            if not isinstance(key, str):
                raise APIReferenceError(
                    "cannot render non-string blueprint key %r" % (key,),
                )
            text = _render_python_value(item, indent=indent + 4)
            item_lines = text.splitlines() or [""]
            rendered.append(
                (
                    child_pad
                    + json.dumps(key, ensure_ascii=False)
                    + ": "
                    + item_lines[0]
                ),
            )
            rendered.extend(child_pad + line for line in item_lines[1:])
            rendered[-1] += ","
        rendered.append(pad + "}")
        return "\n".join(rendered)
    raise APIReferenceError(
        "cannot render blueprint value %r" % (type(value).__name__,),
    )


def _node_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)
    starts: list[int] = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    if not hasattr(node, "end_lineno") or node.end_lineno is None:
        raise APIReferenceError(
            "Python parser did not provide expression end offsets",
        )
    start = starts[node.lineno - 1] + node.col_offset
    end = starts[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _template_envelope(
    source: str,
    *,
    filename: str,
) -> tuple[str, str, str]:
    assignments = _assignment_value_nodes(source, filename=filename)
    api_start, api_end = _node_offsets(source, assignments["APIS_REFERENCE"])
    deprecated_start, deprecated_end = _node_offsets(
        source,
        assignments["DEPRECATED_APIS_REFERENCE"],
    )
    if not (api_start < api_end <= deprecated_start < deprecated_end):
        raise APIReferenceError(
            "API-reference template assignments are in an unsafe order",
        )
    return (
        source[:api_start],
        source[api_end:deprecated_start],
        source[deprecated_end:],
    )


def build_reference_source(
    blueprint: Optional[Mapping[str, object]] = None,
    *,
    template_path: Optional[Path] = None,
    template_source: Optional[str] = None,
    prefix: Optional[str] = None,
    between: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Build complete ``apis_reference.py`` source from the canonical template.

    With no blueprint or envelope overrides, the result is the template byte for
    byte.  This makes the checked-in ``.py.in`` file the single durable source of
    editorial structure.  Advanced callers may derive a mutable blueprint with
    :func:`reference_blueprint_copy`, customize it, and render it back through
    the same template envelope.
    """

    path, source = _read_template(template_path, template_source)
    canonical = _blueprint_from_source(path, source)
    if blueprint is None and prefix is None and between is None and suffix is None:
        return source

    selected = canonical if blueprint is None else copy.deepcopy(dict(blueprint))
    validate_reference_blueprint(selected)
    template_prefix, template_between, template_suffix = _template_envelope(
        source,
        filename=str(path),
    )
    return "".join(
        [
            template_prefix if prefix is None else prefix,
            _render_python_value(selected["api_reference"]),
            template_between if between is None else between,
            _render_python_value(selected.get("deprecated_api_reference", {})),
            template_suffix if suffix is None else suffix,
        ]
    )


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> Mapping[str, object]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.8-3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise APIReferenceError(
                "Python < 3.11 requires 'tomli' to read pyproject.toml. "
                "Install the repository development dependencies."
            ) from exc
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _nested_table(
    data: Mapping[str, object],
    keys: Sequence[str],
) -> Optional[Mapping[str, object]]:
    current: object = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, Mapping) else None


def load_policy(repo_root: Path) -> Policy:
    """Load API-reference maintenance policy from ``pyproject.toml``.

    Missing policy is allowed and falls back to explicit conservative defaults;
    malformed configured values are rejected.
    """

    data = _load_toml(repo_root / PYPROJECT_NAME)
    table = _nested_table(data, POLICY_TABLE)
    if table is None:
        return Policy()

    def _string(key: str, default: str) -> str:
        value = table.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise APIReferenceError(
                "[%s].%s must be a non-empty string" % (".".join(POLICY_TABLE), key),
            )
        return value

    def _bool(key: str, default: bool) -> bool:
        value = table.get(key, default)
        if not isinstance(value, bool):
            raise APIReferenceError(
                "[%s].%s must be boolean" % (".".join(POLICY_TABLE), key),
            )
        return value

    public_source = _string("public_source", "auto")
    if public_source not in PUBLIC_SOURCES:
        raise APIReferenceError(
            "[%s].public_source must be one of: %s"
            % (".".join(POLICY_TABLE), ", ".join(PUBLIC_SOURCES))
        )

    return Policy(
        reference_file=_string("reference_file", DEFAULT_REFERENCE_FILE),
        distribution=_string("distribution", DEFAULT_DISTRIBUTION),
        package_root=_string("package_root", DEFAULT_PACKAGE_ROOT),
        public_source=public_source,
        require_distribution=_bool("require_distribution", True),
        include_module_objects=_bool("include_module_objects", False),
    )


# ---------------------------------------------------------------------------
# Static APIS_REFERENCE parsing -- never executes the docs configuration
# ---------------------------------------------------------------------------


def _literal(
    node: Optional[ast.AST],
    *,
    default: object = None,
) -> object:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return default


def _dict_items(node: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(node, ast.dict):
        raise APIReferenceError(
            "expected dictionary literal in APIS_REFERENCE",
        )
    result: dict[str, ast.AST] = {}
    for key_node, value_node in zip(node.keys, node.values):
        key = _literal(key_node)
        if not isinstance(key, str):
            raise APIReferenceError(
                "APIS_REFERENCE keys must be literal strings",
            )
        result[key] = value_node
    return result


def _string_list(
    node: Optional[ast.AST],
    *,
    field_name: str,
) -> list[str]:
    if node is None:
        return []
    value = _literal(node)
    if not isinstance(value, list) or not all(
        (isinstance(item, str) for item in value),
    ):
        raise APIReferenceError(
            "%s must be a literal list of strings" % field_name,
        )
    return list(value)


def _description_sources(
    node: Optional[ast.AST],
) -> list[str]:
    """Extract module paths from literal ``_get_submodule(parent, child)`` calls."""

    if node is None:
        return []
    values: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Name) or child.func.id != "_get_submodule":
            continue
        if len(child.args) < 2:
            continue
        parent = _literal(child.args[0])
        name = _literal(child.args[1])
        if isinstance(parent, str) and isinstance(name, str):
            if name == "__init__":
                values.append(parent)
            else:
                values.append("%s.%s" % (parent, name))
    return values


def _module_generator_config(
    node: Optional[ast.AST],
) -> tuple[Optional[str], set[str], bool]:
    if node is None:
        return None, set(), True
    value = _literal(node)
    if not isinstance(value, dict):
        raise APIReferenceError(
            "module generator metadata must be a literal dictionary",
        )
    public_source = value.get("public_source")
    if public_source is not None and public_source not in PUBLIC_SOURCES:
        raise APIReferenceError(
            "module generator public_source must be one of %s" % (PUBLIC_SOURCES,),
        )
    ignore = value.get("ignore", [])
    if not isinstance(ignore, list) or not all(
        (isinstance(item, str) for item in ignore),
    ):
        raise APIReferenceError(
            "module generator ignore must be a list of strings",
        )
    sync = value.get("sync", True)
    if not isinstance(sync, bool):
        raise APIReferenceError(
            "module generator sync must be boolean",
        )
    return public_source, set(ignore), sync


def load_reference(
    path: Path,
) -> ReferenceModel:
    """Parse ``APIS_REFERENCE`` without importing or executing the file."""

    path = path.resolve()
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise APIReferenceError(
            "cannot parse %s: %s" % (path, exc),
        ) from exc

    assignment: Optional[ast.AST] = None
    for node in tree.body:
        value: Optional[ast.AST] = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        if any(
            isinstance(target, ast.Name) and target.id == "APIS_REFERENCE"
            for target in targets
        ):
            assignment = value

    if not isinstance(assignment, ast.dict):
        raise APIReferenceError(
            "could not find literal APIS_REFERENCE dictionary assignment",
        )

    modules: dict[str, ModuleReference] = {}
    for key_node, value_node in zip(assignment.keys, assignment.values):
        module_name = _literal(key_node)
        if not isinstance(module_name, str):
            raise APIReferenceError(
                "APIS_REFERENCE module keys must be strings",
            )
        if module_name in modules:
            raise APIReferenceError(
                "duplicate APIS_REFERENCE module: %s" % module_name,
            )
        module_fields = _dict_items(value_node)
        sections_node = module_fields.get("sections")
        if not isinstance(sections_node, ast.list):
            raise APIReferenceError(
                "%s.sections must be a literal list" % module_name,
            )

        public_source, ignore, sync = _module_generator_config(
            module_fields.get("generator"),
        )
        sections: list[SectionReference] = []
        for index, section_node in enumerate(sections_node.elts):
            fields = _dict_items(section_node)
            title_value = _literal(fields.get("title"))
            if title_value is not None and not isinstance(title_value, str):
                raise APIReferenceError(
                    "%s section %d title must be a string or None"
                    % (module_name, index),
                )
            autosummary_node = fields.get("autosummary")
            if not isinstance(autosummary_node, ast.list):
                raise APIReferenceError(
                    "%s section %d autosummary must be a literal list"
                    % (module_name, index),
                )
            classes_node = fields.get("classes")
            if classes_node is not None and not isinstance(classes_node, ast.list):
                raise APIReferenceError(
                    "%s section %d classes must be a literal list"
                    % (module_name, index),
                )
            explicit_sources = _string_list(fields.get("sources"), field_name="sources")
            inferred_sources = _description_sources(fields.get("description"))
            exclude = set(_string_list(fields.get("exclude"), field_name="exclude"))
            sections.append(
                SectionReference(
                    module=module_name,
                    index=index,
                    title=title_value,
                    autosummary=_string_list(
                        autosummary_node,
                        field_name="autosummary",
                    ),
                    classes=_string_list(classes_node, field_name="classes"),
                    sources=list(dict.fromkeys(explicit_sources + inferred_sources)),
                    exclude=exclude,
                    autosummary_node=autosummary_node,
                    classes_node=classes_node,
                )
            )
        modules[module_name] = ModuleReference(
            name=module_name,
            sections=sections,
            public_source=public_source,
            ignore=ignore,
            sync=sync,
        )

    return ReferenceModel(path=path, source=source, modules=modules)


# ---------------------------------------------------------------------------
# Installed package inspection
# ---------------------------------------------------------------------------


def _resolve_name(
    module: object,
    module_name: str,
    name: str,
) -> object:
    """Resolve dotted reference names, importing a submodule prefix when needed.

    A missing simple attribute is definitive and raises :class:`AttributeError`.
    A dotted submodule that cannot be imported is *not* definitive because the
    installed environment may lack an optional/compiled capability; that case
    raises :class:`ResolutionUnavailable` so callers do not auto-prune it.
    """

    current = module
    parts = name.split(".")
    for index, part in enumerate(parts):
        try:
            current = getattr(current, part)
            continue
        except (ImportError, ModuleNotFoundError) as exc:
            raise ResolutionUnavailable(
                "%s could not resolve %s because %s: %s"
                % (module_name, name, type(exc).__name__, exc)
            ) from exc
        except AttributeError:
            if index == 0 and len(parts) == 1:
                raise

        prefix = "%s.%s" % (module_name, ".".join(parts[: index + 1]))
        try:
            current = importlib.import_module(prefix)
            continue
        except (ImportError, ModuleNotFoundError) as exc:
            raise ResolutionUnavailable(
                "%s could not import %s while resolving %s: %s"
                % (module_name, prefix, name, exc)
            ) from exc
    return current


def _object_origin(
    obj: object,
    default_module: str,
) -> str:
    if inspect.ismodule(obj):
        return getattr(obj, "__name__", default_module)
    origin = getattr(obj, "__module__", None)
    return origin if isinstance(origin, str) and origin else default_module


def _public_names(
    module: object,
    mode: str,
    *,
    include_module_objects: bool,
) -> tuple[list[str], str]:
    selected = mode
    explicit = getattr(module, "__all__", None)
    if mode == "auto":
        if isinstance(explicit, (list, tuple)) and all(
            (isinstance(item, str) for item in explicit),
        ):
            selected = "all"
        else:
            selected = "local"

    if selected == "all":
        if not isinstance(explicit, (list, tuple)) or not all(
            (isinstance(item, str) for item in explicit),
        ):
            raise APIReferenceError(
                "public_source='all' requires a string sequence __all__",
            )
        names = list(dict.fromkeys(explicit))
    elif selected == "local":
        module_name = getattr(module, "__name__", "")
        names = []
        for name in sorted(set(dir(module))):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(module, name)
            except Exception:
                continue
            if inspect.ismodule(obj):
                if include_module_objects and getattr(
                    obj,
                    "__name__",
                    "",
                ).startswith(module_name + "."):
                    names.append(name)
                continue
            origin = getattr(obj, "__module__", module_name)
            if (
                not isinstance(origin, str)
                or origin == module_name
                or origin.startswith(
                    module_name + ".",
                )
            ):
                names.append(name)
    else:
        raise APIReferenceError("unknown public source: %s" % mode)

    if not include_module_objects:
        filtered: list[str] = []
        for name in names:
            try:
                if inspect.ismodule(getattr(module, name)):
                    continue
            except Exception:
                pass
            filtered.append(name)
        names = filtered
    return sorted(dict.fromkeys(names)), selected


def _section_source_sets(
    module_ref: ModuleReference,
    imported_module: object,
) -> dict[int, set[str]]:
    sources: dict[int, set[str]] = {}
    for section in module_ref.sections:
        values: set[str] = set(section.sources)
        if not values:
            for name in section.autosummary:
                try:
                    obj = _resolve_name(imported_module, module_ref.name, name)
                except (AttributeError, ResolutionUnavailable):
                    continue
                values.add(_object_origin(obj, module_ref.name))
        sources[section.index] = values
    return sources


def _assign_missing(
    module_ref: ModuleReference,
    imported_module: object,
    missing: Iterable[str],
) -> tuple[dict[str, int], dict[str, list[int]], list[str]]:
    assigned: dict[str, int] = {}
    ambiguous: dict[str, list[int]] = {}
    unassigned: list[str] = []
    section_sources = _section_source_sets(module_ref, imported_module)

    for name in missing:
        try:
            obj = _resolve_name(imported_module, module_ref.name, name)
        except (AttributeError, ResolutionUnavailable):
            unassigned.append(name)
            continue
        origin = _object_origin(obj, module_ref.name)
        matches: list[int] = []
        for section in module_ref.sections:
            if name in section.exclude:
                continue
            prefixes = section_sources.get(section.index, set())
            if any(
                origin == prefix or origin.startswith(prefix + ".")
                for prefix in prefixes
            ):
                matches.append(section.index)

        if len(matches) == 1:
            assigned[name] = matches[0]
        elif (
            not matches
            and len(module_ref.sections) == 1
            and name not in module_ref.sections[0].exclude
        ):
            assigned[name] = module_ref.sections[0].index
        elif len(matches) > 1:
            ambiguous[name] = matches
        else:
            unassigned.append(name)
    return assigned, ambiguous, unassigned


def _structural_errors(model: ReferenceModel) -> list[str]:
    errors: list[str] = []
    for module_ref in model.modules.values():
        seen: dict[str, str] = {}
        for section in module_ref.sections:
            for name in section.autosummary:
                previous = seen.get(name)
                if previous is not None:
                    errors.append(
                        "%s documents %r more than once (%s and %s)"
                        % (module_ref.name, name, previous, section.label)
                    )
                else:
                    seen[name] = section.label
            for cls in section.classes:
                if cls not in section.autosummary:
                    errors.append(
                        "%s section %s lists class %r outside its autosummary"
                        % (module_ref.name, section.label, cls)
                    )
    return errors


def inspect_reference(
    model: ReferenceModel,
    policy: Policy,
    *,
    modules: Optional[set[str]] = None,
    public_source: Optional[str] = None,
    require_distribution: Optional[bool] = None,
    skip_import_errors: bool = False,
) -> InspectionReport:
    """Compare documented names with the importable installed package."""

    report = InspectionReport(modules={}, structural_errors=_structural_errors(model))
    outside_root = sorted(
        name
        for name in model.modules
        if name != policy.package_root
        and not name.startswith(policy.package_root + ".")
    )
    if outside_root:
        report.structural_errors.append(
            "APIS_REFERENCE modules outside configured package_root %r: %s"
            % (policy.package_root, ", ".join(outside_root))
        )
    require_dist = (
        policy.require_distribution
        if require_distribution is None
        else require_distribution
    )
    if require_dist:
        try:
            importlib.metadata.distribution(policy.distribution)
        except importlib.metadata.PackageNotFoundError:
            report.environment_errors.append(
                "distribution %r is not installed; install the built project or pass --allow-uninstalled"
                % policy.distribution
            )
            return report

    requested = set(model.modules) if modules is None else set(modules)
    unknown = sorted(requested - set(model.modules))
    if unknown:
        report.structural_errors.append(
            "unknown APIS_REFERENCE module(s): %s" % ", ".join(unknown),
        )
        return report

    for module_name in sorted(requested):
        module_ref = model.modules[module_name]
        item = ModuleInspection(module=module_name)
        report.modules[module_name] = item
        if not module_ref.sync:
            item.warnings.append("generator.sync=false: module is verify-only")
        try:
            imported = importlib.import_module(module_name)
        # optional modules sometimes exit at import
        except (Exception, SystemExit) as exc:
            item.import_error = "%s: %s" % (type(exc).__name__, exc)
            if skip_import_errors:
                item.warnings.append("import error skipped by caller")
                item.import_error = None
            continue

        mode = public_source or module_ref.public_source or policy.public_source
        try:
            item.public_names, item.public_source = _public_names(
                imported,
                mode,
                include_module_objects=policy.include_module_objects,
            )
        except APIReferenceError as exc:
            item.import_error = str(exc)
            continue

        documented = module_ref.documented
        documented_set = set(documented)
        for name in documented:
            try:
                _resolve_name(imported, module_name, name)
            except ResolutionUnavailable as exc:
                item.unverified.append("%s: %s" % (name, exc))
            except AttributeError:
                item.stale.append(name)

        for section in module_ref.sections:
            for cls in section.classes:
                try:
                    obj = _resolve_name(imported, module_name, cls)
                    if not inspect.isclass(obj):
                        item.warnings.append(
                            "%s classes entry %r resolves to a non-class"
                            % (section.label, cls)
                        )
                except ResolutionUnavailable as exc:
                    item.unverified.append("%s [classes]: %s" % (cls, exc))
                except AttributeError:
                    item.stale_classes.append(cls)

        ignored = module_ref.ignore | set().union(
            *(section.exclude for section in module_ref.sections),
        )
        item.missing = sorted(
            name
            for name in item.public_names
            if name not in documented_set and name not in ignored
        )
        item.assigned, item.ambiguous, item.unassigned = _assign_missing(
            module_ref, imported, item.missing
        )

    return report


# ---------------------------------------------------------------------------
# Safe source rendering
# ---------------------------------------------------------------------------


def _line_comment_guard(
    lines: list[str],
    node: ast.Constant,
    list_node: ast.list,
) -> bool:
    """Return True when removing ``node`` may orphan a nearby explanatory comment."""

    if node.lineno != node.end_lineno:
        return True
    line = lines[node.lineno - 1]
    segment = ast.get_source_segment("".join(lines), node) or ""
    stripped = line.strip()
    if not segment or segment not in line:
        return True
    # Inline comments belong to the entry and can safely leave with it.
    # A directly preceding same-indent comment is treated as editorial context
    # and therefore blocks automatic pruning.
    target_indent = len(line) - len(line.lstrip())
    previous = node.lineno - 2
    while previous >= list_node.lineno - 1:
        candidate = lines[previous]
        if not candidate.strip():
            previous -= 1
            continue
        indent = len(candidate) - len(candidate.lstrip())
        if candidate.lstrip().startswith("#") and indent == target_indent:
            return True
        break
    return not (
        stripped.startswith(("'", '"'))
        and (stripped.endswith(",") or ",  #" in stripped or ", #" in stripped)
    )


def _list_element_lines(
    node: ast.list,
) -> dict[str, ast.Constant]:
    result: dict[str, ast.Constant] = {}
    for child in node.elts:
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            result[child.value] = child
    return result


def render_plan(
    model: ReferenceModel,
    report: InspectionReport,
    *,
    add_missing: bool = True,
    prune_stale: bool = True,
) -> RenderResult:
    """Render safe, deterministic symbol-list updates without writing files."""

    lines = model.source.splitlines(keepends=True)
    removals_by_line: dict[int, str] = {}
    additions_by_line: MutableMapping[int, list[tuple[int, str, str]]] = {}
    result = RenderResult(source=model.source)

    for module_name, inspection in report.modules.items():
        module_ref = model.modules[module_name]
        if inspection.import_error or not module_ref.sync:
            continue

        if prune_stale:
            stale = set(inspection.stale)
            stale_classes = set(inspection.stale_classes)
            for section in module_ref.sections:
                blocked_class_names: set[str] = set()
                if section.classes_node is not None:
                    class_nodes = _list_element_lines(section.classes_node)
                    for name in sorted(stale_classes & set(section.classes)):
                        node = class_nodes.get(name)
                        label = "%s :: %s :: %s [classes]" % (
                            module_name,
                            section.label,
                            name,
                        )
                        if node is None or _line_comment_guard(
                            lines,
                            node,
                            section.classes_node,
                        ):
                            result.blocked_removals.append(label)
                            blocked_class_names.add(name)
                            continue
                        removals_by_line[node.lineno] = label
                        result.removed.append(label)

                nodes = _list_element_lines(section.autosummary_node)
                for name in sorted(stale & set(section.autosummary)):
                    label = "%s :: %s :: %s" % (module_name, section.label, name)
                    # Keep the autosummary entry when the same stale class could
                    # not be safely removed from the inheritance list; otherwise
                    # generation would leave a self-inconsistent section.
                    if name in blocked_class_names:
                        result.blocked_removals.append(label + " [blocked by classes]")
                        continue
                    node = nodes.get(name)
                    if node is None or _line_comment_guard(
                        lines,
                        node,
                        section.autosummary_node,
                    ):
                        result.blocked_removals.append(label)
                        continue
                    removals_by_line[node.lineno] = label
                    result.removed.append(label)

        if add_missing:
            by_section: dict[int, list[str]] = {}
            for name, section_index in inspection.assigned.items():
                by_section.setdefault(section_index, []).append(name)
            for section_index, names in by_section.items():
                section = module_ref.sections[section_index]
                if not names:
                    continue
                existing_nodes = list(section.autosummary_node.elts)
                indent = None
                if existing_nodes:
                    indent = getattr(existing_nodes[-1], "col_offset", None)
                if indent is None:
                    indent = section.autosummary_node.col_offset + 4
                close_line = section.autosummary_node.end_lineno
                if close_line is None:
                    raise APIReferenceError("AST lacks end_lineno for %s" % module_name)
                for name in sorted(names):
                    label = "%s :: %s :: %s" % (module_name, section.label, name)
                    additions_by_line.setdefault(close_line, []).append(
                        (indent, name, label),
                    )
                    result.added.append(label)

    output: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if lineno in removals_by_line:
            continue
        if lineno in additions_by_line:
            for indent, name, _label in additions_by_line[lineno]:
                output.append(" " * indent + repr(name) + ",\n")
        output.append(line)
    result.source = "".join(output)
    return result


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Presentation / CLI
# ---------------------------------------------------------------------------


def _report_dict(
    report: InspectionReport,
) -> dict[str, object]:
    return {
        "structural_errors": report.structural_errors,
        "environment_errors": report.environment_errors,
        "summary": {
            "modules": len(report.modules),
            "stale": report.stale_count,
            "missing": report.missing_count,
            "import_errors": report.import_error_count,
        },
        "modules": {
            name: {
                "public_source": item.public_source,
                "public_names": item.public_names,
                "stale": item.stale,
                "stale_classes": item.stale_classes,
                "missing": item.missing,
                "assigned": item.assigned,
                "ambiguous": item.ambiguous,
                "unassigned": item.unassigned,
                "unverified": item.unverified,
                "import_error": item.import_error,
                "warnings": item.warnings,
            }
            for name, item in sorted(report.modules.items())
        },
    }


def _print_report(
    report: InspectionReport,
    stream: TextIO,
) -> None:
    for error in report.environment_errors:
        print("ENV ERROR: %s" % error, file=stream)
    for error in report.structural_errors:
        print("STRUCTURE ERROR: %s" % error, file=stream)
    for name, item in sorted(report.modules.items()):
        if item.import_error:
            print("IMPORT ERROR %s: %s" % (name, item.import_error), file=stream)
            continue
        if item.stale:
            print("STALE %s: %s" % (name, ", ".join(item.stale)), file=stream)
        if item.stale_classes:
            print(
                "STALE CLASS %s: %s" % (name, ", ".join(item.stale_classes)),
                file=stream,
            )
        if item.missing:
            print("MISSING %s: %s" % (name, ", ".join(item.missing)), file=stream)
        for missing, section_index in sorted(item.assigned.items()):
            print("  assign %s -> section[%d]" % (missing, section_index), file=stream)
        for missing, sections in sorted(item.ambiguous.items()):
            print("  ambiguous %s -> %s" % (missing, sections), file=stream)
        for missing in item.unassigned:
            print("  unassigned %s" % missing, file=stream)
        for unresolved in item.unverified:
            print("UNVERIFIED %s: %s" % (name, unresolved), file=stream)
        for warning in item.warnings:
            print("WARNING %s: %s" % (name, warning), file=stream)
    print(
        "summary: modules=%d stale=%d missing=%d import_errors=%d"
        % (
            len(report.modules),
            report.stale_count,
            report.missing_count,
            report.import_error_count,
        ),
        file=stream,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and safely generate Scikit-Plots API reference symbol lists",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root containing pyproject.toml",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="override the policy API-reference file",
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        default=[],
        help="limit inspection to one APIS_REFERENCE module; repeatable",
    )
    parser.add_argument(
        "--public-source",
        choices=PUBLIC_SOURCES,
        default=None,
        help="override public-name discovery mode",
    )
    parser.add_argument(
        "--allow-uninstalled",
        action="store_true",
        help="do not require installed distribution metadata before imports",
    )
    parser.add_argument(
        "--skip-import-errors",
        action="store_true",
        help="treat unimportable modules as unverified warnings instead of errors",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="verify documented names and optionally coverage",
    )
    check.add_argument(
        "--strict-coverage",
        action="store_true",
        help="treat undocumented public names as errors",
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

    plan = subparsers.add_parser(
        "plan",
        help="show stale, missing, and safe section assignments",
    )
    plan.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

    inventory = subparsers.add_parser(
        "inventory",
        help="show discovered installed public names",
    )
    inventory.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

    generate = subparsers.add_parser(
        "generate",
        help="render a safe candidate update; dry-run diff unless --apply or --output is used",
    )
    generate.add_argument(
        "--apply",
        action="store_true",
        help="atomically replace the canonical reference file",
    )
    generate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write candidate to a separate file",
    )
    generate.add_argument(
        "--no-add-missing",
        action="store_true",
        help="do not add unambiguously assigned public names",
    )
    generate.add_argument(
        "--no-prune-stale",
        action="store_true",
        help="do not remove safely prunable stale names",
    )
    generate.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable action summary",
    )

    rebuild = subparsers.add_parser(
        "rebuild",
        help="recreate the complete API-reference configuration from the canonical .py.in template",
    )
    rebuild.add_argument(
        "--apply",
        action="store_true",
        help="atomically create or replace the canonical reference file",
    )
    rebuild.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full rebuilt file to a separate path",
    )
    rebuild.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero when the canonical file differs from a clean full rebuild",
    )
    rebuild.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable rebuild status",
    )

    return parser


def _resolve_repo_root(value: Optional[Path]) -> Path:
    root = (value or DEFAULT_REPO_ROOT).expanduser().resolve()
    if not root.is_dir() or not (root / PYPROJECT_NAME).is_file():
        raise APIReferenceError(
            "repository root must contain pyproject.toml: %s" % root,
        )
    return root


def _reference_path(
    repo_root: Path,
    override: Optional[Path],
    policy: Policy,
    *,
    must_exist: bool = True,
) -> Path:
    configured = override if override is not None else Path(policy.reference_file)
    candidate = configured.expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise APIReferenceError(
            "reference file must stay inside the repository root: %s" % candidate,
        ) from exc
    if must_exist and not candidate.is_file():
        raise APIReferenceError(
            "reference file does not exist: %s" % candidate,
        )
    if not must_exist and candidate.exists() and not candidate.is_file():
        raise APIReferenceError(
            "reference path exists but is not a file: %s" % candidate,
        )
    return candidate


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """CLI entry point.  Returns a process status and performs no hidden writes."""

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser().parse_args(argv)
    try:
        repo_root = _resolve_repo_root(args.repo_root)
        policy = load_policy(repo_root)
        reference_path = _reference_path(
            repo_root,
            args.reference_file,
            policy,
            must_exist=args.command != "rebuild",
        )

        if args.command == "rebuild":
            if args.apply and args.output is not None:
                raise APIReferenceError(
                    "--apply and --output are mutually exclusive",
                )
            if args.check and (args.apply or args.output is not None):
                raise APIReferenceError(
                    "--check cannot be combined with --apply or --output",
                )
            template_blueprint = reference_blueprint_copy()
            candidate = build_reference_source()
            module_count = len(template_blueprint["api_reference"])
            deprecated_count = len(
                template_blueprint.get("deprecated_api_reference", {}),
            )
            current = (
                reference_path.read_text(encoding="utf-8")
                if reference_path.is_file()
                else ""
            )
            changed = current != candidate
            if args.check:
                if args.json:
                    print(
                        json.dumps(
                            {
                                "changed": changed,
                                "exists": reference_path.is_file(),
                                "reference_file": str(reference_path),
                                "modules": module_count,
                                "deprecated_versions": deprecated_count,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        file=out,
                    )
                elif changed:
                    print(
                        "API reference requires a full rebuild: %s" % reference_path,
                        file=out,
                    )
                else:
                    print(
                        "API reference full rebuild is clean: %s" % reference_path,
                        file=out,
                    )
                return 1 if changed else 0
            if args.apply:
                _atomic_write(reference_path, candidate)
            elif args.output is not None:
                target = args.output.expanduser().resolve()
                if target == reference_path:
                    raise APIReferenceError(
                        "use --apply to replace the canonical reference file",
                    )
                _atomic_write(target, candidate)
            elif not args.json:
                diff = difflib.unified_diff(
                    current.splitlines(keepends=True),
                    candidate.splitlines(keepends=True),
                    fromfile=str(reference_path) if current else "/dev/null",
                    tofile=str(reference_path) + " (full rebuild)",
                )
                out.writelines(diff)
            if args.json:
                print(
                    json.dumps(
                        {
                            "changed": changed,
                            "applied": bool(args.apply),
                            "output": str(args.output) if args.output else None,
                            "reference_file": str(reference_path),
                            "modules": module_count,
                            "deprecated_versions": deprecated_count,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    file=out,
                )
            elif args.apply or args.output is not None:
                destination = reference_path if args.apply else args.output
                print(
                    "rebuilt %s (modules=%d deprecated_versions=%d)"
                    % (destination, module_count, deprecated_count),
                    file=out,
                )
            return 0

        model = load_reference(reference_path)
        module_filter = set(args.modules) if args.modules else None
        report = inspect_reference(
            model,
            policy,
            modules=module_filter,
            public_source=args.public_source,
            require_distribution=False if args.allow_uninstalled else None,
            skip_import_errors=args.skip_import_errors,
        )

        if args.command == "check":
            if args.json:
                print(
                    json.dumps(_report_dict(report), indent=2, sort_keys=True),
                    file=out,
                )
            else:
                _print_report(report, out)
            return 1 if report.has_errors(strict_coverage=args.strict_coverage) else 0

        if args.command == "plan":
            if args.json:
                print(
                    json.dumps(_report_dict(report), indent=2, sort_keys=True),
                    file=out,
                )
            else:
                _print_report(report, out)
            return (
                1
                if (
                    report.environment_errors
                    or report.structural_errors
                    or report.import_error_count
                )
                else 0
            )

        if args.command == "inventory":
            payload = {
                name: {
                    "source": item.public_source,
                    "names": item.public_names,
                    "import_error": item.import_error,
                }
                for name, item in sorted(report.modules.items())
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True), file=out)
            else:
                for name, item in payload.items():
                    if item["import_error"]:
                        print(
                            "%s: IMPORT ERROR %s" % (name, item["import_error"]),
                            file=out,
                        )
                    else:
                        print("%s [%s]" % (name, item["source"]), file=out)
                        for symbol in item["names"]:
                            print("  %s" % symbol, file=out)
            return 1 if report.environment_errors or report.import_error_count else 0

        if args.command == "generate":
            if args.apply and args.output is not None:
                raise APIReferenceError("--apply and --output are mutually exclusive")
            if (
                report.environment_errors
                or report.structural_errors
                or report.import_error_count
            ):
                if args.json:
                    print(
                        json.dumps(
                            _report_dict(report),
                            indent=2,
                            sort_keys=True,
                        ),
                        file=out,
                    )
                else:
                    _print_report(report, err)
                return 1
            rendered = render_plan(
                model,
                report,
                add_missing=not args.no_add_missing,
                prune_stale=not args.no_prune_stale,
            )
            if args.apply:
                _atomic_write(model.path, rendered.source)
            elif args.output is not None:
                target = args.output.expanduser().resolve()
                if target == model.path:
                    raise APIReferenceError(
                        "use --apply to replace the canonical reference file",
                    )
                _atomic_write(target, rendered.source)
            elif not args.json:
                diff = difflib.unified_diff(
                    model.source.splitlines(keepends=True),
                    rendered.source.splitlines(keepends=True),
                    fromfile=str(model.path),
                    tofile=str(model.path) + " (candidate)",
                )
                out.writelines(diff)
            if args.json:
                print(
                    json.dumps(
                        {
                            "changed": rendered.changed,
                            "added": rendered.added,
                            "removed": rendered.removed,
                            "blocked_removals": rendered.blocked_removals,
                            "applied": bool(args.apply),
                            "output": str(args.output) if args.output else None,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    file=out,
                )
            elif args.apply or args.output is not None:
                destination = model.path if args.apply else args.output
                print(
                    "generated %s (added=%d removed=%d blocked_removals=%d)"
                    % (
                        destination,
                        len(rendered.added),
                        len(rendered.removed),
                        len(rendered.blocked_removals),
                    ),
                    file=out,
                )
            return 0

        raise APIReferenceError("unknown command: %s" % args.command)
    except APIReferenceError as exc:
        print("error: %s" % exc, file=err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
