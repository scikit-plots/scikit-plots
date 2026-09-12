"""Family-wide dependency graph and capability ownership primitives.

This module is maintenance-only.  It intentionally understands the runtime tree from
source evidence without importing any Sphinx extension package.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

EDGE_KINDS = {
    "runtime_required",
    "runtime_optional",
    "test_only",
    "maintenance_only",
    "family_related",
}
RUNTIME_EDGE_KINDS = {"runtime_required", "runtime_optional"}
NON_RUNTIME_EDGE_KINDS = EDGE_KINDS - RUNTIME_EDGE_KINDS

_ALLOWED_EVIDENCE_BY_KIND = {
    "runtime_required": {"python_import", "source_reference"},
    "runtime_optional": {"python_import", "source_reference", "architectural_relation"},
    "test_only": {"test_import", "architectural_relation"},
    "maintenance_only": {"maintenance_reference", "architectural_relation"},
    "family_related": {"architectural_relation"},
}
EVIDENCE_KINDS = {
    "python_import",
    "source_reference",
    "architectural_relation",
    "test_import",
    "maintenance_reference",
}
_CAPABILITY_ID = re.compile(r"[a-z][a-z0-9_.-]{2,127}")


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    kind: str
    evidence: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityOwner:
    capability_id: str
    owner_package: str
    description: str
    steward: str
    stability: str


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_packages(runtime_family: Path) -> set[str]:
    return {p.name for p in runtime_family.iterdir() if p.is_dir() and not p.name.startswith(".")}


def runtime_imports(package_root: Path, runtime_family: Path) -> set[str]:
    """Return direct sibling imports found by AST, without importing runtime code."""
    deps: set[str] = set()
    siblings = runtime_packages(runtime_family)
    for path in package_root.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level >= 2:
                    if module:
                        first = module.split(".", 1)[0]
                        if first in siblings:
                            deps.add(first)
                    else:
                        # ``from .. import _sphinx_collection`` has no module field.
                        deps.update(alias.name for alias in node.names if alias.name in siblings)
                elif "_sphinx_ext." in module:
                    tail = module.split("_sphinx_ext.", 1)[1]
                    first = tail.split(".", 1)[0]
                    if first in siblings:
                        deps.add(first)
                elif module.endswith("_sphinx_ext"):
                    # ``from scikitplot._externals._sphinx_ext import _sphinx_collection``.
                    deps.update(alias.name for alias in node.names if alias.name in siblings)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "_sphinx_ext." in alias.name:
                        tail = alias.name.split("_sphinx_ext.", 1)[1]
                        first = tail.split(".", 1)[0]
                        if first in siblings:
                            deps.add(first)
    deps.discard(package_root.name)
    return deps


def runtime_source_references(package_root: Path, token: str) -> bool:
    """Return whether executable Python syntax contains a string reference to token.

    Comments are intentionally ignored so a prose note cannot satisfy a declared
    dynamic-runtime edge.  Imports are handled separately by :func:`runtime_imports`.
    """
    for path in package_root.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if token in node.value:
                    return True
    return False


def observed_runtime_graph(runtime_family: Path) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for package in sorted(runtime_packages(runtime_family)):
        graph[package] = runtime_imports(runtime_family / package, runtime_family)
    return graph


def normalize_dependency_edges(manifest: Mapping[str, object]) -> list[DependencyEdge]:
    """Normalize v2 typed edges and the v1 runtime_requires/family_related shape."""
    source = str(manifest.get("runtime_dir", ""))
    raw = manifest.get("dependency_edges")
    if isinstance(raw, list):
        edges: list[DependencyEdge] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            capabilities = item.get("capabilities", [])
            if not isinstance(capabilities, list):
                capabilities = []
            edges.append(
                DependencyEdge(
                    source=source,
                    target=str(item.get("target", "")),
                    kind=str(item.get("kind", "")),
                    evidence=str(item.get("evidence", "")),
                    capabilities=tuple(str(v) for v in capabilities),
                )
            )
        return edges

    # Backward-compatible v1 normalization.  The checker can therefore inspect older
    # standalone maintenance bundles without silently interpreting them as v2.
    edges = [
        DependencyEdge(source, str(target), "runtime_required", "source_reference", ())
        for target in manifest.get("runtime_requires", []) or []
    ]
    edges.extend(
        DependencyEdge(source, str(target), "family_related", "architectural_relation", ())
        for target in manifest.get("family_related", []) or []
    )
    return edges


def capability_owners(manifest: Mapping[str, object]) -> list[CapabilityOwner]:
    steward = str(manifest.get("runtime_dir", ""))
    raw = manifest.get("capability_ownership", [])
    if not isinstance(raw, list):
        return []
    result: list[CapabilityOwner] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        result.append(
            CapabilityOwner(
                capability_id=str(item.get("id", "")),
                owner_package=str(item.get("owner", "")),
                description=str(item.get("description", "")),
                steward=steward,
                stability=str(item.get("stability", "internal")),
            )
        )
    return result


def find_cycle(graph: Mapping[str, Iterable[str]]) -> list[str] | None:
    visiting: set[str] = set()
    done: set[str] = set()
    trail: list[str] = []

    def visit(node: str):
        if node in visiting:
            i = trail.index(node)
            return trail[i:] + [node]
        if node in done:
            return None
        visiting.add(node)
        trail.append(node)
        for nxt in sorted(graph.get(node, ())):
            found = visit(nxt)
            if found:
                return found
        trail.pop()
        visiting.remove(node)
        done.add(node)
        return None

    for node in sorted(graph):
        found = visit(node)
        if found:
            return found
    return None


def effective_runtime_graph(
    runtime_family: Path, manifests: Sequence[Mapping[str, object]]
) -> dict[str, set[str]]:
    """Combine AST-observed imports with declared dynamic/optional runtime edges."""
    graph = observed_runtime_graph(runtime_family)
    for manifest in manifests:
        for edge in normalize_dependency_edges(manifest):
            if edge.kind in RUNTIME_EDGE_KINDS and edge.source:
                graph.setdefault(edge.source, set()).add(edge.target)
                graph.setdefault(edge.target, set())
    return graph


def validate_manifest_architecture(
    manifest: Mapping[str, object], runtime_family: Path
) -> list[str]:
    """Validate typed edge shape and source evidence for one subsystem."""
    errors: list[str] = []
    source = str(manifest.get("runtime_dir", ""))
    source_dir = runtime_family / source
    packages = runtime_packages(runtime_family)
    edges = normalize_dependency_edges(manifest)
    schema_version = manifest.get("schema_version", 1)

    if schema_version >= 2:
        if not isinstance(manifest.get("dependency_edges"), list):
            errors.append("schema v2 MAINTENANCE.json requires dependency_edges")
        if not isinstance(manifest.get("capability_ownership"), list):
            errors.append("schema v2 MAINTENANCE.json requires capability_ownership")
        legacy = [key for key in ("runtime_requires", "family_related") if key in manifest]
        if legacy:
            errors.append(
                "schema v2 MAINTENANCE.json must not retain legacy dependency arrays: "
                + ", ".join(legacy)
            )

    seen_targets: set[str] = set()
    runtime_declared: set[str] = set()
    non_runtime_declared: set[str] = set()
    owners = capability_owners(manifest)
    local_capability_map = {c.capability_id: c for c in owners if c.capability_id}

    for edge in edges:
        if not edge.target:
            errors.append("dependency edge missing target")
            continue
        if edge.target == source:
            errors.append(f"dependency edge cannot self-reference runtime package: {source}")
        if edge.target in seen_targets:
            errors.append(f"dependency target classified more than once: {edge.target}")
        seen_targets.add(edge.target)
        if edge.kind not in EDGE_KINDS:
            errors.append(f"dependency edge {edge.target} has unknown kind: {edge.kind}")
            continue
        if edge.evidence not in EVIDENCE_KINDS:
            errors.append(f"dependency edge {edge.target} has unknown evidence: {edge.evidence}")
        elif edge.kind in _ALLOWED_EVIDENCE_BY_KIND and edge.evidence not in _ALLOWED_EVIDENCE_BY_KIND[edge.kind]:
            errors.append(
                f"dependency edge {edge.target} uses evidence {edge.evidence} "
                f"incompatible with kind {edge.kind}"
            )
        if schema_version >= 2 and not edge.capabilities:
            errors.append(f"dependency edge {edge.target} must cite at least one capability")
        if len(edge.capabilities) != len(set(edge.capabilities)):
            errors.append(f"dependency edge {edge.target} repeats a capability ID")
        if edge.kind in {"runtime_required", "runtime_optional", "family_related"} and edge.target not in packages:
            if edge.kind == "runtime_required":
                errors.append(f"declared runtime dependency missing: {edge.target}")
            else:
                errors.append(f"dependency target runtime package missing: {edge.target}")
        if edge.kind in RUNTIME_EDGE_KINDS:
            runtime_declared.add(edge.target)
        else:
            non_runtime_declared.add(edge.target)

        # If a capability is declared locally, its owner must match the target that
        # justifies this edge.  Cross-manifest resolution is checked family-wide.
        for capability_id in edge.capabilities:
            owner = local_capability_map.get(capability_id)
            if owner is not None and owner.owner_package != edge.target:
                errors.append(
                    f"capability {capability_id} is owned by {owner.owner_package}, "
                    f"not dependency target {edge.target}"
                )

    owner_ids = [c.capability_id for c in owners]
    if len(owner_ids) != len(set(owner_ids)):
        errors.append("MAINTENANCE.json has duplicate capability ownership IDs")
    allowed_owners = {source, *seen_targets}
    for capability in owners:
        if not _CAPABILITY_ID.fullmatch(capability.capability_id):
            errors.append(f"invalid capability ID: {capability.capability_id or '<empty>'}")
        if capability.owner_package not in packages:
            errors.append(
                f"capability {capability.capability_id} owner runtime package missing: "
                f"{capability.owner_package}"
            )
        if capability.owner_package not in allowed_owners:
            errors.append(
                f"capability {capability.capability_id} owner {capability.owner_package} "
                f"is outside subsystem ownership/dependency boundary"
            )
        if not capability.description.strip():
            errors.append(f"capability {capability.capability_id} missing description")

    if not source_dir.is_dir():
        return errors

    actual = runtime_imports(source_dir, runtime_family)
    for dep in sorted(actual - runtime_declared):
        if dep.startswith(("_sphinx", "_pydata")):
            errors.append(f"undeclared cross-stack runtime import: {dep}")
    for dep in sorted(actual & non_runtime_declared):
        errors.append(
            f"non-runtime dependency classification is imported by runtime: {dep}"
        )

    for edge in edges:
        if edge.kind == "runtime_required":
            if edge.evidence == "python_import" and edge.target not in actual:
                errors.append(
                    f"runtime_required edge lacks declared python import evidence: {edge.target}"
                )
            elif (
                edge.evidence == "source_reference"
                and edge.target not in actual
                and not runtime_source_references(source_dir, edge.target)
            ):
                errors.append(
                    f"runtime_required edge lacks declared source reference evidence: {edge.target}"
                )

    # Local reachable-cycle check preserves the useful subsystem gate while the family
    # gate separately checks the entire effective graph.  Declared dynamic edges are
    # included, so a source-reference extension edge can participate in a cycle.
    graph = effective_runtime_graph(runtime_family, [manifest])
    reachable: set[str] = set()
    stack = [source]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(graph.get(node, ()))
    local_graph = {node: {d for d in graph.get(node, ()) if d in reachable} for node in reachable}
    cycle = find_cycle(local_graph)
    if cycle:
        errors.append("runtime dependency cycle: " + " -> ".join(cycle))

    return errors


def validate_family_architecture(
    manifest_paths: Sequence[Path], runtime_family: Path
) -> list[str]:
    """Validate ownership resolution and the effective runtime DAG family-wide."""
    errors: list[str] = []
    manifests: list[dict] = []
    for path in manifest_paths:
        try:
            manifests.append(load_manifest(path))
        except Exception:
            continue  # subsystem checker provides the precise JSON error

    registry: dict[str, CapabilityOwner] = {}
    for manifest in manifests:
        for capability in capability_owners(manifest):
            previous = registry.get(capability.capability_id)
            if previous is not None:
                errors.append(
                    f"capability {capability.capability_id} has multiple ownership declarations: "
                    f"{previous.owner_package} ({previous.steward}) and "
                    f"{capability.owner_package} ({capability.steward})"
                )
            else:
                registry[capability.capability_id] = capability

    for manifest in manifests:
        for edge in normalize_dependency_edges(manifest):
            for capability_id in edge.capabilities:
                owner = registry.get(capability_id)
                if owner is None:
                    errors.append(
                        f"dependency {edge.source} -> {edge.target} references unknown capability: "
                        f"{capability_id}"
                    )
                elif owner.owner_package != edge.target:
                    errors.append(
                        f"dependency {edge.source} -> {edge.target} cites capability "
                        f"{capability_id} owned by {owner.owner_package}"
                    )

    graph = effective_runtime_graph(runtime_family, manifests)
    cycle = find_cycle(graph)
    if cycle:
        errors.append("effective runtime dependency cycle: " + " -> ".join(cycle))
    return errors


def architecture_snapshot(
    manifest_paths: Sequence[Path], runtime_family: Path
) -> dict[str, object]:
    """Return deterministic architecture data suitable for diagnostics or CI artifacts."""
    manifests = [load_manifest(p) for p in manifest_paths]
    observed = observed_runtime_graph(runtime_family)
    effective = effective_runtime_graph(runtime_family, manifests)
    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "kind": edge.kind,
            "evidence": edge.evidence,
            "capabilities": list(edge.capabilities),
        }
        for manifest in manifests
        for edge in normalize_dependency_edges(manifest)
    ]
    capabilities = [
        {
            "id": c.capability_id,
            "owner": c.owner_package,
            "steward": c.steward,
            "description": c.description,
            "stability": c.stability,
        }
        for manifest in manifests
        for c in capability_owners(manifest)
    ]
    return {
        "schema_version": 1,
        "packages": sorted(runtime_packages(runtime_family)),
        "observed_runtime_imports": {k: sorted(v) for k, v in sorted(observed.items())},
        "declared_edges": sorted(edges, key=lambda x: (x["source"], x["target"], x["kind"])),
        "capabilities": sorted(capabilities, key=lambda x: x["id"]),
        "effective_runtime_graph": {k: sorted(v) for k, v in sorted(effective.items())},
        "cycle": find_cycle(effective),
    }
