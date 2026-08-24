# SPDX-License-Identifier: BSD-3-Clause
"""
Stable internal consumer facade for generated ``_sphinx_llm`` artifacts.

Consumers must use this module or the static manifest contract.  This facade
intentionally imports no implementation from the preserved NVIDIA package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.artifacts import normalize_relative_path

MANIFEST_RELATIVE_PATH = "_llms/manifest.json"


class ArtifactContractError(ValueError):
    """Raised when a published artifact violates the consumer contract."""


@dataclass(frozen=True)
class DocumentArtifact:
    docname: str
    title: str
    description: str | None
    html_path: str | None
    markdown_paths: tuple[str, ...]
    source_kind: str
    fidelity: str
    included_in_llms: bool
    included_in_full: bool
    warnings: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> DocumentArtifact:
        try:
            markdown_paths = tuple(
                normalize_relative_path(str(path)) for path in payload["markdown_paths"]
            )
            html_path = payload.get("html_path")
            if html_path is not None:
                html_path = normalize_relative_path(str(html_path))
            return cls(
                docname=str(payload["docname"]),
                title=str(payload["title"]),
                description=payload.get("description"),
                html_path=html_path,
                markdown_paths=markdown_paths,
                source_kind=str(payload["source_kind"]),
                fidelity=str(payload["fidelity"]),
                included_in_llms=bool(payload["included_in_llms"]),
                included_in_full=bool(payload["included_in_full"]),
                warnings=tuple(str(item) for item in payload.get("warnings", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactContractError(
                f"invalid document manifest record: {exc}"
            ) from exc


@dataclass(frozen=True)
class ArtifactCatalog:
    root: Path
    manifest: dict[str, Any]
    documents: tuple[DocumentArtifact, ...]

    def document(self, docname: str) -> DocumentArtifact | None:
        return next((item for item in self.documents if item.docname == docname), None)

    def for_html_path(self, html_path: str) -> DocumentArtifact | None:
        normalized = normalize_relative_path(html_path)
        candidates = [item for item in self.documents if item.html_path == normalized]
        return _best(candidates)

    def markdown_path(self, docname: str) -> Path | None:
        item = self.document(docname)
        if item is None or not item.markdown_paths:
            return None
        return self.resolve(item.markdown_paths[0])

    def resolve(self, relative_path: str) -> Path:
        rel = normalize_relative_path(relative_path)
        root = self.root.resolve()
        path = (self.root / rel).resolve()
        if root not in path.parents and path != root:
            raise ArtifactContractError(
                f"artifact escapes catalog root: {relative_path!r}"
            )
        return path

    def capabilities(self) -> dict[str, Any]:
        fidelities = sorted({item.fidelity for item in self.documents})
        return {
            "schema_version": self.manifest.get("schema_version"),
            "build_id": self.manifest.get("build_id"),
            "documents": len(self.documents),
            "fidelities": fidelities,
            "has_canonical": "canonical" in fidelities,
            "has_compatibility": "compatibility" in fidelities,
        }


def _best(items: list[DocumentArtifact]) -> DocumentArtifact | None:
    rank = {"canonical": 0, "compatibility": 1, "runtime-fallback": 2}
    return min(
        items,
        key=lambda item: (rank.get(item.fidelity, 99), item.docname),
        default=None,
    )


def load_catalog(
    root: str | Path, *, manifest_path: str = MANIFEST_RELATIVE_PATH
) -> ArtifactCatalog:
    """Load and minimally validate the stable static manifest contract."""

    root_path = Path(root)
    rel = normalize_relative_path(manifest_path)
    manifest_file = (root_path / rel).resolve()
    resolved_root = root_path.resolve()
    if resolved_root not in manifest_file.parents:
        raise ArtifactContractError("manifest path escapes artifact root")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactContractError(f"cannot read artifact manifest: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ArtifactContractError("manifest must contain a documents array")
    documents = tuple(
        DocumentArtifact.from_mapping(item) for item in payload["documents"]
    )
    return ArtifactCatalog(root=resolved_root, manifest=payload, documents=documents)


__all__ = [
    "MANIFEST_RELATIVE_PATH",
    "ArtifactCatalog",
    "ArtifactContractError",
    "DocumentArtifact",
    "load_catalog",
]
