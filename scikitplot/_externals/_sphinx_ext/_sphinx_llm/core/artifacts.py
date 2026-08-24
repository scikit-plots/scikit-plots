# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic artifact hashing, manifest construction, and JSON writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_relative_path(value: str) -> str:
    """Normalize a manifest path and reject absolute/traversal forms."""

    raw = str(value).replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"artifact path must be safe and relative: {value!r}")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise ValueError(f"artifact path must name a file: {value!r}")
    return normalized


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def stable_build_id(
    identity: dict[str, Any], documents: Iterable[dict[str, Any]]
) -> str:
    """Return a reproducible build identity from non-secret inputs and outputs."""

    canonical = {
        "identity": identity,
        "documents": sorted(
            (
                {
                    "docname": item.get("docname"),
                    "source_hash": item.get("source_hash"),
                    "output_hashes": item.get("output_hashes", {}),
                    "fidelity": item.get("fidelity"),
                }
                for item in documents
            ),
            key=lambda item: str(item.get("docname", "")),
        ),
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"sha256:{sha256_text(encoded)}"


def output_hashes(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash only existing, safe files beneath *root*."""

    hashes: dict[str, str] = {}
    resolved_root = root.resolve()
    for raw in relative_paths:
        rel = normalize_relative_path(raw)
        path = (root / rel).resolve()
        if path != resolved_root and resolved_root not in path.parents:
            raise ValueError(f"artifact escaped output root: {raw!r}")
        if path.is_file():
            hashes[rel] = sha256_file(path)
    return dict(sorted(hashes.items()))


def make_manifest(
    *,
    project: str,
    docs_version: str,
    language: str,
    builder: str,
    generator: dict[str, Any],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the canonical machine inventory with deterministic identity."""

    documents = sorted(documents, key=lambda item: str(item.get("docname", "")))
    identity = {
        "project": project,
        "docs_version": docs_version,
        "language": language,
        "builder": builder,
        "generator": generator,
    }
    build_id = stable_build_id(identity, documents)
    return {
        "schema_version": 1,
        "project": project,
        "docs_version": docs_version,
        "language": language,
        "builder": builder,
        "build_id": build_id,
        # Deliberately null: deterministic artifacts must not change solely due
        # to wall-clock time. CI/deployment metadata can timestamp externally.
        "generated_at": None,
        "generator": generator,
        "documents": documents,
    }


def make_provenance(build_id: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "build_id": build_id,
        "artifacts": sorted(artifacts, key=lambda item: str(item.get("path", ""))),
    }


__all__ = [
    "atomic_write_json",
    "make_manifest",
    "make_provenance",
    "normalize_relative_path",
    "output_hashes",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "stable_build_id",
]
