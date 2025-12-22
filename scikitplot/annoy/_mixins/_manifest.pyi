# scikitplot/annoy/_mixins/_manifest.pyi
# from __future__ import annotations

import os
from typing import Any, Mapping, TypedDict

from typing_extensions import Self

__all__: tuple[str, ...] = ("Manifest", "ManifestMixin")

class Manifest(TypedDict, total=False):
    """
    Typed representation of a manifest payload.

    This is a permissive schema used for forward-compatible metadata export.
    """  # noqa: PYI021

    # Manifest format
    manifest_schema_version: int

    # Core Annoy configuration
    f: int | None
    metric: str | None
    on_disk_path: str | None
    prefault: bool | None
    schema_version: int | None
    seed: int | None
    verbose: int | None

    # Optional keys (from Annoy.info include_* flags)
    n_items: int
    n_trees: int
    memory_usage_byte: int
    memory_usage_mib: float

    # Optional high-level persistence knobs (if exposed by the concrete class)
    pickle_mode: str | None
    compress_mode: str | None

class ManifestMixin:
    """Export/import a small manifest describing an index."""  # noqa: PYI021

    _MANIFEST_FORMAT_VERSION: int

    def _low_level(self) -> Any: ...
    def to_manifest(self) -> Manifest: ...
    def to_json(self, path: str | os.PathLike[str] | None = None) -> str: ...
    def to_yaml(self, path: str | os.PathLike[str]) -> None: ...
    @classmethod
    def from_manifest(
        cls: type[Self], manifest: Mapping[str, Any], *, load: bool = True
    ) -> Self: ...
    @classmethod
    def from_json(
        cls: type[Self], path: str | os.PathLike[str], *, load: bool = True
    ) -> Self: ...
    @classmethod
    def from_yaml(
        cls: type[Self], path: str | os.PathLike[str], *, load: bool = True
    ) -> Self: ...
