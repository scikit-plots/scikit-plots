# _manifest.pyi

from typing import Any

from typing_extensions import Self

class ManifestMixin:
    """
    Export/import metadata only.

    Safe for huge indexes.
    """  # noqa: PYI021

    def to_manifest(self) -> dict[str, Any]: ...
    def to_json(self, path: str | None = None) -> str: ...
    def to_yaml(self, path: str) -> None: ...
    @classmethod
    def from_manifest(cls: type[Self], manifest: dict[str, Any]) -> Self: ...
    @classmethod
    def from_json(cls: type[Self], path: str) -> Self: ...
    @classmethod
    def from_yaml(cls: type[Self], path: str) -> Self: ...
