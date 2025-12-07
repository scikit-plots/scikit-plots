from __future__ import annotations

import json
from typing import Any


class ManifestMixin:
    """
    Export/import *metadata* only.

    Safe for huge indexes.
    """

    def to_manifest(self) -> dict[str, Any]:
        return {
            "f": int(self.f),
            "metric": str(self.metric),
            "on_disk_path": getattr(self, "_on_disk_path", None),
            "pickle_mode": getattr(self, "_pickle_mode", "auto"),
            "compress_mode": getattr(self, "_compress_mode", None),
            "prefault": getattr(self, "_prefault", False),
        }

    def to_json(self, path: str | None = None) -> str:
        payload = json.dumps(self.to_manifest(), ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]):
        f = int(manifest["f"])
        metric = str(manifest["metric"])
        obj = cls(f, metric)

        # Restore config if attributes exist
        if "prefault" in manifest and hasattr(obj, "prefault"):
            obj.prefault = bool(manifest["prefault"])
        if "pickle_mode" in manifest and hasattr(obj, "pickle_mode"):
            obj.pickle_mode = str(manifest["pickle_mode"])
        if "compress_mode" in manifest and hasattr(obj, "compress_mode"):
            cm = manifest["compress_mode"]
            obj.compress_mode = None if cm is None else str(cm)

        path = manifest.get("on_disk_path")
        if path:
            obj.load(str(path), prefault=bool(getattr(obj, "_prefault", False)))

        return obj

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return cls.from_manifest(manifest)

    def to_yaml(self, path: str) -> None:
        import yaml  # noqa: PLC0415

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_manifest(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str):
        import yaml  # noqa: PLC0415

        with open(path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        return cls.from_manifest(manifest)
