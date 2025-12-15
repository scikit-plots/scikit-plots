# _pickle.pyi

from typing import Any, Literal

from typing_extensions import Self, TypeAlias

from .. import annoylib

PickleMode: TypeAlias = Literal["auto", "byte", "disk"]
CompressMode: TypeAlias = Literal["zlib", "gzip"] | None

class PathAwareAnnoy(annoylib.Annoy):
    """
    Thin Python subclass that tracks last known on-disk path.
    """  # noqa: PYI021

    _on_disk_path: str | None

    def on_disk_build(self, path: str) -> Self: ...
    def load(self, path: str, prefault: bool = False) -> Self: ...
    def save(self, path: str, prefault: bool = False) -> Self: ...

class PickleMixin(PathAwareAnnoy):
    """
    Strict persistence support for pickle/joblib/cloudpickle.
    """  # noqa: PYI021

    _prefault: bool
    _pickle_mode: PickleMode
    _compress_mode: CompressMode

    def __init__(self, f: int = 0, metric: str = "angular"): ...
    @property
    def prefault(self) -> bool: ...
    @prefault.setter
    def prefault(self, value: object) -> None: ...
    @property
    def on_disk_path(self) -> str | None: ...
    @on_disk_path.setter
    def on_disk_path(self, value: str | None) -> None: ...
    @property
    def pickle_mode(self) -> PickleMode: ...
    @pickle_mode.setter
    def pickle_mode(self, value: PickleMode) -> None: ...
    @property
    def compress_mode(self) -> CompressMode: ...
    @compress_mode.setter
    def compress_mode(self, value: CompressMode) -> None: ...
    def __reduce__(self) -> object: ...
    @classmethod
    def _rebuild(cls: type[Self], state: dict[str, Any]) -> Self: ...
