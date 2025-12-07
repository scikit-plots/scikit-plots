# _io.pyi

from typing import Literal

from typing_extensions import Self, TypeAlias

SerializerBackend: TypeAlias = Literal["pickle", "joblib", "cloudpickle"]

class ObjectIOMixin:
    """
    Persistence for the Python object (Index/Annoy alias), not the raw Annoy index file.
    """  # noqa: PYI021

    def dump_binary(
        self, path: str, *, backend: SerializerBackend = "pickle"
    ) -> None: ...
    def save_to_file(self, path: str) -> None: ...
    @classmethod
    def load_binary(
        cls: type[Self], path: str, *, backend: SerializerBackend = "pickle"
    ) -> Self: ...
    @classmethod
    def load_from_file(cls: type[Self], path: str) -> Self: ...
