# scikitplot/annoy/_mixins/_io.pyi
"""
Typing stubs for :mod:`scikitplot.annoy._mixins._io`.

The runtime module defines two distinct persistence helpers:

* :class:`~.IndexIOMixin` wraps Annoy's native binary index-file I/O
  (low-level :meth:`save` / :meth:`load`).
* :class:`~.PickleIOMixin` provides explicit Python-object serialization
  helpers via ``pickle`` / ``cloudpickle`` / ``joblib``.

Notes
-----
Deserializing untrusted pickle/joblib bytes is unsafe for all supported
backends.
"""  # noqa: PYI021

# from __future__ import annotations

from os import PathLike
from typing import Any, ClassVar, Literal

from typing_extensions import Self, TypeAlias

SerializerBackend: TypeAlias = Literal["pickle", "cloudpickle", "joblib"]

# _T = TypeVar("_T", bound="PickleIOMixin")  # noqa: PYI018, PYI020

__all__ = ["IndexIOMixin", "PickleIOMixin", "SerializerBackend"]

class IndexIOMixin:
    """Mixin adding explicit Annoy-native index file persistence."""  # noqa: PYI021

    def _low_level(self) -> Any: ...
    def save_index(
        self, path: str | PathLike[str], *, prefault: bool | None = ...
    ) -> None: ...
    def load_index(
        self, path: str | PathLike[str], *, prefault: bool | None = ...
    ) -> None: ...
    def save_bundle(
        self,
        directory: str | PathLike[str],
        *,
        index_filename: str = ...,
        manifest_filename: str = ...,
        prefault: bool | None = ...,
    ) -> None: ...
    @classmethod
    def load_bundle(
        cls: type[Self],
        directory: str | PathLike[str],
        *,
        index_filename: str = ...,
        manifest_filename: str = ...,
        prefault: bool | None = ...,
    ) -> Self: ...

class PickleIOMixin:
    """Mixin adding explicit Python-object to pickling dump/load helpers."""  # noqa: PYI021

    # Declared for static type checkers; real value is the concrete subclass.
    _io_expected_type: ClassVar[type[Self]]

    def _low_level(self) -> Any: ...
    @classmethod
    def load_pickle(
        cls: type[Self],
        path: str | PathLike[str],
        *,
        backend: SerializerBackend = ...,
        **kwargs: Any,
    ) -> Self: ...
    def dump_pickle(
        self,
        path: str | PathLike[str],
        *,
        backend: SerializerBackend = ...,
        **kwargs: Any,
    ) -> None: ...
