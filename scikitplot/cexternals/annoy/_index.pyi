# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeVar

T = TypeVar("T", bound="Index")

class AnnoyIndex:
    f: int
    metric: str

    def __init__(self, f: int, metric: str) -> None: ...
    def add_item(self, i: int, vector: any) -> None: ...
    def build(self, n_trees: int) -> None: ...
    def get_nns_by_item(self, i: int, n: int, search_k: int = ...) -> list[int]: ...
    def get_nns_by_vector(self, vector: any, n: int, search_k: int = ...) -> list[int]: ...
    def serialize(self) -> bytes: ...
    def deserialize(self, buf: bytes) -> bool: ...

class Index(AnnoyIndex):
    # ----- class attributes -----
    _lock: any
    _compress: bool
    _compression_type: str

    # ----- init inherited -----
    def __init__(self, f: int, metric: str) -> None: ...

    # ----- properties -----
    @property
    def compress(self) -> bool: ...
    @compress.setter
    def compress(self, value: bool) -> None: ...

    @property
    def compression_type(self) -> str: ...
    @compression_type.setter
    def compression_type(self, value: str) -> None: ...

    # ----- pickling protocol -----
    def __reduce__(self) -> tuple[any, tuple[dict[str, any]]]: ...

    @classmethod
    def _rebuild(cls: type[T], state: dict[str, any]) -> T: ...

    # ----- convenience file helpers -----
    def save_to_file(self, path: str) -> None: ...

    @classmethod
    def load_from_file(cls: type[T], path: str) -> T: ...
