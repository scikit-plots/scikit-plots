# scikitplot/annoy/_mixins/_meta.pyi
#
# flake8: noqa: E301
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Typing stubs for :mod:`~scikitplot.annoy._mixins._meta`."""  # noqa: PYI021

import os
from collections.abc import Mapping
from typing import Any, TypedDict

from typing_extensions import Self

__all__: list[str]  # noqa: PYI035

class IndexMetadata(TypedDict, total=True):
    index_schema_version: int
    params: dict[str, Any]
    info: dict[str, Any] | None
    persistence: dict[str, Any] | None

class MetaMixin:
    _META_SCHEMA_VERSION: int

    def to_metadata(
        self,
        *,
        include_info: bool = ...,
        strict: bool = ...,
    ) -> IndexMetadata: ...
    def to_json(
        self,
        path: str | os.PathLike[str] | None = ...,
        *,
        indent: int = ...,
        sort_keys: bool = ...,
        ensure_ascii: bool = ...,
        include_info: bool = ...,
        strict: bool = ...,
    ) -> str: ...
    @classmethod
    def from_metadata(
        cls: type[Self],
        metadata: Mapping[str, Any],
        *,
        load: bool = ...,
    ) -> Self: ...
    @classmethod
    def from_json(
        cls: type[Self],
        path: str | os.PathLike[str],
        *,
        load: bool = ...,
    ) -> Self: ...
    def to_yaml(
        self,
        path: str | os.PathLike[str] | None = ...,
        *,
        include_info: bool = ...,
        strict: bool = ...,
    ) -> str: ...
    @classmethod
    def from_yaml(
        cls: type[Self],
        path: str | os.PathLike[str],
        *,
        load: bool = ...,
    ) -> Self: ...

class MetadataRoutingMixin:
    def get_metadata_routing(self) -> Any: ...
    @staticmethod
    def build_metadata_router(*, owner: object) -> Any: ...
