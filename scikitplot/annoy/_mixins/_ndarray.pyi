# scikitplot/annoy/_mixins/_ndarray.pyi

"""Typing stubs for NumPy / SciPy / pandas interoperability."""  # noqa: PYI021

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

__all__: tuple[str, ...] = ("NDArrayMixin",)

IdsInput: TypeAlias = Sequence[int] | Iterable[int] | None
SparsePolicy: TypeAlias = Literal["error", "toarray"]
FinitePolicy: TypeAlias = bool | Literal["allow-nan"]

class NDArrayMixin:
    def add_items(
        self,
        X: Any,
        ids: Sequence[int] | Iterable[int] | None = ...,
        *,
        start_id: int | None = ...,
        accept_sparse: SparsePolicy = ...,
        ensure_all_finite: FinitePolicy = ...,
        copy: bool = ...,
        dtype: Any = ...,
        order: Literal["C", "F", "A", "K"] = ...,
        check_unique_ids: bool = ...,
    ) -> NDArray[np.int64]: ...
    @overload
    def get_item_vectors(
        self,
        ids: IdsInput = ...,
        *,
        dtype: Any = ...,
        start: int = ...,
        stop: int | None = ...,
        n_rows: int | None = ...,
        return_ids: Literal[False] = ...,
        validate_vector_len: bool = ...,
    ) -> NDArray[Any]: ...
    @overload
    def get_item_vectors(
        self,
        ids: IdsInput,
        *,
        dtype: Any = ...,
        start: int = ...,
        stop: int | None = ...,
        n_rows: int | None = ...,
        return_ids: Literal[True] = ...,
        validate_vector_len: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[np.int64]]: ...
    def get_item_vectors(
        self,
        ids: IdsInput = ...,
        *,
        dtype: Any = ...,
        start: int = ...,
        stop: int | None = ...,
        n_rows: int | None = ...,
        return_ids: bool = ...,
        validate_vector_len: bool = ...,
    ) -> NDArray[Any] | tuple[NDArray[Any], NDArray[np.int64]]: ...
    def to_numpy(
        self,
        ids: IdsInput = ...,
        *,
        dtype: Any = ...,
        start: int = ...,
        stop: int | None = ...,
        n_rows: int | None = ...,
        validate_vector_len: bool = ...,
    ) -> NDArray[Any]: ...
    def iter_item_vectors(
        self,
        ids: IdsInput = ...,
        *,
        start: int = ...,
        stop: int | None = ...,
        with_ids: bool = ...,
        dtype: Any | None = ...,
    ) -> Iterator[NDArray[Any] | tuple[int, NDArray[Any]]]: ...
    def to_scipy_csr(
        self,
        ids: IdsInput = ...,
        *,
        dtype: Any = ...,
        start: int = ...,
        stop: int | None = ...,
        n_rows: int | None = ...,
        validate_vector_len: bool = ...,
    ) -> Any: ...
    def to_pandas(
        self,
        ids: IdsInput = ...,
        *,
        dtype: Any = ...,
        start: int = ...,
        stop: int | None = ...,
        n_rows: int | None = ...,
        id_location: Literal["index", "column", "both", "none"] = ...,
        id_name: str = ...,
        columns: Sequence[str] | None = ...,
        validate_vector_len: bool = ...,
    ) -> Any: ...
