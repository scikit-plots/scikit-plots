# _ndarray.pyi

from typing import Iterable, Iterator, Sequence

from typing_extensions import TypeAlias

try:
    import numpy as np
except Exception:  # pragma: no cover
    np: TypeAlias = None  # type: ignore[]  # noqa: PYI042

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd: TypeAlias = None  # type: ignore[]  # noqa: PYI042

IdsInput: TypeAlias = Sequence[int] | Iterable[int] | None

class NDArrayExportMixin:
    """
    Export utilities for Annoy-like objects.

    Requires:
    - get_item_vector(i: int) -> Sequence[float]
    - get_n_items() -> int
    - attribute/property f
    """  # noqa: PYI021

    # iteration
    def iter_item_vectors(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        with_ids: bool = True,
    ) -> Iterator[Sequence[float] | tuple[int, Sequence[float]]]: ...

    # in-memory
    def to_numpy(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        dtype: str = "float32",
    ) -> "np.ndarray": ...  # type: ignore[] # noqa: PYI020, UP037

    # on-disk
    def save_vectors_npy(
        self,
        path: str,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        dtype: str = "float32",
        overwrite: bool = True,
    ) -> str: ...

    # pandas (small/medium)
    def to_dataframe(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        columns: list[str] | None = None,
        dtype: str = "float32",
    ) -> "pd.DataFrame": ...  # type: ignore[] # noqa: PYI020, UP037

    # streaming CSV
    def to_csv(
        self,
        path: str,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        header: bool = True,
        delimiter: str = ",",
        float_format: str | None = None,
        columns: list[str] | None = None,
        dtype: str = "float32",
    ) -> str: ...

    # strict id helper (see implementation suggestion below)
    def partition_existing_ids(
        self,
        ids: Sequence[int],
        *,
        missing_exceptions: tuple[type[Exception], ...] = (IndexError,),  # noqa: PYI011
    ) -> tuple[list[int], list[int]]: ...
