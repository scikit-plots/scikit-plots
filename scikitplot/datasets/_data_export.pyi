# scikitplot/_data/_data_export.pyi
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import pandas as pd

SamplingStrategy: TypeAlias = Literal["hash", "random", "stratified"]
WithinGroupStrategy: TypeAlias = Literal["hash", "random", "linspace"]
StrataAllocation: TypeAlias = Literal["proportional", "equal"]
InputFormat: TypeAlias = Literal["auto", "csv", "parquet"]
OutputFormat: TypeAlias = Literal["csv", "parquet"]
RoundingMode: TypeAlias = Literal["floor", "round", "ceil"]

def stable_hash64(text: str) -> int: ...
def resolve_sizes(
    *,
    prepared_rows: int,
    sizes: Sequence[int] | None,
    fractions: Sequence[float] | None,
    percentages: Sequence[float] | None,
    rounding: RoundingMode,
) -> list[int]: ...
def profile_dataframe(df: pd.DataFrame, *, max_columns: int | None = ...) -> dict: ...
def load_dataframe(
    source: str,
    *,
    input_format: InputFormat = ...,
    usecols: Sequence[str] | None = ...,
    query: str | None = ...,
    low_memory: bool = ...,
    csv_sep: str = ...,
    csv_encoding: str | None = ...,
) -> pd.DataFrame: ...
def enforce_required_columns(
    df: pd.DataFrame,
    *,
    required_cols: Sequence[str] | None,
) -> pd.DataFrame: ...
def drop_duplicates_by_id(
    df: pd.DataFrame,
    *,
    id_col: str,
    keep: Literal["first", "last"] = ...,
) -> pd.DataFrame: ...
def apply_keep_drop_columns(
    df: pd.DataFrame,
    *,
    keep_cols: Sequence[str] | None,
    drop_cols: Sequence[str] | None,
) -> pd.DataFrame: ...
def sample_hash_full(df: pd.DataFrame, *, n: int, id_col: str) -> pd.DataFrame: ...
def sample_random(df: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame: ...
def sample_linspace(df: pd.DataFrame, *, n: int) -> pd.DataFrame: ...
def allocate_strata(
    group_sizes: pd.Series, *, n: int, allocation: StrataAllocation
) -> pd.Series: ...
def sample_stratified(
    df: pd.DataFrame,
    *,
    n: int,
    id_col: str | None,
    strata_cols: Sequence[str],
    within: WithinGroupStrategy,
    allocation: StrataAllocation,
    seed: int,
) -> pd.DataFrame: ...

class StreamStats:
    rows_seen_total: int
    rows_seen_eligible: int

def sample_hash_csv_stream(
    source: str,
    *,
    n: int,
    id_col: str,
    usecols: Sequence[str] | None = ...,
    required_cols: Sequence[str] | None = ...,
    query: str | None = ...,
    chunksize: int = ...,
    low_memory: bool = ...,
    csv_sep: str = ...,
    csv_encoding: str | None = ...,
) -> tuple[pd.DataFrame, StreamStats]: ...

class ExportSpec:
    size: int
    strategy: SamplingStrategy

def write_dataset(
    df: pd.DataFrame, *, output_path: Path, fmt: OutputFormat
) -> None: ...
def write_manifest(
    *,
    manifest_path: Path,
    source: str,
    input_format: InputFormat,
    output_file: Path,
    export_spec: ExportSpec,
    prep_params: dict,
    strategy_params: dict,
    rows: dict,
    columns: list[str],
) -> None: ...
def write_json(data: dict, *, output_path: Path) -> None: ...
def main(argv: Sequence[str] | None = ...) -> int: ...
