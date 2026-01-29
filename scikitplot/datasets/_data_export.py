# scikitplot/_data/_data_export.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
data_export.py.

Generic, reproducible dataset subset exporter for ML workflows.

Key features:

- Source-agnostic: reads CSV or Parquet from local path (and CSV/Parquet URLs if supported by pandas).
- Reproducible sampling:
  - hash: deterministic top-N by stable hash of an id column (no RNG needed).
  - random: reproducible via explicit seed.
  - stratified: strict allocation across strata groups (proportional or equal), then deterministic within group.
- Safer filtering:
  - required_cols controls which columns must be non-null (avoid destructive dropna over all columns).
  - optional query for row filtering.
- Traceability:
  - writes manifest JSON describing exact export parameters and row counts.
- Friendly sizing:
  - absolute sizes via --sizes (rows)
  - fractions via --fractions (0..1)
  - percentages via --percentages (0..100)

Design constraints:

- hash sampling REQUIRES an explicit id_col; no guessing.
- stratified sampling REQUIRES explicit strata_cols; no guessing.
- within='hash' REQUIRES explicit id_col; no guessing.
- requested export sizes MUST be <= prepared dataset row count; otherwise error.

Examples
--------
>>> # from scikitplot import datasets
>>> !python -m scikitplot.datasets._data_export \
>>>   --input autoscout24_dataset_20251108.csv \
>>>   --output-dir data/subsets \
>>>   --percentages 0.1 \
>>>   --rounding round \
>>>   --strategy hash \
>>>   --id-col id \
>>>   --dedup \
>>>   --required-cols id price make model registration_date mileage_km_raw vehicle_type fuel_category is_used country_code \
>>>   --keep-cols id price make model model_version registration_date mileage_km_raw vehicle_type body_type fuel_category primary_fuel \
>>>       transmission power_kw power_hp nr_seats nr_doors country_code zip city latitude longitude is_used seller_is_dealer offer_type \
>>>       description equipment_comfort equipment_entertainment  equipment_extra equipment_safety\
>>>   --format csv \
>>>   --write-manifest \
>>>   --write-profile

1) Deterministic subsets (hash) from CSV:

>>> python data_export.py \
>>>     --input dataset.csv \
>>>     --output-dir data/subsets \
>>>     --sizes 1000 10000 40000 \
>>>     --strategy hash \
>>>     --id-col id \
>>>     --required-cols id \
>>>     --format parquet \
>>>     --write-manifest

2) Deterministic 10% subset (computed from prepared rows):

>>> python data_export.py \
>>>     --input dataset.csv \
>>>     --output-dir data/subsets \
>>>     --percentages 10 \
>>>     --strategy hash \
>>>     --id-col id \
>>>     --required-cols id \
>>>     --format parquet \
>>>     --write-manifest

3) Reproducible random sample (seeded RNG):

>>> python data_export.py \
>>>     --input dataset.csv \
>>>     --output-dir data/subsets \
>>>     --sizes 12000 \
>>>     --strategy random \
>>>     --seed 42 \
>>>     --format parquet \
>>>     --write-manifest

4) Stratified sample preserving distributions (proportional allocation):

>>> python data_export.py \
>>>     --input dataset.csv \
>>>     --output-dir data/subsets \
>>>     --percentages 10 \
>>>     --strategy stratified \
>>>     --strata-cols country_code fuel_category is_used \
>>>     --within random \
>>>     --seed 42 \
>>>     --format parquet \
>>>     --write-manifest

5) Stratified sample with *equal* representation per stratum (balanced classes):

>>> python data_export.py \
>>>     --input dataset.csv \
>>>     --output-dir data/subsets \
>>>     --sizes 12000 \
>>>     --strategy stratified \
>>>     --strata-cols label \
>>>     --allocation equal \
>>>     --within linspace \
>>>     --format parquet \
>>>     --write-manifest

6) Streaming hash sample for huge CSV (memory-bounded, supports multiple absolute sizes):

>>> python data_export.py \
>>>     --input huge.csv \
>>>     --output-dir data/subsets \
>>>     --sizes 1000 10000 40000 \
>>>     --strategy hash \
>>>     --id-col id \
>>>     --stream-hash-csv \
>>>     --chunksize 50000 \
>>>     --format parquet \
>>>     --write-manifest
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence  # noqa: F401
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SamplingStrategy = Literal["hash", "random", "stratified"]
WithinGroupStrategy = Literal["hash", "random", "linspace"]
StrataAllocation = Literal["proportional", "equal"]
InputFormat = Literal["auto", "csv", "parquet"]
OutputFormat = Literal["csv", "parquet"]
RoundingMode = Literal["floor", "round", "ceil"]

LOGGER = logging.getLogger("data_export")


# -----------------------------
# Utilities
# -----------------------------
def _utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.

    Parameters
    ----------
    None

    Returns
    -------
    str
        ISO timestamp.

    Raises
    ------
    None

    Seealso
    --------
    datetime.now

    Notes
    -----
    Timestamp is recorded in manifest for export traceability.

    Examples
    --------
    >>> isinstance(_utc_now_iso(), str)
    True
    """
    return datetime.now(timezone.utc).isoformat()


def stable_hash64(text: str) -> int:
    """
    Compute a stable 64-bit hash for a string.

    Parameters
    ----------
    text : str
        Input text to hash (e.g., row id).

    Returns
    -------
    int
        Unsigned 64-bit hash value.

    Raises
    ------
    None

    Seealso
    --------
    hashlib.blake2b

    Notes
    -----
    Uses BLAKE2b with 8-byte digest. This is deterministic across platforms and
    Python versions (unlike Python's built-in hash()).

    Examples
    --------
    >>> stable_hash64("abc") >= 0
    True
    """
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), byteorder="big", signed=False)


def _ensure_cols_exist(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """
    Validate that requested columns exist in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to validate.
    cols : sequence of str
        Columns expected to exist in `df`.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If any requested columns are missing.

    Seealso
    --------
    pandas.DataFrame.columns

    Notes
    -----
    This is strict validation; no implicit fallbacks.

    Examples
    --------
    >>> import numpy as np
    import pandas as pd
    >>> _ensure_cols_exist(pd.DataFrame({"a": [1]}), ["a"])
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def _infer_format(source: str) -> InputFormat:
    r"""
    Infer input format from filename extension (strict, suffix-based).

    Supports:
    - .csv, .csv.gz
    - .parquet, .pq

    Notes
    -----
    For URLs, suffix is inferred from the URL path (query string ignored).

    Raises
    ------
    ValueError
        If extension is not recognized.

    Examples
    --------
    >>> _infer_format("x.csv")
    'csv'
    """
    lower = source.lower()
    path = urlparse(lower).path if "://" in lower else lower

    if path.endswith((".csv", ".csv.gz")):
        return "csv"
    if path.endswith((".parquet", ".pq")):
        return "parquet"

    raise ValueError(
        "Cannot infer input format. Use --input-format csv|parquet explicitly."
    )


def _round_nonnegative(x: float, *, mode: RoundingMode) -> int:
    """
    Round a non-negative float to an integer deterministically.

    Parameters
    ----------
    x : float
        Non-negative value to round.
    mode : {'floor', 'round', 'ceil'}
        Rounding mode.

    Returns
    -------
    int
        Rounded integer.

    Raises
    ------
    ValueError
        If x is negative or mode is invalid.

    Notes
    -----
    For mode='round', uses half-up rounding: floor(x + 0.5).
    """
    if x < 0:
        raise ValueError("x must be non-negative")
    if mode == "floor":
        return math.floor(x)
    if mode == "ceil":
        return math.ceil(x)
    if mode == "round":
        return math.floor(x + 0.5)
    raise ValueError(f"Unsupported rounding mode: {mode}")


def resolve_sizes(  # noqa: PLR0912
    *,
    prepared_rows: int,
    sizes: Sequence[int] | None,
    fractions: Sequence[float] | None,
    percentages: Sequence[float] | None,
    rounding: RoundingMode,
) -> list[int]:
    """
    Resolve requested sizes (absolute + fraction + percentage) into unique row counts.

    Parameters
    ----------
    prepared_rows : int
        Number of rows after preparation filters.
    sizes : sequence of int or None
        Absolute sizes (row counts).
    fractions : sequence of float or None
        Fractions in [0, 1] (e.g., 0.1 means 10%).
    percentages : sequence of float or None
        Percentages in [0, 100].
    rounding : {'floor', 'round', 'ceil'}
        Rounding mode for fraction/percentage conversion.

    Returns
    -------
    list of int
        Sorted unique sizes.

    Raises
    ------
    ValueError
        If any requested size is invalid or exceeds prepared_rows.

    Notes
    -----
    - Fractions/percentages are computed from *prepared_rows* (post-filter).
    - Values > 0 that round to 0 are rejected as a strict safety check.
    """
    if prepared_rows < 0:
        raise ValueError("prepared_rows must be non-negative")

    out: list[int] = []

    if sizes:
        for s in sizes:
            s_int = int(s)
            if s_int < 0:
                raise ValueError("--sizes values must be >= 0")
            out.append(s_int)

    if fractions:
        for f in fractions:
            f_float = float(f)
            if not (0.0 <= f_float <= 1.0):
                raise ValueError("--fractions must be in [0, 1]")
            n = _round_nonnegative(prepared_rows * f_float, mode=rounding)
            if f_float > 0.0 and n < 1:
                raise ValueError(
                    f"fraction={f_float} is too small for prepared_rows={prepared_rows} with rounding={rounding}"
                )
            out.append(n)

    if percentages:
        for p in percentages:
            p_float = float(p)
            if not (0.0 <= p_float <= 100.0):  # noqa: PLR2004
                raise ValueError("--percentages must be in [0, 100]")
            n = _round_nonnegative(prepared_rows * (p_float / 100.0), mode=rounding)
            if p_float > 0.0 and n < 1:
                raise ValueError(
                    f"percentage={p_float} is too small for prepared_rows={prepared_rows} with rounding={rounding}"
                )
            out.append(n)

    if not out:
        raise ValueError(
            "At least one of --sizes / --fractions / --percentages must be provided."
        )

    # Unique + sorted
    uniq = sorted(set(out))
    if any(n > prepared_rows for n in uniq):
        raise ValueError(
            f"Requested size exceeds prepared_rows={prepared_rows}: {uniq}"
        )
    return uniq


def profile_dataframe(df: pd.DataFrame, *, max_columns: int | None = None) -> dict:
    """
    Create a lightweight, deterministic dataset profile.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to profile.
    max_columns : int or None, default=None
        If provided, only profile up to this many columns (left-to-right).
        Use this for very wide datasets.

    Returns
    -------
    dict
        Profile payload suitable for JSON serialization.

    Raises
    ------
    ValueError
        If max_columns is provided and is non-positive.

    Notes
    -----
    The profile is intentionally modest:
    - row/col counts
    - dtypes
    - missing counts
    - numeric summary via describe(include='number')
    - nunique for object/bool columns (can be expensive, but deterministic)
    """
    if max_columns is not None and max_columns <= 0:
        raise ValueError("max_columns must be positive if provided")

    if max_columns is not None:
        cols = list(df.columns)[: int(max_columns)]
        df2 = df[cols]
    else:
        df2 = df

    missing = df2.isna().sum().astype(int).to_dict()
    dtypes = {c: str(t) for c, t in df2.dtypes.items()}

    numeric = df2.select_dtypes(include=["number"])
    numeric_desc = {}
    if numeric.shape[1] > 0 and len(numeric) > 0:
        numeric_desc = numeric.describe().to_dict()

    # nunique for non-numeric
    non_numeric = df2.select_dtypes(exclude=["number"])
    nunique = {}
    if non_numeric.shape[1] > 0:
        nunique = non_numeric.nunique(dropna=True).astype(int).to_dict()

    return {
        "rows": len(df2),
        "cols": int(df2.shape[1]),
        "dtypes": dtypes,
        "missing": missing,
        "nunique_non_numeric": nunique,
        "numeric_describe": numeric_desc,
    }


# -----------------------------
# Loading + preparation
# -----------------------------
def load_dataframe(
    source: str,
    *,
    input_format: InputFormat = "auto",
    usecols: Sequence[str] | None = None,
    query: str | None = None,
    low_memory: bool = False,
    csv_sep: str = ",",
    csv_encoding: str | None = None,
) -> pd.DataFrame:
    """
    Load a dataset from CSV or Parquet into a DataFrame.

    Parameters
    ----------
    source : str
        Path or URL to input dataset.
    input_format : {'auto', 'csv', 'parquet'}, default='auto'
        Input format selection. If 'auto', inferred from suffix.
    usecols : sequence of str or None, default=None
        Optional subset of columns to load (CSV only; Parquet respects columns via `columns=`).
    query : str or None, default=None
        Optional pandas query expression applied after loading.
    low_memory : bool, default=False
        Passed to pandas.read_csv for type inference behavior.
    csv_sep : str, default=','
        CSV delimiter.
    csv_encoding : str or None, default=None
        Optional CSV encoding passed to pandas.read_csv.

    Returns
    -------
    pandas.DataFrame
        Loaded (optionally filtered) dataframe.

    Raises
    ------
    ValueError
        If format is unsupported or query fails.
    KeyError
        If query references unknown columns.

    Seealso
    --------
    pandas.read_csv, pandas.read_parquet, pandas.DataFrame.query

    Notes
    -----
    Query is applied post-load for strict correctness and consistent semantics.
    """
    fmt: InputFormat = input_format
    if fmt == "auto":
        fmt = _infer_format(source)

    LOGGER.info("Loading source=%s format=%s", source, fmt)

    if fmt == "csv":
        df = pd.read_csv(
            source,
            usecols=usecols,
            low_memory=low_memory,
            sep=csv_sep,
            encoding=csv_encoding,
        )
    elif fmt == "parquet":
        df = pd.read_parquet(source, columns=list(usecols) if usecols else None)
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    if query is not None:
        LOGGER.info("Applying query=%r", query)
        df = df.query(query, engine="python")

    return df


def enforce_required_columns(
    df: pd.DataFrame,
    *,
    required_cols: Sequence[str] | None,
) -> pd.DataFrame:
    """
    Drop rows missing values in required columns only.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    required_cols : sequence of str or None
        Columns that must be non-null. If None/empty, no rows are dropped.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.

    Raises
    ------
    KeyError
        If required columns do not exist.

    Seealso
    --------
    pandas.DataFrame.dropna

    Notes
    -----
    This avoids destructive `.dropna()` across all columns.
    """
    if not required_cols:
        return df
    _ensure_cols_exist(df, required_cols)
    before = len(df)
    out = df.dropna(subset=list(required_cols))
    LOGGER.info(
        "Dropped rows with NA in required_cols=%s: %d -> %d",
        list(required_cols),
        before,
        len(out),
    )
    return out


def drop_duplicates_by_id(
    df: pd.DataFrame, *, id_col: str, keep: Literal["first", "last"] = "first"
) -> pd.DataFrame:
    """
    Drop duplicate rows by identifier column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    id_col : str
        Identifier column name.
    keep : {'first', 'last'}, default='first'
        Which duplicate row to keep.

    Returns
    -------
    pandas.DataFrame
        Deduplicated dataframe.

    Raises
    ------
    KeyError
        If `id_col` is missing.

    Seealso
    --------
    pandas.DataFrame.drop_duplicates
    """
    _ensure_cols_exist(df, [id_col])
    before = len(df)
    out = df.drop_duplicates(subset=[id_col], keep=keep)
    LOGGER.info("Dropped duplicates by %s: %d -> %d", id_col, before, len(out))
    return out


def apply_keep_drop_columns(
    df: pd.DataFrame,
    *,
    keep_cols: Sequence[str] | None,
    drop_cols: Sequence[str] | None,
) -> pd.DataFrame:
    """
    Apply strict keep/drop column selection.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    keep_cols : sequence of str or None
        If provided, keep exactly these columns.
    drop_cols : sequence of str or None
        If provided, drop these columns.

    Returns
    -------
    pandas.DataFrame
        Dataframe with adjusted columns.

    Raises
    ------
    ValueError
        If both keep_cols and drop_cols are provided.
    KeyError
        If requested columns do not exist.
    """
    if keep_cols and drop_cols:
        raise ValueError("Provide at most one of keep_cols or drop_cols")

    if keep_cols:
        _ensure_cols_exist(df, list(keep_cols))
        return df.loc[:, list(keep_cols)]
    if drop_cols:
        _ensure_cols_exist(df, list(drop_cols))
        return df.drop(columns=list(drop_cols))

    return df


# -----------------------------
# Sampling (in-memory)
# -----------------------------
def sample_hash_full(df: pd.DataFrame, *, n: int, id_col: str) -> pd.DataFrame:
    """
    Deterministically sample `n` rows by sorting on stable hash of `id_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    n : int
        Number of rows to sample. Must be <= len(df).
    id_col : str
        Identifier column to hash.

    Returns
    -------
    pandas.DataFrame
        Sampled dataframe.

    Raises
    ------
    ValueError
        If `n` is invalid.
    KeyError
        If `id_col` is missing.

    Seealso
    --------
    stable_hash64
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(df):
        raise ValueError(f"n={n} exceeds dataset size={len(df)}")
    _ensure_cols_exist(df, [id_col])

    hashes = df[id_col].astype(str).map(stable_hash64)
    idx = hashes.sort_values(kind="mergesort").index[:n]
    return df.loc[idx].reset_index(drop=True)


def sample_random(df: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    """
    Randomly sample `n` rows with a fixed seed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    n : int
        Number of rows to sample. Must be <= len(df).
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Sampled dataframe.

    Raises
    ------
    ValueError
        If `n` is invalid.

    Seealso
    --------
    pandas.DataFrame.sample
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(df):
        raise ValueError(f"n={n} exceeds dataset size={len(df)}")
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def _linspace_positions(*, population: int, n: int) -> np.ndarray:
    """
    Compute `n` unique, sorted integer positions over ``range(population)`` using linspace.

    Parameters
    ----------
    population : int
        Total number of available rows.
    n : int
        Number of positions to select. Must satisfy ``0 <= n <= population``.

    Returns
    -------
    numpy.ndarray
        Sorted unique integer positions of length `n`.

    Raises
    ------
    ValueError
        If `population` or `n` are invalid.
    RuntimeError
        If the internal repair step fails.

    Notes
    -----
    This is a deterministic, spread-out selection method:
    it selects approximately evenly spaced rows using ``numpy.linspace``.
    """
    if population < 0:
        raise ValueError("population must be non-negative")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > population:
        raise ValueError(f"n={n} exceeds population={population}")
    if n == 0:
        return np.asarray([], dtype=int)
    if n == population:
        return np.arange(population, dtype=int)

    # Round to nearest integer to reduce left-bias; then repair duplicates deterministically.
    pos = np.rint(np.linspace(0, population - 1, num=n)).astype(int)
    pos = np.unique(pos)

    if pos.size == n:
        return pos

    missing = int(n - pos.size)
    all_idx = np.arange(population, dtype=int)
    remaining = np.setdiff1d(all_idx, pos, assume_unique=False)
    pos = np.concatenate([pos, remaining[:missing]])
    pos.sort()

    if pos.size != n:
        raise RuntimeError(
            "linspace position repair failed to produce the requested number of positions."
        )
    return pos


def sample_linspace(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    """
    Deterministically sample `n` rows spread across the dataframe using linspace positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    n : int
        Number of rows to sample. Must be <= len(df).

    Returns
    -------
    pandas.DataFrame
        Sampled dataframe.

    Raises
    ------
    ValueError
        If `n` is invalid.

    Notes
    -----
    Useful when you want a *spread-out* subset along the existing row order
    (or after you explicitly sort the dataframe before calling).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(df):
        raise ValueError(f"n={n} exceeds dataset size={len(df)}")
    pos = _linspace_positions(population=len(df), n=n)
    return df.iloc[pos].reset_index(drop=True)


def _largest_remainder_allocation(group_sizes: pd.Series, *, n: int) -> pd.Series:
    """
    Allocate integer sample counts to groups proportionally using largest remainder.

    Parameters
    ----------
    group_sizes : pandas.Series
        Group sizes indexed by group key.
    n : int
        Total sample size.

    Returns
    -------
    pandas.Series
        Integer allocations per group summing to `n`.

    Raises
    ------
    ValueError
        If `n` is invalid.
    RuntimeError
        If allocation fails due to inconsistent capacities.

    Notes
    -----
    Deterministic tie-breaking using (fraction desc, capacity desc, group key asc).
    """
    total = int(group_sizes.sum())
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > total:
        raise ValueError(f"n={n} exceeds dataset size={total}")

    quotas = group_sizes.astype(float) * (float(n) / float(total))
    floors = quotas.apply(lambda x: int(x))
    alloc = floors.clip(upper=group_sizes.astype(int))

    remaining = n - int(alloc.sum())
    if remaining == 0:
        return alloc.astype(int)

    frac = quotas - floors
    capacity = group_sizes.astype(int) - alloc.astype(int)

    idx_sorted = sorted(
        alloc.index,
        key=lambda k: (-float(frac.loc[k]), -int(capacity.loc[k]), str(k)),
    )

    alloc = alloc.astype(int)
    for k in idx_sorted:
        if remaining == 0:
            break
        cap = int(capacity.loc[k])
        if cap <= 0:
            continue
        take = min(cap, remaining)
        alloc.loc[k] += take
        remaining -= take

    if remaining != 0:
        raise RuntimeError("Allocation failed to distribute all remaining samples.")

    return alloc.astype(int)


def _equal_allocation(group_sizes: pd.Series, *, n: int) -> pd.Series:
    """
    Allocate integer sample counts to groups as evenly as possible.

    Parameters
    ----------
    group_sizes : pandas.Series
        Group sizes indexed by group key.
    n : int
        Total sample size.

    Returns
    -------
    pandas.Series
        Integer allocations per group summing to `n`.

    Raises
    ------
    ValueError
        If `n` is invalid.
    RuntimeError
        If allocation fails due to inconsistent capacities.

    Notes
    -----
    - Starts with ``floor(n / k)`` for each of ``k`` groups, then distributes the remainder
      to groups in deterministic key order.
    - If some groups have insufficient capacity, the shortfall is redistributed
      to other groups with available capacity in deterministic key order.
    """
    total = int(group_sizes.sum())
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > total:
        raise ValueError(f"n={n} exceeds dataset size={total}")

    k = int(group_sizes.shape[0])
    if k == 0:
        return group_sizes.astype(int)

    base = int(n // k)
    rem = int(n % k)

    idx_sorted = sorted(group_sizes.index, key=lambda x: str(x))
    desired = pd.Series(base, index=group_sizes.index, dtype=int)
    if rem:
        desired.loc[idx_sorted[:rem]] += 1

    alloc = desired.clip(upper=group_sizes.astype(int)).astype(int)
    remaining = int(n - alloc.sum())
    if remaining == 0:
        return alloc.astype(int)

    capacity = (group_sizes.astype(int) - alloc.astype(int)).astype(int)
    for key in idx_sorted:
        if remaining == 0:
            break
        cap = int(capacity.loc[key])
        if cap <= 0:
            continue
        take = min(cap, remaining)
        alloc.loc[key] += take
        remaining -= take

    if remaining != 0:
        raise RuntimeError("Allocation failed to distribute all remaining samples.")
    return alloc.astype(int)


def allocate_strata(
    group_sizes: pd.Series, *, n: int, allocation: StrataAllocation
) -> pd.Series:
    """
    Allocate integer sample counts to groups.

    Parameters
    ----------
    group_sizes : pandas.Series
        Group sizes indexed by group key.
    n : int
        Total sample size.
    allocation : {'proportional', 'equal'}
        Allocation mode.

    Returns
    -------
    pandas.Series
        Integer allocations per group summing to `n`.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if allocation == "proportional":
        return _largest_remainder_allocation(group_sizes, n=n)
    if allocation == "equal":
        return _equal_allocation(group_sizes, n=n)
    raise ValueError(f"Unknown allocation mode: {allocation!r}")


def sample_stratified(  # noqa: D417, PLR0912
    df: pd.DataFrame,
    *,
    n: int,
    id_col: str | None,
    strata_cols: Sequence[str],
    within: WithinGroupStrategy,
    allocation: StrataAllocation,
    seed: int,
) -> pd.DataFrame:
    """
    Stratified sampling with strict allocation across strata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    n : int
        Number of rows to sample. Must be <= len(df).
    id_col : str or None
        Identifier column used for stable hashing when within='hash'.
        Can be None when within='random'.
    strata_cols : sequence of str
        Columns defining strata groups (must be provided explicitly).
    within : {'hash', 'random', 'linspace'}
        Within-stratum sampling method.
    seed : int
        Seed used when within='random' and for deterministic final trimming.

    Returns
    -------
    pandas.DataFrame
        Stratified sample of exactly n rows.

    Raises
    ------
    ValueError
        If inputs are invalid or required columns are missing.
    KeyError
        If required columns are missing.
    """
    if not strata_cols:
        raise ValueError("strata_cols must be non-empty for stratified sampling")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(df):
        raise ValueError(f"n={n} exceeds dataset size={len(df)}")

    _ensure_cols_exist(df, list(strata_cols))
    if within == "hash":
        if not id_col:
            raise ValueError("id_col is required when within='hash'")
        _ensure_cols_exist(df, [id_col])

    gb = df.groupby(list(strata_cols), dropna=False, sort=True)
    grp_sizes = gb.size()
    alloc = allocate_strata(grp_sizes, n=n, allocation=allocation)

    pieces: list[pd.DataFrame] = []

    for key, g in gb:
        k_n = int(alloc.loc[key])
        if k_n <= 0:
            continue
        if k_n >= len(g):
            pieces.append(g)
            continue

        if within == "hash":
            pieces.append(sample_hash_full(g, n=k_n, id_col=str(id_col)))
        elif within == "random":
            key_text = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            group_seed = stable_hash64(key_text) ^ int(seed)
            pieces.append(sample_random(g, n=k_n, seed=int(group_seed & 0x7FFFFFFF)))
        else:
            pieces.append(sample_linspace(g, n=k_n))

    out = pd.concat(pieces, ignore_index=True)

    # Strict: enforce exact size n deterministically (safety net)
    if len(out) != n:
        if within == "hash":
            out = sample_hash_full(out, n=n, id_col=str(id_col))
        elif within == "random":
            out = sample_random(out, n=n, seed=int(seed))
        else:
            out = sample_linspace(out, n=n)

    return out.reset_index(drop=True)


# -----------------------------
# Sampling (streaming hash for CSV)
# -----------------------------
@dataclasses.dataclass(frozen=True)
class StreamStats:
    """
    Streaming pass statistics.

    Parameters
    ----------
    rows_seen_total : int
        Total rows read from the CSV (before required_cols/query filters).
    rows_seen_eligible : int
        Total rows remaining after required_cols/query/id_col filters.
    """

    rows_seen_total: int
    rows_seen_eligible: int


def sample_hash_csv_stream(  # noqa: PLR0912
    source: str,
    *,
    n: int,
    id_col: str,
    usecols: Sequence[str] | None = None,
    required_cols: Sequence[str] | None = None,
    query: str | None = None,
    chunksize: int = 50_000,
    low_memory: bool = False,
    csv_sep: str = ",",
    csv_encoding: str | None = None,
) -> tuple[pd.DataFrame, StreamStats]:
    """
    Deterministically sample `n` rows from a CSV using streaming top-N by stable hash.

    Parameters
    ----------
    source : str
        CSV path or URL.
    n : int
        Number of rows to sample.
    id_col : str
        Identifier column for hashing.
    usecols : sequence of str or None, default=None
        Columns to load. If provided, MUST include `id_col` and any `required_cols`.
    required_cols : sequence of str or None, default=None
        Columns that must be non-null.
    query : str or None, default=None
        Optional pandas query applied per chunk.
    chunksize : int, default=50_000
        Chunk size for pandas.read_csv.
    low_memory : bool, default=False
        Passed to pandas.read_csv.
    csv_sep : str, default=','
        CSV delimiter.
    csv_encoding : str or None, default=None
        Optional CSV encoding passed to pandas.read_csv.

    Returns
    -------
    (pandas.DataFrame, StreamStats)
        Sampled dataframe of exactly `n` rows (unless n==0) and streaming stats.

    Raises
    ------
    ValueError
        If `n` is invalid or stream produces fewer than n eligible rows.
    KeyError
        If required columns are missing in chunks.

    Notes
    -----
    - Keeps the `n` smallest hashes using a bounded heap; deterministic and memory-bounded w.r.t. n.
    - Query is evaluated per chunk.

    Examples
    --------
    >>> # df, stats = sample_hash_csv_stream("big.csv", n=10000, id_col="id")
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    needed_cols: set[str] = {id_col}
    if required_cols:
        needed_cols.update(required_cols)

    if usecols is not None:
        missing = [c for c in needed_cols if c not in set(usecols)]
        if missing:
            raise KeyError(f"usecols is missing required columns: {missing}")

    heap: list[tuple[int, int, dict]] = []
    counter = 0

    rows_seen_total = 0
    rows_seen_eligible = 0

    # Stream chunks
    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=low_memory,
        sep=csv_sep,
        encoding=csv_encoding,
    ):
        rows_seen_total += len(chunk)
        _ensure_cols_exist(chunk, list(needed_cols))

        if required_cols:
            chunk = chunk.dropna(subset=list(required_cols))  # noqa: PLW2901
        chunk = chunk.dropna(subset=[id_col])  # noqa: PLW2901

        if query is not None and len(chunk) > 0:
            chunk = chunk.query(query, engine="python")  # noqa: PLW2901

        if len(chunk) == 0:
            continue

        rows_seen_eligible += len(chunk)

        # Compute hashes for the chunk (strict, deterministic)
        h = chunk[id_col].astype(str).map(stable_hash64)

        # Only consider chunk-local best candidates to reduce heap churn
        k = min(n, len(chunk))
        idx_best = h.nsmallest(k, keep="first").index

        for idx in idx_best:
            row = chunk.loc[idx]
            hh = int(h.loc[idx])

            # heap stores (-hash, counter, row_dict)
            item = (-hh, counter, row.to_dict())
            counter += 1

            if len(heap) < n:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    if n == 0:
        return pd.DataFrame(), StreamStats(
            rows_seen_total=rows_seen_total, rows_seen_eligible=rows_seen_eligible
        )

    if len(heap) < n:
        raise ValueError(
            f"Not enough eligible rows after filters/query. needed={n}, got={len(heap)}"
        )

    # Convert heap to DataFrame, sorted by hash asc (deterministic output order)
    rows = [it[2] for it in heap]
    out = pd.DataFrame(rows)
    out["_hash64__"] = out[id_col].astype(str).map(stable_hash64)
    out = (
        out.sort_values("_hash64__", kind="mergesort")
        .head(n)
        .drop(columns=["_hash64__"])
        .reset_index(drop=True)
    )

    return out, StreamStats(
        rows_seen_total=rows_seen_total, rows_seen_eligible=rows_seen_eligible
    )


# -----------------------------
# Writing + manifest
# -----------------------------
def write_dataset(df: pd.DataFrame, *, output_path: Path, fmt: OutputFormat) -> None:
    """
    Write dataframe to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to write.
    output_path : pathlib.Path
        Output file path.
    fmt : {'csv', 'parquet'}
        Output format.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If format is unsupported.

    Seealso
    --------
    pandas.DataFrame.to_csv, pandas.DataFrame.to_parquet
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(output_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


@dataclasses.dataclass(frozen=True)
class ExportSpec:
    """
    Export specification.

    Parameters
    ----------
    size : int
        Number of rows to export.
    strategy : {'hash', 'random', 'stratified'}
        Sampling strategy.

    Notes
    -----
    Immutable by design for reproducible manifest writing.
    """

    size: int
    strategy: SamplingStrategy


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
) -> None:
    """
    Write a JSON manifest describing the export.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Manifest file path.
    source : str
        Input source path/URL.
    input_format : {'auto', 'csv', 'parquet'}
        Input format used.
    output_file : pathlib.Path
        Output dataset file path.
    export_spec : ExportSpec
        Export specification.
    prep_params : dict
        Preparation parameters (query, required_cols, dedup, keep/drop cols, etc.).
    strategy_params : dict
        Sampling parameters (id_col, seed, strata_cols, etc.).
    rows : dict
        Row counters (input_prepared, output, and optional streaming counts).
    columns : list of str
        Exported columns.

    Returns
    -------
    None

    Notes
    -----
    This is the single source of truth for how the subset was built.
    """
    payload = {
        "created_utc": _utc_now_iso(),
        "source": {"path_or_url": source, "format": input_format},
        "prepare": prep_params,
        "export": {
            "size": export_spec.size,
            "strategy": export_spec.strategy,
            "params": strategy_params,
        },
        "rows": rows,
        "columns": columns,
        "output_file": str(output_file),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_json(data: dict, *, output_path: Path) -> None:
    """
    Write a JSON file.

    Parameters
    ----------
    data : dict
        JSON-serializable object.
    output_path : pathlib.Path
        Path to write.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.

    Notes
    -----
    Keep arguments explicit; avoid implicit inference beyond file suffix.
    """
    p = argparse.ArgumentParser(
        description="Generic reproducible dataset subset exporter (CSV/Parquet)."
    )

    p.add_argument("--input", required=True, help="Input dataset path or URL.")
    p.add_argument(
        "--input-format",
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="Input format.",
    )
    p.add_argument(
        "--output-dir", required=True, help="Directory to write exported subsets into."
    )

    # sizing: at least one of these must be provided
    p.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=None,
        help="Absolute subset sizes (rows), e.g. 1000 10000.",
    )
    p.add_argument(
        "--fractions",
        nargs="*",
        type=float,
        default=None,
        help="Fractions in [0,1], e.g. 0.1 for 10%.",
    )
    p.add_argument(
        "--percentages",
        nargs="*",
        type=float,
        default=None,
        help="Percentages in [0,100], e.g. 10 for 10%.",
    )
    p.add_argument(
        "--rounding",
        choices=["floor", "round", "ceil"],
        default="round",
        help="Rounding mode for fractions/percentages.",
    )

    p.add_argument(
        "--strategy",
        choices=["hash", "random", "stratified"],
        default="hash",
        help="Sampling strategy.",
    )
    p.add_argument(
        "--id-col",
        default=None,
        help="Identifier column (required for hash and within=hash).",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for random strategies.")
    p.add_argument(
        "--strata-cols",
        nargs="*",
        default=None,
        help="Strata columns for stratified sampling.",
    )
    p.add_argument(
        "--within",
        choices=["hash", "random", "linspace"],
        default="hash",
        help="Within-stratum sampling strategy.",
    )
    p.add_argument(
        "--allocation",
        choices=["proportional", "equal"],
        default="proportional",
        help="Strata allocation mode for stratified sampling.",
    )

    p.add_argument(
        "--usecols", nargs="*", default=None, help="Optional list of columns to load."
    )
    p.add_argument(
        "--required-cols",
        nargs="*",
        default=None,
        help="Rows must be non-null for these columns.",
    )
    p.add_argument(
        "--dedup",
        action="store_true",
        help="Drop duplicates by id-col (requires --id-col).",
    )
    p.add_argument(
        "--query",
        default=None,
        help="Optional pandas query applied after loading (or per chunk in stream mode).",
    )

    p.add_argument(
        "--keep-cols",
        nargs="*",
        default=None,
        help="Keep exactly these columns (strict).",
    )
    p.add_argument(
        "--drop-cols", nargs="*", default=None, help="Drop these columns (strict)."
    )

    # CSV options
    p.add_argument("--csv-sep", default=",", help="CSV delimiter (default ',').")
    p.add_argument(
        "--csv-encoding", default=None, help="CSV encoding (default pandas default)."
    )
    p.add_argument(
        "--csv-low-memory",
        action="store_true",
        help="Enable pandas read_csv low_memory mode.",
    )

    p.add_argument(
        "--format", choices=["csv", "parquet"], default="parquet", help="Output format."
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write .manifest.json alongside outputs.",
    )
    p.add_argument(
        "--write-profile",
        action="store_true",
        help="Write lightweight .profile.json alongside outputs.",
    )
    p.add_argument(
        "--profile-max-columns",
        type=int,
        default=None,
        help="Limit profile to first N columns.",
    )

    # Streaming option: only applies to CSV + hash strategy
    p.add_argument(
        "--stream-hash-csv",
        action="store_true",
        help="Use streaming hash top-N for CSV (memory-bounded).",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="CSV chunksize for --stream-hash-csv.",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:  # noqa: PLR0912
    """
    Validate CLI args strictly.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required arguments for a strategy are missing or incompatible.

    Notes
    -----
    No fallback guessing. Missing requirements are hard errors.
    """
    if not (args.sizes or args.fractions or args.percentages):
        raise ValueError(
            "Provide at least one of --sizes / --fractions / --percentages."
        )

    if args.keep_cols and args.drop_cols:
        raise ValueError("Provide at most one of --keep-cols or --drop-cols.")

    if args.strategy == "hash":  # noqa: SIM102
        if not args.id_col:
            raise ValueError("--id-col is required for --strategy hash")

    if args.strategy == "stratified":
        if not args.strata_cols:
            raise ValueError("--strata-cols is required for --strategy stratified")
        if args.within == "hash" and not args.id_col:
            raise ValueError(
                "--id-col is required for stratified sampling when --within hash"
            )

    if args.dedup and not args.id_col:
        raise ValueError("--dedup requires --id-col")

    if args.stream_hash_csv:
        if args.strategy != "hash":
            raise ValueError("--stream-hash-csv is only supported with --strategy hash")
        if args.input_format not in ("auto", "csv"):
            raise ValueError(
                "--stream-hash-csv requires CSV input (use --input-format csv or auto with .csv/.csv.gz)"
            )
        if not args.id_col:
            raise ValueError("--id-col is required for --stream-hash-csv")
        # Fractions/percentages require knowing prepared row count; streaming is single-pass.
        if args.fractions or args.percentages:
            raise ValueError(
                "--fractions/--percentages are not supported with --stream-hash-csv (use absolute --sizes)"
            )

    if args.sizes:
        sizes = [int(s) for s in args.sizes]
        if any(s < 0 for s in sizes):
            raise ValueError("--sizes values must be >= 0")
        if len(set(sizes)) != len(sizes):
            raise ValueError("--sizes must be unique")

    if args.fractions:
        fr = [float(x) for x in args.fractions]
        if any((x < 0.0 or x > 1.0) for x in fr):
            raise ValueError("--fractions must be in [0, 1]")

    if args.percentages:
        pc = [float(x) for x in args.percentages]
        if any((x < 0.0 or x > 100.0) for x in pc):  # noqa: PLR2004
            raise ValueError("--percentages must be in [0, 100]")


def _required_columns_for_pipeline(args: argparse.Namespace) -> list[str]:
    """
    Compute columns that must be present for requested operations.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    list of str
        Required columns.

    Notes
    -----
    This enables strict validation for --keep-cols and --usecols.
    """
    required: list[str] = []
    if args.id_col:
        required.append(str(args.id_col))
    if args.required_cols:
        required.extend(list(args.required_cols))
    if args.strategy == "stratified" and args.strata_cols:
        required.extend(list(args.strata_cols))
    return sorted(set(required))


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0912
    """
    CLI entrypoint.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        CLI args. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 success).

    Raises
    ------
    Exception
        Unexpected failures are raised after logging.

    Notes
    -----
    Strict validations prevent silent behavior changes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    _validate_args(args)

    source = str(args.input)
    output_dir = Path(args.output_dir)
    out_fmt: OutputFormat = str(args.format)
    rounding: RoundingMode = str(args.rounding)

    required_for_ops = _required_columns_for_pipeline(args)

    # Strict: if usecols is provided, it must include columns required by operations
    if args.usecols:
        missing = [c for c in required_for_ops if c not in set(args.usecols)]
        if missing:
            raise KeyError(
                f"--usecols is missing required columns for selected operations: {missing}"
            )

    # Strict: if keep-cols is provided, it must include columns required by operations
    if args.keep_cols:
        missing = [c for c in required_for_ops if c not in set(args.keep_cols)]
        if missing:
            raise KeyError(
                f"--keep-cols is missing required columns for selected operations: {missing}"
            )

    prep_params = {
        "usecols": list(args.usecols) if args.usecols else None,
        "required_cols": list(args.required_cols) if args.required_cols else None,
        "dedup": bool(args.dedup),
        "query": args.query,
        "keep_cols": list(args.keep_cols) if args.keep_cols else None,
        "drop_cols": list(args.drop_cols) if args.drop_cols else None,
    }

    # ---------------------------
    # Streaming path (CSV + hash)
    # ---------------------------
    if args.stream_hash_csv:
        # Streaming supports only absolute sizes
        sizes_abs = sorted({int(s) for s in (args.sizes or []) if int(s) >= 0})
        if not sizes_abs:
            raise ValueError(
                "--stream-hash-csv requires at least one absolute size via --sizes"
            )

        max_n = max(sizes_abs)
        df_top, stats = sample_hash_csv_stream(
            source,
            n=max_n,
            id_col=str(args.id_col),
            usecols=list(args.usecols) if args.usecols else None,
            required_cols=list(args.required_cols) if args.required_cols else None,
            query=args.query,
            chunksize=int(args.chunksize),
            low_memory=bool(args.csv_low_memory),
            csv_sep=str(args.csv_sep),
            csv_encoding=args.csv_encoding,
        )

        # Optional: apply keep/drop after sampling (strict column presence validated earlier)
        df_top = apply_keep_drop_columns(
            df_top,
            keep_cols=list(args.keep_cols) if args.keep_cols else None,
            drop_cols=list(args.drop_cols) if args.drop_cols else None,
        )

        if args.write_profile:
            prof = profile_dataframe(df_top, max_columns=args.profile_max_columns)
            write_json(prof, output_path=output_dir / "stream_top.profile.json")

        for n in sizes_abs:
            out = df_top.head(int(n)).reset_index(drop=True)

            stem = f"subset_{args.strategy}_{n}"
            out_file = (
                output_dir / f"{stem}.{out_fmt if out_fmt == 'csv' else 'parquet'}"
            )
            write_dataset(out, output_path=out_file, fmt=out_fmt)

            if args.write_profile:
                prof = profile_dataframe(out, max_columns=args.profile_max_columns)
                write_json(prof, output_path=output_dir / f"{stem}.profile.json")

            if args.write_manifest:
                manifest_file = output_dir / f"{stem}.manifest.json"
                strategy_params = {
                    "id_col": str(args.id_col),
                    "stream": True,
                    "chunksize": int(args.chunksize),
                    "max_n": int(max_n),
                }
                rows = {
                    "input_prepared": None,
                    "output": len(out),
                    "rows_seen_total": int(stats.rows_seen_total),
                    "rows_seen_eligible": int(stats.rows_seen_eligible),
                }
                write_manifest(
                    manifest_path=manifest_file,
                    source=source,
                    input_format=str(args.input_format),
                    output_file=out_file,
                    export_spec=ExportSpec(size=int(n), strategy=str(args.strategy)),
                    prep_params=prep_params,
                    strategy_params=strategy_params,
                    rows=rows,
                    columns=list(out.columns),
                )

            LOGGER.info("Wrote stream subset rows=%d -> %s", len(out), out_file)

        LOGGER.info("Done (stream hash).")
        return 0

    # ---------------------------
    # In-memory path
    # ---------------------------
    df = load_dataframe(
        source,
        input_format=str(args.input_format),
        usecols=list(args.usecols) if args.usecols else None,
        query=args.query,
        low_memory=bool(args.csv_low_memory),
        csv_sep=str(args.csv_sep),
        csv_encoding=args.csv_encoding,
    )

    df = apply_keep_drop_columns(
        df,
        keep_cols=list(args.keep_cols) if args.keep_cols else None,
        drop_cols=list(args.drop_cols) if args.drop_cols else None,
    )

    df = enforce_required_columns(
        df, required_cols=list(args.required_cols) if args.required_cols else None
    )

    if args.id_col:
        # For hash/within-hash, id must be non-null
        df = df.dropna(subset=[str(args.id_col)])

    if args.dedup:
        df = drop_duplicates_by_id(df, id_col=str(args.id_col))

    input_rows = len(df)
    LOGGER.info("Prepared dataset rows=%d cols=%d", input_rows, df.shape[1])

    sizes_final = resolve_sizes(
        prepared_rows=input_rows,
        sizes=args.sizes or None,
        fractions=args.fractions or None,
        percentages=args.percentages or None,
        rounding=rounding,
    )

    if args.write_profile:
        prof = profile_dataframe(df, max_columns=args.profile_max_columns)
        write_json(prof, output_path=output_dir / "prepared.profile.json")

    for n in sizes_final:
        spec = ExportSpec(size=int(n), strategy=str(args.strategy))
        if spec.size > input_rows:
            raise ValueError(
                f"Requested size={spec.size} exceeds prepared dataset rows={input_rows}"
            )

        if spec.strategy == "hash":
            out = sample_hash_full(df, n=spec.size, id_col=str(args.id_col))
            strategy_params = {"id_col": str(args.id_col), "stream": False}
        elif spec.strategy == "random":
            out = sample_random(df, n=spec.size, seed=int(args.seed))
            strategy_params = {"seed": int(args.seed)}
        else:
            out = sample_stratified(
                df,
                n=spec.size,
                id_col=str(args.id_col) if args.id_col else None,
                strata_cols=list(args.strata_cols) if args.strata_cols else [],
                within=str(args.within),
                allocation=str(args.allocation),
                seed=int(args.seed),
            )
            strategy_params = {
                "id_col": str(args.id_col) if args.id_col else None,
                "strata_cols": list(args.strata_cols) if args.strata_cols else [],
                "within": str(args.within),
                "allocation": str(getattr(args, "allocation", "proportional")),
                "seed": int(args.seed),
            }

        stem = f"subset_{spec.strategy}_{spec.size}"
        out_file = output_dir / f"{stem}.{out_fmt if out_fmt == 'csv' else 'parquet'}"
        write_dataset(out, output_path=out_file, fmt=out_fmt)

        if args.write_profile:
            prof = profile_dataframe(out, max_columns=args.profile_max_columns)
            write_json(prof, output_path=output_dir / f"{stem}.profile.json")

        if args.write_manifest:
            manifest_file = output_dir / f"{stem}.manifest.json"
            rows = {"input_prepared": int(input_rows), "output": len(out)}
            write_manifest(
                manifest_path=manifest_file,
                source=source,
                input_format=str(args.input_format),
                output_file=out_file,
                export_spec=spec,
                prep_params=prep_params,
                strategy_params=strategy_params,
                rows=rows,
                columns=list(out.columns),
            )

        LOGGER.info("Wrote %s rows=%d -> %s", spec, len(out), out_file)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
