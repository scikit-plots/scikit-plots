# scikitplot/datasets/tests/test__data_export.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._data_export` — 29 test classes, 140+ test methods.

Coverage map
------------
_utc_now_iso               format, UTC, monotone            → TestUtcNowIso
stable_hash64              determinism, range, encoding     → TestStableHash64
_ensure_cols_exist         present/missing/empty            → TestEnsureColsExist
_infer_format              csv/parquet/gz/url/error         → TestInferFormat
_round_nonnegative         floor/ceil/round, negative error → TestRoundNonnegative
resolve_sizes              abs/frac/pct, dup, bounds        → TestResolveSizes
profile_dataframe          keys, missing, max_cols, JSON    → TestProfileDataframe
load_dataframe             CSV/Parquet/query/usecols        → TestLoadDataframe
enforce_required_columns   None/empty/drop/missing col      → TestEnforceRequiredColumns
drop_duplicates_by_id      first/last/count/missing col     → TestDropDuplicatesById
apply_keep_drop_columns    keep/drop/both/neither           → TestApplyKeepDropColumns
sample_hash_full           determinism/size/n=0/subset      → TestSampleHashFull
sample_random              determinism/seed/no-replace      → TestSampleRandom
_linspace_positions        count/sorted/range/edges         → TestLinspacePositions
sample_linspace            determinism/size/index           → TestSampleLinspace
_largest_remainder_alloc   sums-to-n/capacity/errors        → TestLargestRemainderAlloc
_equal_allocation          balanced/capacity/sums-to-n      → TestEqualAllocation
allocate_strata            dispatch/invalid                 → TestAllocateStrata
sample_stratified          all within/alloc combos          → TestSampleStratified
sample_hash_csv_stream     happy/filter/n=0/errors          → TestSampleHashCsvStream
write_dataset              csv/parquet/dirs/invalid fmt     → TestWriteDataset
write_manifest             JSON/keys/dirs                   → TestWriteManifest
write_json                 content/unicode/dirs             → TestWriteJson
ExportSpec                 frozen/fields                    → TestExportSpec
StreamStats                frozen/fields                    → TestStreamStats
build_parser               defaults/required flags          → TestBuildParser
_validate_args             all strategies/conflicts         → TestValidateArgs
_required_columns          id/required/strata               → TestRequiredColumnsForPipeline
main CLI                   hash/random/stratified/stream    → TestMainCli
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency guard — parquet requires pyarrow or fastparquet
# ---------------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401
    _PARQUET_OK = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_OK = True
    except ImportError:
        _PARQUET_OK = False

_need_parquet = unittest.skipUnless(
    _PARQUET_OK,
    "pyarrow or fastparquet required for Parquet I/O tests",
)


from .._data_export import (
    ExportSpec,
    StreamStats,
    _ensure_cols_exist,
    _equal_allocation,
    _infer_format,
    _largest_remainder_allocation,
    _linspace_positions,
    _required_columns_for_pipeline,
    _round_nonnegative,
    _utc_now_iso,
    _validate_args,
    allocate_strata,
    apply_keep_drop_columns,
    build_parser,
    drop_duplicates_by_id,
    enforce_required_columns,
    load_dataframe,
    main,
    profile_dataframe,
    resolve_sizes,
    sample_hash_csv_stream,
    sample_hash_full,
    sample_linspace,
    sample_random,
    sample_stratified,
    stable_hash64,
    write_dataset,
    write_json,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 20, *, id_col: bool = True) -> pd.DataFrame:
    """Deterministic test DataFrame."""
    data: dict = {"x": list(range(n)), "y": [float(i) * 0.5 for i in range(n)]}
    if id_col:
        data["id"] = [f"row{i:04d}" for i in range(n)]
    return pd.DataFrame(data)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# _utc_now_iso
# ---------------------------------------------------------------------------

class TestUtcNowIso(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(_utc_now_iso(), str)

    def test_contains_utc_offset(self):
        ts = _utc_now_iso()
        self.assertTrue("+00:00" in ts or ts.endswith("Z"), msg=repr(ts))

    def test_non_empty(self):
        self.assertGreater(len(_utc_now_iso()), 0)

    def test_monotone(self):
        t1 = _utc_now_iso()
        t2 = _utc_now_iso()
        self.assertGreaterEqual(t2, t1)


# ---------------------------------------------------------------------------
# stable_hash64
# ---------------------------------------------------------------------------

class TestStableHash64(unittest.TestCase):

    def test_returns_int(self):
        self.assertIsInstance(stable_hash64("abc"), int)

    def test_deterministic(self):
        self.assertEqual(stable_hash64("hello"), stable_hash64("hello"))

    def test_non_negative(self):
        for s in ("", "abc", "12345", "row_0001"):
            with self.subTest(s=s):
                self.assertGreaterEqual(stable_hash64(s), 0)

    def test_different_inputs_differ(self):
        self.assertNotEqual(stable_hash64("row0001"), stable_hash64("row0002"))

    def test_empty_string_accepted(self):
        h = stable_hash64("")
        self.assertIsInstance(h, int)
        self.assertGreaterEqual(h, 0)

    def test_fits_64_bits(self):
        self.assertLessEqual(stable_hash64("test"), 2**64 - 1)

    def test_unicode_accepted(self):
        h = stable_hash64("café")
        self.assertIsInstance(h, int)
        self.assertGreaterEqual(h, 0)


# ---------------------------------------------------------------------------
# _ensure_cols_exist
# ---------------------------------------------------------------------------

class TestEnsureColsExist(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    def test_present_no_error(self):
        _ensure_cols_exist(self.df, ["a"])

    def test_all_present_no_error(self):
        _ensure_cols_exist(self.df, ["a", "b"])

    def test_empty_seq_no_error(self):
        _ensure_cols_exist(self.df, [])

    def test_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            _ensure_cols_exist(self.df, ["c"])

    def test_partial_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            _ensure_cols_exist(self.df, ["a", "missing"])

    def test_error_names_missing_col(self):
        try:
            _ensure_cols_exist(self.df, ["z"])
        except KeyError as exc:
            self.assertIn("z", str(exc))


# ---------------------------------------------------------------------------
# _infer_format
# ---------------------------------------------------------------------------

class TestInferFormat(unittest.TestCase):

    def test_csv(self):
        self.assertEqual(_infer_format("data.csv"), "csv")

    def test_csv_gz(self):
        self.assertEqual(_infer_format("data.csv.gz"), "csv")

    def test_parquet(self):
        self.assertEqual(_infer_format("data.parquet"), "parquet")

    def test_pq(self):
        self.assertEqual(_infer_format("data.pq"), "parquet")

    def test_case_insensitive_csv(self):
        self.assertEqual(_infer_format("DATA.CSV"), "csv")

    def test_case_insensitive_parquet(self):
        self.assertEqual(_infer_format("DATA.PARQUET"), "parquet")

    def test_url_csv(self):
        self.assertEqual(_infer_format("https://example.com/data.csv"), "csv")

    def test_url_parquet(self):
        self.assertEqual(_infer_format("https://example.com/data.parquet"), "parquet")

    def test_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            _infer_format("data.xlsx")

    def test_no_extension_raises_value_error(self):
        with self.assertRaises(ValueError):
            _infer_format("data")


# ---------------------------------------------------------------------------
# _round_nonnegative
# ---------------------------------------------------------------------------

class TestRoundNonnegative(unittest.TestCase):

    def test_floor(self):
        self.assertEqual(_round_nonnegative(2.9, mode="floor"), 2)

    def test_ceil(self):
        self.assertEqual(_round_nonnegative(2.1, mode="ceil"), 3)

    def test_round_half_up_2_5(self):
        self.assertEqual(_round_nonnegative(2.5, mode="round"), 3)

    def test_round_half_up_3_5(self):
        self.assertEqual(_round_nonnegative(3.5, mode="round"), 4)

    def test_zero_floor(self):
        self.assertEqual(_round_nonnegative(0.0, mode="floor"), 0)

    def test_exact_integer_unchanged(self):
        self.assertEqual(_round_nonnegative(5.0, mode="ceil"), 5)

    def test_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            _round_nonnegative(-1.0, mode="floor")

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            _round_nonnegative(1.0, mode="truncate")  # type: ignore[arg-type]

    def test_return_type_is_int(self):
        self.assertIsInstance(_round_nonnegative(3.7, mode="floor"), int)


# ---------------------------------------------------------------------------
# resolve_sizes
# ---------------------------------------------------------------------------

class TestResolveSizes(unittest.TestCase):

    def test_absolute_sizes(self):
        result = resolve_sizes(prepared_rows=100, sizes=[10, 50],
                               fractions=None, percentages=None, rounding="round")
        self.assertEqual(result, [10, 50])

    def test_fractions(self):
        result = resolve_sizes(prepared_rows=100, sizes=None,
                               fractions=[0.1, 0.5], percentages=None, rounding="round")
        self.assertEqual(result, [10, 50])

    def test_percentages(self):
        result = resolve_sizes(prepared_rows=100, sizes=None,
                               fractions=None, percentages=[10.0, 50.0], rounding="round")
        self.assertEqual(result, [10, 50])

    def test_mixed_deduplicated(self):
        result = resolve_sizes(prepared_rows=100, sizes=[10],
                               fractions=[0.1], percentages=None, rounding="round")
        self.assertEqual(result, [10])

    def test_result_sorted(self):
        result = resolve_sizes(prepared_rows=100, sizes=[50, 10, 30],
                               fractions=None, percentages=None, rounding="round")
        self.assertEqual(result, sorted(result))

    def test_no_inputs_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_sizes(prepared_rows=100, sizes=None, fractions=None,
                          percentages=None, rounding="round")

    def test_exceeds_prepared_rows_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_sizes(prepared_rows=10, sizes=[100], fractions=None,
                          percentages=None, rounding="round")

    def test_fraction_out_of_range_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_sizes(prepared_rows=100, sizes=None, fractions=[1.5],
                          percentages=None, rounding="round")

    def test_percentage_out_of_range_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_sizes(prepared_rows=100, sizes=None, fractions=None,
                          percentages=[150.0], rounding="round")

    def test_fraction_too_small_rounds_to_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_sizes(prepared_rows=5, sizes=None, fractions=[0.001],
                          percentages=None, rounding="floor")

    def test_floor_rounding(self):
        result = resolve_sizes(prepared_rows=100, sizes=None, fractions=[0.359],
                               percentages=None, rounding="floor")
        self.assertEqual(result, [35])

    def test_ceil_rounding(self):
        result = resolve_sizes(prepared_rows=100, sizes=None, fractions=[0.351],
                               percentages=None, rounding="ceil")
        self.assertEqual(result, [36])

    def test_zero_size_allowed(self):
        result = resolve_sizes(prepared_rows=100, sizes=[0], fractions=None,
                               percentages=None, rounding="round")
        self.assertIn(0, result)

    def test_negative_prepared_rows_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_sizes(prepared_rows=-1, sizes=[1], fractions=None,
                          percentages=None, rounding="round")


# ---------------------------------------------------------------------------
# profile_dataframe
# ---------------------------------------------------------------------------

class TestProfileDataframe(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1, 2, 3, None],
            "b": ["x", "y", None, "z"],
            "c": [1.0, 2.0, 3.0, 4.0],
        })

    def test_returns_dict(self):
        self.assertIsInstance(profile_dataframe(self.df), dict)

    def test_rows_count(self):
        self.assertEqual(profile_dataframe(self.df)["rows"], 4)

    def test_cols_count(self):
        self.assertEqual(profile_dataframe(self.df)["cols"], 3)

    def test_missing_col_a(self):
        p = profile_dataframe(self.df)
        self.assertEqual(p["missing"]["a"], 1)

    def test_missing_col_b(self):
        p = profile_dataframe(self.df)
        self.assertEqual(p["missing"]["b"], 1)

    def test_missing_col_c_zero(self):
        p = profile_dataframe(self.df)
        self.assertEqual(p["missing"]["c"], 0)

    def test_numeric_describe_present(self):
        self.assertIn("numeric_describe", profile_dataframe(self.df))

    def test_nunique_non_numeric_present(self):
        p = profile_dataframe(self.df)
        self.assertIn("b", p["nunique_non_numeric"])

    def test_max_columns_limits_profiled(self):
        p = profile_dataframe(self.df, max_columns=1)
        self.assertEqual(p["cols"], 1)

    def test_max_columns_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            profile_dataframe(self.df, max_columns=0)

    def test_json_serialisable(self):
        p = profile_dataframe(self.df)
        json.dumps(p)  # must not raise

    def test_empty_dataframe_no_crash(self):
        empty = pd.DataFrame({"a": pd.Series([], dtype=int)})
        p = profile_dataframe(empty)
        self.assertEqual(p["rows"], 0)


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------

@_need_parquet
class TestLoadDataframe(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.df = _make_df(10)
        self.csv_path = str(Path(self.tmpdir.name) / "data.csv")
        self.pq_path = str(Path(self.tmpdir.name) / "data.parquet")
        _write_csv(self.df, Path(self.csv_path))
        _write_parquet(self.df, Path(self.pq_path))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_csv_length(self):
        self.assertEqual(len(load_dataframe(self.csv_path)), 10)

    def test_load_parquet_length(self):
        self.assertEqual(len(load_dataframe(self.pq_path)), 10)

    def test_auto_format_csv(self):
        df = load_dataframe(self.csv_path, input_format="auto")
        self.assertIsInstance(df, pd.DataFrame)

    def test_auto_format_parquet(self):
        df = load_dataframe(self.pq_path, input_format="auto")
        self.assertIsInstance(df, pd.DataFrame)

    def test_usecols_filters_columns(self):
        df = load_dataframe(self.csv_path, usecols=["x", "id"])
        self.assertIn("x", df.columns)
        self.assertNotIn("y", df.columns)

    def test_query_filters_rows(self):
        df = load_dataframe(self.csv_path, query="x < 5")
        self.assertEqual(len(df), 5)
        self.assertTrue((df["x"] < 5).all())

    def test_unsupported_format_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_dataframe(self.csv_path, input_format="xlsx")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# enforce_required_columns
# ---------------------------------------------------------------------------

class TestEnforceRequiredColumns(unittest.TestCase):

    def test_none_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        self.assertEqual(len(enforce_required_columns(df, required_cols=None)), 3)

    def test_empty_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        self.assertEqual(len(enforce_required_columns(df, required_cols=[])), 3)

    def test_drops_na_in_required(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        out = enforce_required_columns(df, required_cols=["a"])
        self.assertEqual(len(out), 2)
        self.assertFalse(out["a"].isna().any())

    def test_non_required_na_not_dropped(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        out = enforce_required_columns(df, required_cols=["a"])
        self.assertEqual(len(out), 3)

    def test_missing_col_raises_key_error(self):
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(KeyError):
            enforce_required_columns(df, required_cols=["nonexistent"])


# ---------------------------------------------------------------------------
# drop_duplicates_by_id
# ---------------------------------------------------------------------------

class TestDropDuplicatesById(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({"id": ["a", "b", "a", "c"], "val": [1, 2, 3, 4]})

    def test_keep_first_row_count(self):
        out = drop_duplicates_by_id(self.df, id_col="id", keep="first")
        self.assertEqual(len(out), 3)

    def test_keep_first_value(self):
        out = drop_duplicates_by_id(self.df, id_col="id", keep="first")
        self.assertEqual(int(out.loc[out["id"] == "a", "val"].iloc[0]), 1)

    def test_keep_last_value(self):
        out = drop_duplicates_by_id(self.df, id_col="id", keep="last")
        self.assertEqual(int(out.loc[out["id"] == "a", "val"].iloc[0]), 3)

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"id": ["x", "y", "z"], "val": [1, 2, 3]})
        self.assertEqual(len(drop_duplicates_by_id(df, id_col="id")), 3)

    def test_missing_col_raises_key_error(self):
        with self.assertRaises(KeyError):
            drop_duplicates_by_id(self.df, id_col="nonexistent")


# ---------------------------------------------------------------------------
# apply_keep_drop_columns
# ---------------------------------------------------------------------------

class TestApplyKeepDropColumns(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    def test_keep_cols(self):
        out = apply_keep_drop_columns(self.df, keep_cols=["a", "b"], drop_cols=None)
        self.assertEqual(list(out.columns), ["a", "b"])

    def test_drop_cols(self):
        out = apply_keep_drop_columns(self.df, keep_cols=None, drop_cols=["c"])
        self.assertNotIn("c", out.columns)

    def test_neither_unchanged(self):
        out = apply_keep_drop_columns(self.df, keep_cols=None, drop_cols=None)
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_both_raises_value_error(self):
        with self.assertRaises(ValueError):
            apply_keep_drop_columns(self.df, keep_cols=["a"], drop_cols=["b"])

    def test_missing_keep_col_raises_key_error(self):
        with self.assertRaises(KeyError):
            apply_keep_drop_columns(self.df, keep_cols=["z"], drop_cols=None)

    def test_missing_drop_col_raises_key_error(self):
        with self.assertRaises(KeyError):
            apply_keep_drop_columns(self.df, keep_cols=None, drop_cols=["z"])


# ---------------------------------------------------------------------------
# sample_hash_full
# ---------------------------------------------------------------------------

class TestSampleHashFull(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(50)

    def test_correct_n(self):
        self.assertEqual(len(sample_hash_full(self.df, n=10, id_col="id")), 10)

    def test_deterministic(self):
        out1 = sample_hash_full(self.df, n=10, id_col="id")
        out2 = sample_hash_full(self.df, n=10, id_col="id")
        pd.testing.assert_frame_equal(out1.reset_index(drop=True),
                                      out2.reset_index(drop=True))

    def test_n_zero_returns_empty(self):
        self.assertEqual(len(sample_hash_full(self.df, n=0, id_col="id")), 0)

    def test_n_equals_len(self):
        self.assertEqual(len(sample_hash_full(self.df, n=len(self.df), id_col="id")), len(self.df))

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_hash_full(self.df, n=-1, id_col="id")

    def test_n_exceeds_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_hash_full(self.df, n=len(self.df) + 1, id_col="id")

    def test_missing_id_col_raises_key_error(self):
        with self.assertRaises(KeyError):
            sample_hash_full(self.df, n=5, id_col="nonexistent")

    def test_result_subset_of_original(self):
        out = sample_hash_full(self.df, n=10, id_col="id")
        self.assertTrue(set(out["id"]).issubset(set(self.df["id"])))

    def test_index_reset(self):
        out = sample_hash_full(self.df, n=5, id_col="id")
        self.assertEqual(list(out.index), list(range(len(out))))


# ---------------------------------------------------------------------------
# sample_random
# ---------------------------------------------------------------------------

class TestSampleRandom(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(50)

    def test_correct_n(self):
        self.assertEqual(len(sample_random(self.df, n=10, seed=42)), 10)

    def test_deterministic_same_seed(self):
        out1 = sample_random(self.df, n=10, seed=42)
        out2 = sample_random(self.df, n=10, seed=42)
        pd.testing.assert_frame_equal(out1.reset_index(drop=True),
                                      out2.reset_index(drop=True))

    def test_different_seeds_differ(self):
        out1 = sample_random(self.df, n=10, seed=1)
        out2 = sample_random(self.df, n=10, seed=2)
        self.assertFalse((out1["x"].values == out2["x"].values).all())

    def test_n_zero_returns_empty(self):
        self.assertEqual(len(sample_random(self.df, n=0, seed=42)), 0)

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_random(self.df, n=-1, seed=42)

    def test_n_exceeds_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_random(self.df, n=len(self.df) + 1, seed=42)

    def test_no_replacement(self):
        out = sample_random(self.df, n=len(self.df), seed=42)
        self.assertEqual(len(out), out["id"].nunique())


# ---------------------------------------------------------------------------
# _linspace_positions
# ---------------------------------------------------------------------------

class TestLinspacePositions(unittest.TestCase):

    def test_n_zero_returns_empty(self):
        self.assertEqual(len(_linspace_positions(population=10, n=0)), 0)

    def test_n_equals_population(self):
        pos = _linspace_positions(population=5, n=5)
        np.testing.assert_array_equal(pos, [0, 1, 2, 3, 4])

    def test_correct_count(self):
        self.assertEqual(len(_linspace_positions(population=100, n=10)), 10)

    def test_sorted(self):
        pos = _linspace_positions(population=50, n=7)
        self.assertTrue((np.diff(pos) > 0).all())

    def test_unique(self):
        pos = _linspace_positions(population=50, n=10)
        self.assertEqual(len(pos), len(set(pos)))

    def test_within_range(self):
        pop = 30
        pos = _linspace_positions(population=pop, n=10)
        self.assertTrue((pos >= 0).all() and (pos < pop).all())

    def test_n_exceeds_population_raises_value_error(self):
        with self.assertRaises(ValueError):
            _linspace_positions(population=5, n=6)

    def test_negative_population_raises_value_error(self):
        with self.assertRaises(ValueError):
            _linspace_positions(population=-1, n=0)

    def test_negative_n_raises_value_error(self):
        with self.assertRaises(ValueError):
            _linspace_positions(population=10, n=-1)

    def test_n_one_single_position(self):
        self.assertEqual(len(_linspace_positions(population=100, n=1)), 1)


# ---------------------------------------------------------------------------
# sample_linspace
# ---------------------------------------------------------------------------

class TestSampleLinspace(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(50)

    def test_correct_n(self):
        self.assertEqual(len(sample_linspace(self.df, n=10)), 10)

    def test_deterministic(self):
        out1 = sample_linspace(self.df, n=10)
        out2 = sample_linspace(self.df, n=10)
        pd.testing.assert_frame_equal(out1, out2)

    def test_n_zero_returns_empty(self):
        self.assertEqual(len(sample_linspace(self.df, n=0)), 0)

    def test_index_reset(self):
        out = sample_linspace(self.df, n=5)
        self.assertEqual(list(out.index), list(range(5)))

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_linspace(self.df, n=-1)

    def test_n_exceeds_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_linspace(self.df, n=len(self.df) + 1)


# ---------------------------------------------------------------------------
# _largest_remainder_allocation
# ---------------------------------------------------------------------------

class TestLargestRemainderAlloc(unittest.TestCase):

    def _grp(self, counts):
        return pd.Series(counts, index=[f"g{i}" for i in range(len(counts))])

    def test_sums_to_n(self):
        self.assertEqual(int(_largest_remainder_allocation(self._grp([40, 40, 20]), n=10).sum()), 10)

    def test_all_nonnegative(self):
        alloc = _largest_remainder_allocation(self._grp([50, 30, 20]), n=15)
        self.assertTrue((alloc >= 0).all())

    def test_n_zero_sums_zero(self):
        self.assertEqual(int(_largest_remainder_allocation(self._grp([10, 20]), n=0).sum()), 0)

    def test_n_exceeds_total_raises_value_error(self):
        with self.assertRaises(ValueError):
            _largest_remainder_allocation(self._grp([10, 10]), n=25)

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            _largest_remainder_allocation(self._grp([10, 10]), n=-1)

    def test_capacity_respected(self):
        grp = self._grp([5, 5, 5])
        alloc = _largest_remainder_allocation(grp, n=6)
        self.assertTrue((alloc <= grp).all())


# ---------------------------------------------------------------------------
# _equal_allocation
# ---------------------------------------------------------------------------

class TestEqualAllocation(unittest.TestCase):

    def _grp(self, counts):
        return pd.Series(counts, index=[f"g{i}" for i in range(len(counts))])

    def test_sums_to_n(self):
        self.assertEqual(int(_equal_allocation(self._grp([100, 100, 100]), n=30).sum()), 30)

    def test_balanced(self):
        alloc = _equal_allocation(self._grp([100, 100, 100]), n=30)
        self.assertTrue((alloc == 10).all())

    def test_respects_capacity(self):
        alloc = _equal_allocation(self._grp([2, 100, 100]), n=12)
        self.assertLessEqual(int(alloc.iloc[0]), 2)
        self.assertEqual(int(alloc.sum()), 12)

    def test_n_zero_sums_zero(self):
        self.assertEqual(int(_equal_allocation(self._grp([10, 10]), n=0).sum()), 0)

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            _equal_allocation(self._grp([10, 10]), n=-1)


# ---------------------------------------------------------------------------
# allocate_strata
# ---------------------------------------------------------------------------

class TestAllocateStrata(unittest.TestCase):

    def _grp(self, counts):
        return pd.Series(counts, index=[f"g{i}" for i in range(len(counts))])

    def test_proportional_sums_to_n(self):
        self.assertEqual(int(allocate_strata(self._grp([60, 40]), n=10, allocation="proportional").sum()), 10)

    def test_equal_sums_to_n(self):
        self.assertEqual(int(allocate_strata(self._grp([100, 100]), n=10, allocation="equal").sum()), 10)

    def test_invalid_allocation_raises_value_error(self):
        with self.assertRaises(ValueError):
            allocate_strata(self._grp([10, 10]), n=5, allocation="random")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sample_stratified
# ---------------------------------------------------------------------------

class TestSampleStratified(unittest.TestCase):

    def setUp(self):
        rows = [{"g": g, "x": i, "id": f"g{g}r{i:02d}"}
                for g in range(4) for i in range(10)]
        self.df = pd.DataFrame(rows)

    def test_within_hash_proportional(self):
        out = sample_stratified(self.df, n=20, id_col="id", strata_cols=["g"],
                                within="hash", allocation="proportional", seed=0)
        self.assertEqual(len(out), 20)

    def test_within_random_proportional(self):
        out = sample_stratified(self.df, n=20, id_col=None, strata_cols=["g"],
                                within="random", allocation="proportional", seed=42)
        self.assertEqual(len(out), 20)

    def test_within_linspace_equal(self):
        out = sample_stratified(self.df, n=16, id_col=None, strata_cols=["g"],
                                within="linspace", allocation="equal", seed=0)
        self.assertEqual(len(out), 16)

    def test_all_groups_represented(self):
        out = sample_stratified(self.df, n=20, id_col=None, strata_cols=["g"],
                                within="random", allocation="proportional", seed=1)
        self.assertEqual(out["g"].nunique(), 4)

    def test_n_zero_returns_empty(self):
        out = sample_stratified(self.df, n=0, id_col=None, strata_cols=["g"],
                                within="random", allocation="proportional", seed=0)
        self.assertEqual(len(out), 0)

    def test_empty_strata_cols_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_stratified(self.df, n=10, id_col=None, strata_cols=[],
                              within="random", allocation="proportional", seed=0)

    def test_within_hash_no_id_col_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_stratified(self.df, n=10, id_col=None, strata_cols=["g"],
                              within="hash", allocation="proportional", seed=0)

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_stratified(self.df, n=-1, id_col=None, strata_cols=["g"],
                              within="random", allocation="proportional", seed=0)

    def test_index_reset(self):
        out = sample_stratified(self.df, n=8, id_col=None, strata_cols=["g"],
                                within="linspace", allocation="equal", seed=0)
        self.assertEqual(list(out.index), list(range(len(out))))

    def test_within_random_no_id_col_regression(self):
        """Regression: within='random' must not require id_col."""
        df = pd.DataFrame({"g": [0, 0, 1, 1], "x": [10, 11, 12, 13]})
        out = sample_stratified(df, n=2, id_col=None, strata_cols=["g"],
                                within="random", allocation="proportional", seed=7)
        self.assertEqual(len(out), 2)


# ---------------------------------------------------------------------------
# sample_hash_csv_stream
# ---------------------------------------------------------------------------

@_need_parquet
class TestSampleHashCsvStream(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.df = _make_df(100)
        self.csv_path = str(Path(self.tmpdir.name) / "big.csv")
        _write_csv(self.df, Path(self.csv_path))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_correct_n(self):
        out, _ = sample_hash_csv_stream(self.csv_path, n=20, id_col="id")
        self.assertEqual(len(out), 20)

    def test_stream_stats_total_rows(self):
        _, stats = sample_hash_csv_stream(self.csv_path, n=10, id_col="id")
        self.assertEqual(stats.rows_seen_total, 100)

    def test_stream_stats_eligible_rows(self):
        _, stats = sample_hash_csv_stream(self.csv_path, n=10, id_col="id")
        self.assertEqual(stats.rows_seen_eligible, 100)

    def test_deterministic(self):
        out1, _ = sample_hash_csv_stream(self.csv_path, n=20, id_col="id")
        out2, _ = sample_hash_csv_stream(self.csv_path, n=20, id_col="id")
        pd.testing.assert_frame_equal(out1, out2)

    def test_n_zero_returns_empty(self):
        out, _ = sample_hash_csv_stream(self.csv_path, n=0, id_col="id")
        self.assertEqual(len(out), 0)

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_hash_csv_stream(self.csv_path, n=-1, id_col="id")

    def test_chunksize_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            sample_hash_csv_stream(self.csv_path, n=10, id_col="id", chunksize=0)

    def test_required_cols_filters_na_rows(self):
        df2 = self.df.copy()
        df2.loc[0:9, "x"] = None
        csv2 = str(Path(self.tmpdir.name) / "na.csv")
        _write_csv(df2, Path(csv2))
        _, stats = sample_hash_csv_stream(csv2, n=10, id_col="id", required_cols=["x"])
        self.assertEqual(stats.rows_seen_eligible, 90)

    def test_usecols_must_include_id_col(self):
        with self.assertRaises(KeyError):
            sample_hash_csv_stream(self.csv_path, n=5, id_col="id", usecols=["x", "y"])

    def test_result_subset_of_original(self):
        out, _ = sample_hash_csv_stream(self.csv_path, n=10, id_col="id")
        self.assertTrue(set(out["id"]).issubset(set(self.df["id"])))


# ---------------------------------------------------------------------------
# write_dataset
# ---------------------------------------------------------------------------

@_need_parquet
class TestWriteDataset(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.df = _make_df(5)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writes_csv(self):
        out = Path(self.tmpdir.name) / "out.csv"
        write_dataset(self.df, output_path=out, fmt="csv")
        self.assertTrue(out.exists())
        self.assertEqual(len(pd.read_csv(out)), 5)

    def test_writes_parquet(self):
        out = Path(self.tmpdir.name) / "out.parquet"
        write_dataset(self.df, output_path=out, fmt="parquet")
        self.assertTrue(out.exists())
        self.assertEqual(len(pd.read_parquet(out)), 5)

    def test_creates_parent_directories(self):
        out = Path(self.tmpdir.name) / "sub" / "deep" / "out.csv"
        write_dataset(self.df, output_path=out, fmt="csv")
        self.assertTrue(out.exists())

    def test_unsupported_format_raises_value_error(self):
        out = Path(self.tmpdir.name) / "out.xlsx"
        with self.assertRaises(ValueError):
            write_dataset(self.df, output_path=out, fmt="xlsx")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

@_need_parquet
class TestWriteManifest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.manifest = Path(self.tmpdir.name) / "m.json"
        self.out_file = Path(self.tmpdir.name) / "out.parquet"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, **kw):
        defaults = dict(
            manifest_path=self.manifest,
            source="data.csv",
            input_format="csv",
            output_file=self.out_file,
            export_spec=ExportSpec(size=100, strategy="hash"),
            prep_params={"usecols": None},
            strategy_params={"id_col": "id"},
            rows={"input_prepared": 1000, "output": 100},
            columns=["id", "x"],
        )
        defaults.update(kw)
        write_manifest(**defaults)

    def test_creates_file(self):
        self._write()
        self.assertTrue(self.manifest.exists())

    def test_valid_json(self):
        self._write()
        data = json.loads(self.manifest.read_text())
        self.assertIsInstance(data, dict)

    def test_created_utc_present(self):
        self._write()
        self.assertIn("created_utc", json.loads(self.manifest.read_text()))

    def test_source_path(self):
        self._write()
        data = json.loads(self.manifest.read_text())
        self.assertEqual(data["source"]["path_or_url"], "data.csv")

    def test_export_size(self):
        self._write()
        data = json.loads(self.manifest.read_text())
        self.assertEqual(data["export"]["size"], 100)

    def test_columns_present(self):
        self._write()
        data = json.loads(self.manifest.read_text())
        self.assertEqual(data["columns"], ["id", "x"])

    def test_creates_parent_dirs(self):
        path = Path(self.tmpdir.name) / "sub" / "m.json"
        write_manifest(
            manifest_path=path, source="s", input_format="csv",
            output_file=self.out_file,
            export_spec=ExportSpec(size=1, strategy="hash"),
            prep_params={}, strategy_params={}, rows={}, columns=[],
        )
        self.assertTrue(path.exists())


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------

class TestWriteJson(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_creates_file(self):
        p = Path(self.tmpdir.name) / "out.json"
        write_json({"k": "v"}, output_path=p)
        self.assertTrue(p.exists())

    def test_valid_json_content(self):
        p = Path(self.tmpdir.name) / "out.json"
        write_json({"a": 1, "b": [2, 3]}, output_path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data["a"], 1)

    def test_unicode_written(self):
        p = Path(self.tmpdir.name) / "uni.json"
        write_json({"name": "café"}, output_path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "café")

    def test_creates_parent_dirs(self):
        p = Path(self.tmpdir.name) / "deep" / "sub" / "data.json"
        write_json({}, output_path=p)
        self.assertTrue(p.exists())


# ---------------------------------------------------------------------------
# ExportSpec
# ---------------------------------------------------------------------------

class TestExportSpec(unittest.TestCase):

    def test_fields(self):
        spec = ExportSpec(size=500, strategy="hash")
        self.assertEqual(spec.size, 500)
        self.assertEqual(spec.strategy, "hash")

    def test_frozen(self):
        spec = ExportSpec(size=100, strategy="random")
        with self.assertRaises(Exception):
            spec.size = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StreamStats
# ---------------------------------------------------------------------------

class TestStreamStats(unittest.TestCase):

    def test_fields(self):
        s = StreamStats(rows_seen_total=1000, rows_seen_eligible=950)
        self.assertEqual(s.rows_seen_total, 1000)
        self.assertEqual(s.rows_seen_eligible, 950)

    def test_frozen(self):
        s = StreamStats(rows_seen_total=1, rows_seen_eligible=1)
        with self.assertRaises(Exception):
            s.rows_seen_total = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser(unittest.TestCase):

    def setUp(self):
        self.parser = build_parser()

    def _parse(self, args):
        return self.parser.parse_args(args)

    def test_missing_input_exits(self):
        with self.assertRaises(SystemExit):
            self._parse(["--output-dir", "out"])

    def test_missing_output_dir_exits(self):
        with self.assertRaises(SystemExit):
            self._parse(["--input", "data.csv"])

    def test_default_strategy_hash(self):
        ns = self._parse(["--input", "x.csv", "--output-dir", "out", "--sizes", "10"])
        self.assertEqual(ns.strategy, "hash")

    def test_default_rounding_round(self):
        ns = self._parse(["--input", "x.csv", "--output-dir", "out", "--sizes", "10"])
        self.assertEqual(ns.rounding, "round")

    def test_default_format_parquet(self):
        ns = self._parse(["--input", "x.csv", "--output-dir", "out", "--sizes", "10"])
        self.assertEqual(ns.format, "parquet")

    def test_default_seed_42(self):
        ns = self._parse(["--input", "x.csv", "--output-dir", "out", "--sizes", "10"])
        self.assertEqual(ns.seed, 42)

    def test_sizes_parsed_as_ints(self):
        ns = self._parse(["--input", "x.csv", "--output-dir", "out", "--sizes", "100", "500"])
        self.assertEqual(ns.sizes, [100, 500])

    def test_fractions_parsed_as_floats(self):
        ns = self._parse(["--input", "x.csv", "--output-dir", "out", "--fractions", "0.1", "0.5"])
        self.assertAlmostEqual(ns.fractions[0], 0.1)


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------

class TestValidateArgs(unittest.TestCase):

    def _ns(self, **overrides):
        defaults = dict(
            input="data.csv", output_dir="out",
            sizes=[100], fractions=None, percentages=None,
            rounding="round", strategy="hash", id_col="id",
            seed=42, strata_cols=None, within="hash",
            allocation="proportional", usecols=None,
            required_cols=None, dedup=False, query=None,
            keep_cols=None, drop_cols=None, csv_sep=",",
            csv_encoding=None, csv_low_memory=False,
            format="parquet", write_manifest=False,
            write_profile=False, profile_max_columns=None,
            stream_hash_csv=False, chunksize=50_000,
            log_level="INFO", input_format="auto",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_valid_hash_no_error(self):
        _validate_args(self._ns())

    def test_no_sizes_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(sizes=None, fractions=None, percentages=None))

    def test_hash_no_id_col_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(strategy="hash", id_col=None))

    def test_stratified_no_strata_cols_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(strategy="stratified", strata_cols=None))

    def test_stratified_within_hash_no_id_col_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(strategy="stratified", strata_cols=["g"],
                                    within="hash", id_col=None))

    def test_dedup_no_id_col_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(dedup=True, id_col=None))

    def test_stream_non_hash_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(stream_hash_csv=True, strategy="random", sizes=[100]))

    def test_stream_no_id_col_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(stream_hash_csv=True, strategy="hash", id_col=None, sizes=[100]))

    def test_stream_with_fractions_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(stream_hash_csv=True, strategy="hash",
                                    id_col="id", sizes=None, fractions=[0.1]))

    def test_keep_and_drop_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(keep_cols=["a"], drop_cols=["b"]))

    def test_negative_sizes_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(sizes=[-1]))

    def test_fractions_out_of_range_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(sizes=None, fractions=[1.5]))

    def test_percentages_out_of_range_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_args(self._ns(sizes=None, percentages=[150.0]))


# ---------------------------------------------------------------------------
# _required_columns_for_pipeline
# ---------------------------------------------------------------------------

class TestRequiredColumnsForPipeline(unittest.TestCase):

    def _ns(self, **kw):
        defaults = dict(id_col=None, required_cols=None,
                        strategy="random", strata_cols=None, within="random")
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_no_columns_empty_list(self):
        self.assertEqual(_required_columns_for_pipeline(self._ns()), [])

    def test_id_col_included(self):
        self.assertIn("id", _required_columns_for_pipeline(self._ns(id_col="id")))

    def test_required_cols_included(self):
        result = _required_columns_for_pipeline(self._ns(required_cols=["price", "make"]))
        self.assertIn("price", result)
        self.assertIn("make", result)

    def test_strata_cols_included_for_stratified(self):
        result = _required_columns_for_pipeline(
            self._ns(strategy="stratified", strata_cols=["country"]))
        self.assertIn("country", result)

    def test_no_duplicates(self):
        result = _required_columns_for_pipeline(
            self._ns(id_col="id", required_cols=["id", "price"]))
        self.assertEqual(len(result), len(set(result)))


# ---------------------------------------------------------------------------
# main() — CLI integration
# ---------------------------------------------------------------------------

@_need_parquet
class TestMainCli(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        rows = [{"id": f"r{i:04d}", "group": i % 4, "x": float(i)} for i in range(200)]
        self.df = pd.DataFrame(rows)
        self.csv_path = str(Path(self.tmpdir.name) / "data.csv")
        _write_csv(self.df, Path(self.csv_path))
        self.out_dir = str(Path(self.tmpdir.name) / "out")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self, extra: list) -> int:
        return main(["--input", self.csv_path, "--output-dir", self.out_dir] + extra)

    def test_hash_strategy_returns_zero(self):
        rc = self._run(["--sizes", "50", "--strategy", "hash",
                        "--id-col", "id", "--format", "parquet"])
        self.assertEqual(rc, 0)

    def test_hash_produces_one_file(self):
        self._run(["--sizes", "50", "--strategy", "hash",
                   "--id-col", "id", "--format", "parquet"])
        self.assertEqual(len(list(Path(self.out_dir).glob("*.parquet"))), 1)

    def test_hash_correct_row_count(self):
        self._run(["--sizes", "50", "--strategy", "hash",
                   "--id-col", "id", "--format", "parquet"])
        out = pd.read_parquet(list(Path(self.out_dir).glob("*.parquet"))[0])
        self.assertEqual(len(out), 50)

    def test_random_strategy_returns_zero(self):
        rc = self._run(["--sizes", "40", "--strategy", "random",
                        "--seed", "42", "--format", "csv"])
        self.assertEqual(rc, 0)

    def test_stratified_strategy_returns_zero(self):
        rc = self._run(["--sizes", "40", "--strategy", "stratified",
                        "--strata-cols", "group", "--within", "linspace",
                        "--allocation", "proportional", "--format", "parquet"])
        self.assertEqual(rc, 0)

    def test_write_manifest_creates_json(self):
        self._run(["--sizes", "30", "--strategy", "hash", "--id-col", "id",
                   "--format", "parquet", "--write-manifest"])
        manifests = list(Path(self.out_dir).glob("*.manifest.json"))
        self.assertEqual(len(manifests), 1)
        data = json.loads(manifests[0].read_text())
        self.assertIn("created_utc", data)

    def test_multiple_sizes_multiple_files(self):
        self._run(["--sizes", "30", "60", "--strategy", "hash",
                   "--id-col", "id", "--format", "parquet"])
        self.assertEqual(len(list(Path(self.out_dir).glob("*.parquet"))), 2)

    def test_percentage_flag_correct_rows(self):
        self._run(["--percentages", "25", "--strategy", "hash",
                   "--id-col", "id", "--format", "parquet"])
        out = pd.read_parquet(list(Path(self.out_dir).glob("*.parquet"))[0])
        self.assertEqual(len(out), 50)  # 25% of 200

    def test_stream_hash_csv_returns_zero(self):
        rc = self._run(["--sizes", "30", "--strategy", "hash", "--id-col", "id",
                        "--stream-hash-csv", "--chunksize", "50", "--format", "parquet"])
        self.assertEqual(rc, 0)

    def test_dedup_removes_duplicates(self):
        df2 = self.df.copy()
        df2.loc[0, "id"] = df2.loc[1, "id"]
        csv2 = str(Path(self.tmpdir.name) / "dupes.csv")
        _write_csv(df2, Path(csv2))
        out_dir2 = str(Path(self.tmpdir.name) / "out2")
        main(["--input", csv2, "--output-dir", out_dir2,
              "--sizes", "100", "--strategy", "hash", "--id-col", "id",
              "--dedup", "--format", "parquet"])
        out = pd.read_parquet(list(Path(out_dir2).glob("*.parquet"))[0])
        self.assertEqual(out["id"].nunique(), len(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
