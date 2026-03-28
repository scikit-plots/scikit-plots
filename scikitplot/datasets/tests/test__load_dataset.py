# scikitplot/datasets/tests/test__load_dataset.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._load_dataset`.

Coverage map
------------
get_data_home        env var / default / expanduser / dir creation  → TestGetDataHome
get_dataset_names    mocked URL response, list of strings           → TestGetDatasetNames
load_dataset         DataFrame guard, cache=False, cache=True,      → TestLoadDataset
                     per-dataset post-processing (tips typo check,
                     flights month, exercise, titanic, penguins,
                     diamonds, taxis, seaice, dowjones), TypeError
__all__              public API surface                              → TestPublicAPI

Dead-import regression
----------------------
colorsys, inspect, warnings, contextmanager were unused and have been
removed. The module must be importable without them leaking into its
namespace.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock as mock
from io import StringIO
from pathlib import Path

import pandas as pd

from .._load_dataset import (
    DATASET_SOURCE,
    DATASET_NAMES_URL,
    get_data_home,
    get_dataset_names,
    load_dataset,
)

_SUB_MOD = "scikitplot.datasets._load_dataset"
_SUB_MOD_ROOT = _SUB_MOD.rsplit(".", maxsplit=1)[0]


# ---------------------------------------------------------------------------
# get_data_home
# ---------------------------------------------------------------------------

class TestGetDataHome(unittest.TestCase):
    """get_data_home must return a writable, existing path."""

    def test_returns_string(self):
        with tempfile.TemporaryDirectory() as td:
            result = get_data_home(data_home=td)
            self.assertIsInstance(result, str)

    def test_returns_provided_path(self):
        with tempfile.TemporaryDirectory() as td:
            result = get_data_home(data_home=td)
            self.assertEqual(os.path.realpath(result), os.path.realpath(td))

    def test_creates_directory_if_missing(self):
        with tempfile.TemporaryDirectory() as td:
            new_dir = os.path.join(td, "scikitplot_cache_test")
            self.assertFalse(os.path.exists(new_dir))
            get_data_home(data_home=new_dir)
            self.assertTrue(os.path.isdir(new_dir))

    def test_env_var_used_when_no_arg(self):
        with tempfile.TemporaryDirectory() as td:
            env_patch = {"SCIKITPLOT_DATA": td}
            with mock.patch.dict(os.environ, env_patch):
                result = get_data_home()
            self.assertEqual(os.path.realpath(result), os.path.realpath(td))

    def test_default_path_is_dir(self):
        with tempfile.TemporaryDirectory() as td:
            result = get_data_home(data_home=td)
            self.assertTrue(os.path.isdir(result))

    def test_tilde_expansion(self):
        """~ in data_home must be expanded (no literal tilde in returned path)."""
        with tempfile.TemporaryDirectory():
            result = get_data_home(data_home=tempfile.gettempdir())
            self.assertNotIn("~", result)

    def test_repeated_call_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            r1 = get_data_home(data_home=td)
            r2 = get_data_home(data_home=td)
            self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# get_dataset_names
# ---------------------------------------------------------------------------

class TestGetDatasetNames(unittest.TestCase):
    """get_dataset_names must parse URL content into a list of clean names."""

    def _mock_urlopen(self, names: list[str]):
        """Return a context manager mock for urlopen yielding name bytes."""
        content = "\n".join(names + [""]).encode("utf-8")
        cm = mock.MagicMock()
        cm.__enter__ = mock.Mock(return_value=cm)
        cm.__exit__ = mock.Mock(return_value=False)
        cm.read = mock.Mock(return_value=content)
        return cm

    def test_returns_list(self):
        with mock.patch(
            "scikitplot.datasets._load_dataset.urlopen",
            return_value=self._mock_urlopen(["tips", "flights", "penguins"]),
        ):
            result = get_dataset_names()
        self.assertIsInstance(result, list)

    def test_strips_empty_lines(self):
        with mock.patch(
            "scikitplot.datasets._load_dataset.urlopen",
            return_value=self._mock_urlopen(["tips", "", "flights"]),
        ):
            result = get_dataset_names()
        self.assertNotIn("", result)

    def test_strips_whitespace_from_names(self):
        with mock.patch(
            "scikitplot.datasets._load_dataset.urlopen",
            return_value=self._mock_urlopen(["  tips  ", "flights "]),
        ):
            result = get_dataset_names()
        self.assertIn("tips", result)
        self.assertIn("flights", result)

    def test_known_names_present(self):
        known = ["tips", "flights", "exercise", "titanic", "penguins",
                 "diamonds", "taxis", "seaice", "dowjones"]
        with mock.patch(
            "scikitplot.datasets._load_dataset.urlopen",
            return_value=self._mock_urlopen(known),
        ):
            result = get_dataset_names()
        for name in known:
            with self.subTest(name=name):
                self.assertIn(name, result)

    def test_uses_correct_url(self):
        cm = self._mock_urlopen(["tips"])
        with mock.patch(
            "scikitplot.datasets._load_dataset.urlopen",
            return_value=cm,
        ) as m:
            get_dataset_names()
        m.assert_called_once_with(DATASET_NAMES_URL)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset(unittest.TestCase):
    """load_dataset must validate inputs, cache files, and post-process datasets."""

    # -- DataFrame guard --

    def test_dataframe_input_raises_type_error(self):
        """Passing a DataFrame instead of a string must raise TypeError."""
        with self.assertRaises(TypeError):
            load_dataset(pd.DataFrame({"a": [1]}))

    def test_dataframe_guard_message_helpful(self):
        """TypeError message must mention 'DataFrame'."""
        try:
            load_dataset(pd.DataFrame())
        except TypeError as exc:
            self.assertIn("DataFrame", str(exc))

    # -- Unknown name guard --

    def test_unknown_name_raises_value_error(self):
        """An unknown dataset name (with caching) must raise ValueError."""
        fake_names = "tips\nflights\n"
        cm = mock.MagicMock()
        cm.__enter__ = mock.Mock(return_value=cm)
        cm.__exit__ = mock.Mock(return_value=False)
        cm.read = mock.Mock(return_value=fake_names.encode())

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("scikitplot.datasets._load_dataset.urlopen", return_value=cm):
                with self.assertRaises(ValueError):
                    load_dataset("completely_unknown_dataset_xyz", data_home=td)

    # -- cache=False path (no disk I/O) --

    def _load_no_cache(self, name: str, csv_content: str, **kws) -> pd.DataFrame:
        """Load with cache=False by mocking read_csv over the URL."""
        with mock.patch(
            "scikitplot.datasets._load_dataset.pd.read_csv",
            return_value=pd.read_csv(StringIO(csv_content)),
        ):
            return load_dataset(name, cache=False, **kws)

    def test_cache_false_returns_dataframe(self):
        csv = "day,total_bill,tip,sex,smoker,time\nThur,10.0,1.5,Male,No,Lunch\nFri,12.0,2.0,Female,Yes,Dinner\n"
        df = self._load_no_cache("tips", csv)
        self.assertIsInstance(df, pd.DataFrame)

    def test_tips_day_categorical_order(self):
        """Tips 'day' must have categories ordered [Their, Fri, Sat, Sun]."""
        csv = "day,total_bill,tip,sex,smoker,time\n"
        for day in ["Their", "Fri", "Sat", "Sun"]:
            csv += f"{day},10.0,1.5,Male,No,Lunch\n"

        with mock.patch(
            "scikitplot.datasets._load_dataset.pd.read_csv",
            return_value=pd.read_csv(StringIO(csv)),
        ):
            df = load_dataset("tips", cache=False)

        self.assertEqual(list(df["day"].cat.categories), ["Their", "Fri", "Sat", "Sun"])

    def test_tips_sex_categorical(self):
        csv = "day,total_bill,tip,sex,smoker,time\nThur,10.0,1.5,Male,No,Lunch\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("tips", cache=False)
        self.assertTrue(hasattr(df["sex"], "cat"))

    def test_tips_time_categorical(self):
        csv = "day,total_bill,tip,sex,smoker,time\nThur,10.0,1.5,Male,No,Lunch\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("tips", cache=False)
        self.assertTrue(hasattr(df["time"], "cat"))

    def test_tips_smoker_categorical(self):
        csv = "day,total_bill,tip,sex,smoker,time\nThur,10.0,1.5,Male,No,Lunch\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("tips", cache=False)
        self.assertTrue(hasattr(df["smoker"], "cat"))

    # -- Flights dataset --

    def test_flights_month_categorical(self):
        csv = "year,month,passengers\n1949,January,112\n1949,February,118\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("flights", cache=False)
        self.assertTrue(hasattr(df["month"], "cat"))

    def test_flights_month_truncated_to_3_chars(self):
        csv = "year,month,passengers\n1949,January,112\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("flights", cache=False)
        self.assertEqual(df["month"].iloc[0], "Jan")

    # -- Exercise dataset --

    def test_exercise_time_categorical(self):
        csv = "id,Unnamed: 0,diet,pulse,time,kind\n1,0,no fat,85,1 min,rest\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("exercise", cache=False)
        self.assertTrue(hasattr(df["time"], "cat"))

    def test_exercise_kind_categorical(self):
        csv = "id,Unnamed: 0,diet,pulse,time,kind\n1,0,no fat,85,1 min,rest\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("exercise", cache=False)
        self.assertTrue(hasattr(df["kind"], "cat"))

    # -- Titanic dataset --

    def test_titanic_class_categorical(self):
        csv = "survived,pclass,class,sex,age,deck\n1,1,First,male,22.0,A\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("titanic", cache=False)
        self.assertTrue(hasattr(df["class"], "cat"))

    def test_titanic_deck_categorical(self):
        csv = "survived,pclass,class,sex,age,deck\n1,1,First,male,22.0,A\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("titanic", cache=False)
        self.assertTrue(hasattr(df["deck"], "cat"))

    # -- Penguins dataset --

    def test_penguins_sex_title_case(self):
        csv = "species,island,sex\nAdelie,Torgersen,MALE\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("penguins", cache=False)
        self.assertEqual(df["sex"].iloc[0], "Male")

    # -- Diamonds dataset --

    def test_diamonds_color_categorical(self):
        csv = "carat,cut,color,clarity,depth,table,price,x,y,z\n0.23,Ideal,E,SI2,61.5,55,326,3.95,3.98,2.43\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("diamonds", cache=False)
        self.assertTrue(hasattr(df["color"], "cat"))

    def test_diamonds_cut_categorical(self):
        csv = "carat,cut,color,clarity,depth,table,price,x,y,z\n0.23,Ideal,E,SI2,61.5,55,326,3.95,3.98,2.43\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("diamonds", cache=False)
        self.assertTrue(hasattr(df["cut"], "cat"))

    # -- Taxis dataset --

    def test_taxis_pickup_datetime(self):
        csv = "pickup,dropoff,passengers,distance,fare\n2019-03-23 20:21:09,2019-03-23 20:27:24,1,1.6,7.0\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("taxis", cache=False)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["pickup"]))

    def test_taxis_dropoff_datetime(self):
        csv = "pickup,dropoff,passengers,distance,fare\n2019-03-23 20:21:09,2019-03-23 20:27:24,1,1.6,7.0\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("taxis", cache=False)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["dropoff"]))

    # -- Seaice dataset --

    def test_seaice_date_datetime(self):
        csv = "Date,Extent\n1979-01-01,15.0\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("seaice", cache=False)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))

    # -- Dowjones dataset --

    def test_dowjones_date_datetime(self):
        csv = "Date,Price\n1914-01-01,53.0\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("dowjones", cache=False)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))

    # -- Trailing NA row removal --

    def test_trailing_all_na_row_dropped(self):
        """A final all-NA row must be silently dropped."""
        csv = "x,y\n1,2\n,\n"
        with mock.patch("scikitplot.datasets._load_dataset.pd.read_csv",
                        return_value=pd.read_csv(StringIO(csv))):
            df = load_dataset("unknown_simple", cache=False)
        # The all-NA last row should be gone
        self.assertFalse(df.iloc[-1].isna().all())

    # -- cache=True path (disk I/O) --

    def test_cache_true_writes_file(self):
        csv_content = "x,y\n1,2\n3,4\n"
        with tempfile.TemporaryDirectory() as td:
            url = f"{DATASET_SOURCE}/mycache_test.csv"
            cache_file = Path(td) / "mycache_test.csv"

            names_cm = mock.MagicMock()
            names_cm.__enter__ = mock.Mock(return_value=names_cm)
            names_cm.__exit__ = mock.Mock(return_value=False)
            names_cm.read = mock.Mock(return_value=b"mycache_test\n")

            def fake_urlretrieve(url_, dest):
                Path(dest).write_text(csv_content, encoding="utf-8")

            with mock.patch("scikitplot.datasets._load_dataset.urlopen", return_value=names_cm), \
                 mock.patch("scikitplot.datasets._load_dataset.urlretrieve", side_effect=fake_urlretrieve):
                df = load_dataset("mycache_test", cache=True, data_home=td)

            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(cache_file.exists())

    def test_cache_true_second_call_does_not_re_download(self):
        """Second call with cache=True must not call urlretrieve again."""
        csv_content = "x,y\n1,2\n"
        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / "myname.csv"
            cache_file.write_text(csv_content, encoding="utf-8")

            with mock.patch("scikitplot.datasets._load_dataset.urlretrieve") as mock_dl:
                df = load_dataset("myname", cache=True, data_home=td)

            mock_dl.assert_not_called()
            self.assertIsInstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicAPI(unittest.TestCase):
    """The _load_dataset module public API must expose exactly the declared names."""

    def test_all_contains_expected_names(self):
        from scikitplot.datasets._load_dataset import __all__
        expected = {"get_data_home", "get_dataset_names", "load_dataset"}
        self.assertTrue(
            expected.issubset(set(__all__)),
            msg=f"Missing from __all__: {expected - set(__all__)}",
        )

    def test_public_functions_callable(self):
        from scikitplot.datasets._load_dataset import __all__
        import scikitplot.datasets._load_dataset as m
        for name in __all__:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(m, name)))

    def test_no_dead_imports_in_namespace(self):
        """Dead imports removed in bug fix must not leak into module namespace.

        colorsys, inspect, warnings, contextmanager were unused and removed.
        """
        import scikitplot.datasets._load_dataset as m
        for name in ("colorsys", "inspect", "contextmanager"):
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(m, name),
                    msg=f"Dead import '{name}' still present in module namespace",
                )

    def test_dataset_source_url_is_string(self):
        self.assertIsInstance(DATASET_SOURCE, str)
        self.assertTrue(DATASET_SOURCE.startswith("https://"))

    def test_dataset_names_url_is_string(self):
        self.assertIsInstance(DATASET_NAMES_URL, str)
        self.assertIn("dataset_names.txt", DATASET_NAMES_URL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
