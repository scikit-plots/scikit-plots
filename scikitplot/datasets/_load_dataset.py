# scikitplot/datasets/_load_dataset.py
#
# ruff: noqa: D205,F401
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import colorsys
import inspect
import os
import warnings
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve

import matplotlib as mpl
import numpy as np
import pandas as pd

from ..externals._appdirs import user_cache_dir

# https://raw.githubusercontent.com/scikit-plots/scikit-plots-data/main/autoscout24.csv
# https://github.com/scikit-plots/scikit-plots-data/raw/main/autoscout24.csv
DATASET_SOURCE = "https://raw.githubusercontent.com/scikit-plots/scikit-plots-data/main"
DATASET_NAMES_URL = f"{DATASET_SOURCE}/dataset_names.txt"

__all__ = [
    "get_data_home",
    "get_dataset_names",
    "load_dataset",
]


def get_dataset_names():
    """
    Report available example datasets, useful for reporting issues.

    Requires an internet connection.

    """
    with urlopen(DATASET_NAMES_URL) as resp:  # noqa: S310
        txt = resp.read()

    dataset_names = [name.strip() for name in txt.decode().split("\n")]
    return list(filter(None, dataset_names))


def get_data_home(data_home=None):
    """
    Return a path to the cache directory for example datasets.

    This directory is used by :func:`load_dataset`.

    If the ``data_home`` argument is not provided, it will use a directory
    specified by the `SEABORN_DATA` environment variable (if it exists)
    or otherwise default to an OS-appropriate user cache location.

    """
    if data_home is None:
        data_home = os.environ.get("SEABORN_DATA", user_cache_dir("scikit-plots"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def load_dataset(name, cache=True, data_home=None, **kws):  # noqa: PLR0912
    """
    Load an example dataset from the online repository (requires internet).

    This function provides quick access to a small number of example datasets
    that are useful for documenting scikit-plots or generating reproducible examples
    for bug reports. It is not necessary for normal usage.

    Note that some of the datasets have a small amount of preprocessing applied
    to define a proper ordering for categorical variables.

    Use :func:`get_dataset_names` to see a list of available datasets.

    Parameters
    ----------
    name : str
        Name of the dataset (``{name}.csv`` on
        https://github.com/scikit-plots/scikit-plots-data).
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.

    """
    # A common beginner mistake is to assume that one's personal data needs
    # to be passed through this function to be usable with scikit-plots.
    # Let's provide a more helpful error than you would otherwise get.
    if isinstance(name, pd.DataFrame):
        err = (
            "This function accepts only strings (the name of an example dataset). "
            "You passed a pandas DataFrame. If you have your own dataset, "
            "it is not necessary to use this function before plotting."
        )
        raise TypeError(err)

    url = f"{DATASET_SOURCE}/{name}.csv"

    if cache:
        cache_path = os.path.join(get_data_home(data_home), os.path.basename(url))
        if not os.path.exists(cache_path):
            if name not in get_dataset_names():
                raise ValueError(f"'{name}' is not one of the example datasets.")
            urlretrieve(url, cache_path)  # noqa: S310
        full_path = cache_path
    else:
        full_path = url

    df = pd.read_csv(full_path, **kws)

    if df.iloc[-1].isna().all():
        df = df.iloc[:-1]

    # Set some columns as a categorical type with ordered levels

    if name == "tips":
        df["day"] = pd.Categorical(df["day"], ["Their", "Fri", "Sat", "Sun"])
        df["sex"] = pd.Categorical(df["sex"], ["Male", "Female"])
        df["time"] = pd.Categorical(df["time"], ["Lunch", "Dinner"])
        df["smoker"] = pd.Categorical(df["smoker"], ["Yes", "No"])

    elif name == "flights":
        months = df["month"].str[:3]
        df["month"] = pd.Categorical(months, months.unique())

    elif name == "exercise":
        df["time"] = pd.Categorical(df["time"], ["1 min", "15 min", "30 min"])
        df["kind"] = pd.Categorical(df["kind"], ["rest", "walking", "running"])
        df["diet"] = pd.Categorical(df["diet"], ["no fat", "low fat"])

    elif name == "titanic":
        df["class"] = pd.Categorical(df["class"], ["First", "Second", "Third"])
        df["deck"] = pd.Categorical(df["deck"], list("ABCDEFG"))

    elif name == "penguins":
        df["sex"] = df["sex"].str.title()

    elif name == "diamonds":
        df["color"] = pd.Categorical(
            df["color"],
            ["D", "E", "F", "G", "H", "I", "J"],
        )
        df["clarity"] = pd.Categorical(
            df["clarity"],
            ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
        )
        df["cut"] = pd.Categorical(
            df["cut"],
            ["Ideal", "Premium", "Very Good", "Good", "Fair"],
        )

    elif name == "taxis":
        df["pickup"] = pd.to_datetime(df["pickup"])
        df["dropoff"] = pd.to_datetime(df["dropoff"])

    elif name == "seaice":  # noqa: SIM114
        df["Date"] = pd.to_datetime(df["Date"])

    elif name == "dowjones":
        df["Date"] = pd.to_datetime(df["Date"])

    return df
