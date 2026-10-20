# scikitplot/feature_selection/tests/_helpers.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared, deterministic test frames for the feature_selection suite."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mixed_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a frame with one column of every family and several traits."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "customer_id": [f"c{i:05d}" for i in range(n)],
            "income": rng.lognormal(10.0, 1.0, n),
            "height": rng.normal(1.7, 0.1, n),
            "age": rng.integers(18, 80, n).astype(float),
            "kids": rng.integers(0, 4, n),
            "city": rng.choice(
                ["ankara", "izmir", "bursa", "van"], n, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "vip": rng.choice([True, False], n),
            "const": 1.0,
            "empty": np.nan,
            "grade": pd.Categorical(
                rng.choice(["lo", "mid", "hi"], n),
                categories=["lo", "mid", "hi"],
                ordered=True,
            ),
            "joined": pd.date_range("2020-01-01", periods=n, freq="D"),
            "churn": rng.integers(0, 2, n),
        }
    )
    frame.loc[::9, "age"] = np.nan
    return frame


def target_policy() -> dict:
    """Return a policy dict naming the target and the identifier."""
    return {"roles": {"target": ["churn"], "identifier": ["customer_id"]}}
