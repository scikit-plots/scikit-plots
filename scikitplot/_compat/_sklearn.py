from __future__ import annotations

import sklearn

from ..externals._seaborn.utils import _version_predates


def learning_curve_params(val: dict | None):
    # dep use params 1.6
    if _version_predates(sklearn, "1.6.0"):
        return {"fit_params": val}
    return {"params": val}
