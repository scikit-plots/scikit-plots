# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Import/layout contract for :mod:`_cf_worker.__init__`."""
from __future__ import annotations
import importlib

def test_package_init_is_importable_without_side_effect_api():
    mod = importlib.import_module("scikitplot._externals._sphinx_ext._sphinx_ai_assistant._cf_worker")
    assert mod.__name__.endswith("._cf_worker")
    assert not getattr(mod, "__all__", ())
