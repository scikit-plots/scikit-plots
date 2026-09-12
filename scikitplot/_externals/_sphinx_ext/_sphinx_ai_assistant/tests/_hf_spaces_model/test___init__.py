# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Import/layout contract for :mod:`_hf_spaces_model.__init__`."""
from __future__ import annotations
import importlib

def test_package_init_is_importable_without_eager_service_startup():
    mod = importlib.import_module("scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_model")
    assert mod.__name__.endswith("._hf_spaces_model")
    assert not getattr(mod, "__all__", ())
