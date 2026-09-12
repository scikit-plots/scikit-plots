# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Public export contract for :mod:`_hf_spaces_proxy._providers.__init__`."""
from __future__ import annotations
import importlib

def test_provider_package_exports_are_unique_and_resolvable():
    mod = importlib.import_module("scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers")
    names = tuple(mod.__all__)
    assert names
    assert len(names) == len(set(names))
    assert all(hasattr(mod, name) for name in names)

def test_provider_package_does_not_export_private_credentials():
    mod = importlib.import_module("scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers")
    lowered = {name.lower() for name in mod.__all__}
    assert not any(token in name for name in lowered for token in ("secret", "token", "password", "credential"))
