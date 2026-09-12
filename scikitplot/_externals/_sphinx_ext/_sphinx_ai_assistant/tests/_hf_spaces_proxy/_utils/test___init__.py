# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Package-layout contract owned by :mod:`_hf_spaces_proxy._utils.__init__`."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import importlib

UTILS = RUNTIME_ROOT / "_hf_spaces_proxy" / "_utils"
EXPECTED_HELPERS = {
    "__init__.py",
    "_chat_contract.py",
    "_contribution_ledger.py",
    "_dataset_schema.py",
    "_rate_limit.py",
    "_provider_artifact.py",
    "_provider_artifact_lifecycle.py",
    "_resource_contract.py",
    "_resource_transport.py",
    "_redis_security.py",
    "_share_contract.py",
    "_share_store.py",
    "_shared_logic.py",
    "_storage.py",
    "_stub_model.py",
    "_telemetry.py",
    "_zip_artifact.py",
    "_zip_workspace.py",
    "deduplicate_dataset_v1.py",
}


def test_utils_init_is_import_light_and_exports_nothing_eagerly() -> None:
    mod = importlib.import_module(
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils"
    )
    assert tuple(mod.__all__) == ()


def test_utils_contains_complete_helper_package() -> None:
    assert {p.name for p in UTILS.glob("*.py")} == EXPECTED_HELPERS
