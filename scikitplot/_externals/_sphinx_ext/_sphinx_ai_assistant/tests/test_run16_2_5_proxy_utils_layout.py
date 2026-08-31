# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 16.2.5 proxy helper-package layout regression guards."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
UTILS = PROXY / "_utils"

EXPECTED_ROOT_PY = {"app.py", "deduplicate_dataset.py"}
EXPECTED_HELPERS = {
    "__init__.py",
    "_chat_contract.py",
    "_contribution_ledger.py",
    "_dataset_schema.py",
    "_rate_limit.py",
    "_redis_security.py",
    "_share_contract.py",
    "_share_store.py",
    "_shared_logic.py",
    "_storage.py",
    "_stub_model.py",
    "_telemetry.py",
    "deduplicate_dataset_v1.py",
}


def test_proxy_root_keeps_only_supported_python_entrypoints() -> None:
    assert {p.name for p in PROXY.glob("*.py")} == EXPECTED_ROOT_PY


def test_proxy_utils_contains_complete_helper_package() -> None:
    assert {p.name for p in UTILS.glob("*.py")} == EXPECTED_HELPERS


def test_docker_copies_utils_as_one_runtime_package() -> None:
    docker = (PROXY / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=1000:1000 _utils ./_utils" in docker
    assert "COPY --chown=1000:1000 app.py deduplicate_dataset.py ./" in docker
    for name in EXPECTED_HELPERS - {"deduplicate_dataset_v1.py"}:
        assert (UTILS / name).is_file()


def test_top_level_hf_space_import_resolves_utils() -> None:
    code = "import app; from _utils import _shared_logic; print(app.PROXY_VERSION, _shared_logic.PROXY_VERSION)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROXY,
        check=True,
        capture_output=True,
        text=True,
    )
    left, right = proc.stdout.strip().split()
    assert left == right
    assert tuple(int(part) for part in left.split(".")) >= (6, 8, 0)


def test_direct_deduplicator_resolves_utils() -> None:
    code = "import deduplicate_dataset as d; print(d._SCHEMA_AVAILABLE)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROXY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "True"
