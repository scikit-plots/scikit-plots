# tests/test_discovery_contract.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Contract guard for the server -> client config-discovery sync.

The HF Space proxy (``_hf_spaces_proxy/app.py``, ``GET /``) emits a JSON
manifest; the browser widget (``_static/ai-assistant.js``,
``_fetchProxyDatasetInfo``) consumes a subset of it. Those two sides are
hand-written independently, so a rename on either side silently breaks
auto-discovery with no error.

This test locks the live sync surface to ``_maintenance/discovery_contract.json``.
For every field the client consumes it asserts:

1. the server still emits the JSON key literal (present in ``app.py``), and
2. the client still reads the mapped access expression (present in the JS).

If either drifts, this test fails — turning "keep app.py and the JS in sync"
from a manual discipline into a verifiable CI gate, consistent with the
submodule's always-green-before-packaging policy.

The check is presence-based (substring of the source), which is robust to
formatting and needs no server boot, browser, or JS engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Submodule root = parent of the tests/ directory.
_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT = _ROOT / "_maintenance" / "discovery_contract.json"
_APP_PY = _ROOT / "_hf_spaces_proxy" / "app.py"
_JS = _ROOT / "_static" / "ai-assistant.js"


def _load_contract() -> dict:
    return json.loads(_CONTRACT.read_text(encoding="utf-8"))


def test_contract_file_exists_and_parses() -> None:
    """The canonical contract file is present and valid JSON."""
    assert _CONTRACT.is_file(), f"missing contract file: {_CONTRACT}"
    data = _load_contract()
    assert data.get("version") == 1
    assert isinstance(data.get("consumed_fields"), list)
    assert data["consumed_fields"], "contract lists no consumed fields"


@pytest.mark.skipif(not _APP_PY.is_file(), reason="server app.py not present in this layout")
def test_server_still_emits_every_consumed_key() -> None:
    """Each consumed field's server key literal must appear in app.py."""
    src = _APP_PY.read_text(encoding="utf-8")
    missing = [
        f["server_path"]
        for f in _load_contract()["consumed_fields"]
        if f["server_key_literal"] not in src
    ]
    assert not missing, (
        "app.py no longer emits these discovery keys the client depends on: "
        + ", ".join(missing)
        + " — update app.py or the contract together."
    )


@pytest.mark.skipif(not _JS.is_file(), reason="ai-assistant.js not present in this layout")
def test_client_still_reads_every_consumed_key() -> None:
    """Each consumed field's client access expression must appear in the JS."""
    src = _JS.read_text(encoding="utf-8")
    missing = [
        f["server_path"]
        for f in _load_contract()["consumed_fields"]
        if f["client_access_expr"] not in src
    ]
    assert not missing, (
        "ai-assistant.js no longer reads these discovery keys: "
        + ", ".join(missing)
        + " — the server->client sync has drifted."
    )


@pytest.mark.skipif(
    not (_APP_PY.is_file() and _JS.is_file()),
    reason="both server and client sources required",
)
def test_token_values_never_appear_in_manifest() -> None:
    """
    Defence-in-depth: the discovery manifest must expose token PRESENCE and
    TYPE only, never a value. The server keys the client reads are all
    ``*_type`` / ``*_repo`` / ``*_ready`` / ``*_enabled`` / ``*_mode`` — none
    is a raw ``token`` value field. This guards against a future edit that
    starts leaking a secret through discovery.
    """
    forbidden = ('"hf_token":', '"hf_write_token":', '"hf_dataset_token":')
    src = _APP_PY.read_text(encoding="utf-8")
    leaked = [f for f in forbidden if f in src]
    assert not leaked, f"manifest appears to expose a raw token value: {leaked}"
