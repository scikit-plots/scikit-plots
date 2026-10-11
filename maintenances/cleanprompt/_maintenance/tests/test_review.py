"""
The reviewer must not present a stale record as the current state.

Notes
-----
**Developer notes.** The internal review of 2026-10-10 (CP-NEW-01) found
``check_contract`` failing on a stale evidence fingerprint while
``review_subsystem`` printed ``PASS`` and exited 0. Both directions are
asserted here by supplying the current fingerprint, so the tests hold whether
or not this checkout's evidence happens to be fresh.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import review_subsystem  # noqa: E402
from check_contract import MAINTENANCE, discover_repo, read_json  # noqa: E402

ROOT = discover_repo(Path(__file__).resolve())
RECORDED = read_json(
    ROOT.joinpath(*MAINTENANCE) / "_maintenance" / "EVIDENCE.json"
).get("runtime_tree_fingerprint")


def _run(monkeypatch, current):
    monkeypatch.setattr(review_subsystem, "tree_fingerprint", lambda _root: current)
    out = io.StringIO()
    with redirect_stdout(out):
        status = review_subsystem.main(["--repo", str(ROOT)])
    return status, out.getvalue(), review_subsystem.review(ROOT)


def test_a_stale_record_is_reported_and_exits_3(monkeypatch):
    status, text, report = _run(monkeypatch, "0" * 64)
    assert report["evidence_current"] is False
    assert status == review_subsystem.EXIT_STALE
    assert "STALE" in text


def test_a_current_record_exits_0(monkeypatch):
    status, text, report = _run(monkeypatch, RECORDED)
    assert report["evidence_current"] is True
    assert status == 0
    assert "STALE" not in text


def test_the_report_names_both_fingerprints(monkeypatch):
    _, _, report = _run(monkeypatch, "f" * 64)
    assert report["recorded_fingerprint"] == RECORDED
    assert report["current_fingerprint"] == "f" * 64
