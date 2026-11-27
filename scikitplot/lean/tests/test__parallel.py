# lean/tests/test__parallel.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Names resolved in worker processes give exactly the edges resolved here
========================================================================

With ``workers > 1`` on Linux, a step that needs every edge resolves them
in forked workers, in runs of consecutive declarations taken back in order.
Every report must be the one ``workers=1`` gives; where forking is not
used (another platform, other threads, too few declarations), the index
resolves the names itself.
"""

from __future__ import annotations

import json
import sys
import threading

import pytest

from .. import _parallel
from .._index import LeanIndex
from .test__order import PROJECT, _write

linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="forked workers are used on Linux"
)


def _reports(index):
    return json.dumps(
        [
            index.summary(max_items=50),
            index.discover(top=50),
            [index.dossier(n, max_items=50) for n in ("top", "top2", "viaL", "zed")],
        ],
        sort_keys=True,
        default=str,
    )


@linux
@pytest.mark.parametrize("graph", ["python", "numpy"])
def test_workers_give_the_reports_one_process_gives(tmp_path, monkeypatch, graph):
    if graph == "numpy":
        pytest.importorskip("numpy")
    _write(tmp_path)
    want = _reports(LeanIndex.from_folder(tmp_path, graph=graph))
    monkeypatch.setattr(_parallel, "MIN_NODES", 1)  # a small project forks too
    calls = []
    resolve = _parallel.resolve
    monkeypatch.setattr(
        _parallel, "resolve", lambda *a: calls.append(len(a[1])) or resolve(*a)
    )
    index = LeanIndex.from_folder(tmp_path, graph=graph, workers=2)
    if not _parallel.usable(2, 1):  # e.g. a test runner with threads
        pytest.skip("forking is not safe in this process")
    assert _reports(index) == want
    assert calls and calls[0] == len(index._name)  # every edge came from workers


def test_forking_is_used_only_where_it_is_safe_and_pays(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert not _parallel.usable(1, 10**6)  # one worker: here
    assert not _parallel.usable(2, _parallel.MIN_NODES - 1)  # too few to pay
    monkeypatch.setattr(sys, "platform", "darwin")
    assert not _parallel.usable(2, 10**6)  # fork is unsafe on macOS
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(threading, "active_count", lambda: 2)
    assert not _parallel.usable(2, 10**6)  # another thread: no fork


def test_runs_split_nodes_into_consecutive_runs():
    assert _parallel._runs([], 3) == []
    assert _parallel._runs([0, 1, 2, 3, 4], 2) == [(0, 2), (2, 4), (4, 5)]
    assert _parallel._runs([0, 1, 5, 6, 9], 10) == [(0, 2), (5, 7), (9, 10)]


def test_workers_is_validated():
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="workers"):
            LeanIndex(workers=bad)
    assert LeanIndex(workers=None)._workers >= 1
