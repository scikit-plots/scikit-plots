# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for ANNOY-FD-001 (guide 6.8).

The on-disk build file descriptor `_fd` used `0` as the "not open" sentinel and
tested it via truthiness (`if (_fd)`). File descriptor `0` is valid (e.g. when
stdin is closed), so an index whose on-disk file landed on fd 0 was treated as
"not open" — `unload()` skipped the close, leaking the descriptor. The sentinel
is now `-1` and every check compares against it.

This test closes fd 0 in a *subprocess* (so it cannot disrupt the pytest runner),
builds an on-disk index — which then opens onto fd 0 — and asserts that
`unload()` actually closes it. With the old truthiness the descriptor would still
be open after unload.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import sys, types, os, random, tempfile
    sys.modules["scikitplot.api"] = types.ModuleType("scikitplot.api")
    sys.modules["scikitplot.api"].__path__ = []
    import scikitplot
    from scikitplot.config import get_config
    scikitplot.get_config = get_config
    from scikitplot.cexternals._annoy import annoylib as A

    d = 5
    p = tempfile.mktemp(suffix=".ann")
    random.seed(0)
    os.close(0)                       # free fd 0 so open() will reuse it
    idx = A.AnnoyIndex(d, "euclidean")
    idx.on_disk_build(p)              # opens onto fd 0
    for i in range(60):
        idx.add_item(i, [random.random() for _ in range(d)])
    idx.build(10)
    used_fd0 = os.path.exists("/proc/self/fd/0")
    q = [random.random() for _ in range(d)]
    r1 = idx.get_nns_by_vector(q, 8)
    idx.unload()
    still_open = os.path.exists("/proc/self/fd/0")
    idx2 = A.AnnoyIndex(d, "euclidean"); idx2.load(p)
    r2 = idx2.get_nns_by_vector(q, 8); idx2.unload()
    os.remove(p)
    print(f"{int(used_fd0)} {int(still_open)} {int(r1 == r2)}")
    """
)


@pytest.mark.skipif(not os.path.exists("/proc/self/fd"), reason="needs /proc")
def test_fd_zero_is_closed_on_unload():
    env = {"PYTHONPATH": os.pathsep.join(sys.path), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    used_fd0, still_open, nns_match = map(int, proc.stdout.split()[-3:])
    if not used_fd0:
        pytest.skip("on_disk_build did not land on fd 0 in this environment")
    assert still_open == 0, "fd 0 leaked: unload() did not close it (sentinel bug)"
    assert nns_match == 1, "results changed after on-disk round-trip"
