# lean/tests/test__graph_scipy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The ``graph="scipy"`` backend gives exactly the standard library's reports
==========================================================================

The same checks as for ``graph="numpy"``: reports compared as JSON between
backends, and SciPy's breadth-first search compared with a FIFO search on
random graphs, with several roots in any order and with nodes not followed.
Skipped without NumPy and SciPy.
"""

from __future__ import annotations

import random
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy.sparse.csgraph")

from .._graph_scipy import breadth_first  # noqa: E402
from .._index import LeanIndex  # noqa: E402
from .test__discovery import GENERATORS  # noqa: E402
from .test__graph_numpy import _fifo, _generated, _project, _reports  # noqa: E402
from .test__index import PROJECT, VISIBLE  # noqa: E402


@pytest.mark.parametrize(
    ("files", "targets"),
    [
        (PROJECT, ["main", "main2", "Proj.bad"]),
        (VISIBLE, ["main", "mid"]),
        (GENERATORS, ["plain"]),
    ],
    ids=["project", "visibility", "generators"],
)
def test_hand_written_projects_give_identical_reports(tmp_path, files, targets):
    root = _project(tmp_path, files)
    assert _reports(root, "scipy", targets) == _reports(root, "python", targets)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generated_projects_give_identical_reports(tmp_path, seed):
    files, names = _generated(seed)
    root = _project(tmp_path, files)
    targets = [names[-1], names[len(names) // 2], names[3]]
    assert _reports(root, "scipy", targets) == _reports(root, "python", targets)


@pytest.mark.parametrize("seed", range(8))
def test_breadth_first_is_the_fifo_search(seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 300)
    rows = [
        sorted({rng.randrange(n) * 4 + rng.randrange(3) for _ in range(rng.randrange(6))})
        for _ in range(n)
    ]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=indptr[1:])
    codes = np.array([c for r in rows for c in r], dtype=np.int32)
    roots = rng.sample(range(n), k=min(n, rng.randrange(1, 6)))  # any order
    follow = np.array([rng.random() < 0.8 for _ in range(n)])
    # Ascending roots take SciPy's search (as LeanIndex passes them); any
    # other order the exact NumPy one. Both must be the FIFO search.
    for order in (sorted(roots), roots):
        for mask in (None, follow):
            got = breadth_first(indptr, codes, order, n, follow=mask)
            want = _fifo(indptr, codes, order, mask)
            assert list(got.items()) == list(want.items())  # nodes, order, parents, how


def test_no_roots_and_no_edges():
    indptr = np.zeros(4, dtype=np.int64)
    codes = np.zeros(0, dtype=np.int32)
    assert breadth_first(indptr, codes, [], 3) == {}
    assert breadth_first(indptr, codes, [2, 0], 3) == {2: (-1, -1), 0: (-1, -1)}


def test_without_scipy_the_option_says_what_to_install():
    code = (
        "import sys; sys.modules['scipy'] = None\n"
        "from scikitplot.lean import LeanIndex\n"
        "LeanIndex(graph='numpy')  # NumPy alone is enough for it\n"
        "try:\n"
        "    LeanIndex(graph='scipy')\n"
        "except ImportError as exc:\n"
        "    assert 'pip install numpy scipy' in str(exc), exc\n"
        "    print('ok')\n"
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]


def test_graph_names_the_three_backends():
    with pytest.raises(ValueError, match="'python', 'numpy', 'scipy'"):
        LeanIndex(graph="networkx")
