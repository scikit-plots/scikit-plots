# lean/tests/test__graph_numpy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The ``graph="numpy"`` backend gives exactly the standard library's reports
==========================================================================

Every report (summary, dossiers before and after the edges are packed,
discovery) is compared as JSON between the two backends, on hand-written
projects and on generated ones; the vectorised breadth-first search is
compared with a FIFO search on random graphs. Skipped without NumPy.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import textwrap
from collections import deque

import pytest

np = pytest.importorskip("numpy")

from .._graph_numpy import breadth_first, use_counts  # noqa: E402
from .._index import LeanIndex  # noqa: E402
from .test__discovery import GENERATORS  # noqa: E402
from .test__index import PROJECT, VISIBLE  # noqa: E402


def _project(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


def _generated(seed, files=12, per_file=25):
    """A project of chained modules whose declarations name earlier ones."""
    rng = random.Random(seed)
    names: list[str] = []
    out = {}
    for f in range(files):
        lines = [f"import G.M{f - 1}"] if f else []
        if f > 1 and rng.random() < 0.5:
            lines.append(f"import G.M{rng.randrange(f - 1)}")
        lines.append(f"namespace G{f}")
        for d in range(per_file):
            name = f"t{f}_{d}"
            used = rng.sample(names, k=min(len(names), rng.randrange(4))) if names else []
            proof = " ∧ ".join(used) if used else "True"
            body = "sorry" if rng.random() < 0.02 else "trivial"
            kind = "instance" if rng.random() < 0.03 else "theorem"
            if kind == "instance":
                lines.append(f"instance i{f}_{d} : Inhabited Nat := {body}")
            else:
                lines.append(f"theorem {name} : {proof} := {body}")
            names.append(f"G{f}.{name}")
        lines.append(f"end G{f}")
        out[f"G/M{f}.lean"] = "\n".join(lines) + "\n"
    return out, names


def _reports(root, graph, targets):
    index = LeanIndex.from_folder(root, graph=graph)
    before = [index.dossier(t, max_items=1000) for t in targets]
    summary = index.summary(max_items=1000)
    found = index.discover(top=1000)
    after = [index.dossier(t, max_items=1000) for t in targets]  # packed, for numpy
    return json.dumps([before, summary, found, after], sort_keys=True)


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
    assert _reports(root, "numpy", targets) == _reports(root, "python", targets)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generated_projects_give_identical_reports(tmp_path, seed):
    files, names = _generated(seed)
    root = _project(tmp_path, files)
    targets = [names[-1], names[len(names) // 2], names[3]]
    assert _reports(root, "numpy", targets) == _reports(root, "python", targets)


def _fifo(indptr, codes, roots, follow=None):
    reached = dict.fromkeys(roots, (-1, -1))
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        if follow is not None and not follow[node]:
            continue
        for code in codes[indptr[node] : indptr[node + 1]].tolist():
            target, how = divmod(code, 4)
            if target not in reached:
                reached[target] = (node, how)
                queue.append(target)
    return reached


@pytest.mark.parametrize("seed", range(6))
def test_breadth_first_is_the_fifo_search(seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 300)
    rows = [sorted({rng.randrange(n) * 4 + rng.randrange(3) for _ in range(rng.randrange(6))}) for _ in range(n)]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=indptr[1:])
    codes = np.array([c for r in rows for c in r], dtype=np.int32)
    roots = [rng.randrange(n) for _ in range(rng.randrange(1, 4))]
    follow = np.array([rng.random() < 0.8 for _ in range(n)])
    for mask in (None, follow):
        got = breadth_first(indptr, codes, roots, n, follow=mask)
        want = _fifo(indptr, codes, roots, mask)
        assert list(got.items()) == list(want.items())  # same parents, same order


def test_use_counts_count_files_once():
    # Node 2 is named twice from file 0 (nodes 0, 1) and once from file 1.
    indptr = np.array([0, 1, 2, 3, 3])
    codes = np.array([8, 8, 8], dtype=np.int32)
    file_of = np.array([0, 0, 1, 1])
    in_degree, in_files = use_counts(indptr, codes, file_of, 4)
    assert in_degree.tolist() == [0, 0, 3, 0] and in_files.tolist() == [0, 0, 2, 0]


def test_packing_frees_the_per_declaration_arrays_and_a_new_file_unpacks(tmp_path):
    root = _project(tmp_path, PROJECT)
    index = LeanIndex.from_folder(root, graph="numpy")
    index.summary()  # nothing here adds declarations: no whole-graph step
    assert index._packed is None
    index.discover()  # counting uses needs every edge: packed
    assert index._packed is not None and index._edges == {}
    extra = tmp_path / "Extra.lean"
    extra.write_text("theorem extra : True := trivial\n", encoding="utf-8")
    index.add_file(extra, label="Extra.lean")
    assert index._packed is None
    assert index.dossier("main2")["flags_found"] == ["sorry"]


def test_an_empty_project(tmp_path):
    index = LeanIndex.from_folder(tmp_path, graph="numpy")
    assert index.summary()["declarations"] == 0
    assert index.discover()["headline_results"]["count"] == 0


def test_graph_is_validated():
    with pytest.raises(ValueError, match="graph"):
        LeanIndex(graph="scipy")


def test_without_numpy_the_option_says_what_to_install():
    code = (
        "import sys; sys.modules['numpy'] = None\n"
        "from scikitplot.lean import LeanIndex\n"
        "LeanIndex()  # the default needs nothing\n"
        "try:\n"
        "    LeanIndex(graph='numpy')\n"
        "except ImportError as exc:\n"
        "    assert 'pip install numpy' in str(exc), exc\n"
        "    print('ok')\n"
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]
