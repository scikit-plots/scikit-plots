# lean/tests/test__order.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The same files give the same reports, in any order of adding them
=================================================================

Reports list declarations, break ties and choose among equally short
paths by declaration number, and numbers follow the order files were
added. Files added out of label order are renumbered before the next
query, so every report equals :meth:`LeanIndex.from_folder`'s; files added
in order are never renumbered. Whole-project caches are keyed by a
generation every new file bumps, so a file without declarations (it can
change what the others see) is never answered from a stale cache.
"""

from __future__ import annotations

import itertools
import json
import random
import textwrap

import pytest

from .._index import LeanIndex

# Ties on purpose: two headline theorems per file, two equally short paths
# from `top` to a flagged leaf, a name declared in two files, a flagged
# instance, a statement shape shared across files and a file that cannot
# be read.
PROJECT = {
    "Proj/Base.lean": """
        def leafA : Nat := sorry
        def leafB : Nat := leafA
        instance flaky : Inhabited Nat := ⟨sorry⟩
        theorem shared (n : Nat) : n = n := rfl
    """,
    "Proj/Left.lean": """
        import Proj.Base
        def viaL : Nat := leafB
        theorem shapeL (m : Nat) : m = m := rfl
        theorem dup : True := trivial
    """,
    "Proj/Right.lean": """
        import Proj.Base
        def viaR : Nat := leafB
        theorem dup : True := trivial
    """,
    "Proj/Top.lean": """
        import Proj.Left
        import Proj.Right
        theorem top : viaR = viaL := sorry
        theorem top2 : True := trivial
    """,
    "Proj/Zed.lean": """
        import Proj.Base
        theorem zed (k : Nat) : k = k := rfl
    """,
}


def _write(root):
    paths = []
    for name, source in PROJECT.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
        paths.append((path, name))
    return paths


def _reports(index):
    found = index.discover(top=50)
    names = ["top", "top2", "viaL", "dup", "zed"]
    return json.dumps(
        [
            index.summary(max_items=50),
            found,
            [index.dossier(n, max_items=50) for n in names],
            [index.find(n) for n in names],
        ],
        sort_keys=True,
        default=str,
    )


@pytest.mark.parametrize("graph", ["python", "numpy", "scipy"])
def test_every_order_of_adding_files_gives_the_folder_reports(tmp_path, graph):
    if graph != "python":
        pytest.importorskip("numpy")
        if graph == "scipy":
            pytest.importorskip("scipy.sparse.csgraph")
    files = _write(tmp_path)
    want = _reports(LeanIndex.from_folder(tmp_path, graph=graph))
    orders = list(itertools.permutations(files))
    for order in random.Random(0).sample(orders, 12) + [orders[-1]]:
        index = LeanIndex(graph=graph)
        for i, (path, label) in enumerate(order):
            index.add_file(path, label=label)
            if i == 2:  # queries in between, then more files
                index.summary()
                index.discover(top=3)
                index.dossier("leafB")
        assert _reports(index) == want


def test_files_added_in_order_are_never_renumbered(tmp_path):
    _write(tmp_path)
    index = LeanIndex.from_folder(tmp_path)
    assert index._in_order
    names = list(index._name)
    index.summary()
    assert index._name == names


def test_a_file_without_declarations_refreshes_the_project_caches(tmp_path):
    # Run imports Proj.Mid; while Mid is missing Run's imports are unknown
    # and it may see `mk`. Mid, empty, makes them known: `mk` is not seen.
    proj = tmp_path / "Proj"
    proj.mkdir()
    (proj / "Gen.lean").write_text("def mk : IO Unit := addDecl 0\n", encoding="utf-8")
    (proj / "Run.lean").write_text("import Proj.Mid\n#eval mk\n", encoding="utf-8")
    index = LeanIndex()
    for name in ("Gen", "Run"):
        index.add_file(proj / f"{name}.lean", label=f"Proj/{name}.lean")
    assert index.summary()["generators"]["count"] == 1
    (proj / "Mid.lean").write_text("-- nothing\n", encoding="utf-8")
    index.add_file(proj / "Mid.lean", label="Proj/Mid.lean")
    fresh = LeanIndex.from_folder(tmp_path).summary()["generators"]
    assert index.summary()["generators"] == fresh
    assert fresh["count"] == 0


# Every attribute of an index, by what renumbering does with it. A new
# attribute fails this test until it is placed here, and _canonical
# renumbers it (per node, per file) or drops it (cache).
_PER_NODE = {
    "_name", "_kind", "_file", "_start", "_end", "_namespace", "_opens",
    "_ref_ptr", "_ref_data", "_ctx_of",
    "_flags", "_attributes", "_shape_of", "_private", "_claims", "_by_name",
    "_vacuous",
}  # fmt: skip
_PER_FILE = {
    "_labels", "_label_ids", "_paths", "_imports", "_file_lines", "_first_node",
    "_layer_of",
    "_problems",
}  # fmt: skip
_CACHES = {
    "_by_short", "_edges", "_reach", "_expanded", "_packed", "_follow",
    "_contexts", "_lines_cache", "_clash_cache", "_module_cache",
    "_import_files_cache", "_generator_cache", "_gaps", "_generation",
    "_clash_modules",
}  # fmt: skip
_ORDER_FREE = {
    "_kinds", "_intern", "_learned", "_skipped", "_in_order", "_commands", "_workers",
    "_ref_names", "_ref_ids", "_ctx_keys", "_ctx_ids", "_flat", "_layers",
    "_dialect", "_graph", "_max_bytes", "_max_depth", "_source_root",
    "_suffix", "_visibility",
}  # fmt: skip


def test_renumbering_knows_every_attribute(tmp_path):
    _write(tmp_path)
    index = LeanIndex.from_folder(tmp_path)
    index.summary()
    index.discover()
    index.dossier("top")
    known = _PER_NODE | _PER_FILE | _CACHES | _ORDER_FREE
    assert set(vars(index)) <= known, sorted(set(vars(index)) - known)
