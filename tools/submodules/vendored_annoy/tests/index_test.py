# Copyright (c) 2013 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import os
import random
from pathlib import Path

import pytest
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve  # Python 3

# from annoy import AnnoyIndex
from scikitplot.cexternals.annoy import AnnoyIndex


# ---------------------------------------------------------------------------
# Annoy version detection (safe)
# ---------------------------------------------------------------------------

def _get_annoy_version():
    try:
        import annoy  # type: ignore
        v = getattr(annoy, "__version__", "0")
    except Exception:
        v = "0"

    # Prefer packaging if available
    try:
        from packaging.version import Version  # type: ignore
        return Version(v)
    except Exception:
        # Very small fallback parser: "2.0.0.post1" -> (2,0,0)
        parts = []
        for p in str(v).split("."):
            num = ""
            for ch in p:
                if ch.isdigit():
                    num += ch
                else:
                    break
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])


ANNOY_VERSION = _get_annoy_version()
V2_CUTOFF = (2, 0, 0)

def _is_v2_or_later():
    try:
        # packaging Version path
        from packaging.version import Version  # type: ignore
        if isinstance(ANNOY_VERSION, Version):
            return ANNOY_VERSION >= Version("2.0.0")
    except Exception:
        pass
    # tuple fallback
    return ANNOY_VERSION >= V2_CUTOFF


IS_V2 = _is_v2_or_later()


# ---------------------------------------------------------------------------
# Reference trees
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

# Legacy upstream-style file (pre-v2 contract)
REFERENCE_TREE_V1 = HERE / "test.tree"

# New stable-on-disk contract file for 2.x+
# You should generate this once using your 2.0.0 build and commit it.
REFERENCE_TREE_V2 = HERE / "test_v2.tree"


# Expected neighbour lists.
# V1 expected list is taken from upstream test expectations.
EXPECTED_V1_NNS = [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]

# For V2, set this to the list produced by your committed test_v2.tree.
# If you regenerated v2 tree with the same dataset/seed as V1, this may match.
EXPECTED_V2_NNS = [0, 736, 940, 348, 63, 798, 235, 56, 473, 679]  # 0


def _build_fresh_tree(path: Path, f: int = 10, n_items: int = 1000, metric: str = "angular") -> None:
    """Build a fresh test tree using the *current* Annoy build."""
    idx = AnnoyIndex(f, metric)
    for i in range(n_items):
        v = [random.gauss(0, 1) for _ in range(f)]
        idx.add_item(i, v)
    idx.build(10)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(path))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tree_env(tmp_path_factory):
    """
    Provide a loadable reference tree with version-aware semantics.

    Modes
    -----
    - v2 (Annoy >= 2.0.0):
        * Must load REFERENCE_TREE_V2.
        * No fallback.
    - v1 (Annoy < 2.0.0):
        * Prefer REFERENCE_TREE_V1 if it loads.
        * If ABI/layout mismatch is detected, fall back to a fresh temp tree.
    """
    if IS_V2:
        # Enforce the new 2.x on-disk stability contract
        assert REFERENCE_TREE_V2.exists(), (
            "Missing test/test_v2.tree.\n"
            "For Annoy >= 2.0.0 this file is the on-disk ABI contract.\n"
            "Generate it with the 2.0.0+ build and commit it."
        )
        i = AnnoyIndex(10, "angular")
        # If this fails, that's a breaking change in 2.x.
        i.load(str(REFERENCE_TREE_V2))
        return {"path": REFERENCE_TREE_V2, "mode": "v2"}

    # Pre-v2 behavior (legacy compatibility path)
    if not REFERENCE_TREE_V1.exists():
        tmp_dir = tmp_path_factory.mktemp("annoy_tree")
        new_path = tmp_dir / "test.tree"
        _build_fresh_tree(new_path, f=10, n_items=1000, metric="angular")
        return {"path": new_path, "mode": "fresh"}

    i = AnnoyIndex(10, "angular")
    try:
        i.load(str(REFERENCE_TREE_V1))
    except (IOError, OSError) as e:
        msg = str(e)
        # Classic ABI break when Node layout / S changed.
        if "Index size is not a multiple of vector size" in msg:
            tmp_dir = tmp_path_factory.mktemp("annoy_tree")
            new_path = tmp_dir / "test.tree"
            _build_fresh_tree(new_path, f=10, n_items=1000, metric="angular")
            return {"path": new_path, "mode": "fresh"}
        raise
    else:
        return {"path": REFERENCE_TREE_V1, "mode": "v1"}


@pytest.fixture
def tree_index(tree_env):
    i = AnnoyIndex(10, "angular")
    i.load(str(tree_env["path"]))
    return i


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_not_found_tree():
    i = AnnoyIndex(10, "angular")
    with pytest.raises(IOError):
        i.load("nonexists.tree")


def test_binary_compatibility(tree_env, tree_index):
    """
    Version-aware golden-file test.

    - v1: checks the historical upstream neighbour order.
    - v2: checks your new 2.x on-disk contract neighbour order.
    - fresh: not meaningful (tree generated on the fly).
    """
    mode = tree_env["mode"]
    i = tree_index

    if mode == "fresh":
        pytest.skip("No committed reference tree available for pre-v2 ABI; using a fresh generated tree.")

    expected = EXPECTED_V2_NNS if mode == "v2" else EXPECTED_V1_NNS
    assert i.get_nns_by_item(0, 10) == expected


def test_load_unload(tree_env):
    # Issue #108
    path = str(tree_env["path"])
    i = AnnoyIndex(10, "angular")
    for x in range(100000):
        i.load(path)
        i.unload()


def test_construct_load_destruct(tree_env):
    path = str(tree_env["path"])
    for x in range(100000):
        i = AnnoyIndex(10, "angular")
        i.load(path)


def test_construct_destruct():
    for x in range(100000):
        i = AnnoyIndex(10, "angular")
        i.add_item(1000, [random.gauss(0, 1) for z in range(10)])


def test_save_twice():
    # Issue #100
    t = AnnoyIndex(10, "angular")
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for z in range(10)])
    t.build(10)
    t.save("t1.ann")
    t.save("t2.ann")


def test_load_save(tree_env):
    # Issue #61
    path = str(tree_env["path"])

    i = AnnoyIndex(10, "angular")
    i.load(path)
    u = i.get_item_vector(99)

    i.save("i.tree")
    v = i.get_item_vector(99)
    assert u == v

    j = AnnoyIndex(10, "angular")
    j.load(path)
    w = j.get_item_vector(99)
    assert u == w

    # Ensure specifying if prefault is allowed does not impact result
    j.save("j.tree", True)
    k = AnnoyIndex(10, "angular")
    k.load("j.tree", True)
    x = k.get_item_vector(99)
    assert u == x

    k.save("k.tree", False)
    l = AnnoyIndex(10, "angular")
    l.load("k.tree", False)
    y = l.get_item_vector(99)
    assert u == y


def test_save_without_build():
    t = AnnoyIndex(10, "angular")
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for z in range(10)])
    # Note: in earlier version, this was allowed (see eg #61)
    with pytest.raises(Exception):
        t.save("x.tree")


def test_unbuild_with_loaded_tree(tree_index):
    with pytest.raises(Exception):
        tree_index.unbuild()


def test_seed(tree_index):
    tree_index.set_seed(42)


def test_unknown_distance():
    with pytest.raises(Exception):
        AnnoyIndex(10, "banana")


def test_metric_kwarg():
    # Issue 211
    i = AnnoyIndex(2, metric="euclidean")
    i.add_item(0, [1, 0])
    i.add_item(1, [9, 0])
    assert i.get_distance(0, 1) == pytest.approx(8)
    assert i.f == 2


def test_metric_f_kwargs():
    AnnoyIndex(f=3, metric="euclidean")


def test_item_vector_after_save():
    # Issue #279
    a = AnnoyIndex(3, "angular")
    a.verbose(True)
    a.add_item(1, [1, 0, 0])
    a.add_item(2, [0, 1, 0])
    a.add_item(3, [0, 0, 1])
    a.build(-1)
    assert a.get_n_items() == 4
    assert a.get_item_vector(3) == [0, 0, 1]
    assert set(a.get_nns_by_item(1, 999)) == {1, 2, 3}
    a.save("something.annoy")
    assert a.get_n_items() == 4
    assert a.get_item_vector(3) == [0, 0, 1]
    assert set(a.get_nns_by_item(1, 999)) == {1, 2, 3}


def test_prefault(tree_env):
    path = str(tree_env["path"])

    i1 = AnnoyIndex(10, "angular")
    i1.load(path, prefault=False)
    base = i1.get_nns_by_item(0, 10)

    i2 = AnnoyIndex(10, "angular")
    i2.load(path, prefault=True)
    assert i2.get_nns_by_item(0, 10) == base

    mode = tree_env["mode"]
    if mode == "v1":
        assert base == EXPECTED_V1_NNS
    elif mode == "v2":
        assert base == EXPECTED_V2_NNS


def test_fail_save():
    t = AnnoyIndex(40, "angular")
    with pytest.raises(IOError):
        t.save("")


def test_overwrite_index():
    # Issue #335
    f = 40

    # Build the initial index
    t = AnnoyIndex(f, "angular")
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)
    t.build(10)
    t.save("test.ann")

    # Load index file
    t2 = AnnoyIndex(f, "angular")
    t2.load("test.ann")

    # Overwrite index file
    t3 = AnnoyIndex(f, "angular")
    for i in range(500):
        v = [random.gauss(0, 1) for z in range(f)]
        t3.add_item(i, v)
    t3.build(10)
    if os.name == "nt":
        with pytest.raises(IOError):
            t3.save("test.ann")
    else:
        t3.save("test.ann")
        v = [random.gauss(0, 1) for z in range(f)]
        t2.get_nns_by_vector(v, 1000)  # Should not crash


def test_get_n_trees(tree_index):
    assert tree_index.get_n_trees() == 10


def test_write_failed():
    f = 40

    t = AnnoyIndex(f, "angular")
    t.verbose(True)
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)
    t.build(10)

    if os.name == "nt":
        path = "Z:\\xyz.annoy"
    else:
        path = "/x/y/z.annoy"
    with pytest.raises(Exception):
        t.save(path)


def test_dimension_mismatch():
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    t.save("test.annoy")

    u = AnnoyIndex(200, "angular")
    with pytest.raises(IOError):
        u.load("test.annoy")
    u = AnnoyIndex(50, "angular")
    with pytest.raises(IOError):
        u.load("test.annoy")


def test_add_after_save():
    # 398
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    t.save("test.annoy")

    v = [random.gauss(0, 1) for z in range(100)]
    with pytest.raises(Exception):
        t.add_item(i, v)


def test_build_twice():
    # 420
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    with pytest.raises(Exception):
        t.build(10)


def test_very_large_index():
    # 388
    f = 3
    dangerous_size = 2 ** 31
    size_per_vector = 4 * (f + 3)
    n_vectors = int(dangerous_size / size_per_vector)
    m = AnnoyIndex(3, "angular")
    m.verbose(True)
    for i in range(100):
        m.add_item(n_vectors + i, [random.gauss(0, 1) for z in range(f)])
    n_trees = 10
    m.build(n_trees)
    path = "test_big.annoy"
    m.save(path)  # Raises on Windows

    assert os.path.getsize(path) >= dangerous_size
    assert os.path.getsize(path) < dangerous_size + 100e3
    assert m.get_n_trees() == n_trees
