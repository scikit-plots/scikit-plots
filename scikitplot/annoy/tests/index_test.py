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

import pytest

# from annoy import AnnoyIndex
from scikitplot.annoy import AnnoyIndex

# Read only: the committed fixture ``test64.tree``. Every file a test writes
# goes to ``tmp_path`` (Rule 30), and nothing is downloaded.
HERE = os.path.dirname(__file__)  # "tests"


def test_not_found_tree():
    i = AnnoyIndex(10, "angular")
    with pytest.raises(IOError):
        i.load(f"{HERE}/nonexists.tree")


def test_binary_compatibility():
    # i = Index(10, "angular")
    # i.load(f"{HERE}/test.tree")
    # # This might change in the future if we change the search algorithm, but in that case let's update the test
    # result = i.get_nns_by_item(0, 10)
    # assert result == [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]  # 32-bit

    i = AnnoyIndex(10, "angular")
    i.load(f"{HERE}/test64.tree")
    # This might change in the future if we change the search algorithm, but in that case let's update the test
    result = i.get_nns_by_item(0, 10)
    assert result == [0, 85, 42, 11, 54, 38, 26, 53, 66, 3]


def test_load_unload():
    # Issue #108
    i = AnnoyIndex(10, "angular")
    for x in range(100000):
        i.load(f"{HERE}/test64.tree")
        i.unload()


def test_construct_load_destruct():
    for x in range(100000):
        i = AnnoyIndex(10, "angular")
        i.load(f"{HERE}/test64.tree")


def test_construct_destruct():
    for x in range(100000):
        i = AnnoyIndex(10, "angular")
        i.add_item(1000, [random.gauss(0, 1) for z in range(10)])


def test_save_twice(tmp_path):
    # Issue #100
    t = AnnoyIndex(10, "angular")
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for z in range(10)])
    t.build(10)
    t.save(str(tmp_path / "t1.ann"))
    t.save(str(tmp_path / "t2.ann"))


def test_load_save(tmp_path):
    # Issue #61
    i = AnnoyIndex(10, "angular")
    i.load(f"{HERE}/test64.tree")
    u = i.get_item(99)
    i.save(str(tmp_path / "i.tree"))
    v = i.get_item(99)
    assert u == v
    j = AnnoyIndex(10, "angular")
    j.load(f"{HERE}/test64.tree")
    w = i.get_item(99)
    assert u == w
    # Ensure specifying if prefault is allowed does not impact result
    j.save(str(tmp_path / "j.tree"), True)
    k = AnnoyIndex(10, "angular")
    k.load(str(tmp_path / "j.tree"), True)
    x = k.get_item(99)
    assert u == x
    k.save(str(tmp_path / "k.tree"), False)
    l = AnnoyIndex(10, "angular")
    l.load(str(tmp_path / "k.tree"), False)
    y = l.get_item(99)
    assert u == y


def test_save_without_build(tmp_path):
    t = AnnoyIndex(10, "angular")
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for z in range(10)])
    # Note: in earlier version, this was allowed (see eg #61)
    with pytest.raises(Exception):
        t.save(str(tmp_path / "x.tree"))


def test_unbuild_with_loaded_tree():
    i = AnnoyIndex(10, "angular")
    i.load(f"{HERE}/test64.tree")
    with pytest.raises(Exception):
        i.unbuild()


def test_seed():
    i = AnnoyIndex(10, "angular")
    i.load(f"{HERE}/test64.tree")
    i.set_seed(42)


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


def test_item_vector_after_save(tmp_path):
    # Issue #279
    a = AnnoyIndex(3, "angular")
    a.set_verbose(True)
    a.add_item(1, [1, 0, 0])
    a.add_item(2, [0, 1, 0])
    a.add_item(3, [0, 0, 1])
    a.build(-1)
    assert a.get_n_items() == 4
    assert a.get_item(3) == [0, 0, 1]
    assert set(a.get_nns_by_item(1, 999)) == set([1, 2, 3])
    a.save(str(tmp_path / "something.annoy"))
    assert a.get_n_items() == 4
    assert a.get_item(3) == [0, 0, 1]
    assert set(a.get_nns_by_item(1, 999)) == set([1, 2, 3])


def test_prefault():
    i = AnnoyIndex(10, "angular")
    i.load(f"{HERE}/test64.tree", prefault=True)
    result = i.get_nns_by_item(0, 10)
    # assert result == [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]  # 32-bit
    assert result == [0, 85, 42, 11, 54, 38, 26, 53, 66, 3]



def test_fail_save():
    t = AnnoyIndex(40, "angular")
    with pytest.raises(IOError):
        t.save("")


def test_overwrite_index(tmp_path):
    # Issue #335
    f = 40

    # Build the initial index
    t = AnnoyIndex(f, "angular")
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)
    t.build(10)
    t.save(str(tmp_path / "test.ann"))

    # Load index file
    t2 = AnnoyIndex(f, "angular")
    t2.load(str(tmp_path / "test.ann"))

    # Overwrite index file
    t3 = AnnoyIndex(f, "angular")
    for i in range(500):
        v = [random.gauss(0, 1) for z in range(f)]
        t3.add_item(i, v)
    t3.build(10)
    if os.name == "nt":
        # Can't overwrite on Windows
        with pytest.raises(IOError):
            t3.save(str(tmp_path / "test.ann"))
    else:
        t3.save(str(tmp_path / "test.ann"))
        # Get nearest neighbors
        v = [random.gauss(0, 1) for z in range(f)]
        t2.get_nns_by_vector(v, 1000)  # Should not crash


def test_get_n_trees():
    i = AnnoyIndex(10, "angular")
    i.load(f"{HERE}/test64.tree")
    assert i.get_n_trees() == 10


def test_write_failed():
    f = 40

    # Build the initial index
    t = AnnoyIndex(f, "angular")
    t.set_verbose(True)
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


def test_dimension_mismatch(tmp_path):
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    t.save(str(tmp_path / "test.annoy"))

    u = AnnoyIndex(200, "angular")
    with pytest.raises(IOError):
        u.load(str(tmp_path / "test.annoy"))
    u = AnnoyIndex(50, "angular")
    with pytest.raises(IOError):
        u.load(str(tmp_path / "test.annoy"))


def test_add_after_save(tmp_path):
    # 398
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    t.save(str(tmp_path / "test.annoy"))

    # Used to segfault:
    v = [random.gauss(0, 1) for z in range(100)]
    with pytest.raises(Exception):
        t.add_item(i, v)


def test_build_twice():
    # 420
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    # Used to segfault:
    with pytest.raises(Exception):
        t.build(10)


# python -c "import struct; print(struct.calcsize('P')*8)"
# Unit	Bytes
# 1 KB	10³
# 1 MB	10⁶
# 1 GB	10⁹
# 1 TB	10¹²
# 1 PB	10¹⁵
# 1 EB	10¹⁸
# required by test	9.22 EB
def test_very_large_index(tmp_path):
    # 388
    #
    # The saved file is about 3.2 GB. It was written into this directory, the
    # installed package, and never removed: every environment that ran the
    # suite kept one, and three of them at once filled the disk ("Unable to
    # write: No space left on device"). It now goes to pytest's temporary
    # directory and is removed before the test returns, pass or fail.
    f = 3
    dangerous_size = 2**31  # 2**63 2**31
    size_per_vector = 4 * (f + 3)
    n_vectors = int(dangerous_size / size_per_vector)
    m = AnnoyIndex(3, "angular")
    m.set_verbose(True)
    for i in range(100):
        m.add_item(n_vectors + i, [random.gauss(0, 1) for z in range(f)])
    n_trees = 10
    m.build(n_trees)
    path = os.fspath(tmp_path / "test_big.annoy")
    try:
        m.save(path)  # Raises on Windows

        # Sanity check size of index
        assert os.path.getsize(path) >= dangerous_size
        assert os.path.getsize(path) >= dangerous_size + 100e3

        # Sanity check number of trees
        assert m.get_n_trees() == n_trees
    finally:
        # The index is mapped from the file after save(); release it first,
        # because a mapped file cannot be removed on Windows.
        m.unload()
        if os.path.exists(path):
            os.remove(path)
