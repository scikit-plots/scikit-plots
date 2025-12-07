
import os
import random

# assert i.get_distance(0, 1) == pytest.approx(0.0, rel=1e-12, abs=0.0)
# npt.assert_allclose(i.get_distance(0, 1), 0, rtol=1e-12)
# npt.assert_allclose(i.get_distance(0, 1), 0.0, rtol=1e-12, atol=0.0)
import numpy.testing as npt

# from annoy import AnnoyIndex
from scikitplot.annoy import AnnoyIndex

HERE = os.path.dirname(__file__)  # "tests"


def test_serialize_index():
    f = 32

    index = AnnoyIndex(f, 'angular')

    for iteration in range(1000):
        vector = [random.gauss(0, 1) for _ in range(f)]
        index.add_item(iteration, vector)

    index.build(10)

    _ = index.serialize()


def test_deserialize_index():
    f = 32

    index = AnnoyIndex(f, 'angular')

    for iteration in range(1000):
        vector = [random.gauss(0, 1) for _ in range(f)]
        index.add_item(iteration, vector)

    index.build(10)

    data = index.serialize()

    index2 = AnnoyIndex(f, 'angular')

    index2.deserialize(data)

    index_item_count = index.get_n_items()

    assert index_item_count == index2.get_n_items()
    assert index.get_n_trees() == index2.get_n_trees()
    assert index.get_nns_by_item(0, index_item_count) == index2.get_nns_by_item(0, index_item_count)

def test_serialize_after_load():
    f = 32

    index1 = AnnoyIndex(f, 'angular')

    for iteration in range(1000):
        vector = [random.gauss(0, 1) for _ in range(f)]
        index1.add_item(iteration, vector)

    index1.build(10)

    save_path = f"{HERE}/test_serialize.tree"
    index1.save(save_path)

    index2 = AnnoyIndex(f, 'angular')
    index2.load(save_path)

    assert index1.serialize() == index2.serialize()
    assert index1.get_n_items() == index2.get_n_items()
    assert index1.get_n_trees() == index2.get_n_trees()
    assert index1.get_nns_by_item(0, index1.get_n_items()) == index2.get_nns_by_item(0, index1.get_n_items())
