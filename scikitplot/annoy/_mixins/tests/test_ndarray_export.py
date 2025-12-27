# tests/test_ndarray_export.py

import os
import pytest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _get_index_class():
    # Prefer public alias if you expose it
    try:
        from scikitplot.annoy import Index
        return Index
    except Exception:
        pass


@pytest.mark.skipif(np is None, reason="NumPy required for NDArrayMixin tests")
def test_iter_and_to_numpy_basic():
    Index = _get_index_class()

    idx = Index(3, "euclidean")
    idx.add_item(0, [1.0, 0.0, 0.0])
    idx.add_item(1, [0.0, 1.0, 0.0])
    idx.add_item(2, [0.0, 0.0, 1.0])
    idx.build(1)

    got = list(idx.iter_item_vectors(start=0, stop=3, with_ids=True))
    assert got[0][0] == 0
    assert list(got[0][1]) == [1.0, 0.0, 0.0]

    arr = idx.to_numpy(start=0, stop=3, dtype="float32")
    assert arr.shape == (3, 3)
    assert arr.dtype == np.float32


@pytest.mark.skipif(np is None, reason="NumPy required for NDArrayMixin tests")
def test_ids_must_be_sized_sequence():
    Index = _get_index_class()
    idx = Index(2, "euclidean")
    idx.add_item(0, [1.0, 1.0])

    # generator is not sized -> strict error
    gen = (i for i in [0])
    with pytest.raises(TypeError):
        idx.to_numpy(ids=gen)  # type: ignore


@pytest.mark.skipif(np is None, reason="NumPy required for NDArrayMixin tests")
def test_save_vectors_npy_roundtrip(tmp_path):
    Index = _get_index_class()

    idx = Index(2, "euclidean")
    for i in range(5):
        idx.add_item(i, [float(i), float(i + 1)])
    idx.build(1)

    out = tmp_path / "vectors.npy"
    path = idx.save_vectors_npy(str(out), start=0, stop=5, dtype="float32")
    assert os.path.exists(path)

    arr = np.load(path, mmap_mode="r")
    assert arr.shape == (5, 2)
    assert float(arr[0, 0]) == 0.0
    assert float(arr[4, 0]) == 4.0


def test_to_dataframe_optional_dependency():
    Index = _get_index_class()
    idx = Index(2, "euclidean")
    idx.add_item(0, [1.0, 2.0])

    pandas = pytest.importorskip("pandas")
    np_mod = pytest.importorskip("numpy")

    df = idx.to_dataframe(start=0, stop=1, include_id=True)
    assert list(df.columns)[0] == "id"
    assert df.shape[0] == 1


def test_partition_existing_ids_strict_behavior():
    Index = _get_index_class()
    idx = Index(2, "euclidean")
    idx.add_item(0, [1.0, 2.0])

    # If you added partition_existing_ids to NDArrayMixin:
    existing, missing = idx.partition_existing_ids([0, 999999])  # type: ignore
    assert 0 in existing
    assert 999999 in missing
