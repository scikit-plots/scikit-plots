import numpy as np

from .._file import humansize, humansize_vector


def test_scalar():
    assert humansize(1024) == "1 KB"
    assert humansize(1536) == "1.5 KB"

def test_vector():
    out = humansize_vector([1024, 2048])
    assert list(out) == ["1 KB", "2 KB"]

def test_numpy():
    arr = np.array([1024, 1024**2])
    out = humansize_vector(arr)
    assert out[1] == "1 MB"

def test_negative():
    assert humansize(-2048) == "-2 KB"
