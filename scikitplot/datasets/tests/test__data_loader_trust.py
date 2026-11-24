"""Pickle-family files load only when trusted (``scikitplot.datasets``)."""

from __future__ import annotations

import hashlib
import io
import pickle

import numpy as np
import pandas as pd
import pytest

from .. import _data_loader as dl


class _Marker:
    """Unpickling this writes a file: proof that code from the file ran."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (_touch, (self.path,))


def _touch(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ran")
    return "loaded"


@pytest.fixture
def hostile(tmp_path):
    marker = tmp_path / "marker"
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps(_Marker(str(marker))))
    return path, marker


LOADERS = [dl.load_pickle, dl.load_joblib, dl.load_pandas_pickle]


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("trusted", [False, None, "true"])
def test_untrusted_is_refused_before_reading(hostile, loader, trusted):
    path, marker = hostile
    with pytest.raises(ValueError, match=r"load_data.*trusted=True"):
        loader(path, trusted=trusted)
    assert not marker.exists()


def test_cloudpickle_is_gated_too(hostile):
    pytest.importorskip("cloudpickle")
    path, marker = hostile
    with pytest.raises(ValueError, match="disabled by default"):
        dl.load_cloudpickle(path)
    assert not marker.exists()


@pytest.mark.parametrize("entry", ["load_data", "load_file_from_path"])
def test_public_entry_points_are_gated(hostile, entry):
    path, marker = hostile
    with pytest.raises((ValueError, RuntimeError), match="trusted=True"):
        getattr(dl, entry)(str(path))
    assert not marker.exists()


def test_digest_mismatch_is_refused_before_unpickling(hostile):
    path, marker = hostile
    with pytest.raises(ValueError, match="does not match"):
        dl.load_pickle(path, trusted=True, expected_sha256="0" * 64)
    assert not marker.exists()


def test_trusted_with_matching_digest_loads(hostile):
    path, marker = hostile
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert dl.load_data(str(path), trusted=True, expected_sha256=digest) == "loaded"
    assert marker.exists()


def test_stream_input_is_gated_and_verified(hostile):
    path, marker = hostile
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="disabled by default"):
        dl.load_pickle(io.BytesIO(raw))
    with pytest.raises(ValueError, match=f"{len(raw)} bytes SHA-256"):
        dl.load_pickle(io.BytesIO(raw), trusted=True, expected_sha256="0" * 64)
    assert not marker.exists()


def test_safe_pandas_pickle_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "frame.pd.pkl"
    df.to_pickle(path)
    pd.testing.assert_frame_equal(dl.load_pandas_pickle(path, trusted=True), df)


def test_numpy_without_objects_needs_no_trust(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.arange(3))
    np.testing.assert_array_equal(dl.load_numpy(path), np.arange(3))
    # allow_pickle=False passed explicitly is not a duplicate argument.
    np.testing.assert_array_equal(dl.load_numpy(path, allow_pickle=False), np.arange(3))


def test_numpy_object_arrays_need_trust(tmp_path):
    path = tmp_path / "o.npy"
    np.save(path, np.array([{"k": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="disabled by default"):
        dl.load_numpy(path, allow_pickle=True)
    with pytest.raises(ValueError):  # numpy refuses objects without allow_pickle
        dl.load_numpy(path)
    assert dl.load_numpy(path, allow_pickle=True, trusted=True)[0] == {"k": 1}
