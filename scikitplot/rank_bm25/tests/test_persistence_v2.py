"""
bm25-index/2: arrays persisted, verified on load, old generations still load.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["alpha", "beta"], ["beta", "gamma", "gamma"], [], ["delta"], ["alpha", "alpha"]]
IDS = list("abcde")
QUERIES = [["alpha"], ["gamma", "beta"], ["zeta"], ["alpha", "alpha", "delta"]]
V1 = pathlib.Path(__file__).parent / "data" / "v1-index"


def _generation(root):
    return root / json.loads((root / "current.json").read_text())["generation"]


@pytest.mark.parametrize("cls", [BM25Okapi, BM25L, BM25Plus])
@pytest.mark.parametrize("mmap_mode", [None, "r"])
def test_v2_round_trip_is_bit_identical(tmp_path, cls, mmap_mode):
    built = cls(CORPUS, doc_ids=IDS, b=0.6)
    built.save(tmp_path / "i")
    loaded = cls.load(tmp_path / "i", mmap_mode=mmap_mode)
    for query in QUERIES:
        assert np.array_equal(loaded.get_scores(query), built.get_scores(query))
        assert loaded.get_top_ids(query, n=5) == built.get_top_ids(query, n=5)
        assert loaded.get_batch_scores(query, [0, 3]) == built.get_batch_scores(query, [0, 3])
    assert loaded.identity == built.identity


def test_the_json_holds_no_per_document_dicts(tmp_path):
    BM25Okapi(CORPUS, doc_ids=IDS).save(tmp_path / "i")
    generation = _generation(tmp_path / "i")
    state = json.loads((generation / "index.json").read_text())
    assert state["schema"] == "bm25-index/2" and "doc_freqs" not in state
    assert sorted(p.name for p in generation.glob("*.npy")) == [
        "indptr.npy",
        "rows.npy",
        "tfs.npy",
    ]
    assert np.load(generation / "rows.npy").dtype == np.int32


def test_dicts_are_rebuilt_only_when_needed(tmp_path):
    built = BM25Plus(CORPUS, doc_ids=IDS)
    built.save(tmp_path / "i")
    loaded = BM25Plus.load(tmp_path / "i")
    loaded.get_top_ids(["alpha"], n=2)
    assert loaded.__dict__.get("_doc_freqs") is None  # the engine answered
    assert loaded.doc_freqs == built.doc_freqs  # rebuilt on demand, equal
    reference = BM25Plus.load(tmp_path / "i")
    reference._backend = "reference"
    assert np.array_equal(reference.get_scores(["gamma"]), built.get_scores(["gamma"]))


@pytest.mark.parametrize("name", ["rows", "tfs", "indptr"])
def test_a_changed_array_is_refused(tmp_path, name):
    BM25Okapi(CORPUS, doc_ids=IDS).save(tmp_path / "i")
    path = _generation(tmp_path / "i") / f"{name}.npy"
    array = np.load(path)
    array[0] = array[0] + 1
    np.save(path, array)
    with pytest.raises(ValueError, match="does not match what was saved"):
        BM25Okapi.load(tmp_path / "i")


def test_a_pickled_array_is_refused(tmp_path):
    BM25Okapi(CORPUS, doc_ids=IDS).save(tmp_path / "i")
    path = _generation(tmp_path / "i") / "tfs.npy"
    np.save(path, np.array([object()], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="not a plain array"):
        BM25Okapi.load(tmp_path / "i")


def test_a_missing_array_is_refused(tmp_path):
    BM25Okapi(CORPUS, doc_ids=IDS).save(tmp_path / "i")
    (_generation(tmp_path / "i") / "tfs.npy").unlink()
    with pytest.raises(ValueError, match="missing tfs.npy"):
        BM25Okapi.load(tmp_path / "i")


def test_a_v1_generation_written_by_the_previous_version_loads():
    if not V1.is_dir():  # pragma: no cover - fixture not shipped
        pytest.skip("v1 fixture not present")
    loaded = BM25Plus.load(V1)
    fresh = BM25Plus(CORPUS[:4], doc_ids=list("abcd"), b=0.5)
    for query in QUERIES:
        assert np.array_equal(loaded.get_scores(query), fresh.get_scores(query), equal_nan=True)
    assert loaded.identity == fresh.identity


def test_v2_is_smaller_than_the_json_it_replaces(tmp_path):
    rng = np.random.default_rng(0)
    corpus = [[f"w{x}" for x in rng.integers(0, 2000, 60)] for _ in range(3000)]
    BM25Okapi(corpus).save(tmp_path / "i")
    size = sum(p.stat().st_size for p in _generation(tmp_path / "i").iterdir())
    as_v1 = len(json.dumps([dict(f) for f in BM25Okapi(corpus).doc_freqs], indent=2))
    assert size < as_v1


def test_unknown_mmap_mode_is_refused(tmp_path):
    BM25Okapi(CORPUS, doc_ids=IDS).save(tmp_path / "i")
    with pytest.raises(ValueError, match="mmap_mode"):
        BM25Okapi.load(tmp_path / "i", mmap_mode="w+")
