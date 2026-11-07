"""
bm25-index/2: arrays persisted; recipe, parameters and identity verified on load.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from .._rank_bm25 import BM25, BM25L, BM25Okapi, BM25Plus

CORPUS = [
    ["alpha", "beta"],
    ["beta", "gamma", "gamma"],
    [],
    ["delta"],
    ["alpha", "alpha"],
]
IDS = list("abcde")
QUERIES = [["alpha"], ["gamma", "beta"], ["zeta"], ["alpha", "alpha", "delta"]]
V1 = pathlib.Path(__file__).parent / "data" / "v1-index"


def _generation(root):
    return (
        root
        / json.loads((root / "current.json").read_text(encoding="utf-8"))["generation"]
    )


@pytest.mark.parametrize("cls", [BM25Okapi, BM25L, BM25Plus])
@pytest.mark.parametrize("mmap_mode", [None, "r"])
def test_v2_round_trip_is_bit_identical(tmp_path, cls, mmap_mode):
    built = cls(CORPUS, doc_ids=IDS, b=0.6)
    built.save(tmp_path / "i")
    loaded = cls.load(tmp_path / "i", mmap_mode=mmap_mode)
    for query in QUERIES:
        assert np.array_equal(loaded.get_scores(query), built.get_scores(query))
        assert loaded.get_top_ids(query, n=5) == built.get_top_ids(query, n=5)
        assert loaded.get_batch_scores(query, [0, 3]) == built.get_batch_scores(
            query, [0, 3]
        )
    assert loaded.identity == built.identity


def test_the_json_holds_no_per_document_dicts(tmp_path):
    BM25Okapi(CORPUS, doc_ids=IDS).save(tmp_path / "i")
    generation = _generation(tmp_path / "i")
    state = json.loads((generation / "index.json").read_text(encoding="utf-8"))
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


def test_a_generation_from_before_this_version_is_refused():
    """``bm25-index/1`` (recipe ``bm25plus-delta/1``) is rebuilt, never rescored."""
    if not V1.is_dir():  # pragma: no cover - fixture not shipped
        pytest.skip("v1 fixture not present")
    with pytest.raises(ValueError, match=r"schema 'bm25-index/1'.*Rebuild"):
        BM25Plus.load(V1)


# -- what load verifies -------------------------------------------------------


def _saved(tmp_path, cls=BM25Plus, **params):
    cls(CORPUS, doc_ids=IDS, **params).save(tmp_path / "i")
    return tmp_path / "i", _generation(tmp_path / "i") / "index.json"


def _edit(index_json, **changes):
    state = json.loads(index_json.read_text(encoding="utf-8"))
    state.update(changes)
    index_json.write_text(json.dumps(state), encoding="utf-8")
    return state


@pytest.mark.parametrize(
    ("built", "loader", "owner"),
    [(BM25Okapi, BM25Plus, "BM25Okapi"), (BM25Plus, BM25L, "BM25Plus")],
)
def test_another_scorers_index_is_refused_naming_its_class(
    tmp_path, built, loader, owner
):
    root, _ = _saved(tmp_path, built)
    with pytest.raises(ValueError, match=f"load it with {owner}"):
        loader.load(root)


def test_an_earlier_recipe_version_is_refused(tmp_path):
    root, index_json = _saved(tmp_path)
    _edit(index_json, recipe="bm25plus-delta/1")
    with pytest.raises(ValueError, match=r"'bm25plus-delta/1'.*rebuild it"):
        BM25Plus.load(root)


def test_the_generation_directory_is_checked_like_the_root(tmp_path):
    _root, index_json = _saved(tmp_path)
    _edit(index_json, recipe="unknown/9")
    with pytest.raises(ValueError, match="'unknown/9'"):
        BM25Plus.load(index_json.parent)


@pytest.mark.parametrize(
    "change",
    [
        {"parameters": {"k1": 1.5, "b": 0.6, "delta": 2.0}},
        {
            "analyzer": {
                "name": "edited",
                "version": "1",
                "language": None,
                "stopwords": [],
                "stemmer": None,
                "lowercase": True,
            }
        },
        {"identity": "0" * 64},
    ],
    ids=["parameter", "analyzer", "identity"],
)
def test_edited_metadata_is_refused_by_the_identity(tmp_path, change):
    root, index_json = _saved(tmp_path)
    _edit(index_json, **change)
    with pytest.raises(ValueError, match="recorded identity does not match"):
        BM25Plus.load(root)


@pytest.mark.parametrize(
    ("parameters", "error", "match"),
    [
        ({"k1": 1.5, "b": 7.0, "delta": 1.0}, ValueError, "b"),
        ({"k1": 1.5, "b": 0.5}, ValueError, "exactly"),
        ({"k1": 1.5, "b": 0.5, "delta": 1.0, "epsilon": 0.2}, ValueError, "exactly"),
        ({"k1": "1.5", "b": 0.5, "delta": 1.0}, TypeError, "k1"),
    ],
)
def test_parameters_are_validated_at_load(tmp_path, parameters, error, match):
    """A consistent but out-of-domain declaration is refused before identity."""
    root, index_json = _saved(tmp_path)
    _edit(index_json, parameters=parameters)
    with pytest.raises(error, match=match):
        BM25Plus.load(root)


def test_the_base_class_cannot_load(tmp_path):
    root, _ = _saved(tmp_path)
    with pytest.raises(TypeError, match="scorer class"):
        BM25.load(root)


def test_a_valid_index_still_loads_from_root_and_generation(tmp_path):
    root, index_json = _saved(tmp_path, b=0.4)
    built = BM25Plus(CORPUS, doc_ids=IDS, b=0.4)
    for where in (root, index_json.parent):
        loaded = BM25Plus.load(where)
        assert loaded.identity == built.identity
        assert np.array_equal(loaded.get_scores(["alpha"]), built.get_scores(["alpha"]))


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
