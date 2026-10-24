# scikitplot/corpus/_embeddings/tests/test__cache_policy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The embedding cache policy (Run 3, R3-5).

A disk-cache hit must be what the engine would compute. An unpinned model
used to be served the old model's vectors after an upstream change, the cache
grew without bound, and the multimodal cache keyed on the source path and
its mtime. Each test below names the defect it closes.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import types

import numpy as np
import pytest

from .. import _embedding
from .._cache_policy import (
    CachePolicy,
    CacheStatus,
    enforce_bounds,
    resolve_hub_revision,
)
from .._embedding import EmbeddingEngine, _make_cache_key
from .._multimodal_embedding import MultimodalEmbeddingEngine

logger = _embedding.logger  # logging.getLogger(_embedding.__name__)

#: Hub commits (40 hex); only a commit pins a sentence-transformers model.
COMMIT = "0123456789abcdef0123456789abcdef01234567"
RESOLVED = "fedcba9876543210fedcba9876543210fedcba98"


class _Upstream:
    """A stand-in for the hub's current model: its output can change."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.full((len(texts), 4), self.value, dtype=np.float32)


def _engine(cache_dir, upstream=None, **kwargs) -> EmbeddingEngine:
    """A sentence-transformers engine whose model is ``upstream``."""
    engine = EmbeddingEngine(cache_dir=cache_dir, normalize=False, **kwargs)
    object.__setattr__(engine, "_embed_fn", upstream or _Upstream())
    return engine


def _entries(cache_dir: pathlib.Path) -> list[str]:
    return sorted(p.name for p in cache_dir.glob("*.npy"))


class TestPolicy:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"unpinned": "never"}, "unpinned must be one of"),
            ({"unpinned": "resolve"}, "requires a revision_resolver"),
            ({"revision_resolver": str}, "only with unpinned='resolve'"),
            ({"unpinned": "resolve", "revision_resolver": "x"}, "must be callable"),
            ({"max_entries": 0}, "max_entries must be a positive int"),
            ({"max_entries": True}, "max_entries must be a positive int"),
            ({"max_bytes": -1}, "max_bytes must be a positive int"),
            ({"max_bytes": 1.5}, "max_bytes must be a positive int"),
        ],
    )
    def test_refuses_what_it_cannot_honour(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CachePolicy(**kwargs)

    def test_the_default_bypasses_unpinned_and_is_unbounded(self):
        policy = CachePolicy()
        assert (policy.unpinned, policy.bounded) == ("bypass", False)

    def test_a_revision_must_be_a_string(self, tmp_path):
        """Re-review 1: ``None`` reached the key and raised AttributeError."""
        with pytest.raises(TypeError, match="model_revision must be a str"):
            EmbeddingEngine(backend="openai", model_revision=None, cache_dir=tmp_path)

    def test_an_engine_refuses_another_type(self, tmp_path):
        with pytest.raises(TypeError, match="cache_policy must be a CachePolicy"):
            EmbeddingEngine(cache_dir=tmp_path, cache_policy={"unpinned": "allow"})
        with pytest.raises(TypeError, match="cache_policy must be a CachePolicy"):
            MultimodalEmbeddingEngine(cache_policy="allow")


class TestUnpinned:
    def test_an_upstream_change_is_never_served_from_the_cache(self, tmp_path):
        """The defect: the old model's vectors after an upstream change."""
        upstream = _Upstream(1.0)
        first = _engine(tmp_path, upstream).embed_cached(["hello"])
        upstream.value = 2.0
        second = _engine(tmp_path, upstream).embed_cached(["hello"])
        assert first.status is second.status is CacheStatus.BYPASS_UNPINNED
        assert second.embeddings[0, 0] == 2.0
        assert _entries(tmp_path) == []

    def test_the_bypass_is_logged_once(self, tmp_path, captured_logger, caplog):
        engine = _engine(tmp_path)

        with captured_logger(logging.INFO, logger):
            engine.embed_cached(["a"])
            engine.embed_cached(["b"])

        notices = [
            r for r in caplog.records
            if "not pinned to a commit" in r.getMessage()  # r.message
        ]
        assert len(notices) == 1

    def test_allow_is_the_old_behaviour_by_choice(self, tmp_path):
        upstream = _Upstream()
        engine = _engine(tmp_path, upstream, cache_policy=CachePolicy(unpinned="allow"))
        stored = engine.embed_cached(["hello"])
        hit = engine.embed_cached(["hello"])
        assert (stored.status, hit.status) == (CacheStatus.STORED, CacheStatus.HIT)
        assert hit.revision is None
        assert upstream.calls == 1
        # The same key as before the policy existed: old entries stay valid.
        assert hit.key == _make_cache_key(
            engine.model_name,
            ["hello"],
            normalize=False,
            dtype="float32",
            backend="sentence_transformers",
        )

    def test_a_pinned_revision_is_cached_whatever_the_policy_says(self, tmp_path):
        engine = _engine(tmp_path, model_revision=COMMIT)
        assert engine.embed_cached(["x"]).status is CacheStatus.STORED
        hit = engine.embed_cached(["x"])
        assert (hit.status, hit.revision) == (CacheStatus.HIT, COMMIT)


class TestWhatPins:
    """Review 5: a branch name counted as pinned and was served stale."""

    def test_a_branch_moves_so_it_is_not_pinned(self, tmp_path):
        upstream = _Upstream(1.0)
        _engine(tmp_path, upstream, model_revision="main").embed_cached(["x"])
        upstream.value = 2.0
        again = _engine(tmp_path, upstream, model_revision="main").embed_cached(["x"])
        assert again.status is CacheStatus.BYPASS_UNPINNED
        assert again.embeddings[0, 0] == 2.0

    def test_a_branch_is_cached_by_choice(self, tmp_path):
        allow = CachePolicy(unpinned="allow")
        engine = _engine(tmp_path, model_revision="main", cache_policy=allow)
        assert engine.embed_cached(["x"]).status is CacheStatus.STORED
        assert engine.embed_cached(["x"]).revision == "main"

    def test_another_backend_trusts_a_stated_revision(self, tmp_path):
        engine = EmbeddingEngine(
            backend="openai", model_revision="2024-05", cache_dir=tmp_path
        )
        object.__setattr__(engine, "_embed_fn", _Upstream())
        assert engine.embed_cached(["x"]).status is CacheStatus.STORED
        unstated = EmbeddingEngine(backend="openai", cache_dir=tmp_path)
        object.__setattr__(unstated, "_embed_fn", _Upstream())
        assert unstated.embed_cached(["x"]).status is CacheStatus.BYPASS_UNPINNED

    def test_resolve_applies_only_where_the_commit_is_loaded(self, tmp_path):
        """Review 7: the resolver's answer keyed an API model it never loads."""
        asked = []
        policy = CachePolicy(
            unpinned="resolve", revision_resolver=lambda n: asked.append(n) or COMMIT
        )
        engine = EmbeddingEngine(
            backend="openai", cache_dir=tmp_path, cache_policy=policy
        )
        object.__setattr__(engine, "_embed_fn", _Upstream())
        assert engine.embed_cached(["x"]).status is CacheStatus.BYPASS_UNPINNED
        assert asked == []


class TestResolve:
    def _policy(self, resolver):
        return CachePolicy(unpinned="resolve", revision_resolver=resolver)

    def test_the_resolved_revision_is_loaded_keyed_and_recorded(
        self, tmp_path, monkeypatch
    ):
        asked, loaded = [], []

        def resolver(name):
            asked.append(name)
            return RESOLVED

        def loader(**kwargs):
            loaded.append(kwargs["revision"])
            return _Upstream()

        monkeypatch.setattr(_embedding, "_make_sentence_transformers_fn", loader)
        engine = EmbeddingEngine(
            cache_dir=tmp_path, normalize=False, cache_policy=self._policy(resolver)
        )
        first = engine.embed_cached(["x"])
        second = engine.embed_cached(["x"])
        assert (first.status, second.status) == (CacheStatus.STORED, CacheStatus.HIT)
        assert second.revision == RESOLVED
        assert asked == [engine.model_name]
        assert loaded == [RESOLVED]
        assert engine.embedding_manifest(4).revision == RESOLVED

    @pytest.mark.parametrize(
        "answer",
        [RuntimeError("offline"), "", None, 7, "main", "ABCDEF" * 7],
    )
    def test_a_failed_resolution_bypasses_and_loads_unpinned(
        self, tmp_path, monkeypatch, captured_logger, caplog, answer
    ):
        def resolver(name):
            if isinstance(answer, Exception):
                raise answer
            return answer

        loaded = []

        def loader(**kwargs):
            loaded.append(kwargs["revision"])
            return _Upstream()

        monkeypatch.setattr(_embedding, "_make_sentence_transformers_fn", loader)
        engine = EmbeddingEngine(
            cache_dir=tmp_path,
            normalize=False,
            cache_policy=self._policy(resolver),
        )

        with captured_logger(logging.WARNING, logger):
            result = engine.embed_cached(["x"])
            engine.embed_cached(["x"])

        assert result.status is CacheStatus.BYPASS_UNPINNED
        assert loaded == [None]
        assert _entries(tmp_path) == []

        resolution_warnings = [
            record for record in caplog.records
            if record.name == logger.name and "resolve" in record.getMessage()
        ]
        assert len(resolution_warnings) == 1
        assert engine.embedding_manifest(4).revision is None


class TestCustomAndDisabled:
    def test_an_unidentified_provider_is_computed(self, tmp_path):
        engine = EmbeddingEngine(
            backend="custom", custom_fn=_Upstream(), cache_dir=tmp_path
        )
        assert engine.embed_cached(["x"]).status is CacheStatus.BYPASS_UNIDENTIFIED
        assert _entries(tmp_path) == []

    def test_an_identified_provider_is_cached(self, tmp_path):
        engine = EmbeddingEngine(
            backend="custom",
            custom_fn=_Upstream(),
            custom_fn_id="u/1",
            cache_dir=tmp_path,
        )
        assert engine.embed_cached(["x"]).status is CacheStatus.STORED
        assert engine.embed_cached(["x"]).status is CacheStatus.HIT

    def test_disabled(self, tmp_path):
        engine = _engine(tmp_path, enable_cache=False, model_revision=COMMIT)
        result = engine.embed_cached(["x"])
        assert (result.status, result.key) == (CacheStatus.DISABLED, None)

    def test_embed_with_cache_keeps_its_contract(self, tmp_path):
        engine = _engine(tmp_path, model_revision=COMMIT)
        assert engine.embed_with_cache(["x"])[1] is False
        assert engine.embed_with_cache(["x"])[1] is True

    def test_a_failed_write_is_reported(self, tmp_path, monkeypatch):
        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(_embedding, "_save_to_cache", fail)
        engine = _engine(tmp_path, model_revision=COMMIT)
        assert engine.embed_cached(["x"]).status is CacheStatus.STORE_FAILED


class TestBounds:
    """Least recently used eviction (the cache used to grow without bound)."""

    def test_the_least_recently_used_entry_goes_first(self, tmp_path):
        policy = CachePolicy(max_entries=2)
        engine = _engine(tmp_path, model_revision=COMMIT, cache_policy=policy)
        a = engine.embed_cached(["a"]).key
        b = engine.embed_cached(["b"]).key
        os.utime(tmp_path / f"{a}.npy", ns=(1_000, 1_000))
        os.utime(tmp_path / f"{b}.npy", ns=(2_000, 2_000))
        assert engine.embed_cached(["a"]).status is CacheStatus.HIT  # a is now recent
        c = engine.embed_cached(["c"]).key
        assert _entries(tmp_path) == sorted([f"{a}.npy", f"{c}.npy"])

    def test_a_stored_entry_that_vanished_costs_no_extra_eviction(self, tmp_path):
        """Re-review 4: it was counted although the scan did not see it."""
        names = [f"{ch * 64}.npy" for ch in "abc"]
        for i, name in enumerate(names):
            (tmp_path / name).write_bytes(b"x")
            os.utime(tmp_path / name, ns=(1_000 + i, 1_000 + i))
        gone = tmp_path / f"{'d' * 64}.npy"
        assert enforce_bounds(tmp_path, CachePolicy(max_entries=3), stored=gone)
        assert _entries(tmp_path) == names

    def test_ties_are_broken_by_name(self, tmp_path):
        names = [f"{ch * 64}.npy" for ch in "cab"]
        for name in names:
            (tmp_path / name).write_bytes(b"x")
            os.utime(tmp_path / name, ns=(5_000, 5_000))
        stored = tmp_path / names[0]
        assert enforce_bounds(tmp_path, CachePolicy(max_entries=2), stored=stored)
        assert _entries(tmp_path) == sorted([names[0], names[2]])  # "a..." went

    def test_an_entry_larger_than_the_quota_is_not_kept(self, tmp_path):
        engine = _engine(
            tmp_path, model_revision=COMMIT, cache_policy=CachePolicy(max_bytes=8)
        )
        assert engine.embed_cached(["x"]).status is CacheStatus.TOO_LARGE
        assert _entries(tmp_path) == []

    def test_max_bytes_evicts_until_it_holds(self, tmp_path):
        engine = _engine(tmp_path, model_revision=COMMIT)
        first = engine.embed_cached(["a"]).key
        size = (tmp_path / f"{first}.npy").stat().st_size
        os.utime(tmp_path / f"{first}.npy", ns=(1_000, 1_000))
        bounded = _engine(
            tmp_path, model_revision=COMMIT, cache_policy=CachePolicy(max_bytes=size)
        )
        second = bounded.embed_cached(["b"])
        assert second.status is CacheStatus.STORED
        assert _entries(tmp_path) == [f"{second.key}.npy"]

    def test_only_cache_entries_are_ever_removed(self, tmp_path):
        keep = ["notes.txt", f"{'A' * 64}.npy", f"{'a' * 63}.npy", "x.npy"]
        for name in keep:
            (tmp_path / name).write_bytes(b"x" * 100)
        (tmp_path / "sub").mkdir()
        engine = _engine(
            tmp_path, model_revision=COMMIT, cache_policy=CachePolicy(max_entries=1)
        )
        engine.embed_cached(["a"])
        engine.embed_cached(["b"])
        assert all((tmp_path / name).exists() for name in keep)
        assert len([n for n in _entries(tmp_path) if len(n) == 68 and n.islower()]) == 1

    def test_an_entry_removed_by_another_process_is_skipped(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, model_revision=COMMIT)
        old = engine.embed_cached(["a"]).key
        os.utime(tmp_path / f"{old}.npy", ns=(1_000, 1_000))
        real_unlink = pathlib.Path.unlink

        def racing_unlink(self, *args, **kwargs):
            real_unlink(self, *args, **kwargs)
            raise FileNotFoundError(self)

        monkeypatch.setattr(pathlib.Path, "unlink", racing_unlink)
        bounded = _engine(
            tmp_path, model_revision=COMMIT, cache_policy=CachePolicy(max_entries=1)
        )
        assert bounded.embed_cached(["b"]).status is CacheStatus.STORED

    def test_a_read_only_entry_still_hits(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path, model_revision=COMMIT)
        engine.embed_cached(["a"])

        def refuse(*args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr(os, "utime", refuse)
        assert engine.embed_cached(["a"]).status is CacheStatus.HIT


class TestConcurrency:
    """Review 1-4: eviction by another process or thread, and odd files."""

    def test_an_entry_that_vanishes_before_the_load_is_a_miss(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, model_revision=COMMIT)
        key = engine.embed_cached(["a"]).key
        real_stat = pathlib.Path.stat

        def vanishing_stat(self, *args, **kwargs):
            if self.name == f"{key}.npy":
                self.unlink(missing_ok=True)
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "stat", vanishing_stat)
        assert engine.embed_cached(["a"]).status is CacheStatus.STORED

    def test_an_oversized_result_evicts_nothing(self, tmp_path):
        engine = _engine(tmp_path, model_revision=COMMIT)
        small = [engine.embed_cached([f"s{i}"]).key for i in range(3)]
        size = (tmp_path / f"{small[0]}.npy").stat().st_size
        big = _engine(
            tmp_path,
            model_revision=COMMIT,
            cache_policy=CachePolicy(max_bytes=size * 3),
            upstream=lambda t: np.ones((len(t), 512), np.float32),
        )
        assert big.embed_cached(["big"]).status is CacheStatus.TOO_LARGE
        assert _entries(tmp_path) == sorted(f"{k}.npy" for k in small)

    def test_a_stuck_old_entry_keeps_the_new_one(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path, model_revision=COMMIT)
        old = engine.embed_cached(["a"]).key
        real_unlink = pathlib.Path.unlink

        def stuck(self, *args, **kwargs):
            if self.name == f"{old}.npy":
                raise PermissionError("in use")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "unlink", stuck)
        bounded = _engine(
            tmp_path, model_revision=COMMIT, cache_policy=CachePolicy(max_entries=1)
        )
        new = bounded.embed_cached(["b"])
        assert new.status is CacheStatus.STORED
        assert _entries(tmp_path) == sorted([f"{old}.npy", f"{new.key}.npy"])

    def test_a_cache_dir_that_is_a_file_is_a_failed_store(self, tmp_path):
        """Review: stat() on <file>/<key>.npy raised NotADirectoryError."""
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        engine = _engine(blocker, model_revision=COMMIT)
        assert engine.embed_cached(["a"]).status is CacheStatus.STORE_FAILED

    def test_an_unlistable_directory_does_not_fail_the_call(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(
            tmp_path, model_revision=COMMIT, cache_policy=CachePolicy(max_entries=1)
        )

        def refuse(path):
            raise PermissionError(path)

        monkeypatch.setattr(os, "scandir", refuse)
        assert engine.embed_cached(["a"]).status is CacheStatus.STORED


class TestResolveHubRevision:
    @pytest.mark.parametrize(
        "name", ["all-MiniLM-L6-v2", "a/b/c", "/b", "a/", "", None]
    )
    def test_only_a_full_hub_id_is_resolved(self, name):
        with pytest.raises(ValueError, match="full Hub id"):
            resolve_hub_revision(name)

    def _fake_hub(self, monkeypatch, sha):
        calls = []
        module = types.ModuleType("huggingface_hub")

        def model_info(repo_id, timeout=None):
            calls.append((repo_id, timeout))
            return types.SimpleNamespace(sha=sha)

        module.model_info = model_info
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        return calls

    def test_a_local_directory_is_not_a_hub_id(self, tmp_path, monkeypatch):
        """Review 7: sentence-transformers loads the directory, not the id."""
        (tmp_path / "models" / "tuned").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        self._fake_hub(monkeypatch, "0" * 40)
        with pytest.raises(ValueError, match="local directory"):
            resolve_hub_revision("models/tuned")

    def test_the_commit_is_returned(self, monkeypatch):
        calls = self._fake_hub(monkeypatch, "0123abcd")
        assert resolve_hub_revision("owner/model", timeout=5.0) == "0123abcd"
        assert calls == [("owner/model", 5.0)]

    def test_no_commit_raises(self, monkeypatch):
        self._fake_hub(monkeypatch, None)
        with pytest.raises(ValueError, match="no commit"):
            resolve_hub_revision("owner/model")


class _Doc:
    def __init__(self, text, modality="text", raw_tensor=None):
        self.text, self.normalized_text = text, None
        self.modality, self.raw_tensor, self.embedding = modality, raw_tensor, None

    def replace(self, **kwargs):
        doc = _Doc(self.text, self.modality, self.raw_tensor)
        doc.embedding = self.embedding
        for key, value in kwargs.items():
            setattr(doc, key, value)
        return doc


class TestMultimodal:
    """The multimodal cache keyed on path + mtime + count."""

    def _engine(self, tmp_path, monkeypatch, fn=None, **kwargs):
        kwargs.setdefault("cache_policy", CachePolicy(unpinned="allow"))
        engine = MultimodalEmbeddingEngine(
            cache_dir=tmp_path, normalize=False, **kwargs
        )
        calls = []

        def embed_texts(texts):
            calls.append(list(texts))
            return (fn or (lambda t: np.array([[len(x), 1.0] for x in t], np.float32)))(
                texts
            )

        monkeypatch.setattr(engine, "embed_texts", embed_texts)
        return engine, calls

    def test_changed_content_is_recomputed(self, tmp_path, monkeypatch):
        engine, calls = self._engine(tmp_path, monkeypatch)
        src = tmp_path / "src.txt"
        src.write_text("x", encoding="utf-8")
        engine.embed_documents_with_cache([_Doc("ab")], src)
        out = engine.embed_documents_with_cache([_Doc("abcdef")], src)
        assert out[0].embedding.tolist() == [6.0, 1.0]
        assert len(calls) == 2

    def test_equal_content_hits_without_a_path_and_keeps_the_dtype(
        self, tmp_path, monkeypatch
    ):
        engine, calls = self._engine(
            tmp_path, monkeypatch, fn=lambda t: np.ones((len(t), 3), np.float64)
        )
        first, s1 = engine._embed_documents_cached([_Doc("a"), _Doc("b")], None)
        second, s2 = engine._embed_documents_cached([_Doc("a"), _Doc("b")], None)
        assert (s1, s2) == (CacheStatus.STORED, CacheStatus.HIT)
        assert len(calls) == 1
        assert second[0].embedding.dtype == first[0].embedding.dtype == np.float64

    def test_normalisation_is_part_of_the_key(self, tmp_path, monkeypatch):
        plain, _ = self._engine(tmp_path, monkeypatch)
        other = MultimodalEmbeddingEngine(
            cache_dir=tmp_path,
            normalize=True,
            cache_policy=CachePolicy(unpinned="allow"),
        )
        assert plain._cache_key([_Doc("a")]) != other._cache_key([_Doc("a")])

    def test_a_tensor_is_part_of_the_key(self, tmp_path, monkeypatch):
        engine, _ = self._engine(tmp_path, monkeypatch)
        image = np.zeros((2, 2, 3), np.uint8)
        changed = image.copy()
        changed[0, 0, 0] = 1
        keys = {
            engine._cache_key([_Doc("c", "image", tensor)])
            for tensor in (
                image,
                changed,
                image.astype(np.int16),
                image.reshape(3, 2, 2),
            )
        }
        assert len(keys) == 4
        assert engine._cache_key([_Doc("c", "image", np.array([object()]))]) is None
        assert engine._cache_key([_Doc("c", "image", [[1, 2], [3]])]) is None

    def test_an_unembedded_document_does_not_raise(self, tmp_path, monkeypatch):
        """``np.stack`` raised on the document left without an embedding."""
        engine, _ = self._engine(tmp_path, monkeypatch)
        out, status = engine._embed_documents_cached(
            [_Doc("ab"), _Doc("caption", "image")], None
        )
        assert status is CacheStatus.NOT_STORABLE
        assert out[1].embedding is None
        assert _entries(tmp_path) == []

    def test_a_custom_backend_and_an_unpinned_model_bypass(self, tmp_path):
        custom = MultimodalEmbeddingEngine(
            text_backend="custom",
            text_custom_fn=lambda t: np.zeros((len(t), 2), np.float32),
            cache_dir=tmp_path,
            cache_policy=CachePolicy(unpinned="allow"),
        )
        assert custom._cache_bypass() is CacheStatus.BYPASS_UNIDENTIFIED
        assert MultimodalEmbeddingEngine()._cache_bypass() is (
            CacheStatus.BYPASS_UNPINNED
        )

    def test_bounds_apply(self, tmp_path, monkeypatch):
        engine, _ = self._engine(
            tmp_path,
            monkeypatch,
            cache_policy=CachePolicy(unpinned="allow", max_entries=1),
        )
        engine._embed_documents_cached([_Doc("a")], None)
        engine._embed_documents_cached([_Doc("b")], None)
        assert len(_entries(tmp_path)) == 1
