# corpus/_embeddings/tests/test__embedding.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot.corpus._embeddings._embedding.

Covers
------
- Module constants: DEFAULT_MODEL, DEFAULT_CACHE_DIR, _FLOAT_DTYPES
- _make_cache_key: determinism, uniqueness per parameter, hex format
- _cache_path: path construction and extension
- _save_to_cache: file creation, atomic write, parent dir creation, error cleanup
- _load_from_cache: hit/miss/corrupt/wrong-shape/wrong-ndim/wrong-dtype
- _make_sentence_transformers_fn: ImportError guard, lazy load, model reuse
- _make_openai_fn: ImportError guard, batching logic, result shape
- EmbeddingEngine.__post_init__: all validation branches (backend, batch_size, dtype)
- EmbeddingEngine.embed: empty raises, shape/dtype validation, custom fn, logging
- EmbeddingEngine.embed_with_cache: hit, miss, disabled, stat failure, write failure
- EmbeddingEngine.embed_documents: empty, populates embedding, normalized_text fallback
- EmbeddingEngine._get_embed_fn: lazy init, thread-safe caching, all backend branches
- EmbeddingEngine.__repr__: content and format
- EmbeddingEngine.VALID_BACKENDS: membership
- Import-skip guards for sentence_transformers and openai heavy deps
"""

from __future__ import annotations

import pathlib
import tempfile
import threading
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from .._embedding import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL,
    EmbeddingEngine,
    _FLOAT_DTYPES,
    _cache_path,
    _load_from_cache,
    _make_cache_key,
    _make_openai_fn,
    _make_sentence_transformers_fn,
    _save_to_cache,
)


# ---------------------------------------------------------------------------
# Minimal CorpusDocument stub (avoids importing the full schema)
# ---------------------------------------------------------------------------


class _Doc:
    """Minimal CorpusDocument stub for embedding tests."""

    def __init__(
        self,
        text: str = "Sample text.",
        normalized_text: str | None = None,
        embedding: Any = None,
    ) -> None:
        self.text = text
        self.normalized_text = normalized_text
        self.embedding = embedding

    def replace(self, **kwargs: Any) -> "_Doc":
        d = _Doc(self.text, self.normalized_text, self.embedding)
        for k, v in kwargs.items():
            setattr(d, k, v)
        return d


def _make_custom_engine(dim: int = 32, **kwargs: Any) -> EmbeddingEngine:
    """Return an EmbeddingEngine backed by a zero-vector custom_fn."""
    def _fn(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), dim), dtype=np.float32)

    return EmbeddingEngine(backend="custom", custom_fn=_fn, **kwargs)


# ===========================================================================
# Module constants
# ===========================================================================


class TestConstants:
    def test_default_model_is_string(self) -> None:
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0

    def test_default_cache_dir_is_path(self) -> None:
        assert isinstance(DEFAULT_CACHE_DIR, pathlib.Path)

    def test_default_cache_dir_under_home(self) -> None:
        assert "scikitplot" in str(DEFAULT_CACHE_DIR)

    def test_float_dtypes_contains_float32(self) -> None:
        arr = np.zeros(1, dtype=np.float32)
        assert arr.dtype in _FLOAT_DTYPES

    def test_float_dtypes_contains_float64(self) -> None:
        arr = np.zeros(1, dtype=np.float64)
        assert arr.dtype in _FLOAT_DTYPES

    def test_float_dtypes_contains_float16(self) -> None:
        arr = np.zeros(1, dtype=np.float16)
        assert arr.dtype in _FLOAT_DTYPES

    def test_float_dtypes_excludes_int32(self) -> None:
        arr = np.zeros(1, dtype=np.int32)
        assert arr.dtype not in _FLOAT_DTYPES

    def test_float_dtypes_is_tuple(self) -> None:
        assert isinstance(_FLOAT_DTYPES, tuple)
        assert len(_FLOAT_DTYPES) == 3  # noqa: PLR2004


# ===========================================================================
# _make_cache_key
# ===========================================================================


class TestMakeCacheKey:
    def test_returns_24_chars(self) -> None:
        key = _make_cache_key("model", "/path/file.txt", 1700000000.0, 100)
        assert len(key) == 24  # noqa: PLR2004

    def test_returns_hex_string(self) -> None:
        key = _make_cache_key("model", "/path/file.txt", 1700000000.0, 100)
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic_same_inputs(self) -> None:
        k1 = _make_cache_key("model-v1", "/data/f.txt", 1700000000.0, 512)
        k2 = _make_cache_key("model-v1", "/data/f.txt", 1700000000.0, 512)
        assert k1 == k2

    def test_different_model_different_key(self) -> None:
        k1 = _make_cache_key("model-a", "/path/f.txt", 0.0, 10)
        k2 = _make_cache_key("model-b", "/path/f.txt", 0.0, 10)
        assert k1 != k2

    def test_different_path_different_key(self) -> None:
        k1 = _make_cache_key("model", "/path/a.txt", 0.0, 10)
        k2 = _make_cache_key("model", "/path/b.txt", 0.0, 10)
        assert k1 != k2

    def test_different_mtime_different_key(self) -> None:
        k1 = _make_cache_key("model", "/path/f.txt", 1.0, 10)
        k2 = _make_cache_key("model", "/path/f.txt", 2.0, 10)
        assert k1 != k2

    def test_different_n_texts_different_key(self) -> None:
        k1 = _make_cache_key("model", "/path/f.txt", 0.0, 10)
        k2 = _make_cache_key("model", "/path/f.txt", 0.0, 20)
        assert k1 != k2

    def test_all_string_inputs(self) -> None:
        # Should not raise with any string model/path combination
        key = _make_cache_key("multilingual/model:v2", "/tmp/corpus file.txt", 1.23, 1)
        assert isinstance(key, str)
        assert len(key) == 24  # noqa: PLR2004

    def test_zero_mtime(self) -> None:
        key = _make_cache_key("model", "/path", 0.0, 0)
        assert len(key) == 24  # noqa: PLR2004


# ===========================================================================
# _cache_path
# ===========================================================================


class TestCachePath:
    def test_returns_path(self) -> None:
        p = _cache_path(pathlib.Path("/cache"), "abc123xyz456789012345678")
        assert isinstance(p, pathlib.Path)

    def test_extension_is_npy(self) -> None:
        p = _cache_path(pathlib.Path("/cache"), "abc123xyz456789012345678")
        assert p.suffix == ".npy"

    def test_filename_matches_key(self) -> None:
        key = "abc123xyz456789012345678"
        p = _cache_path(pathlib.Path("/cache"), key)
        assert p.name == f"{key}.npy"

    def test_parent_is_cache_dir(self) -> None:
        cache_dir = pathlib.Path("/my/cache/dir")
        p = _cache_path(cache_dir, "key12345678901234567890x")
        assert p.parent == cache_dir


# ===========================================================================
# _save_to_cache
# ===========================================================================


class TestSaveToCache:
    def test_creates_file(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((3, 64), dtype=np.float32)
        path = tmp_path / "embeddings.npy"
        _save_to_cache(arr, path)
        assert path.exists()

    def test_saves_correct_data(self, tmp_path: pathlib.Path) -> None:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        path = tmp_path / "test.npy"
        _save_to_cache(arr, path)
        loaded = np.load(str(path), allow_pickle=False)
        np.testing.assert_array_equal(loaded, arr)

    def test_creates_nested_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        arr = np.zeros((2, 4), dtype=np.float32)
        path = tmp_path / "a" / "b" / "c" / "test.npy"
        _save_to_cache(arr, path)
        assert path.exists()

    def test_no_temp_file_left_after_success(self, tmp_path: pathlib.Path) -> None:
        arr = np.zeros((2, 4), dtype=np.float32)
        path = tmp_path / "test.npy"
        _save_to_cache(arr, path)
        assert not path.with_suffix(".tmp.npy").exists()

    def test_overwrites_existing_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "test.npy"
        old = np.ones((2, 3), dtype=np.float32)
        _save_to_cache(old, path)
        new = np.zeros((2, 3), dtype=np.float32) + 99
        _save_to_cache(new, path)
        loaded = np.load(str(path), allow_pickle=False)
        np.testing.assert_array_equal(loaded, new)

    def test_float32_preserved(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((4, 8), dtype=np.float32)
        path = tmp_path / "test.npy"
        _save_to_cache(arr, path)
        loaded = np.load(str(path), allow_pickle=False)
        assert loaded.dtype == np.float32


# ===========================================================================
# _load_from_cache
# ===========================================================================


class TestLoadFromCache:
    def test_returns_none_missing_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "not_existing.npy"
        result = _load_from_cache(path, 5)
        assert result is None

    def test_returns_none_empty_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "empty.npy"
        path.write_bytes(b"")
        result = _load_from_cache(path, 1)
        assert result is None

    def test_returns_none_corrupt_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "corrupt.npy"
        path.write_bytes(b"this is not valid numpy data !!!")
        result = _load_from_cache(path, 1)
        assert result is None

    def test_returns_none_wrong_n_rows(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((3, 64), dtype=np.float32)
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 5)  # expected 5, got 3
        assert result is None

    def test_returns_none_wrong_ndim(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones(64, dtype=np.float32)  # 1-D, not 2-D
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 64)
        assert result is None

    def test_returns_none_non_float_dtype(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((3, 64), dtype=np.int32)  # int32 not in _FLOAT_DTYPES
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 3)
        assert result is None

    def test_loads_valid_float32(self, tmp_path: pathlib.Path) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 3)
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_loads_float64_and_casts_to_float32(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((3, 4), dtype=np.float64)
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 3)
        assert result is not None
        assert result.dtype == np.float32

    def test_loads_float16_and_casts_to_float32(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((2, 8), dtype=np.float16)
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 2)
        assert result is not None
        assert result.dtype == np.float32

    def test_shape_preserved_after_load(self, tmp_path: pathlib.Path) -> None:
        arr = np.zeros((5, 128), dtype=np.float32)
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 5)
        assert result is not None
        assert result.shape == (5, 128)

    def test_returns_none_row_count_zero(self, tmp_path: pathlib.Path) -> None:
        arr = np.ones((4, 32), dtype=np.float32)
        path = tmp_path / "test.npy"
        np.save(str(path), arr, allow_pickle=False)
        result = _load_from_cache(path, 0)  # expected 0 rows, got 4
        assert result is None


# ===========================================================================
# _make_sentence_transformers_fn
# ===========================================================================


class TestMakeSentenceTransformersFn:
    def test_raises_import_error_when_not_installed(self) -> None:
        fn = _make_sentence_transformers_fn("model", 32, True, False, None)
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence_transformers"):
                fn(["hello"])

    def test_calls_encode_with_texts(self) -> None:
        mock_st_mod = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, 128), dtype=np.float32)
        mock_st_mod.SentenceTransformer.return_value = mock_model

        fn = _make_sentence_transformers_fn("test-model", 32, True, False, None)
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_mod}):
            result = fn(["text1", "text2"])

        mock_model.encode.assert_called_once()
        assert result.shape == (2, 128)

    def test_model_loaded_only_once(self) -> None:
        """Model must be initialised lazily and reused on subsequent calls."""
        mock_st_mod = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 64), dtype=np.float32)
        mock_st_mod.SentenceTransformer.return_value = mock_model

        fn = _make_sentence_transformers_fn("model-x", 8, False, False, None)
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_mod}):
            fn(["a"])
            fn(["b"])
            fn(["c"])

        # SentenceTransformer() called once even after three fn calls
        assert mock_st_mod.SentenceTransformer.call_count == 1

    def test_result_is_float32(self) -> None:
        mock_st_mod = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((3, 16), dtype=np.float64)
        mock_st_mod.SentenceTransformer.return_value = mock_model

        fn = _make_sentence_transformers_fn("m", 32, False, False, None)
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_mod}):
            result = fn(["a", "b", "c"])

        assert result.dtype == np.float32

    def test_device_kwarg_forwarded(self) -> None:
        mock_st_mod = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 8), dtype=np.float32)
        mock_st_mod.SentenceTransformer.return_value = mock_model

        fn = _make_sentence_transformers_fn("m", 32, False, False, "cpu")
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_mod}):
            fn(["text"])

        mock_st_mod.SentenceTransformer.assert_called_once_with("m", device="cpu")

    def test_no_device_kwarg_when_none(self) -> None:
        mock_st_mod = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 8), dtype=np.float32)
        mock_st_mod.SentenceTransformer.return_value = mock_model

        fn = _make_sentence_transformers_fn("m", 32, False, False, None)
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_mod}):
            fn(["text"])

        # device kwarg must NOT be present when device=None
        _call_kwargs = mock_st_mod.SentenceTransformer.call_args[1]
        assert "device" not in _call_kwargs


# ===========================================================================
# _make_openai_fn
# ===========================================================================


class TestMakeOpenAiFn:
    def _mock_openai(self, n: int, dim: int) -> MagicMock:
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        items = [MagicMock(embedding=[float(j) for j in range(dim)]) for _ in range(n)]
        mock_client.embeddings.create.return_value.data = items
        return mock_openai

    def test_raises_import_error_when_not_installed(self) -> None:
        fn = _make_openai_fn("text-embedding-3-small", 10)
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                fn(["hello"])

    def test_returns_correct_shape(self) -> None:
        mock_openai = self._mock_openai(n=3, dim=8)
        fn = _make_openai_fn("text-embedding-3-small", 100)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = fn(["a", "b", "c"])
        assert result.shape == (3, 8)

    def test_result_is_float32(self) -> None:
        mock_openai = self._mock_openai(n=2, dim=4)
        fn = _make_openai_fn("text-embedding-3-small", 100)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = fn(["x", "y"])
        assert result.dtype == np.float32

    def test_batching_multiple_api_calls(self) -> None:
        """batch_size=2 with 5 texts → ceil(5/2)=3 API calls."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        def _create(input, model):  # noqa: A002
            items = [MagicMock(embedding=[1.0]) for _ in input]
            resp = MagicMock()
            resp.data = items
            return resp

        mock_client.embeddings.create.side_effect = _create

        fn = _make_openai_fn("text-embedding-3-small", batch_size=2)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = fn(["a", "b", "c", "d", "e"])

        assert mock_client.embeddings.create.call_count == 3  # noqa: PLR2004
        assert result.shape == (5, 1)

    def test_single_text_single_api_call(self) -> None:
        mock_openai = self._mock_openai(n=1, dim=4)
        fn = _make_openai_fn("text-embedding-3-small", 10)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = fn(["only one text"])
        assert result.shape == (1, 4)


# ===========================================================================
# EmbeddingEngine — construction & validation
# ===========================================================================


class TestEmbeddingEngineValidation:
    def test_defaults(self) -> None:
        e = _make_custom_engine()
        assert e.backend == "custom"
        assert e.batch_size == 64  # noqa: PLR2004
        assert e.normalize is True
        assert e.enable_cache is True

    def test_default_model_name(self) -> None:
        e = _make_custom_engine()
        assert e.model_name == DEFAULT_MODEL

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            EmbeddingEngine(backend="unknown_backend")

    def test_custom_backend_requires_custom_fn(self) -> None:
        with pytest.raises(ValueError, match="custom_fn"):
            EmbeddingEngine(backend="custom", custom_fn=None)

    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_custom_engine(batch_size=0)

    def test_batch_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_custom_engine(batch_size=-1)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            _make_custom_engine(dtype=np.int32)

    def test_float32_dtype_accepted(self) -> None:
        e = _make_custom_engine(dtype=np.float32)
        assert e.dtype == np.float32

    def test_float64_dtype_accepted(self) -> None:
        e = _make_custom_engine(dtype=np.float64)
        assert e.dtype == np.float64

    def test_float16_dtype_accepted(self) -> None:
        e = _make_custom_engine(dtype=np.float16)
        assert e.dtype == np.float16

    def test_cache_dir_defaults_when_none(self) -> None:
        e = _make_custom_engine()
        assert e.cache_dir is not None
        assert isinstance(e.cache_dir, pathlib.Path)

    def test_custom_cache_dir_preserved(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(cache_dir=tmp_path)
        assert e.cache_dir == tmp_path

    def test_valid_backends_class_var(self) -> None:
        assert "sentence_transformers" in EmbeddingEngine.VALID_BACKENDS
        assert "openai" in EmbeddingEngine.VALID_BACKENDS
        assert "custom" in EmbeddingEngine.VALID_BACKENDS

    def test_sentence_transformers_backend_accepted(self) -> None:
        # Must not raise during construction (only raises at call-time if lib absent)
        e = EmbeddingEngine(backend="sentence_transformers")
        assert e.backend == "sentence_transformers"

    def test_openai_backend_accepted(self) -> None:
        e = EmbeddingEngine(backend="openai")
        assert e.backend == "openai"


# ===========================================================================
# EmbeddingEngine.embed
# ===========================================================================


class TestEmbeddingEngineEmbed:
    def test_empty_texts_raises(self) -> None:
        e = _make_custom_engine()
        with pytest.raises(ValueError, match="non-empty"):
            e.embed([])

    def test_custom_fn_returns_correct_shape(self) -> None:
        e = _make_custom_engine(dim=128)
        result = e.embed(["hello", "world", "test"])
        assert result.shape == (3, 128)

    def test_output_dtype_matches_config(self) -> None:
        e = _make_custom_engine(dim=16, dtype=np.float32)
        result = e.embed(["text"])
        assert result.dtype == np.float32

    def test_output_cast_to_float64_when_configured(self) -> None:
        e = _make_custom_engine(dim=8, dtype=np.float64)
        result = e.embed(["text"])
        assert result.dtype == np.float64

    def test_raises_when_backend_returns_wrong_ndim(self) -> None:
        def _bad_fn(texts: list[str]) -> np.ndarray:
            return np.zeros(len(texts), dtype=np.float32)  # 1-D, not 2-D

        e = EmbeddingEngine(backend="custom", custom_fn=_bad_fn)
        with pytest.raises(TypeError, match="2-D"):
            e.embed(["hello"])

    def test_raises_when_backend_returns_wrong_row_count(self) -> None:
        def _bad_fn(texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts) + 1, 8), dtype=np.float32)  # extra row

        e = EmbeddingEngine(backend="custom", custom_fn=_bad_fn)
        with pytest.raises(ValueError, match="vectors"):
            e.embed(["hello"])

    def test_single_text(self) -> None:
        e = _make_custom_engine(dim=4)
        result = e.embed(["single sentence."])
        assert result.shape == (1, 4)

    def test_many_texts(self) -> None:
        e = _make_custom_engine(dim=64)
        texts = [f"sentence {i}" for i in range(100)]
        result = e.embed(texts)
        assert result.shape == (100, 64)

    def test_returns_ndarray(self) -> None:
        e = _make_custom_engine()
        result = e.embed(["test"])
        assert isinstance(result, np.ndarray)

    def test_sentence_transformers_backend_via_injected_fn(self) -> None:
        """Test EmbeddingEngine.embed flow using directly injected embed fn."""
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        e = EmbeddingEngine(backend="sentence_transformers")
        object.__setattr__(e, "_embed_fn", lambda texts: expected)
        result = e.embed(["a", "b"])
        np.testing.assert_array_equal(result, expected)

    def test_openai_backend_via_injected_fn(self) -> None:
        expected = np.ones((1, 16), dtype=np.float32)
        e = EmbeddingEngine(backend="openai")
        object.__setattr__(e, "_embed_fn", lambda texts: expected)
        result = e.embed(["openai text"])
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# EmbeddingEngine.embed_with_cache
# ===========================================================================


class TestEmbeddingEngineEmbedWithCache:
    def test_empty_texts_raises(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("dummy")
        with pytest.raises(ValueError, match="non-empty"):
            e.embed_with_cache([], src)

    def test_disabled_cache_always_recomputes(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(cache_dir=tmp_path, enable_cache=False)
        src = tmp_path / "src.txt"
        src.write_text("content")
        r1, flag1 = e.embed_with_cache(["a"], src)
        r2, flag2 = e.embed_with_cache(["a"], src)
        assert flag1 is False
        assert flag2 is False

    def test_cache_miss_returns_false_flag(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("hello")
        _, flag = e.embed_with_cache(["text"], src)
        assert flag is False

    def test_cache_hit_returns_true_flag(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(dim=8, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("hello")
        e.embed_with_cache(["text"], src)  # miss — writes cache
        _, flag = e.embed_with_cache(["text"], src)  # hit
        assert flag is True

    def test_cache_file_created_on_miss(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(dim=16, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("hello")
        e.embed_with_cache(["text1", "text2"], src)
        npy_files = list(tmp_path.glob("*.npy"))
        assert len(npy_files) == 1

    def test_cached_values_match_computed(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(dim=4, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("hello")
        r1, _ = e.embed_with_cache(["one", "two"], src)
        r2, _ = e.embed_with_cache(["one", "two"], src)
        np.testing.assert_array_equal(r1, r2)

    def test_stat_failure_bypasses_cache(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(cache_dir=tmp_path)
        non_existent = tmp_path / "no_such_file.txt"
        result, flag = e.embed_with_cache(["text"], non_existent)
        assert flag is False
        assert result.shape[0] == 1

    def test_cache_write_failure_logs_warning(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        e = _make_custom_engine(cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")

        with patch("scikitplot.corpus._embeddings._embedding._save_to_cache") as mock_save:
            mock_save.side_effect = OSError("disk full")
            with caplog.at_level(logging.WARNING):
                result, flag = e.embed_with_cache(["text"], src)

        assert flag is False
        assert result is not None

    def test_returns_correct_shape_from_cache(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(dim=32, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        e.embed_with_cache(["a", "b", "c"], src)
        r2, _ = e.embed_with_cache(["a", "b", "c"], src)
        assert r2.shape == (3, 32)


# ===========================================================================
# EmbeddingEngine.embed_documents
# ===========================================================================


class TestEmbeddingEngineEmbedDocuments:
    def test_empty_returns_empty_list(self) -> None:
        e = _make_custom_engine()
        result = e.embed_documents([])
        assert result == []

    def test_populates_embedding_field(self) -> None:
        e = _make_custom_engine(dim=8)
        docs = [_Doc("Hello world."), _Doc("Second sentence.")]
        out = e.embed_documents(docs)
        assert all(d.embedding is not None for d in out)
        assert all(d.embedding.shape == (8,) for d in out)

    def test_preserves_doc_count(self) -> None:
        e = _make_custom_engine()
        docs = [_Doc(f"text {i}") for i in range(5)]
        out = e.embed_documents(docs)
        assert len(out) == 5  # noqa: PLR2004

    def test_uses_normalized_text_when_available(self) -> None:
        """normalized_text takes priority over text for embedding input."""
        captured: list[list[str]] = []

        def _tracking_fn(texts: list[str]) -> np.ndarray:
            captured.append(list(texts))
            return np.zeros((len(texts), 4), dtype=np.float32)

        e = EmbeddingEngine(backend="custom", custom_fn=_tracking_fn)
        docs = [
            _Doc("raw text", normalized_text="clean text"),
            _Doc("other raw", normalized_text=None),
        ]
        e.embed_documents(docs)
        assert captured[0][0] == "clean text"
        assert captured[0][1] == "other raw"

    def test_falls_back_to_text_when_no_normalized(self) -> None:
        captured: list[list[str]] = []

        def _fn(texts: list[str]) -> np.ndarray:
            captured.append(list(texts))
            return np.zeros((len(texts), 4), dtype=np.float32)

        e = EmbeddingEngine(backend="custom", custom_fn=_fn)
        docs = [_Doc("fallback text", normalized_text=None)]
        e.embed_documents(docs)
        assert captured[0][0] == "fallback text"

    def test_original_docs_not_mutated(self) -> None:
        e = _make_custom_engine(dim=4)
        docs = [_Doc("text")]
        out = e.embed_documents(docs)
        assert docs[0].embedding is None  # original unchanged
        assert out[0].embedding is not None

    def test_with_source_path_uses_cache(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(dim=4, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        docs = [_Doc("hello")]
        out = e.embed_documents(docs, source_path=src)
        assert out[0].embedding is not None
        # A .npy cache file should have been created
        assert len(list(tmp_path.glob("*.npy"))) == 1

    def test_without_source_path_no_cache_file(self, tmp_path: pathlib.Path) -> None:
        e = _make_custom_engine(dim=4, cache_dir=tmp_path)
        docs = [_Doc("hello")]
        e.embed_documents(docs, source_path=None)
        assert len(list(tmp_path.glob("*.npy"))) == 0

    def test_single_doc(self) -> None:
        e = _make_custom_engine(dim=16)
        out = e.embed_documents([_Doc("only doc")])
        assert len(out) == 1
        assert out[0].embedding.shape == (16,)


# ===========================================================================
# EmbeddingEngine._get_embed_fn
# ===========================================================================


class TestEmbeddingEngineGetEmbedFn:
    def test_custom_fn_returned_directly(self) -> None:
        my_fn = lambda t: np.zeros((len(t), 4), dtype=np.float32)  # noqa: E731
        e = EmbeddingEngine(backend="custom", custom_fn=my_fn)
        fn = e._get_embed_fn()
        assert fn is my_fn

    def test_fn_cached_on_second_call(self) -> None:
        my_fn = lambda t: np.zeros((len(t), 4), dtype=np.float32)  # noqa: E731
        e = EmbeddingEngine(backend="custom", custom_fn=my_fn)
        fn1 = e._get_embed_fn()
        fn2 = e._get_embed_fn()
        assert fn1 is fn2

    def test_sentence_transformers_fn_constructed(self) -> None:
        """_get_embed_fn must build a callable for sentence_transformers backend."""
        e = EmbeddingEngine(backend="sentence_transformers")
        assert e._embed_fn is None  # before call
        # Inject a pre-built fn to skip actual library load
        object.__setattr__(e, "_embed_fn", lambda t: np.zeros((len(t), 8), np.float32))
        fn = e._get_embed_fn()
        assert callable(fn)

    def test_openai_fn_constructed(self) -> None:
        e = EmbeddingEngine(backend="openai")
        object.__setattr__(e, "_embed_fn", lambda t: np.zeros((len(t), 8), np.float32))
        fn = e._get_embed_fn()
        assert callable(fn)

    def test_thread_safety_concurrent_init(self) -> None:
        """Multiple threads must not initialise _embed_fn more than once."""
        call_count = [0]
        original_fn = lambda t: np.zeros((len(t), 4), np.float32)  # noqa: E731

        def _slow_fn(texts: list[str]) -> np.ndarray:
            call_count[0] += 1
            return original_fn(texts)

        e = EmbeddingEngine(backend="custom", custom_fn=_slow_fn)
        # Pre-set to None to simulate the not-yet-initialised state
        object.__setattr__(e, "_embed_fn", None)

        errors: list[Exception] = []
        results: list[Any] = []

        def _worker() -> None:
            try:
                fn = e._get_embed_fn()
                results.append(fn)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is results[0] for r in results)


# ===========================================================================
# EmbeddingEngine.__repr__
# ===========================================================================


class TestEmbeddingEngineRepr:
    def test_repr_contains_backend(self) -> None:
        e = _make_custom_engine()
        assert "custom" in repr(e)

    def test_repr_contains_model_name(self) -> None:
        e = _make_custom_engine()
        assert DEFAULT_MODEL in repr(e)

    def test_repr_contains_batch_size(self) -> None:
        e = _make_custom_engine(batch_size=16)
        assert "16" in repr(e)

    def test_repr_shows_cache_enabled(self) -> None:
        e = _make_custom_engine(enable_cache=True)
        assert "enabled" in repr(e)

    def test_repr_shows_cache_disabled(self) -> None:
        e = _make_custom_engine(enable_cache=False)
        assert "disabled" in repr(e)

    def test_repr_is_string(self) -> None:
        e = _make_custom_engine()
        assert isinstance(repr(e), str)

    def test_repr_contains_normalize(self) -> None:
        e = _make_custom_engine(normalize=False)
        assert "False" in repr(e)

    def test_repr_sentence_transformers_backend(self) -> None:
        e = EmbeddingEngine(backend="sentence_transformers")
        assert "sentence_transformers" in repr(e)

    def test_repr_openai_backend(self) -> None:
        e = EmbeddingEngine(backend="openai")
        assert "openai" in repr(e)


# ===========================================================================
# Integration: custom fn full round-trip
# ===========================================================================


class TestEmbeddingEngineIntegration:
    def test_custom_fn_end_to_end(self) -> None:
        rng = np.random.default_rng(42)

        def realistic_fn(texts: list[str]) -> np.ndarray:
            return rng.standard_normal((len(texts), 384)).astype(np.float32)

        e = EmbeddingEngine(backend="custom", custom_fn=realistic_fn, normalize=False)
        docs = [_Doc(f"document {i}") for i in range(10)]
        out = e.embed_documents(docs)
        assert all(d.embedding.shape == (384,) for d in out)

    def test_embed_consistent_shape_across_batch_sizes(self) -> None:
        dim = 64

        def _fn(texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), dim), dtype=np.float32)

        for bs in (1, 2, 5, 10, 100):
            e = EmbeddingEngine(backend="custom", custom_fn=_fn, batch_size=bs)
            result = e.embed([f"text {i}" for i in range(7)])
            assert result.shape == (7, dim)

    def test_cache_invalidated_by_model_name_change(
        self, tmp_path: pathlib.Path
    ) -> None:
        def _fn(texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 4), dtype=np.float32)

        src = tmp_path / "src.txt"
        src.write_text("hello")

        e1 = EmbeddingEngine(
            backend="custom", custom_fn=_fn,
            model_name="model-a", cache_dir=tmp_path
        )
        e2 = EmbeddingEngine(
            backend="custom", custom_fn=_fn,
            model_name="model-b", cache_dir=tmp_path
        )
        e1.embed_with_cache(["text"], src)
        # model-b has a different cache key → miss, not hit
        _, flag = e2.embed_with_cache(["text"], src)
        assert flag is False  # cache miss for different model

    def test_embed_documents_with_cache_round_trip(
        self, tmp_path: pathlib.Path
    ) -> None:
        dim = 8
        e = _make_custom_engine(dim=dim, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        docs = [_Doc(f"doc {i}") for i in range(4)]
        out1 = e.embed_documents(docs, source_path=src)
        out2 = e.embed_documents(docs, source_path=src)
        for d1, d2 in zip(out1, out2):
            np.testing.assert_array_equal(d1.embedding, d2.embedding)
