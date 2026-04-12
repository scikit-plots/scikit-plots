# corpus/_embeddings/tests/test__multimodal_embedding.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot.corpus._embeddings._multimodal_embedding.

Covers
------
- Module constants: DEFAULT_IMAGE_MODEL, DEFAULT_AUDIO_MODEL, DEFAULT_TEXT_MODEL
- _make_linear_projection: shape, normalisation, determinism, zero-vector safety
- _make_clip_fn: ImportError guard when transformers/torch absent
- _make_open_clip_fn: ImportError guard when open_clip absent
- _make_whisper_encoder_fn: ImportError guard when transformers/torch absent
- _make_wav2vec_fn: ImportError guard when transformers/torch absent
- MultimodalEmbeddingEngine.__post_init__: all validation branches
- MultimodalEmbeddingEngine.embed_texts: empty, custom backend, delegation
- MultimodalEmbeddingEngine.embed_images: empty, custom backend
- MultimodalEmbeddingEngine.embed_audio: empty, custom backend
- MultimodalEmbeddingEngine.embed_video: empty, single frame, multi frame, sampling
- MultimodalEmbeddingEngine.embed_documents: text/image/audio/video dispatch,
  multimodal fusion (mean/concat/text_only/image_only), unknown modality fallback,
  missing raw_tensor fallback, projection_dim applied, custom_projection_fn
- MultimodalEmbeddingEngine._maybe_project: skip when None, skip same dim,
  random projection, custom projection
- MultimodalEmbeddingEngine._get_projection_fn: caching behaviour
- MultimodalEmbeddingEngine._embed_multimodal: all fusion strategies, dim mismatch
- MultimodalEmbeddingEngine.embed_documents_with_cache: hit, miss, stat failure, disabled
- MultimodalEmbeddingEngine.__repr__: content and format
- LLMTrainingExporter construction and repr
- LLMTrainingExporter._ensure_embedded: all-embedded, partial, engine-None raises
- LLMTrainingExporter.to_openai_finetuning_jsonl: writes JSONL, skips empty,
  response_fn, system prompt, include_embeddings, skip_empty=False, metadata
- LLMTrainingExporter.to_embedding_matrix: shape, metadata dict, output file,
  missing embeddings raise, pandas fallback
- LLMTrainingExporter.to_huggingface_training_dataset: ImportError when no transformers
- LLMTrainingExporter.log_to_mlflow: ImportError when no mlflow
- LLMTrainingExporter._mask_tokens: shape, mask-token presence
- Import-skip markers for clip, open_clip, whisper, wav2vec, openai, sentence-transformers
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .._multimodal_embedding import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    LLMTrainingExporter,
    MultimodalEmbeddingEngine,
    _make_clip_fn,
    _make_linear_projection,
    _make_open_clip_fn,
    _make_wav2vec_fn,
    _make_whisper_encoder_fn,
)


# ---------------------------------------------------------------------------
# Extended CorpusDocument stub for multimodal tests
# ---------------------------------------------------------------------------


class _Doc:
    """Minimal CorpusDocument stub covering all multimodal engine fields."""

    def __init__(
        self,
        text: str = "Sample text.",
        normalized_text: str | None = None,
        embedding: Any = None,
        modality: str = "text",
        raw_tensor: Any = None,
        doc_id: str = "deadbeef01234567",
        source_file: str = "test.txt",
        source_type: str = "article",
        chunk_index: int = 0,
        content_hash: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.text = text
        self.normalized_text = normalized_text
        self.embedding = embedding
        self.modality = modality
        self.raw_tensor = raw_tensor
        self.doc_id = doc_id
        self.source_file = source_file
        self.source_type = source_type
        self.chunk_index = chunk_index
        self.content_hash = content_hash
        self.metadata = metadata or {}

    def replace(self, **kwargs: Any) -> "_Doc":
        d = _Doc(
            text=self.text,
            normalized_text=self.normalized_text,
            embedding=self.embedding,
            modality=self.modality,
            raw_tensor=self.raw_tensor,
            doc_id=self.doc_id,
            source_file=self.source_file,
            source_type=self.source_type,
            chunk_index=self.chunk_index,
            content_hash=self.content_hash,
            metadata=dict(self.metadata),
        )
        for k, v in kwargs.items():
            setattr(d, k, v)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _custom_engine(
    text_dim: int = 64,
    image_dim: int = 32,
    audio_dim: int = 48,
    normalize: bool = False,
    **kwargs: Any,
) -> MultimodalEmbeddingEngine:
    """Return a MultimodalEmbeddingEngine backed entirely by custom callables."""
    return MultimodalEmbeddingEngine(
        text_backend="custom",
        text_custom_fn=lambda t: np.ones((len(t), text_dim), dtype=np.float32),
        image_backend="custom",
        image_custom_fn=lambda a: np.full((len(a), image_dim), 2.0, dtype=np.float32),
        audio_backend="custom",
        audio_custom_fn=lambda w: np.full((len(w), audio_dim), 3.0, dtype=np.float32),
        normalize=normalize,
        **kwargs,
    )


# ===========================================================================
# Module constants
# ===========================================================================


class TestConstants:
    def test_default_image_model_is_string(self) -> None:
        assert isinstance(DEFAULT_IMAGE_MODEL, str)
        assert "clip" in DEFAULT_IMAGE_MODEL.lower()

    def test_default_audio_model_is_string(self) -> None:
        assert isinstance(DEFAULT_AUDIO_MODEL, str)
        assert "whisper" in DEFAULT_AUDIO_MODEL.lower()

    def test_default_text_model_is_string(self) -> None:
        assert isinstance(DEFAULT_TEXT_MODEL, str)
        assert len(DEFAULT_TEXT_MODEL) > 0


# ===========================================================================
# _make_linear_projection
# ===========================================================================


class TestMakeLinearProjection:
    def test_output_shape(self) -> None:
        proj = _make_linear_projection(128, 64, normalize=False)
        v = np.ones((5, 128), dtype=np.float32)
        out = proj(v)
        assert out.shape == (5, 64)

    def test_normalize_true_produces_unit_norms(self) -> None:
        proj = _make_linear_projection(128, 64, normalize=True)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 128)).astype(np.float32)
        out = proj(v)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_normalize_false_does_not_unit_normalise(self) -> None:
        proj = _make_linear_projection(64, 32, normalize=False)
        v = np.ones((3, 64), dtype=np.float32) * 10.0
        out = proj(v)
        norms = np.linalg.norm(out, axis=1)
        # Not all norms should be 1.0 when normalize=False
        assert not np.allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_same_dims(self) -> None:
        p1 = _make_linear_projection(128, 64, normalize=False)
        p2 = _make_linear_projection(128, 64, normalize=False)
        v = np.ones((3, 128), dtype=np.float32)
        np.testing.assert_array_equal(p1(v), p2(v))

    def test_different_dims_different_result(self) -> None:
        p1 = _make_linear_projection(128, 64, normalize=False)
        p2 = _make_linear_projection(128, 32, normalize=False)
        v = np.ones((3, 128), dtype=np.float32)
        assert p1(v).shape != p2(v).shape

    def test_zero_vector_handled(self) -> None:
        proj = _make_linear_projection(16, 8, normalize=True)
        v = np.zeros((2, 16), dtype=np.float32)
        out = proj(v)
        assert out.shape == (2, 8)
        # Zero input → zero output (no NaN)
        assert not np.any(np.isnan(out))

    def test_output_is_float32(self) -> None:
        proj = _make_linear_projection(32, 16, normalize=False)
        v = np.ones((2, 32), dtype=np.float64)
        out = proj(v)
        assert out.dtype == np.float32

    def test_single_vector(self) -> None:
        proj = _make_linear_projection(64, 32, normalize=False)
        v = np.ones((1, 64), dtype=np.float32)
        out = proj(v)
        assert out.shape == (1, 32)


# ===========================================================================
# Heavy-dep backend factory ImportError guards
# ===========================================================================


class TestMakeClipFnImportError:
    def test_raises_import_error_transformers(self) -> None:
        fn = _make_clip_fn("openai/clip-vit-base-patch32", 8, True, None)
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(ImportError, match="CLIP"):
                fn([np.zeros((4, 4, 3), dtype=np.uint8)])

    def test_raises_import_error_torch_only(self) -> None:
        fn = _make_clip_fn("openai/clip-vit-base-patch32", 8, True, None)
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises((ImportError, Exception)):
                fn([np.zeros((4, 4, 3), dtype=np.uint8)])


class TestMakeOpenClipFnImportError:
    def test_raises_import_error_when_not_installed(self) -> None:
        fn = _make_open_clip_fn("ViT-B-32", 8, True, None)
        with patch.dict("sys.modules", {"open_clip": None}):
            with pytest.raises(ImportError, match="open_clip"):
                fn([np.zeros((4, 4, 3), dtype=np.uint8)])


class TestMakeWhisperEncoderFnImportError:
    def test_raises_import_error_when_not_installed(self) -> None:
        fn = _make_whisper_encoder_fn("base", 4, True, None)
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(ImportError, match="Whisper"):
                fn([np.zeros(16000, dtype=np.float32)])


class TestMakeWav2VecFnImportError:
    def test_raises_import_error_when_not_installed(self) -> None:
        fn = _make_wav2vec_fn("facebook/wav2vec2-base", 4, True, None)
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(ImportError, match="wav2vec2"):
                fn([np.zeros(16000, dtype=np.float32)])


# ===========================================================================
# MultimodalEmbeddingEngine — construction & validation
# ===========================================================================


class TestMultimodalEmbeddingEngineValidation:
    def test_default_text_backend(self) -> None:
        e = MultimodalEmbeddingEngine()
        assert e.text_backend == "sentence_transformers"

    def test_default_image_backend(self) -> None:
        e = MultimodalEmbeddingEngine()
        assert e.image_backend == "clip"

    def test_default_audio_backend(self) -> None:
        e = MultimodalEmbeddingEngine()
        assert e.audio_backend == "whisper"

    def test_default_fusion(self) -> None:
        e = MultimodalEmbeddingEngine()
        assert e.multimodal_fusion == "mean"

    def test_default_normalize_is_true(self) -> None:
        e = MultimodalEmbeddingEngine()
        assert e.normalize is True

    def test_default_batch_size(self) -> None:
        e = MultimodalEmbeddingEngine()
        assert e.batch_size == 32  # noqa: PLR2004

    def test_invalid_text_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="text_backend"):
            MultimodalEmbeddingEngine(text_backend="bert")

    def test_invalid_image_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="image_backend"):
            MultimodalEmbeddingEngine(image_backend="vit")

    def test_invalid_audio_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="audio_backend"):
            MultimodalEmbeddingEngine(audio_backend="speechbrain")

    def test_invalid_fusion_raises(self) -> None:
        with pytest.raises(ValueError, match="multimodal_fusion"):
            MultimodalEmbeddingEngine(multimodal_fusion="weighted_sum")

    def test_custom_text_backend_requires_fn(self) -> None:
        with pytest.raises(ValueError, match="text_custom_fn"):
            MultimodalEmbeddingEngine(text_backend="custom", text_custom_fn=None)

    def test_custom_image_backend_requires_fn(self) -> None:
        with pytest.raises(ValueError, match="image_custom_fn"):
            MultimodalEmbeddingEngine(image_backend="custom", image_custom_fn=None)

    def test_custom_audio_backend_requires_fn(self) -> None:
        with pytest.raises(ValueError, match="audio_custom_fn"):
            MultimodalEmbeddingEngine(audio_backend="custom", audio_custom_fn=None)

    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _custom_engine(batch_size=0)

    def test_batch_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _custom_engine(batch_size=-5)

    def test_all_valid_text_backends_accepted(self) -> None:
        for backend in MultimodalEmbeddingEngine.VALID_TEXT_BACKENDS:
            if backend != "custom":
                e = MultimodalEmbeddingEngine(text_backend=backend)
                assert e.text_backend == backend

    def test_all_valid_image_backends_accepted(self) -> None:
        for backend in MultimodalEmbeddingEngine.VALID_IMAGE_BACKENDS:
            if backend != "custom":
                e = MultimodalEmbeddingEngine(image_backend=backend)
                assert e.image_backend == backend

    def test_all_valid_audio_backends_accepted(self) -> None:
        for backend in MultimodalEmbeddingEngine.VALID_AUDIO_BACKENDS:
            if backend != "custom":
                e = MultimodalEmbeddingEngine(audio_backend=backend)
                assert e.audio_backend == backend

    def test_all_valid_fusion_strategies_accepted(self) -> None:
        for fusion in MultimodalEmbeddingEngine.VALID_FUSION:
            e = _custom_engine(multimodal_fusion=fusion)
            assert e.multimodal_fusion == fusion

    def test_projection_dim_none_by_default(self) -> None:
        e = _custom_engine()
        assert e.projection_dim is None


# ===========================================================================
# MultimodalEmbeddingEngine.embed_texts
# ===========================================================================


class TestMultimodalEmbeddingEngineEmbedTexts:
    def test_empty_returns_empty_array(self) -> None:
        e = _custom_engine()
        out = e.embed_texts([])
        assert out.shape[0] == 0

    def test_custom_backend_shape(self) -> None:
        e = _custom_engine(text_dim=64)
        out = e.embed_texts(["hello", "world"])
        assert out.shape == (2, 64)

    def test_custom_backend_dtype(self) -> None:
        e = _custom_engine()
        out = e.embed_texts(["test"])
        assert out.dtype == np.float32

    def test_text_engine_lazily_initialised(self) -> None:
        e = _custom_engine()
        assert e._text_engine is None
        e.embed_texts(["trigger init"])
        assert e._text_engine is not None

    def test_text_engine_reused_across_calls(self) -> None:
        e = _custom_engine()
        e.embed_texts(["a"])
        te1 = e._text_engine
        e.embed_texts(["b"])
        te2 = e._text_engine
        assert te1 is te2


# ===========================================================================
# MultimodalEmbeddingEngine.embed_images
# ===========================================================================


class TestMultimodalEmbeddingEngineEmbedImages:
    def test_empty_returns_empty_array(self) -> None:
        e = _custom_engine()
        out = e.embed_images([])
        assert out.shape[0] == 0

    def test_custom_backend_shape(self) -> None:
        e = _custom_engine(image_dim=32)
        arrays = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        out = e.embed_images(arrays)
        assert out.shape == (3, 32)

    def test_custom_backend_dtype(self) -> None:
        e = _custom_engine()
        arrays = [np.zeros((4, 4, 3), dtype=np.uint8)]
        out = e.embed_images(arrays)
        assert out.dtype == np.float32

    def test_image_fn_lazily_initialised(self) -> None:
        e = _custom_engine()
        assert e._image_fn is None
        e.embed_images([np.zeros((4, 4, 3), np.uint8)])
        assert e._image_fn is not None

    def test_image_fn_reused(self) -> None:
        e = _custom_engine()
        arr = np.zeros((4, 4, 3), np.uint8)
        e.embed_images([arr])
        fn1 = e._image_fn
        e.embed_images([arr])
        assert e._image_fn is fn1


# ===========================================================================
# MultimodalEmbeddingEngine.embed_audio
# ===========================================================================


class TestMultimodalEmbeddingEngineEmbedAudio:
    def test_empty_returns_empty_array(self) -> None:
        e = _custom_engine()
        out = e.embed_audio([])
        assert out.shape[0] == 0

    def test_custom_backend_shape(self) -> None:
        e = _custom_engine(audio_dim=48)
        waveforms = [np.zeros(16000, dtype=np.float32) for _ in range(2)]
        out = e.embed_audio(waveforms)
        assert out.shape == (2, 48)

    def test_custom_backend_dtype(self) -> None:
        e = _custom_engine()
        out = e.embed_audio([np.zeros(1000, np.float32)])
        assert out.dtype == np.float32

    def test_audio_fn_lazily_initialised(self) -> None:
        e = _custom_engine()
        assert e._audio_fn is None
        e.embed_audio([np.zeros(100, np.float32)])
        assert e._audio_fn is not None


# ===========================================================================
# MultimodalEmbeddingEngine.embed_video
# ===========================================================================


class TestMultimodalEmbeddingEngineEmbedVideo:
    def test_empty_returns_empty_array(self) -> None:
        e = _custom_engine()
        out = e.embed_video([])
        assert out.shape[0] == 0

    def test_multi_frame_video_shape(self) -> None:
        e = _custom_engine(image_dim=32, normalize=False)
        frames = np.zeros((16, 4, 4, 3), dtype=np.uint8)
        out = e.embed_video([frames])
        assert out.shape == (1, 32)

    def test_single_frame_video(self) -> None:
        """3-D (H,W,C) input must be handled as single frame."""
        e = _custom_engine(image_dim=32, normalize=False)
        single = np.zeros((4, 4, 3), dtype=np.uint8)  # (H,W,C)
        out = e.embed_video([single])
        assert out.shape == (1, 32)

    def test_multiple_videos(self) -> None:
        e = _custom_engine(image_dim=16, normalize=False)
        videos = [np.zeros((8, 4, 4, 3), np.uint8) for _ in range(3)]
        out = e.embed_video(videos)
        assert out.shape == (3, 16)

    def test_output_is_float32(self) -> None:
        e = _custom_engine(normalize=False)
        out = e.embed_video([np.zeros((4, 4, 4, 3), np.uint8)])
        assert out.dtype == np.float32

    def test_n_sample_frames_respected(self) -> None:
        """With n_sample_frames=2, exactly 2 frames sampled per video."""
        call_counts: list[int] = []

        def _counting_image_fn(arrays: list) -> np.ndarray:
            call_counts.append(len(arrays))
            return np.ones((len(arrays), 8), dtype=np.float32)

        e = MultimodalEmbeddingEngine(
            text_backend="custom",
            text_custom_fn=lambda t: np.zeros((len(t), 8), np.float32),
            image_backend="custom",
            image_custom_fn=_counting_image_fn,
            audio_backend="custom",
            audio_custom_fn=lambda w: np.zeros((len(w), 8), np.float32),
            normalize=False,
        )
        frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
        e.embed_video([frames], n_sample_frames=2)
        assert call_counts[-1] == 2  # noqa: PLR2004

    def test_normalize_unit_norm(self) -> None:
        e = _custom_engine(image_dim=8, normalize=True)
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        out = e.embed_video([frames])
        norm = float(np.linalg.norm(out[0]))
        assert abs(norm - 1.0) < 1e-5 or np.isclose(norm, 0.0)


# ===========================================================================
# MultimodalEmbeddingEngine.embed_documents (dispatch by modality)
# ===========================================================================


class TestMultimodalEmbeddingEngineEmbedDocuments:
    def test_empty_returns_empty_list(self) -> None:
        e = _custom_engine()
        assert e.embed_documents([]) == []

    def test_text_modality_dispatched(self) -> None:
        e = _custom_engine(text_dim=64, normalize=False)
        docs = [_Doc("Hello world.", modality="text")]
        out = e.embed_documents(docs)
        assert out[0].embedding.shape == (64,)
        np.testing.assert_allclose(out[0].embedding, np.ones(64))

    def test_image_modality_dispatched(self) -> None:
        e = _custom_engine(image_dim=32, normalize=False)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        docs = [_Doc("", modality="image", raw_tensor=img)]
        out = e.embed_documents(docs)
        assert out[0].embedding.shape == (32,)
        np.testing.assert_allclose(out[0].embedding, np.full(32, 2.0))

    def test_audio_modality_dispatched(self) -> None:
        e = _custom_engine(audio_dim=48, normalize=False)
        waveform = np.zeros(16000, dtype=np.float32)
        docs = [_Doc("", modality="audio", raw_tensor=waveform)]
        out = e.embed_documents(docs)
        assert out[0].embedding.shape == (48,)
        np.testing.assert_allclose(out[0].embedding, np.full(48, 3.0))

    def test_video_modality_dispatched(self) -> None:
        e = _custom_engine(image_dim=32, normalize=False)
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        docs = [_Doc("", modality="video", raw_tensor=frames)]
        out = e.embed_documents(docs)
        assert out[0].embedding is not None
        assert out[0].embedding.shape[0] == 32  # noqa: PLR2004

    def test_unknown_modality_falls_back_to_text(self) -> None:
        e = _custom_engine(text_dim=16, normalize=False)
        docs = [_Doc("fallback", modality="unknown_xyz")]
        out = e.embed_documents(docs)
        assert out[0].embedding.shape == (16,)

    def test_image_missing_raw_tensor_uses_zeros(self) -> None:
        """IMAGE doc with raw_tensor=None must not crash — uses zero array."""
        e = _custom_engine(image_dim=8, normalize=False)
        docs = [_Doc("", modality="image", raw_tensor=None)]
        out = e.embed_documents(docs)
        assert out[0].embedding is not None
        assert out[0].embedding.shape == (8,)

    def test_audio_missing_raw_tensor_uses_zeros(self) -> None:
        e = _custom_engine(audio_dim=8, normalize=False)
        docs = [_Doc("", modality="audio", raw_tensor=None)]
        out = e.embed_documents(docs)
        assert out[0].embedding is not None

    def test_video_missing_raw_tensor_uses_zeros(self) -> None:
        e = _custom_engine(image_dim=8, normalize=False)
        docs = [_Doc("", modality="video", raw_tensor=None)]
        out = e.embed_documents(docs)
        assert out[0].embedding is not None

    def test_preserves_original_doc_order(self) -> None:
        e = _custom_engine(text_dim=4, image_dim=4, normalize=False)
        docs = [
            _Doc("text 0", modality="text"),
            _Doc("img 1", modality="image", raw_tensor=np.zeros((4, 4, 3), np.uint8)),
            _Doc("text 2", modality="text"),
        ]
        out = e.embed_documents(docs)
        assert len(out) == 3  # noqa: PLR2004
        assert out[0].embedding is not None
        assert out[1].embedding is not None
        assert out[2].embedding is not None

    def test_original_docs_not_mutated(self) -> None:
        e = _custom_engine()
        docs = [_Doc("hello")]
        out = e.embed_documents(docs)
        assert docs[0].embedding is None  # original unchanged
        assert out[0].embedding is not None

    def test_projection_dim_applied_to_output(self) -> None:
        e = _custom_engine(text_dim=64, normalize=False, projection_dim=16)
        docs = [_Doc("text", modality="text")]
        out = e.embed_documents(docs)
        assert out[0].embedding.shape == (16,)

    def test_custom_projection_fn_used(self) -> None:
        def my_proj(vecs: np.ndarray) -> np.ndarray:
            return np.zeros((vecs.shape[0], 8), dtype=np.float32)

        e = _custom_engine(
            text_dim=64, normalize=False,
            projection_dim=8, custom_projection_fn=my_proj
        )
        docs = [_Doc("text"), _Doc("another")]
        out = e.embed_documents(docs)
        for d in out:
            assert d.embedding.shape == (8,)
            np.testing.assert_array_equal(d.embedding, np.zeros(8))


# ===========================================================================
# MultimodalEmbeddingEngine — fusion strategies
# ===========================================================================


class TestMultimodalEmbeddingEngineFusion:
    def test_fusion_text_only(self) -> None:
        e = _custom_engine(text_dim=4, image_dim=4, normalize=False,
                           multimodal_fusion="text_only")
        img = np.zeros((4, 4, 3), np.uint8)
        docs = [_Doc("text", modality="multimodal", raw_tensor=img)]
        out = e.embed_documents(docs)
        # text embeddings are all-ones; image would be all-twos
        np.testing.assert_allclose(out[0].embedding, np.ones(4))

    def test_fusion_image_only(self) -> None:
        e = _custom_engine(text_dim=4, image_dim=4, normalize=False,
                           multimodal_fusion="image_only")
        img = np.zeros((4, 4, 3), np.uint8)
        docs = [_Doc("text", modality="multimodal", raw_tensor=img)]
        out = e.embed_documents(docs)
        np.testing.assert_allclose(out[0].embedding, np.full(4, 2.0))

    def test_fusion_concat(self) -> None:
        e = _custom_engine(text_dim=4, image_dim=4, normalize=False,
                           multimodal_fusion="concat")
        img = np.zeros((4, 4, 3), np.uint8)
        docs = [_Doc("text", modality="multimodal", raw_tensor=img)]
        out = e.embed_documents(docs)
        # concat of (4,) + (4,) → (8,)
        assert out[0].embedding.shape == (8,)

    def test_fusion_mean_same_dim(self) -> None:
        e = _custom_engine(text_dim=4, image_dim=4, normalize=False,
                           multimodal_fusion="mean")
        img = np.zeros((4, 4, 3), np.uint8)
        docs = [_Doc("text", modality="multimodal", raw_tensor=img)]
        out = e.embed_documents(docs)
        # mean of ones(4) and twos(4) → 1.5
        np.testing.assert_allclose(out[0].embedding, np.full(4, 1.5), atol=1e-6)

    def test_fusion_mean_mismatched_dims_uses_projection(self) -> None:
        """mean fusion with different text/image dims must not crash."""
        e = _custom_engine(text_dim=8, image_dim=4, normalize=False,
                           multimodal_fusion="mean")
        img = np.zeros((4, 4, 3), np.uint8)
        docs = [_Doc("text", modality="multimodal", raw_tensor=img)]
        out = e.embed_documents(docs)
        # Output dim matches text_dim after projection of image
        assert out[0].embedding.shape == (8,)


# ===========================================================================
# MultimodalEmbeddingEngine._maybe_project
# ===========================================================================


class TestMaybeProject:
    def test_skips_when_projection_dim_none(self) -> None:
        e = _custom_engine(projection_dim=None)
        v = np.ones((3, 64), dtype=np.float32)
        out = e._maybe_project(v)
        assert out.shape == (3, 64)
        np.testing.assert_array_equal(out, v)

    def test_skips_when_already_correct_dim(self) -> None:
        e = _custom_engine(projection_dim=32, normalize=False)
        v = np.ones((4, 32), dtype=np.float32)
        out = e._maybe_project(v)
        assert out.shape == (4, 32)

    def test_applies_projection_when_needed(self) -> None:
        e = _custom_engine(projection_dim=16, normalize=False)
        v = np.ones((3, 64), dtype=np.float32)
        out = e._maybe_project(v)
        assert out.shape == (3, 16)

    def test_empty_array_skipped(self) -> None:
        e = _custom_engine(projection_dim=16)
        v = np.empty((0, 64), dtype=np.float32)
        out = e._maybe_project(v)
        assert out.shape[0] == 0

    def test_custom_projection_fn_used(self) -> None:
        custom_proj = lambda v: np.zeros((v.shape[0], 8), np.float32)  # noqa: E731
        e = _custom_engine(projection_dim=8, normalize=False,
                           custom_projection_fn=custom_proj)
        v = np.ones((2, 64), dtype=np.float32)
        out = e._maybe_project(v)
        np.testing.assert_array_equal(out, np.zeros((2, 8)))


# ===========================================================================
# MultimodalEmbeddingEngine._get_projection_fn (caching)
# ===========================================================================


class TestGetProjectionFn:
    def test_returns_callable(self) -> None:
        e = _custom_engine()
        fn = e._get_projection_fn(64, 32)
        assert callable(fn)

    def test_same_dims_returns_same_fn(self) -> None:
        e = _custom_engine()
        fn1 = e._get_projection_fn(64, 32)
        fn2 = e._get_projection_fn(64, 32)
        assert fn1 is fn2

    def test_different_dims_different_fn(self) -> None:
        e = _custom_engine()
        fn1 = e._get_projection_fn(64, 32)
        fn2 = e._get_projection_fn(64, 16)
        assert fn1 is not fn2


# ===========================================================================
# MultimodalEmbeddingEngine.embed_documents_with_cache
# ===========================================================================


class TestMultimodalEmbeddingEngineEmbedDocumentsWithCache:
    def test_disabled_cache_bypasses_cache(self, tmp_path: pathlib.Path) -> None:
        e = _custom_engine(enable_cache=False, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        docs = [_Doc("text")]
        out = e.embed_documents_with_cache(docs, src)
        assert out[0].embedding is not None
        assert len(list(tmp_path.glob("*.npy"))) == 0

    def test_stat_failure_bypasses_cache(self, tmp_path: pathlib.Path) -> None:
        e = _custom_engine(cache_dir=tmp_path)
        missing = tmp_path / "no_such_file.txt"
        docs = [_Doc("text")]
        out = e.embed_documents_with_cache(docs, missing)
        assert out[0].embedding is not None

    def test_cache_miss_creates_file(self, tmp_path: pathlib.Path) -> None:
        e = _custom_engine(cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        docs = [_Doc("hello"), _Doc("world")]
        e.embed_documents_with_cache(docs, src)
        npy_files = list(tmp_path.glob("*.npy"))
        assert len(npy_files) == 1

    def test_cache_hit_reloads(self, tmp_path: pathlib.Path) -> None:
        e = _custom_engine(text_dim=8, normalize=False, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        docs = [_Doc("hello")]
        out1 = e.embed_documents_with_cache(docs, src)
        out2 = e.embed_documents_with_cache(docs, src)
        np.testing.assert_array_equal(out1[0].embedding, out2[0].embedding)

    def test_corrupt_cache_recomputes(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        e = _custom_engine(text_dim=4, normalize=False, cache_dir=tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("data")
        docs = [_Doc("hello")]
        # Write a corrupt .npy file under the right cache key pattern
        for npy in tmp_path.glob("*.npy"):
            npy.write_bytes(b"garbage bytes")
        with caplog.at_level(logging.WARNING):
            out = e.embed_documents_with_cache(docs, src)
        assert out[0].embedding is not None


# ===========================================================================
# MultimodalEmbeddingEngine.__repr__
# ===========================================================================


class TestMultimodalEmbeddingEngineRepr:
    def test_repr_is_string(self) -> None:
        e = _custom_engine()
        assert isinstance(repr(e), str)

    def test_repr_contains_text_backend(self) -> None:
        e = _custom_engine()
        assert "custom" in repr(e)

    def test_repr_contains_fusion(self) -> None:
        e = _custom_engine(multimodal_fusion="concat")
        assert "concat" in repr(e)

    def test_repr_contains_projection_dim(self) -> None:
        e = _custom_engine(projection_dim=128)
        assert "128" in repr(e)

    def test_repr_contains_none_projection(self) -> None:
        e = _custom_engine(projection_dim=None)
        assert "None" in repr(e)


# ===========================================================================
# LLMTrainingExporter — construction
# ===========================================================================


class TestLLMTrainingExporterConstruction:
    def test_default_engine_is_none(self) -> None:
        exp = LLMTrainingExporter()
        assert exp.engine is None

    def test_default_system_prompt(self) -> None:
        exp = LLMTrainingExporter()
        assert isinstance(exp.default_system_prompt, str)
        assert len(exp.default_system_prompt) > 0

    def test_custom_system_prompt(self) -> None:
        exp = LLMTrainingExporter(default_system_prompt="Be concise.")
        assert exp.default_system_prompt == "Be concise."

    def test_with_engine(self) -> None:
        engine = _custom_engine()
        exp = LLMTrainingExporter(engine=engine)
        assert exp.engine is engine


# ===========================================================================
# LLMTrainingExporter._ensure_embedded
# ===========================================================================


class TestLLMTrainingExporterEnsureEmbedded:
    def test_no_embed_when_all_have_embedding(self) -> None:
        emb = np.zeros(4, dtype=np.float32)
        docs = [_Doc("a", embedding=emb), _Doc("b", embedding=emb)]
        exp = LLMTrainingExporter()
        out = exp._ensure_embedded(docs)
        assert out is docs  # no-op, same list returned

    def test_raises_when_engine_none_and_missing_embeddings(self) -> None:
        docs = [_Doc("no embedding")]
        exp = LLMTrainingExporter(engine=None)
        with pytest.raises(ValueError, match="engine=None"):
            exp._ensure_embedded(docs)

    def test_embeds_missing_documents(self) -> None:
        engine = _custom_engine(text_dim=4, normalize=False)
        exp = LLMTrainingExporter(engine=engine)
        docs = [
            _Doc("has emb", embedding=np.ones(4, np.float32)),
            _Doc("needs emb"),
        ]
        out = exp._ensure_embedded(docs)
        assert out[1].embedding is not None

    def test_already_embedded_kept_unchanged(self) -> None:
        engine = _custom_engine(text_dim=4, normalize=False)
        exp = LLMTrainingExporter(engine=engine)
        original_emb = np.full(4, 99.0, dtype=np.float32)
        docs = [_Doc("text", embedding=original_emb)]
        out = exp._ensure_embedded(docs)
        np.testing.assert_array_equal(out[0].embedding, original_emb)


# ===========================================================================
# LLMTrainingExporter.to_openai_finetuning_jsonl
# ===========================================================================


class TestLLMTrainingExporterOpenAIJsonl:
    def _make_docs(self, n: int = 3) -> list[_Doc]:
        return [_Doc(f"Document number {i}.") for i in range(n)]

    def test_creates_jsonl_file(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(self._make_docs(), path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter()
        path = tmp_path / "nested" / "dir" / "out.jsonl"
        exp.to_openai_finetuning_jsonl(self._make_docs(), path)
        assert path.exists()

    def test_returns_path(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter()
        result = exp.to_openai_finetuning_jsonl(self._make_docs(), tmp_path / "f.jsonl")
        assert isinstance(result, pathlib.Path)

    def test_correct_number_of_lines(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(self._make_docs(5), path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5  # noqa: PLR2004

    def test_each_line_is_valid_json(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(self._make_docs(), path)
        for line in path.read_text().strip().split("\n"):
            obj = json.loads(line)
            assert "messages" in obj

    def test_system_message_present(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter(default_system_prompt="Test prompt.")
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(self._make_docs(1), path)
        obj = json.loads(path.read_text().strip())
        roles = [m["role"] for m in obj["messages"]]
        assert "system" in roles
        system_content = next(m["content"] for m in obj["messages"] if m["role"] == "system")
        assert system_content == "Test prompt."

    def test_user_message_contains_doc_text(self, tmp_path: pathlib.Path) -> None:
        docs = [_Doc("My unique text content.")]
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(docs, path)
        obj = json.loads(path.read_text().strip())
        user_content = next(m["content"] for m in obj["messages"] if m["role"] == "user")
        assert "My unique text content." in user_content

    def test_response_fn_adds_assistant_message(self, tmp_path: pathlib.Path) -> None:
        docs = [_Doc("Question text.")]
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(docs, path, response_fn=lambda d: "My answer.")
        obj = json.loads(path.read_text().strip())
        roles = [m["role"] for m in obj["messages"]]
        assert "assistant" in roles
        asst = next(m["content"] for m in obj["messages"] if m["role"] == "assistant")
        assert asst == "My answer."

    def test_empty_response_fn_no_assistant(self, tmp_path: pathlib.Path) -> None:
        docs = [_Doc("text")]
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(docs, path, response_fn=lambda d: "")
        obj = json.loads(path.read_text().strip())
        roles = [m["role"] for m in obj["messages"]]
        assert "assistant" not in roles

    def test_skip_empty_true_omits_empty_text(self, tmp_path: pathlib.Path) -> None:
        docs = [_Doc(""), _Doc("  "), _Doc("Valid text.")]
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(docs, path, skip_empty=True)
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) == 1

    def test_skip_empty_false_includes_empty(self, tmp_path: pathlib.Path) -> None:
        docs = [_Doc(""), _Doc("text")]
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(docs, path, skip_empty=False)
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) == 2  # noqa: PLR2004

    def test_include_embeddings(self, tmp_path: pathlib.Path) -> None:
        emb = np.ones(4, dtype=np.float32)
        docs = [_Doc("text", embedding=emb)]
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(docs, path, include_embeddings=True)
        obj = json.loads(path.read_text().strip())
        assert "embedding" in obj
        assert len(obj["embedding"]) == 4  # noqa: PLR2004

    def test_metadata_in_output(self, tmp_path: pathlib.Path) -> None:
        doc = _Doc("text", doc_id="abc123", chunk_index=7)
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl([doc], path)
        obj = json.loads(path.read_text().strip())
        assert "metadata" in obj
        assert obj["metadata"]["doc_id"] == "abc123"
        assert obj["metadata"]["chunk_index"] == 7  # noqa: PLR2004

    def test_custom_user_field(self, tmp_path: pathlib.Path) -> None:
        doc = _Doc("original text")
        doc.custom_field = "alternate content"
        exp = LLMTrainingExporter()
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl([doc], path, user_field="custom_field")
        obj = json.loads(path.read_text().strip())
        user_msg = next(m["content"] for m in obj["messages"] if m["role"] == "user")
        assert "alternate content" in user_msg

    def test_custom_system_prompt_arg(self, tmp_path: pathlib.Path) -> None:
        exp = LLMTrainingExporter(default_system_prompt="default")
        path = tmp_path / "out.jsonl"
        exp.to_openai_finetuning_jsonl(
            [_Doc("text")], path, system_prompt="override prompt"
        )
        obj = json.loads(path.read_text().strip())
        sys_content = next(m["content"] for m in obj["messages"] if m["role"] == "system")
        assert sys_content == "override prompt"


# ===========================================================================
# LLMTrainingExporter.to_embedding_matrix
# ===========================================================================


class TestLLMTrainingExporterEmbeddingMatrix:
    def _embedded_docs(self, n: int = 4, dim: int = 8) -> list[_Doc]:
        return [_Doc(f"doc {i}", embedding=np.full(dim, float(i), np.float32))
                for i in range(n)]

    def test_correct_matrix_shape(self) -> None:
        docs = self._embedded_docs(n=5, dim=16)
        exp = LLMTrainingExporter()
        matrix, _ = exp.to_embedding_matrix(docs)
        assert matrix.shape == (5, 16)

    def test_matrix_dtype_is_float32(self) -> None:
        docs = self._embedded_docs()
        exp = LLMTrainingExporter()
        matrix, _ = exp.to_embedding_matrix(docs)
        assert matrix.dtype == np.float32

    def test_matrix_values_match_embeddings(self) -> None:
        docs = self._embedded_docs(n=3, dim=4)
        exp = LLMTrainingExporter()
        matrix, _ = exp.to_embedding_matrix(docs)
        for i, d in enumerate(docs):
            np.testing.assert_array_equal(matrix[i], d.embedding)

    def test_returns_metadata_as_dict_without_pandas(self) -> None:
        docs = self._embedded_docs(n=2)
        exp = LLMTrainingExporter()
        with patch.dict("sys.modules", {"pandas": None}):
            matrix, meta = exp.to_embedding_matrix(docs)
        assert isinstance(meta, dict)
        assert "doc_id" in meta

    def test_metadata_with_pandas(self) -> None:
        pd = pytest.importorskip("pandas")
        docs = self._embedded_docs(n=2)
        exp = LLMTrainingExporter()
        _, meta = exp.to_embedding_matrix(docs)
        # If pandas is installed, should be a DataFrame
        assert hasattr(meta, "columns") or isinstance(meta, dict)

    def test_saves_npy_file_when_output_path_set(self, tmp_path: pathlib.Path) -> None:
        docs = self._embedded_docs(n=3, dim=4)
        exp = LLMTrainingExporter()
        out = tmp_path / "matrix"
        exp.to_embedding_matrix(docs, output_path=out)
        assert (tmp_path / "matrix.npy").exists()

    def test_raises_when_missing_embeddings_no_engine(self) -> None:
        docs = [_Doc("no embedding")]
        exp = LLMTrainingExporter(engine=None)
        with pytest.raises(ValueError, match="engine=None"):
            exp.to_embedding_matrix(docs)

    def test_embeds_via_engine_when_missing(self) -> None:
        engine = _custom_engine(text_dim=4, normalize=False)
        exp = LLMTrainingExporter(engine=engine)
        docs = [_Doc("needs embedding")]
        matrix, _ = exp.to_embedding_matrix(docs)
        assert matrix.shape == (1, 4)

    def test_include_metadata_false(self) -> None:
        """When include_metadata=False, meta has no column keys (dict or empty DF)."""
        docs = self._embedded_docs(n=2)
        exp = LLMTrainingExporter()
        matrix, meta = exp.to_embedding_matrix(docs, include_metadata=False)
        # When pandas installed: empty DataFrame; else: empty dict.
        if isinstance(meta, dict):
            assert len(meta) == 0
        else:
            assert len(meta.columns) == 0

    def test_still_raises_after_embed_if_no_embedding(self) -> None:
        """Engine that returns docs without embeddings → ValueError."""
        # Engine that returns original docs without setting embedding
        bad_engine = MagicMock()
        bad_engine.embed_documents.return_value = [_Doc("still no emb")]
        exp = LLMTrainingExporter(engine=bad_engine)
        docs = [_Doc("no embedding")]
        with pytest.raises((ValueError, AttributeError)):
            exp.to_embedding_matrix(docs)


# ===========================================================================
# LLMTrainingExporter.to_huggingface_training_dataset
# ===========================================================================


class TestLLMTrainingExporterHFDataset:
    def test_raises_import_error_when_no_transformers(self) -> None:
        exp = LLMTrainingExporter()
        docs = [_Doc("test text")]
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                exp.to_huggingface_training_dataset(docs)

    def test_returns_dict_when_datasets_not_installed(self) -> None:
        """When transformers IS installed but datasets is not, returns plain dict."""
        transformers = pytest.importorskip("transformers")
        exp = LLMTrainingExporter()
        docs = [_Doc("A simple sentence for tokenization.")]
        with patch.dict("sys.modules", {"datasets": None}):
            result = exp.to_huggingface_training_dataset(
                docs, tokenizer_name="gpt2", max_length=64
            )
        # Should be a dict (datasets not installed fallback)
        assert isinstance(result, dict)
        assert "input_ids" in result


# ===========================================================================
# LLMTrainingExporter.log_to_mlflow
# ===========================================================================


class TestLLMTrainingExporterMLflow:
    def test_raises_import_error_when_no_mlflow(self) -> None:
        engine = _custom_engine(text_dim=4, normalize=False)
        exp = LLMTrainingExporter(engine=engine)
        emb = np.zeros(4, np.float32)
        docs = [_Doc("text", embedding=emb)]
        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(ImportError, match="mlflow"):
                exp.log_to_mlflow(docs)


# ===========================================================================
# LLMTrainingExporter._mask_tokens
# ===========================================================================


class TestLLMTrainingExporterMaskTokens:
    def test_output_same_length(self) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 103
        ids = list(range(20))
        result = LLMTrainingExporter._mask_tokens(ids, tokenizer)
        assert len(result) == len(ids)

    def test_all_tokens_are_ints(self) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 103
        ids = list(range(50))
        result = LLMTrainingExporter._mask_tokens(ids, tokenizer)
        assert all(isinstance(t, int) for t in result)

    def test_mask_token_present_in_output(self) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 999
        ids = list(range(200))  # large list → statistically certain to have masks
        result = LLMTrainingExporter._mask_tokens(ids, tokenizer, mask_prob=0.5)
        assert 999 in result

    def test_zero_mask_prob_keeps_all_tokens(self) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 103
        ids = [1, 2, 3, 4, 5]
        result = LLMTrainingExporter._mask_tokens(ids, tokenizer, mask_prob=0.0)
        assert result == ids

    def test_full_mask_prob_all_masked(self) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 42
        ids = [1, 2, 3, 4, 5]
        result = LLMTrainingExporter._mask_tokens(ids, tokenizer, mask_prob=1.0)
        assert all(t == 42 for t in result)

    def test_fallback_mask_id_when_no_attr(self) -> None:
        tokenizer = MagicMock(spec=[])  # no mask_token_id attr
        # getattr fallback: None or 103
        ids = [1, 2, 3]
        # Should not raise
        result = LLMTrainingExporter._mask_tokens(ids, tokenizer, mask_prob=1.0)
        assert len(result) == 3  # noqa: PLR2004


# ===========================================================================
# LLMTrainingExporter.__repr__
# ===========================================================================


class TestLLMTrainingExporterRepr:
    def test_repr_without_engine(self) -> None:
        exp = LLMTrainingExporter()
        r = repr(exp)
        assert isinstance(r, str)
        assert "None" in r

    def test_repr_with_engine(self) -> None:
        engine = _custom_engine()
        exp = LLMTrainingExporter(engine=engine)
        r = repr(exp)
        assert "MultimodalEmbeddingEngine" in r

    def test_repr_is_string(self) -> None:
        assert isinstance(repr(LLMTrainingExporter()), str)


# ===========================================================================
# Integration: full round-trip with custom backends
# ===========================================================================


class TestMultimodalIntegration:
    def test_text_docs_round_trip(self) -> None:
        engine = _custom_engine(text_dim=8, normalize=False)
        exporter = LLMTrainingExporter(engine=engine)
        docs = [_Doc(f"Document {i}.") for i in range(5)]
        matrix, meta = exporter.to_embedding_matrix(docs)
        assert matrix.shape == (5, 8)

    def test_mixed_modality_round_trip(self) -> None:
        engine = _custom_engine(text_dim=4, image_dim=4, normalize=False)
        docs = [
            _Doc("text doc", modality="text"),
            _Doc("img doc", modality="image",
                 raw_tensor=np.zeros((4, 4, 3), np.uint8)),
        ]
        embedded = engine.embed_documents(docs)
        assert all(d.embedding is not None for d in embedded)
        assert all(d.embedding.shape == (4,) for d in embedded)

    def test_jsonl_export_from_embedded_docs(self, tmp_path: pathlib.Path) -> None:
        engine = _custom_engine(text_dim=4, normalize=False)
        exporter = LLMTrainingExporter(engine=engine)
        docs = [_Doc(f"sentence {i}") for i in range(3)]
        embedded = engine.embed_documents(docs)
        path = tmp_path / "train.jsonl"
        exporter.to_openai_finetuning_jsonl(embedded, path, include_embeddings=True)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # noqa: PLR2004
        for line in lines:
            obj = json.loads(line)
            assert "embedding" in obj
            assert len(obj["embedding"]) == 4  # noqa: PLR2004

    def test_projection_preserves_doc_count(self) -> None:
        engine = _custom_engine(text_dim=64, normalize=False, projection_dim=8)
        docs = [_Doc(f"doc {i}") for i in range(10)]
        out = engine.embed_documents(docs)
        assert len(out) == 10  # noqa: PLR2004
        assert all(d.embedding.shape == (8,) for d in out)

    def test_embed_matrix_output_consistent_with_embed_documents(self) -> None:
        engine = _custom_engine(text_dim=4, normalize=False)
        docs = [_Doc(f"text {i}") for i in range(4)]
        embedded = engine.embed_documents(docs)
        matrix = np.stack([d.embedding for d in embedded])
        exp = LLMTrainingExporter(engine=engine)
        m2, _ = exp.to_embedding_matrix(embedded)
        np.testing.assert_array_equal(matrix, m2)
