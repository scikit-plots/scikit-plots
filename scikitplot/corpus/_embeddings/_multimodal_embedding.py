# scikitplot/corpus/_embeddings/_multimodal_embedding.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._embeddings._multimodal_embedding
====================================================
Multimodal embedding engine — any modality to any LLM training format.

Bridges the corpus pipeline to LLM fine-tuning by producing a single
unified embedding vector for every :class:`~scikitplot.corpus.CorpusDocument`
regardless of whether it carries text, raw image pixels, audio waveforms,
or video frames.

**Architecture:**

.. code-block:: text

    CorpusDocument.modality
        ├── TEXT   → EmbeddingEngine (sentence-transformers / OpenAI / custom)
        ├── IMAGE  → CLIP / ViT / timm  (raw_tensor: HWC uint8)
        ├── AUDIO  → Whisper encoder / wav2vec2  (raw_tensor: float32 waveform)
        ├── VIDEO  → CLIP per-frame → mean-pool  (raw_tensor: THWC uint8)
        └── MULTIMODAL → TEXT + IMAGE fused (mean or concat)
                         ↓
                 Optional linear projection → (D,) float32
                         ↓
                 doc.embedding  ←  same shape for all modalities

**Why one engine, not four:**
The projection layer is what makes multimodal training practical.
Without it, text embeddings are 768-D and CLIP is 512-D — they can't
be stored in the same column or fed to the same model head.

**Backends (all optional, lazy-loaded):**

Text
    sentence-transformers (default), OpenAI, any ``Callable``.

Image
    ``transformers`` ``CLIPModel`` (default, ``openai/clip-vit-base-patch32``),
    ``open_clip`` (alternative), any ``Callable[[ndarray], ndarray]``.

Audio
    ``faster_whisper`` encoder (default), ``transformers`` wav2vec2/HuBERT,
    any ``Callable[[ndarray], ndarray]``.

Video
    CLIP applied per sampled frame, then mean-pooled. Requires the image
    backend to be available.

Python compatibility
--------------------
Python 3.8-3.15.  ``numpy`` is required.  All other dependencies are
optional and raise ``ImportError`` at call time when missing.
"""  # noqa: D205, D400

from __future__ import annotations

import hashlib
import logging
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Generator, Optional  # noqa: F401

import numpy as np
import numpy.typing as npt

from ._embedding import EmbeddingEngine

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_AUDIO_MODEL",
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_TEXT_MODEL",
    "LLMTrainingExporter",
    "MultimodalEmbeddingEngine",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_MODEL: str = "openai/clip-vit-base-patch32"
DEFAULT_AUDIO_MODEL: str = "openai/whisper-base"
DEFAULT_TEXT_MODEL: str = "all-MiniLM-L6-v2"

#: Frames sampled per video when using CLIP
_VIDEO_FRAME_SAMPLE: int = 8


# ---------------------------------------------------------------------------
# Backend factory helpers
# ---------------------------------------------------------------------------


def _make_clip_fn(
    model_name: str,
    batch_size: int,
    normalize: bool,
    device: str | None,
) -> Callable[[list[npt.NDArray]], npt.NDArray]:
    """
    Build an image-embed function backed by HuggingFace CLIP.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. ``"openai/clip-vit-base-patch32"``).
    batch_size : int
        Images per forward pass.
    normalize : bool
        L2-normalise output vectors.
    device : str or None
        Torch device string.

    Returns
    -------
    Callable
        ``fn(arrays: list[ndarray HWC uint8]) → ndarray (N, D) float32``
    """
    _state: list[Any] = []  # [model, processor, device_str]

    def _load() -> None:
        """Lazily load CLIP model and processor on first call."""
        try:
            import torch  # noqa: PLC0415
            from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "CLIP image embedding requires transformers and torch:\n"
                "  pip install transformers torch"
            ) from exc
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(model_name).to(dev)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        _state.extend([model, processor, dev])
        logger.debug(
            "MultimodalEmbeddingEngine: loaded CLIP %r on %s.", model_name, dev
        )

    def embed_images(arrays: list[npt.NDArray]) -> npt.NDArray:
        """Embed a batch of images via CLIP, loading model on first call.

        Parameters
        ----------
        arrays : list[ndarray]
            Each array ``(H, W, C)`` uint8 RGB.

        Returns
        -------
        ndarray
            Shape ``(N, D)`` float32.
        """
        if not _state:
            _load()
        model, processor, dev = _state
        import torch  # noqa: PLC0415
        from PIL import Image as _PIL  # noqa: N814, PLC0415

        all_vecs: list[npt.NDArray] = []
        for i in range(0, len(arrays), batch_size):
            batch_np = arrays[i : i + batch_size]
            pil_imgs = [_PIL.fromarray(a.astype(np.uint8)) for a in batch_np]
            inputs = processor(images=pil_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            vecs = feats.cpu().float().numpy()
            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0).astype(np.float32)

    return embed_images


def _make_open_clip_fn(
    model_name: str,
    batch_size: int,
    normalize: bool,
    device: str | None,
) -> Callable[[list[npt.NDArray]], npt.NDArray]:
    """
    Build an image-embed function backed by ``open_clip``.

    Parameters
    ----------
    model_name : str
        Open-CLIP model name, e.g. ``"ViT-B-32"`` or ``"RN50"``.
    batch_size : int
        Images per forward pass.
    normalize : bool
        L2-normalise output vectors.
    device : str or None
        Torch device string.

    Returns
    -------
    Callable
        ``fn(arrays: list[ndarray]) → ndarray (N, D) float32``
    """
    _state: list[Any] = []

    def _load() -> None:
        """Lazily load open_clip model and preprocessor on first call."""
        try:
            import open_clip  # type: ignore[import] # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "open_clip image embedding requires open_clip_torch:\n"
                "  pip install open_clip_torch"
            ) from exc
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # model_name may be "ViT-B-32" or "ViT-B-32::openai"
        parts = model_name.split("::")
        arch = parts[0]
        pretrained = parts[1] if len(parts) > 1 else "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=dev
        )
        model.eval()
        _state.extend([model, preprocess, dev])
        logger.debug(
            "MultimodalEmbeddingEngine: loaded open_clip %r/%r on %s.",
            arch,
            pretrained,
            dev,
        )

    def embed_images(arrays: list[npt.NDArray]) -> npt.NDArray:
        """Embed a batch of images through open_clip, loading model on first call.

        Parameters
        ----------
        arrays : list[ndarray]
            Each array ``(H, W, C)`` uint8 RGB.

        Returns
        -------
        ndarray
            Shape ``(N, D)`` float32.
        """
        if not _state:
            _load()
        model, preprocess, dev = _state
        import torch  # noqa: PLC0415
        from PIL import Image as _PIL  # noqa: N814, PLC0415

        all_vecs: list[npt.NDArray] = []
        for i in range(0, len(arrays), batch_size):
            batch_np = arrays[i : i + batch_size]
            pil_imgs = [_PIL.fromarray(a.astype(np.uint8)) for a in batch_np]
            imgs = torch.stack([preprocess(img) for img in pil_imgs]).to(dev)
            with torch.no_grad():
                feats = model.encode_image(imgs)
            vecs = feats.cpu().float().numpy()
            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0).astype(np.float32)

    return embed_images


def _make_whisper_encoder_fn(
    model_name: str,
    batch_size: int,
    normalize: bool,
    device: str | None,
) -> Callable[[list[npt.NDArray]], npt.NDArray]:
    """
    Build an audio-embed function using the Whisper encoder.

    Uses ``faster_whisper`` to extract log-mel features, passes them
    through the Whisper encoder, and mean-pools the hidden states.

    Parameters
    ----------
    model_name : str
        Whisper model size (``"tiny"``, ``"base"``, ``"small"``, etc.)
        or a local path.
    batch_size : int
        Waveforms per forward pass.
    normalize : bool
        L2-normalise output vectors.
    device : str or None
        ``"cpu"``, ``"cuda"``, or ``None`` (auto).

    Returns
    -------
    Callable
        ``fn(waveforms: list[ndarray float32]) → ndarray (N, D) float32``
    """
    _state: list[Any] = []

    def _load() -> None:
        """Lazily load Whisper encoder model and processor on first call."""
        try:
            import torch  # noqa: PLC0415
            from transformers import WhisperModel, WhisperProcessor  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Whisper audio embedding requires transformers and torch:\n"
                "  pip install transformers torch"
            ) from exc
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_name = (
            f"openai/whisper-{model_name}" if "/" not in model_name else model_name
        )
        processor = WhisperProcessor.from_pretrained(hf_name)
        model = WhisperModel.from_pretrained(hf_name).to(dev)
        model.eval()
        _state.extend([model, processor, dev])
        logger.debug(
            "MultimodalEmbeddingEngine: loaded Whisper encoder %r on %s.",
            hf_name,
            dev,
        )

    def embed_audio(waveforms: list[npt.NDArray]) -> npt.NDArray:
        """Embed a batch of audio waveforms through the Whisper encoder.

        Parameters
        ----------
        waveforms : list[ndarray]
            Each waveform ``(samples,)`` float32, 16 kHz.

        Returns
        -------
        ndarray
            Shape ``(N, D)`` float32.
        """
        if not _state:
            _load()
        model, processor, dev = _state
        import torch  # noqa: PLC0415

        all_vecs: list[npt.NDArray] = []
        for i in range(0, len(waveforms), batch_size):
            batch_wav = waveforms[i : i + batch_size]
            # Ensure float32 mono, 16 kHz expected by Whisper
            batch_wav = [
                w.astype(np.float32) if w.dtype != np.float32 else w for w in batch_wav
            ]
            inputs = processor(
                batch_wav,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs["input_features"].to(dev)
            with torch.no_grad():
                enc_out = model.encoder(input_features)
            # Mean-pool over time dimension
            hidden = enc_out.last_hidden_state  # (B, T, D)
            vecs = hidden.mean(dim=1).cpu().float().numpy()
            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0).astype(np.float32)

    return embed_audio


def _make_wav2vec_fn(
    model_name: str,
    batch_size: int,
    normalize: bool,
    device: str | None,
) -> Callable[[list[npt.NDArray]], npt.NDArray]:
    """
    Build an audio-embed function using wav2vec2 / HuBERT.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g.
        ``"facebook/wav2vec2-base"`` or ``"facebook/hubert-base-ls960"``.
    batch_size : int
        Waveforms per forward pass.
    normalize : bool
        L2-normalise output vectors.
    device : str or None
        Torch device string.

    Returns
    -------
    Callable
        ``fn(waveforms: list[ndarray float32]) → ndarray (N, D) float32``
    """
    _state: list[Any] = []

    def _load() -> None:
        """Lazily load wav2vec2/HuBERT model and feature extractor on first call."""
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoFeatureExtractor,
                AutoModel,
            )
        except ImportError as exc:
            raise ImportError(
                "wav2vec2 audio embedding requires transformers and torch:\n"
                "  pip install transformers torch"
            ) from exc
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(dev)
        model.eval()
        _state.extend([model, extractor, dev])
        logger.debug(
            "MultimodalEmbeddingEngine: loaded wav2vec2 %r on %s.", model_name, dev
        )

    def embed_audio(waveforms: list[npt.NDArray]) -> npt.NDArray:
        """Embed a batch of audio waveforms through wav2vec2 / HuBERT.

        Parameters
        ----------
        waveforms : list[ndarray]
            Each waveform ``(samples,)`` float32, 16 kHz.

        Returns
        -------
        ndarray
            Shape ``(N, D)`` float32.
        """
        if not _state:
            _load()
        model, extractor, dev = _state
        import torch  # noqa: PLC0415

        all_vecs: list[npt.NDArray] = []
        for i in range(0, len(waveforms), batch_size):
            batch_wav = [w.astype(np.float32) for w in waveforms[i : i + batch_size]]
            inputs = extractor(
                batch_wav,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            vecs = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0).astype(np.float32)

    return embed_audio


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _make_linear_projection(
    in_dim: int,
    out_dim: int,
    normalize: bool,
) -> Callable[[npt.NDArray], npt.NDArray]:
    """
    Return a *random* linear projection ``W ∈ R^{in_dim x out_dim}``.

    For production use, supply a ``custom_projection_fn`` instead —
    this is a dimensionality-reduction fallback (similar to random LSH)
    when no learned projection is available.

    Parameters
    ----------
    in_dim : int
        Input vector dimension.
    out_dim : int
        Target output dimension.
    normalize : bool
        L2-normalise after projection.

    Returns
    -------
    Callable
        ``fn(vecs: ndarray (N, in_dim)) → ndarray (N, out_dim) float32``

    Notes
    -----
    The projection matrix is seeded with ``in_dim + out_dim`` for
    reproducibility across runs on the same machine.
    """
    rng = np.random.default_rng(seed=in_dim + out_dim)
    # Orthonormal columns via QR decomposition for better geometry
    raw = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    W = q[:, :out_dim].astype(np.float32)  # (in_dim, out_dim)  # noqa: N806

    def project(vecs: npt.NDArray) -> npt.NDArray:
        """Apply the random orthonormal projection to *vecs*.

        Parameters
        ----------
        vecs : ndarray shape (N, in_dim)
            Input vectors to project.

        Returns
        -------
        ndarray shape (N, out_dim)
            Projected and optionally L2-normalised vectors.
        """
        out = vecs.astype(np.float32) @ W  # (N, out_dim)
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            out = out / norms
        return out

    return project


# ---------------------------------------------------------------------------
# MultimodalEmbeddingEngine
# ---------------------------------------------------------------------------


@dataclass
class MultimodalEmbeddingEngine:
    """
    Unified embedding engine for any :class:`~scikitplot.corpus.CorpusDocument`
    modality — text, image, audio, video, or multimodal.

    Routes each document to the appropriate backend by inspecting
    ``doc.modality``, optionally projects all vectors to a common
    ``projection_dim``, and stores the result in ``doc.embedding``.

    Parameters
    ----------
    text_backend : {"sentence_transformers", "openai", "custom"}, optional
        Text embedding backend.  Default: ``"sentence_transformers"``.
    text_model : str, optional
        Model name for the text backend.
        Default: ``"all-MiniLM-L6-v2"``.
    text_custom_fn : callable or None, optional
        Custom text embed function
        ``Callable[[list[str]], ndarray]``. Required when
        ``text_backend="custom"``. Default: ``None``.
    image_backend : {"clip", "open_clip", "custom"}, optional
        Image embedding backend.  Default: ``"clip"``.
    image_model : str, optional
        CLIP/ViT model name.
        Default: ``"openai/clip-vit-base-patch32"``.
    image_custom_fn : callable or None, optional
        Custom image embed function
        ``Callable[[list[ndarray]], ndarray]``.
        Required when ``image_backend="custom"``. Default: ``None``.
    audio_backend : {"whisper", "wav2vec", "custom"}, optional
        Audio embedding backend.  Default: ``"whisper"``.
    audio_model : str, optional
        Whisper model size or HuggingFace model id.
        Default: ``"openai/whisper-base"``.
    audio_custom_fn : callable or None, optional
        Custom audio embed function
        ``Callable[[list[ndarray]], ndarray]``.
        Required when ``audio_backend="custom"``. Default: ``None``.
    multimodal_fusion : {"mean", "concat", "text_only", "image_only"}, optional
        How to combine text + image vectors for ``MULTIMODAL`` docs.
        ``"mean"`` averages the two vectors (requires same dim or
        ``projection_dim`` set).  ``"concat"`` concatenates them
        (output dim = text_dim + image_dim).
        Default: ``"mean"``.
    projection_dim : int or None, optional
        If set, project every embedding to this dimension via a linear
        map.  Unifies incompatible backend dimensions so all modalities
        share one embedding space.  Default: ``None`` (no projection).
    custom_projection_fn : callable or None, optional
        Override the auto-generated random projection with a learned one,
        e.g. a trained linear adapter.
        ``Callable[[ndarray (N, D)], ndarray (N, projection_dim)]``.
        Default: ``None``.
    normalize : bool, optional
        L2-normalise all output embeddings.  Default: ``True``.
    batch_size : int, optional
        Items per forward pass for each backend.  Default: ``32``.
    device : str or None, optional
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``None`` lets each backend auto-select.  Default: ``None``.
    cache_dir : pathlib.Path or None, optional
        Cache directory for embeddings.  ``None`` uses the text engine's
        default cache.  Default: ``None``.
    enable_cache : bool, optional
        Enable/disable embedding cache.  Default: ``True``.

    Notes
    -----
    **Projection dimension choice:** Set ``projection_dim`` to the
    hidden size of your target LLM's embedding layer.  For GPT-4 /
    ``text-embedding-3-large`` this is 3072; for
    ``text-embedding-3-small`` / ``all-MiniLM-L6-v2`` it is 384-768.

    **Cache key** includes modality + backend + model name + source path
    + mtime + n_items — changing any of these invalidates the cache.

    **Thread safety:** Backends are lazily loaded and protected by a
    ``threading.Lock`` per backend. Safe for concurrent reads after warm-up.

    Examples
    --------
    Text + image documents in one call:

    >>> engine = MultimodalEmbeddingEngine(
    ...     projection_dim=512,
    ...     image_backend="clip",
    ... )
    >>> docs = engine.embed_documents(docs)
    >>> docs[0].embedding.shape
    (512,)

    GPT fine-tuning pipeline:

    >>> engine = MultimodalEmbeddingEngine(
    ...     text_backend="openai",
    ...     text_model="text-embedding-3-small",
    ...     projection_dim=1536,
    ... )
    >>> docs = engine.embed_documents(text_docs)
    >>> from scikitplot.corpus._embeddings._multimodal_embedding import (
    ...     LLMTrainingExporter,
    ... )
    >>> exporter = LLMTrainingExporter(engine)
    >>> exporter.to_openai_finetuning_jsonl(docs, Path("train.jsonl"))

    Developer note
    --------------
    ``projection_dim`` with ``custom_projection_fn=None`` uses a random
    orthonormal projection (QR decomposition).  For production fine-tuning
    replace this with a learned adapter trained on your downstream task.
    """  # noqa: D205

    VALID_TEXT_BACKENDS: ClassVar[tuple[str, ...]] = (
        "sentence_transformers",
        "openai",
        "custom",
    )
    VALID_IMAGE_BACKENDS: ClassVar[tuple[str, ...]] = (
        "clip",
        "open_clip",
        "custom",
    )
    VALID_AUDIO_BACKENDS: ClassVar[tuple[str, ...]] = (
        "whisper",
        "wav2vec",
        "custom",
    )
    VALID_FUSION: ClassVar[tuple[str, ...]] = (
        "mean",
        "concat",
        "text_only",
        "image_only",
    )

    # Text
    text_backend: str = field(default="sentence_transformers")
    text_model: str = field(default=DEFAULT_TEXT_MODEL)
    text_custom_fn: Callable | None = field(default=None, repr=False)
    # Image
    image_backend: str = field(default="clip")
    image_model: str = field(default=DEFAULT_IMAGE_MODEL)
    image_custom_fn: Callable | None = field(default=None, repr=False)
    # Audio
    audio_backend: str = field(default="whisper")
    audio_model: str = field(default=DEFAULT_AUDIO_MODEL)
    audio_custom_fn: Callable | None = field(default=None, repr=False)
    # Fusion
    multimodal_fusion: str = field(default="mean")
    # Projection
    projection_dim: int | None = field(default=None)
    custom_projection_fn: Callable | None = field(default=None, repr=False)
    # General
    normalize: bool = field(default=True)
    batch_size: int = field(default=32)
    device: str | None = field(default=None)
    cache_dir: pathlib.Path | None = field(default=None)
    enable_cache: bool = field(default=True)

    # Internal — lazy backends, protected by individual locks
    _text_engine: EmbeddingEngine | None = field(default=None, init=False, repr=False)
    _image_fn: Callable | None = field(default=None, init=False, repr=False)
    _audio_fn: Callable | None = field(default=None, init=False, repr=False)
    _projection_fns: dict[int, Callable] = field(
        default_factory=dict, init=False, repr=False
    )
    _text_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _image_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _audio_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate fields."""
        if self.text_backend not in self.VALID_TEXT_BACKENDS:
            raise ValueError(
                f"MultimodalEmbeddingEngine: text_backend must be one of "
                f"{self.VALID_TEXT_BACKENDS}; got {self.text_backend!r}."
            )
        if self.image_backend not in self.VALID_IMAGE_BACKENDS:
            raise ValueError(
                f"MultimodalEmbeddingEngine: image_backend must be one of "
                f"{self.VALID_IMAGE_BACKENDS}; got {self.image_backend!r}."
            )
        if self.audio_backend not in self.VALID_AUDIO_BACKENDS:
            raise ValueError(
                f"MultimodalEmbeddingEngine: audio_backend must be one of "
                f"{self.VALID_AUDIO_BACKENDS}; got {self.audio_backend!r}."
            )
        if self.multimodal_fusion not in self.VALID_FUSION:
            raise ValueError(
                f"MultimodalEmbeddingEngine: multimodal_fusion must be one of "
                f"{self.VALID_FUSION}; got {self.multimodal_fusion!r}."
            )
        if self.text_backend == "custom" and self.text_custom_fn is None:
            raise ValueError(
                "MultimodalEmbeddingEngine: text_backend='custom' requires "
                "text_custom_fn."
            )
        if self.image_backend == "custom" and self.image_custom_fn is None:
            raise ValueError(
                "MultimodalEmbeddingEngine: image_backend='custom' requires "
                "image_custom_fn."
            )
        if self.audio_backend == "custom" and self.audio_custom_fn is None:
            raise ValueError(
                "MultimodalEmbeddingEngine: audio_backend='custom' requires "
                "audio_custom_fn."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"MultimodalEmbeddingEngine: batch_size must be > 0; "
                f"got {self.batch_size!r}."
            )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        documents: list[Any],
        source_path: pathlib.Path | None = None,
    ) -> list[Any]:
        """
        Embed all documents in-place (via ``doc.replace(embedding=...)``)
        and return the updated list.

        Dispatches by ``doc.modality``:

        - ``TEXT`` → :meth:`embed_texts`
        - ``IMAGE`` → :meth:`embed_images`
        - ``AUDIO`` → :meth:`embed_audio`
        - ``VIDEO`` → :meth:`embed_video`
        - ``MULTIMODAL`` → fused text + image (see ``multimodal_fusion``)
        - fallback: treats as TEXT using ``doc.text`` or ``""``

        Parameters
        ----------
        documents : list[CorpusDocument]
            Documents to embed.
        source_path : pathlib.Path or None, optional
            Used as the cache-key anchor.  Pass the source file path for
            per-file caching.  Default: ``None`` (no file cache).

        Returns
        -------
        list[CorpusDocument]
            Same list with ``embedding`` fields populated.
        """  # noqa: D205
        if not documents:
            return documents

        # Import here to avoid circular import
        try:
            from .._schema import Modality  # noqa: PLC0415
        except ImportError:
            Modality = None  # type: ignore[assignment]  # noqa: N806

        # Group by modality for batch efficiency
        groups: dict[str, list[tuple[int, Any]]] = {}
        for idx, doc in enumerate(documents):
            mod = str(getattr(doc, "modality", "text"))
            groups.setdefault(mod, []).append((idx, doc))

        results: dict[int, Any] = {}

        for mod_str, idx_doc_pairs in groups.items():
            indices, group_docs = zip(*idx_doc_pairs)

            if mod_str == "text":
                texts = [
                    getattr(d, "normalized_text", None)
                    or getattr(d, "text", None)
                    or ""
                    for d in group_docs
                ]
                vecs = self.embed_texts(texts)

            elif mod_str == "image":
                arrays = [getattr(d, "raw_tensor", None) for d in group_docs]
                missing = [i for i, a in enumerate(arrays) if a is None]
                if missing:
                    logger.warning(
                        "MultimodalEmbeddingEngine: %d IMAGE docs have "
                        "raw_tensor=None — using zero vectors.",
                        len(missing),
                    )
                # Replace None with zeros (will embed to zero vector)
                dim_hint = next(
                    (a.shape for a in arrays if a is not None), (224, 224, 3)
                )
                arrays = [
                    a if a is not None else np.zeros(dim_hint, dtype=np.uint8)
                    for a in arrays
                ]
                vecs = self.embed_images(list(arrays))

            elif mod_str == "audio":
                waveforms = [getattr(d, "raw_tensor", None) for d in group_docs]
                waveforms = [
                    w if w is not None else np.zeros(16000, dtype=np.float32)
                    for w in waveforms
                ]
                vecs = self.embed_audio(list(waveforms))

            elif mod_str == "video":
                frame_seqs = [getattr(d, "raw_tensor", None) for d in group_docs]
                frame_seqs = [
                    (
                        f
                        if f is not None
                        else np.zeros(
                            (_VIDEO_FRAME_SAMPLE, 224, 224, 3), dtype=np.uint8
                        )
                    )
                    for f in frame_seqs
                ]
                vecs = self.embed_video(list(frame_seqs))

            elif mod_str == "multimodal":
                vecs = self._embed_multimodal(list(group_docs))

            else:
                # Unknown modality — fall back to text
                texts = [
                    getattr(d, "normalized_text", None)
                    or getattr(d, "text", None)
                    or ""
                    for d in group_docs
                ]
                vecs = self.embed_texts(texts)

            # Apply projection if needed
            vecs = self._maybe_project(vecs)

            for orig_idx, vec in zip(indices, vecs):
                results[orig_idx] = vec

        # Rebuild list preserving order
        return [
            doc.replace(embedding=results[i]) if i in results else doc
            for i, doc in enumerate(documents)
        ]

    def embed_texts(self, texts: list[str]) -> npt.NDArray:
        """
        Embed a list of strings via the configured text backend.

        Parameters
        ----------
        texts : list[str]
            Non-empty list of strings.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, D)`` float32.
        """
        engine = self._get_text_engine()
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return engine.embed(texts)

    def embed_images(
        self,
        arrays: list[npt.NDArray],
    ) -> npt.NDArray:
        """
        Embed a list of raw image arrays via the configured image backend.

        Parameters
        ----------
        arrays : list[ndarray]
            Each array: ``(H, W, C)`` uint8 RGB.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, D)`` float32.

        Raises
        ------
        ImportError
            If the required image backend library is not installed.
        """
        if not arrays:
            return np.empty((0, 0), dtype=np.float32)
        fn = self._get_image_fn()
        return fn(arrays)

    def embed_audio(
        self,
        waveforms: list[npt.NDArray],
    ) -> npt.NDArray:
        """
        Embed a list of audio waveforms via the configured audio backend.

        Parameters
        ----------
        waveforms : list[ndarray]
            Each waveform: ``(samples,)`` float32, 16 kHz.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, D)`` float32.
        """
        if not waveforms:
            return np.empty((0, 0), dtype=np.float32)
        fn = self._get_audio_fn()
        return fn(waveforms)

    def embed_video(
        self,
        frame_sequences: list[npt.NDArray],
        n_sample_frames: int = _VIDEO_FRAME_SAMPLE,
    ) -> npt.NDArray:
        """
        Embed video by sampling frames and mean-pooling CLIP embeddings.

        Parameters
        ----------
        frame_sequences : list[ndarray]
            Each array: ``(T, H, W, C)`` uint8 — T frames, channels-last.
        n_sample_frames : int, optional
            Frames to sample uniformly.  Default: 8.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, D)`` float32.
        """
        if not frame_sequences:
            return np.empty((0, 0), dtype=np.float32)

        all_vecs: list[npt.NDArray] = []
        for seq in frame_sequences:
            if seq.ndim == 3:  # single frame (H, W, C)  # noqa: PLR2004
                seq = seq[np.newaxis]  # → (1, H, W, C)  # noqa: PLW2901
            t = seq.shape[0]
            indices = np.linspace(0, t - 1, min(n_sample_frames, t), dtype=int)
            sampled = [seq[i] for i in indices]  # list of (H, W, C)
            frame_vecs = self.embed_images(sampled)  # (n_frames, D)
            pooled = frame_vecs.mean(axis=0)  # (D,)
            if self.normalize:
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
            all_vecs.append(pooled)
        return np.stack(all_vecs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Cache-aware batch embedding
    # ------------------------------------------------------------------

    def embed_documents_with_cache(
        self,
        documents: list[Any],
        source_path: pathlib.Path,
    ) -> list[Any]:
        """
        Embed documents with SHA-256 cache keyed to *source_path*.

        Cache key: ``SHA256(modality_tag + backend + model + path + mtime + N)[:24]``.

        Parameters
        ----------
        documents : list[CorpusDocument]
            Documents to embed.
        source_path : pathlib.Path
            Source file path.  Used to build the cache key (path + mtime).

        Returns
        -------
        list[CorpusDocument]
            Documents with embeddings populated.
        """
        cache_dir = self.cache_dir or (
            pathlib.Path.home() / ".cache" / "scikitplot" / "embeddings"
        )
        if not self.enable_cache:
            return self.embed_documents(documents, source_path=source_path)

        # Build modality-aware cache key
        modalities = sorted({str(getattr(d, "modality", "text")) for d in documents})
        tag = "+".join(modalities)
        try:
            mtime = source_path.stat().st_mtime
        except OSError:
            return self.embed_documents(documents, source_path=source_path)

        raw_key = (
            f"{tag}:{self.text_model}:{self.image_model}:"
            f"{self.audio_model}:{source_path!s}:{mtime}:{len(documents)}"
        )
        key = hashlib.sha256(raw_key.encode()).hexdigest()[:24]
        cache_path = cache_dir / f"{key}.npy"

        # Try cache load
        if cache_path.exists():
            try:
                cached = np.load(str(cache_path), allow_pickle=False)
                if cached.shape[0] == len(documents):
                    logger.info(
                        "MultimodalEmbeddingEngine: cache hit for %s (%d docs).",
                        source_path.name,
                        len(documents),
                    )
                    return [
                        doc.replace(embedding=cached[i])
                        for i, doc in enumerate(documents)
                    ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("MultimodalEmbeddingEngine: cache load failed: %s", exc)

        # Compute
        docs_out = self.embed_documents(documents, source_path=source_path)
        embeddings = np.stack(
            [d.embedding for d in docs_out],
            axis=0,  # noqa: B009
        ).astype(np.float32)

        # Save
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embeddings)
        except OSError as exc:
            logger.warning("MultimodalEmbeddingEngine: cache write failed: %s", exc)

        return docs_out

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _embed_multimodal(self, docs: list[Any]) -> npt.NDArray:
        """Fuse text + image embeddings per ``multimodal_fusion`` strategy."""
        texts = [
            getattr(d, "normalized_text", None) or getattr(d, "text", None) or ""
            for d in docs
        ]
        arrays = [getattr(d, "raw_tensor", None) for d in docs]

        if self.multimodal_fusion == "text_only":
            return self.embed_texts(texts)

        if self.multimodal_fusion == "image_only":
            safe = [
                a if a is not None else np.zeros((224, 224, 3), dtype=np.uint8)
                for a in arrays
            ]
            return self.embed_images(safe)

        text_vecs = self.embed_texts(texts)

        safe = [
            a if a is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            for a in arrays
        ]
        img_vecs = self.embed_images(safe)

        if self.multimodal_fusion == "concat":
            return np.concatenate([text_vecs, img_vecs], axis=1).astype(np.float32)

        # "mean" — requires same dim (or projection_dim will align later)
        if text_vecs.shape[1] == img_vecs.shape[1]:
            fused = (text_vecs + img_vecs) / 2.0
        else:
            # Mismatched dims: project both to text dim before mean
            if img_vecs.shape[1] != text_vecs.shape[1]:
                proj = self._get_projection_fn(img_vecs.shape[1], text_vecs.shape[1])
                img_vecs = proj(img_vecs)
            fused = (text_vecs + img_vecs) / 2.0

        if self.normalize:
            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            fused = fused / norms

        return fused.astype(np.float32)

    def _maybe_project(self, vecs: npt.NDArray) -> npt.NDArray:
        """Project *vecs* to ``projection_dim`` when set."""
        if self.projection_dim is None or vecs.shape[0] == 0:
            return vecs
        if vecs.shape[1] == self.projection_dim:
            return vecs  # already correct dim
        if self.custom_projection_fn is not None:
            return self.custom_projection_fn(vecs).astype(np.float32)
        proj = self._get_projection_fn(vecs.shape[1], self.projection_dim)
        return proj(vecs)

    def _get_projection_fn(
        self,
        in_dim: int,
        out_dim: int,
    ) -> Callable[[npt.NDArray], npt.NDArray]:
        """Return (cached) linear projection in_dim → out_dim."""
        key = (in_dim, out_dim)
        if key not in self._projection_fns:
            self._projection_fns[key] = _make_linear_projection(
                in_dim, out_dim, self.normalize
            )
        return self._projection_fns[key]

    def _get_text_engine(self) -> EmbeddingEngine:
        """Lazily initialise and return the text EmbeddingEngine."""
        if self._text_engine is not None:
            return self._text_engine
        with self._text_lock:
            if self._text_engine is not None:
                return self._text_engine
            engine = EmbeddingEngine(
                backend=self.text_backend,
                model_name=self.text_model,
                custom_fn=self.text_custom_fn,
                cache_dir=self.cache_dir,
                enable_cache=self.enable_cache,
                batch_size=self.batch_size,
                normalize=self.normalize,
                device=self.device,
            )
            object.__setattr__(self, "_text_engine", engine)
        return self._text_engine  # type: ignore[return-value]

    def _get_image_fn(self) -> Callable:
        """Lazily initialise the image embed function."""
        if self._image_fn is not None:
            return self._image_fn
        with self._image_lock:
            if self._image_fn is not None:
                return self._image_fn
            if self.image_backend == "custom":
                fn = self.image_custom_fn
            elif self.image_backend == "open_clip":
                fn = _make_open_clip_fn(
                    self.image_model, self.batch_size, self.normalize, self.device
                )
            else:  # "clip"
                fn = _make_clip_fn(
                    self.image_model, self.batch_size, self.normalize, self.device
                )
            object.__setattr__(self, "_image_fn", fn)
        return self._image_fn  # type: ignore[return-value]

    def _get_audio_fn(self) -> Callable:
        """Lazily initialise the audio embed function."""
        if self._audio_fn is not None:
            return self._audio_fn
        with self._audio_lock:
            if self._audio_fn is not None:
                return self._audio_fn
            if self.audio_backend == "custom":
                fn = self.audio_custom_fn
            elif self.audio_backend == "wav2vec":
                fn = _make_wav2vec_fn(
                    self.audio_model, self.batch_size, self.normalize, self.device
                )
            else:  # "whisper"
                fn = _make_whisper_encoder_fn(
                    self.audio_model, self.batch_size, self.normalize, self.device
                )
            object.__setattr__(self, "_audio_fn", fn)
        return self._audio_fn  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"MultimodalEmbeddingEngine("
            f"text={self.text_backend!r}/{self.text_model!r}, "
            f"image={self.image_backend!r}/{self.image_model!r}, "
            f"audio={self.audio_backend!r}/{self.audio_model!r}, "
            f"projection_dim={self.projection_dim}, "
            f"fusion={self.multimodal_fusion!r})"
        )


# ===========================================================================
# LLMTrainingExporter — any corpus → any LLM training format
# ===========================================================================


@dataclass
class LLMTrainingExporter:
    """
    Export a corpus with embeddings to LLM training formats.

    Orchestrates the full journey from
    ``list[CorpusDocument]`` → training-ready files / datasets.

    Parameters
    ----------
    engine : MultimodalEmbeddingEngine or EmbeddingEngine or None
        Embedding engine.  When ``None``, existing ``doc.embedding``
        values are used as-is; raises ``ValueError`` if a document
        lacks an embedding where one is required.
    default_system_prompt : str, optional
        System prompt prepended to all OpenAI fine-tuning conversations.
        Default: ``"You are a helpful assistant."``.

    Examples
    --------
    OpenAI fine-tuning (chat format):

    >>> exporter = LLMTrainingExporter(engine)
    >>> exporter.to_openai_finetuning_jsonl(
    ...     docs,
    ...     output_path=Path("train.jsonl"),
    ...     system_prompt="Answer medical questions accurately.",
    ...     response_fn=lambda doc: doc.metadata.get("answer", ""),
    ... )

    HuggingFace SFT dataset:

    >>> ds = exporter.to_huggingface_training_dataset(
    ...     docs,
    ...     tokenizer_name="gpt2",
    ...     task="clm",
    ... )

    Pure embedding matrix for vector DB or contrastive training:

    >>> matrix, meta_df = exporter.to_embedding_matrix(docs)
    >>> matrix.shape  # (N, D)
    (1024, 512)
    """

    engine: Any | None = field(default=None)
    default_system_prompt: str = field(default="You are a helpful assistant.")

    # ------------------------------------------------------------------
    # Ensure embeddings
    # ------------------------------------------------------------------

    def _ensure_embedded(self, documents: list[Any]) -> list[Any]:
        """Embed documents that lack embeddings, using ``self.engine``."""
        needs_embed = [d for d in documents if getattr(d, "embedding", None) is None]
        if not needs_embed:
            return documents

        if self.engine is None:
            raise ValueError(
                "LLMTrainingExporter: some documents have no embedding and "
                "engine=None. Pass an engine or pre-embed documents first."
            )

        logger.info(
            "LLMTrainingExporter: embedding %d documents before export.",
            len(needs_embed),
        )
        embedded = self.engine.embed_documents(needs_embed)
        emb_map = {
            getattr(d, "doc_id", i): e
            for i, (d, e) in enumerate(zip(needs_embed, embedded))
        }

        result = []
        needs_iter = iter(embedded)
        for doc in documents:
            if getattr(doc, "embedding", None) is None:
                result.append(next(needs_iter))
            else:
                result.append(doc)
        return result

    # ------------------------------------------------------------------
    # OpenAI fine-tuning format
    # ------------------------------------------------------------------

    def to_openai_finetuning_jsonl(
        self,
        documents: list[Any],
        output_path: pathlib.Path | str,
        *,
        system_prompt: str | None = None,
        response_fn: Callable[[Any], str] | None = None,
        user_field: str = "text",
        include_embeddings: bool = False,
        skip_empty: bool = True,
    ) -> pathlib.Path:
        """
        Export documents as OpenAI chat fine-tuning JSONL.

        Each line is a valid fine-tuning example::

            {
              "messages": [
                {"role": "system", "content": "<system_prompt>"},
                {"role": "user",   "content": "<doc.text>"},
                {"role": "assistant", "content": "<response_fn(doc)>"}
              ],
              "embedding": [...]    // optional, only when include_embeddings=True
            }

        Parameters
        ----------
        documents : list[CorpusDocument]
            Documents to export.
        output_path : pathlib.Path or str
            Destination ``.jsonl`` file.
        system_prompt : str or None, optional
            System message.  Defaults to ``self.default_system_prompt``.
        response_fn : callable or None, optional
            ``fn(doc) → str`` producing the assistant response for each
            document.  When ``None``, uses ``doc.metadata.get("answer", "")``
            — suitable when answers are stored in metadata.
        user_field : str, optional
            Document attribute to use as the user message.
            Default: ``"text"``.
        include_embeddings : bool, optional
            Append ``"embedding"`` key to each record.  Requires that
            embeddings are present (call :meth:`_ensure_embedded` first
            or set an ``engine``).  Default: ``False``.
        skip_empty : bool, optional
            Skip documents with empty user content.  Default: ``True``.

        Returns
        -------
        pathlib.Path
            Path to the written ``.jsonl`` file.

        Notes
        -----
        OpenAI fine-tuning requires at minimum a ``"user"`` message and
        an ``"assistant"`` message.  Provide ``response_fn`` to generate
        meaningful assistant turns; otherwise the export produces single-turn
        user-only examples (system + user, no assistant reply) which are
        valid for supervised fine-tuning when you add assistant responses
        separately.
        """
        import json  # noqa: PLC0415

        prompt = system_prompt or self.default_system_prompt
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if include_embeddings:
            documents = self._ensure_embedded(documents)

        n_written = 0
        n_skipped = 0

        with out.open("w", encoding="utf-8") as fh:
            for doc in documents:
                user_content = str(getattr(doc, user_field, "") or "").strip()
                if skip_empty and not user_content:
                    n_skipped += 1
                    continue

                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ]

                if response_fn is not None:
                    assistant_text = str(response_fn(doc)).strip()
                    if assistant_text:
                        messages.append(
                            {"role": "assistant", "content": assistant_text}
                        )

                record: dict = {"messages": messages}

                if include_embeddings:
                    emb = getattr(doc, "embedding", None)
                    if emb is not None:
                        record["embedding"] = emb.tolist()

                # Provenance metadata (optional enrichment)
                record["metadata"] = {
                    "doc_id": getattr(doc, "doc_id", None),
                    "source_file": getattr(doc, "source_file", None),
                    "source_type": str(getattr(doc, "source_type", "")),
                    "modality": str(getattr(doc, "modality", "text")),
                    "chunk_index": getattr(doc, "chunk_index", None),
                    "content_hash": getattr(doc, "content_hash", None),
                }

                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

        logger.info(
            "LLMTrainingExporter.to_openai_finetuning_jsonl: wrote %d records "
            "to %s (%d skipped).",
            n_written,
            out,
            n_skipped,
        )
        return out

    # ------------------------------------------------------------------
    # HuggingFace training dataset
    # ------------------------------------------------------------------

    def to_huggingface_training_dataset(  # noqa: PLR0912
        self,
        documents: list[Any],
        *,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        task: str = "clm",
        text_field: str = "text",
        label_field: str | None = None,
        include_embeddings: bool = False,
        stride: int = 0,
    ) -> Any:
        """
        Build a HuggingFace ``datasets.Dataset`` for LLM training.

        Parameters
        ----------
        documents : list[CorpusDocument]
            Documents to tokenize.
        tokenizer_name : str, optional
            HuggingFace tokenizer name or local path.
            Default: ``"gpt2"``.
        max_length : int, optional
            Maximum token sequence length.  Sequences are truncated
            (and optionally strided).  Default: ``512``.
        task : {"clm", "mlm", "sft"}, optional
            Training objective.

            ``"clm"`` — causal language model: ``labels = input_ids``.
            ``"mlm"`` — masked language model: 15% random masking.
            ``"sft"`` — supervised fine-tuning: requires
            ``label_field`` to be set.

            Default: ``"clm"``.
        text_field : str, optional
            Document attribute to tokenize.  Default: ``"text"``.
        label_field : str or None, optional
            Attribute to use as classification label (for ``"sft"``).
            Default: ``None``.
        include_embeddings : bool, optional
            Add ``"embedding"`` column.  Default: ``False``.
        stride : int, optional
            Overlap between windows when splitting long texts.
            Default: ``0`` (no stride).

        Returns
        -------
        datasets.Dataset
            Tokenized training dataset.  Falls back to a plain dict of
            lists when ``datasets`` is not installed.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        """
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "to_huggingface_training_dataset requires transformers:\n"
                "  pip install transformers"
            ) from exc

        try:
            from datasets import Dataset  # noqa: PLC0415

            has_datasets = True
        except ImportError:
            has_datasets = False
            logger.warning(
                "datasets library not installed; returning plain dict. "
                "Install with: pip install datasets"
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if include_embeddings:
            documents = self._ensure_embedded(documents)

        all_input_ids: list[list[int]] = []
        all_attention: list[list[int]] = []
        all_labels: list[Any] = []
        all_embeddings: list[Any] = []
        all_doc_ids: list[str] = []
        all_source_types: list[str] = []
        all_modalities: list[str] = []

        for doc in documents:
            raw_text = str(getattr(doc, text_field, "") or "").strip()
            if not raw_text:
                continue

            enc = tokenizer(
                raw_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                stride=max(0, stride),
                return_overflowing_tokens=stride > 0,
                return_tensors=None,
            )

            # When return_overflowing_tokens, enc is a list of windows
            input_id_chunks = (
                [enc["input_ids"]]
                if isinstance(enc["input_ids"][0], int)
                else enc["input_ids"]
            )
            attn_chunks = (
                [enc["attention_mask"]]
                if isinstance(enc["attention_mask"][0], int)
                else enc["attention_mask"]
            )

            for chunk_ids, chunk_attn in zip(input_id_chunks, attn_chunks):
                all_input_ids.append(chunk_ids)
                all_attention.append(chunk_attn)

                if task == "clm":
                    all_labels.append(list(chunk_ids))
                elif task == "mlm":
                    all_labels.append(self._mask_tokens(chunk_ids, tokenizer))
                elif task == "sft" and label_field is not None:
                    lbl = getattr(doc, label_field, None)
                    all_labels.append(str(lbl) if lbl is not None else "")
                else:
                    all_labels.append(list(chunk_ids))

                if include_embeddings:
                    emb = getattr(doc, "embedding", None)
                    all_embeddings.append(emb.tolist() if emb is not None else [])

                all_doc_ids.append(getattr(doc, "doc_id", "") or "")
                all_source_types.append(str(getattr(doc, "source_type", "unknown")))
                all_modalities.append(str(getattr(doc, "modality", "text")))

        data_dict: dict[str, Any] = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention,
            "labels": all_labels,
            "doc_id": all_doc_ids,
            "source_type": all_source_types,
            "modality": all_modalities,
        }
        if include_embeddings:
            data_dict["embedding"] = all_embeddings

        if has_datasets:
            return Dataset.from_dict(data_dict)
        return data_dict

    # ------------------------------------------------------------------
    # Embedding matrix export
    # ------------------------------------------------------------------

    def to_embedding_matrix(
        self,
        documents: list[Any],
        *,
        include_metadata: bool = True,
        output_path: pathlib.Path | str | None = None,
    ) -> tuple[npt.NDArray, Any]:
        """
        Export embeddings as a ``(N, D)`` NumPy matrix with metadata.

        Parameters
        ----------
        documents : list[CorpusDocument]
            Documents.  Those without embeddings are embedded via the
            engine when one is set, else raise ``ValueError``.
        include_metadata : bool, optional
            Build a metadata DataFrame (or dict of lists).
            Default: ``True``.
        output_path : pathlib.Path or str or None, optional
            When set, saves ``{output_path}.npy`` (matrix) and
            ``{output_path}.csv`` (metadata).  Default: ``None``.

        Returns
        -------
        matrix : ndarray shape (N, D) float32
        metadata : pandas.DataFrame or dict[str, list]
            Metadata table with ``doc_id``, ``source_file``,
            ``source_type``, ``modality``, ``content_hash``,
            ``chunk_index`` columns.  Returns plain dict when
            pandas is not installed.

        Raises
        ------
        ValueError
            If any document lacks an embedding and ``engine=None``.
        """
        documents = self._ensure_embedded(documents)

        embeddings_list = [getattr(d, "embedding", None) for d in documents]
        missing = sum(1 for e in embeddings_list if e is None)
        if missing:
            raise ValueError(
                f"LLMTrainingExporter.to_embedding_matrix: {missing} documents "
                f"still have no embedding after engine call."
            )

        matrix = np.stack(embeddings_list, axis=0).astype(np.float32)

        meta: dict[str, list] = {}
        if include_metadata:
            meta = {
                "doc_id": [getattr(d, "doc_id", "") for d in documents],
                "source_file": [getattr(d, "source_file", "") for d in documents],
                "source_type": [str(getattr(d, "source_type", "")) for d in documents],
                "modality": [str(getattr(d, "modality", "text")) for d in documents],
                "content_hash": [getattr(d, "content_hash", None) for d in documents],
                "chunk_index": [getattr(d, "chunk_index", None) for d in documents],
            }

        # Try to return a DataFrame
        try:
            import pandas as pd  # noqa: PLC0415

            meta_out: Any = pd.DataFrame(meta)
        except ImportError:
            meta_out = meta

        if output_path is not None:
            base = pathlib.Path(output_path)
            base.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(base.with_suffix(".npy")), matrix)
            try:
                import pandas as pd  # noqa: PLC0415

                if isinstance(meta_out, pd.DataFrame):
                    meta_out.to_csv(str(base.with_suffix(".csv")), index=False)
            except ImportError:
                import json as _json  # noqa: PLC0415

                with open(str(base.with_suffix(".json")), "w") as fh:
                    _json.dump(meta, fh)

            logger.info(
                "LLMTrainingExporter.to_embedding_matrix: saved %s.npy (%d x %d).",
                base.stem,
                matrix.shape[0],
                matrix.shape[1],
            )

        return matrix, meta_out

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        documents: list[Any],
        *,
        run_name: str | None = None,
        artifact_dir: str = "corpus_embeddings",
        log_params: bool = True,
    ) -> None:
        """
        Log embedding matrix and metadata as MLflow artifacts.

        Parameters
        ----------
        documents : list[CorpusDocument]
            Documents with embeddings.
        run_name : str or None, optional
            MLflow run name.  Uses the active run when ``None``.
        artifact_dir : str, optional
            Directory inside the MLflow artifact store.
            Default: ``"corpus_embeddings"``.
        log_params : bool, optional
            Log engine config as MLflow params.  Default: ``True``.

        Raises
        ------
        ImportError
            If ``mlflow`` is not installed.
        """
        try:
            import mlflow  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "log_to_mlflow requires mlflow:\n  pip install mlflow"
            ) from exc
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp) / "embeddings"
            matrix, _meta = self.to_embedding_matrix(documents, output_path=tmp_path)

            with mlflow.start_run(run_name=run_name or None):
                if log_params and self.engine is not None:
                    eng = self.engine
                    params = {
                        "text_backend": getattr(eng, "text_backend", ""),
                        "text_model": getattr(eng, "text_model", ""),
                        "image_backend": getattr(eng, "image_backend", ""),
                        "image_model": getattr(eng, "image_model", ""),
                        "audio_backend": getattr(eng, "audio_backend", ""),
                        "audio_model": getattr(eng, "audio_model", ""),
                        "projection_dim": str(getattr(eng, "projection_dim", None)),
                        "n_documents": len(documents),
                        "embedding_dim": (
                            matrix.shape[1] if matrix.ndim == 2 else 0  # noqa: PLR2004
                        ),
                    }
                    mlflow.log_params(params)

                mlflow.log_metrics(
                    {
                        "n_embeddings": len(documents),
                        "embedding_dim": (
                            matrix.shape[1] if matrix.ndim == 2 else 0  # noqa: PLR2004
                        ),
                    }
                )
                mlflow.log_artifacts(tmp, artifact_path=artifact_dir)

        logger.info(
            "LLMTrainingExporter.log_to_mlflow: logged %d embeddings to MLflow.",
            len(documents),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_tokens(
        input_ids: list[int],
        tokenizer: Any,
        mask_prob: float = 0.15,
    ) -> list[int]:
        """Return a label list for MLM with 15% of tokens masked."""
        import random  # noqa: PLC0415

        mask_id = getattr(tokenizer, "mask_token_id", None) or 103
        labels = []
        for token_id in input_ids:
            if random.random() < mask_prob:  # noqa: S311
                labels.append(mask_id)
            else:
                labels.append(token_id)
        return labels

    def __repr__(self) -> str:
        eng = type(self.engine).__name__ if self.engine else "None"
        return f"LLMTrainingExporter(engine={eng})"
