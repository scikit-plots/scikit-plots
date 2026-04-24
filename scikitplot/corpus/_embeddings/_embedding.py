# scikitplot/corpus/_embeddings/_embedding.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._embedding
=============================
Multi-backend text and multimodal embedding engine with file-based caching.

Produces dense vector representations of text chunks. The embedding step
is optional in the pipeline (``embed=False`` skips it entirely) and is
triggered per-document or in batches. Results are cached to ``.npy``
files keyed by a SHA-256 hash of ``(model_name, input_path, mtime, n_texts)``
so that re-running the pipeline on an unchanged corpus is O(1).

Original issues fixed (from remarx ``embeddings.py``):

1. **Model reload on every call** — models are cached in a thread-safe
   per-instance dict; the same ``EmbeddingEngine`` object reuses loaded
   models across all calls.
2. **Cache next to source file** — the cache directory is configurable
   via ``cache_dir``; defaults to ``~/.cache/scikitplot/embeddings``.
3. **``allow_pickle=True``** — replaced with ``allow_pickle=False``
   throughout; pickled ``npy`` files are a known attack vector.
4. **No batch size control** — ``batch_size`` parameter prevents OOM on
   large corpora or small GPU memory.
5. **Only sentence-transformers** — supports ``sentence_transformers``,
   ``openai`` (text-embedding-3-small/large), and any ``Callable``
   accepting ``list[str] → np.ndarray``.
6. **Cache key ignores model name change** — cache key now includes the
   model name so swapping models invalidates stale caches.
7. **No shape/dtype validation on cache load** — loaded arrays are
   validated for shape, dtype, and row count before use.

Python compatibility:

Python 3.8-3.15. ``numpy`` is required. ``sentence_transformers``,
``openai``, and ``tiktoken`` are optional; graceful ``ImportError`` at
call time when not installed.
"""  # noqa: D205, D400

from __future__ import annotations

import hashlib
import logging
import pathlib
import threading
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL",
    "EmbeddingEngine",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: A callable that accepts a list of strings and returns a 2-D float32 array.
EmbedFn = Callable[[List[str]], npt.NDArray[np.float32]]

# Default model — multilingual, good baseline for mixed-language corpora
DEFAULT_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"

# Default cache directory under the user's home
DEFAULT_CACHE_DIR: pathlib.Path = (
    pathlib.Path.home() / ".cache" / "scikitplot" / "embeddings"
)

# Accepted float dtypes for embeddings
_FLOAT_DTYPES = (np.float16, np.float32, np.float64)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_cache_key(
    model_name: str,
    input_path: str,
    source_mtime: float,
    n_texts: int,
) -> str:
    """
    Compute a deterministic 24-character hex cache key.

    The key encodes ``(model_name, input_path, source_mtime, n_texts)``
    so that any change to the model, source file path, modification time,
    or document count invalidates the cached embeddings.

    Parameters
    ----------
    model_name : str
        Name or identifier of the embedding model.
    input_path : str
        Absolute path to the source file that produced the texts.
    source_mtime : float
        Modification time of the source file (``pathlib.Path.stat().st_mtime``).
    n_texts : int
        Number of texts that were embedded.

    Returns
    -------
    str
        24-character lowercase hex string.

    Examples
    --------
    >>> k = _make_cache_key("model-v1", "/data/file.txt", 1700000000.0, 512)
    >>> len(k)
    24
    """
    raw = f"{model_name}|{input_path}|{source_mtime:.6f}|{n_texts}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _cache_path(cache_dir: pathlib.Path, key: str) -> pathlib.Path:
    """Return the ``.npy`` path for a given cache key.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory holding cached embedding arrays.
    key : str
        24-character hex cache key from :func:`_make_cache_key`.

    Returns
    -------
    pathlib.Path
        ``cache_dir / f"{key}.npy"``.
    """
    return cache_dir / f"{key}.npy"


def _save_to_cache(
    embeddings: npt.NDArray[np.float32],
    path: pathlib.Path,
) -> None:
    """
    Save ``embeddings`` to ``path`` as a numpy ``.npy`` file.

    Parameters
    ----------
    embeddings : numpy.ndarray
        2-D float array of shape ``(n_texts, dim)``.
    path : pathlib.Path
        Target file path. Parent directories are created if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then rename atomically to avoid
    # leaving a corrupt partial file if the process is interrupted.
    tmp_path = path.with_suffix(".tmp.npy")
    try:
        np.save(str(tmp_path), embeddings, allow_pickle=False)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.debug("EmbeddingEngine: cached %d embeddings to %s.", len(embeddings), path)


def _load_from_cache(
    path: pathlib.Path,
    expected_n: int,
) -> npt.NDArray[np.float32] | None:
    """
    Load and validate a cached embedding array.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``.npy`` cache file.
    expected_n : int
        Expected number of rows. Used to detect stale or truncated caches.

    Returns
    -------
    numpy.ndarray or None
        The loaded array, or ``None`` if the cache is missing, invalid,
        or has the wrong shape.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        arr = np.load(str(path), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "EmbeddingEngine: failed to load cache %s — will regenerate. %s",
            path,
            exc,
        )
        return None

    # Validate shape
    if arr.ndim != 2:  # noqa: PLR2004
        logger.warning(
            "EmbeddingEngine: cache %s has wrong ndim=%d — will regenerate.",
            path,
            arr.ndim,
        )
        return None
    if arr.shape[0] != expected_n:
        logger.warning(
            "EmbeddingEngine: cache %s has %d rows but expected %d — will regenerate.",
            path,
            arr.shape[0],
            expected_n,
        )
        return None
    # Validate dtype
    if arr.dtype not in _FLOAT_DTYPES:
        logger.warning(
            "EmbeddingEngine: cache %s has dtype=%s — will regenerate.",
            path,
            arr.dtype,
        )
        return None

    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def _make_sentence_transformers_fn(
    model_name: str,
    batch_size: int,
    normalize: bool,
    show_progress_bar: bool,
    device: str | None,
) -> EmbedFn:
    """
    Build an ``EmbedFn`` backed by ``sentence_transformers``.

    The model is loaded once when the function is first called (lazy),
    then reused on all subsequent calls via a closure.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path.
    batch_size : int
        Encoding batch size.
    normalize : bool
        Whether to L2-normalise output vectors.
    show_progress_bar : bool
        Show tqdm progress bar during encoding.
    device : str or None
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``). ``None``
        lets sentence_transformers choose.

    Returns
    -------
    EmbedFn
        Callable accepting ``list[str]`` and returning ``np.ndarray``.
    """
    _model: list[Any] = []  # mutable container for lazy init

    def embed(texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed texts with sentence-transformers, loading model on first call."""
        if not _model:
            try:
                from sentence_transformers import (  # type: ignore[] # noqa: PLC0415
                    SentenceTransformer,
                )
            except ImportError as exc:
                raise ImportError(
                    "sentence_transformers is required for this embedding"
                    " backend.\nInstall it with: pip install sentence-transformers"
                ) from exc
            kwargs: dict[str, Any] = {}
            if device is not None:
                kwargs["device"] = device
            _model.append(SentenceTransformer(model_name, **kwargs))
            logger.debug(
                "EmbeddingEngine: loaded sentence_transformers model %r.", model_name
            )

        model = _model[0]
        result = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return result.astype(np.float32, copy=False)

    return embed


def _make_openai_fn(
    model_name: str,
    batch_size: int,
) -> EmbedFn:
    """
    Build an ``EmbedFn`` backed by the OpenAI embeddings API.

    Parameters
    ----------
    model_name : str
        OpenAI embedding model name (e.g. ``"text-embedding-3-small"``).
    batch_size : int
        Maximum texts per API request.

    Returns
    -------
    EmbedFn
        Callable accepting ``list[str]`` and returning ``np.ndarray``.
    """

    def embed(texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed texts via the OpenAI embeddings API in batches."""
        try:
            import openai  # type: ignore[] # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai is required for the OpenAI embedding backend.\n"
                "Install it with: pip install openai"
            ) from exc

        client = openai.OpenAI()
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=model_name)
            all_vecs.extend(item.embedding for item in response.data)

        return np.array(all_vecs, dtype=np.float32)

    return embed


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingEngine:
    """
    Multi-backend sentence embedding engine with SHA-256 file caching.

    Produces a 2-D ``float32`` numpy array of shape ``(n_texts, dim)``
    for a list of input strings. Embeddings are cached to ``.npy`` files
    keyed by ``(model_name, input_path, mtime, n_texts)`` so that
    unchanged corpora are served from disk in O(1).

    Parameters
    ----------
    model_name : str, optional
        Embedding model identifier. Interpretation depends on ``backend``.
        For ``sentence_transformers``, any HuggingFace model name.
        For ``openai``, any OpenAI embedding model name.
        Ignored when ``backend="custom"``.
        Default: ``"paraphrase-multilingual-mpnet-base-v2"``.
    backend : {"sentence_transformers", "openai", "custom"}, optional
        Which embedding backend to use. Default: ``"sentence_transformers"``.
    custom_fn : callable or None, optional
        User-supplied ``Callable[[list[str]], np.ndarray]``. Required when
        ``backend="custom"``. Ignored otherwise.
    cache_dir : pathlib.Path or None, optional
        Directory for ``.npy`` cache files. Created if absent. ``None``
        uses ``~/.cache/scikitplot/embeddings``. Pass
        ``pathlib.Path(os.devnull)`` to disable caching.
    enable_cache : bool, optional
        Set to ``False`` to completely disable file caching (always
        re-computes). Default: ``True``.
    batch_size : int, optional
        Number of texts per encoding batch. Relevant for
        ``sentence_transformers`` and ``openai`` backends. Default: ``64``.
    normalize : bool, optional
        L2-normalise output vectors to unit norm (required for cosine /
        inner-product similarity search). Default: ``True``.
    dtype : numpy.dtype, optional
        Output dtype. Default: ``numpy.float32``.
    show_progress_bar : bool, optional
        Show a tqdm progress bar during encoding (sentence_transformers
        only). Default: ``False``.
    device : str or None, optional
        PyTorch device for sentence_transformers (``"cpu"``, ``"cuda"``,
        ``"mps"``). ``None`` lets the library choose. Default: ``None``.

    Attributes
    ----------
    VALID_BACKENDS : tuple of str
        Class variable. All accepted backend names.

    Raises
    ------
    ValueError
        If ``backend="custom"`` but ``custom_fn`` is ``None``.
    ValueError
        If ``batch_size`` or ``dtype`` are invalid.
    ImportError
        At call time if the required backend library is not installed.

    See Also
    --------
    scikitplot.corpus.pipeline.CorpusPipeline : Integrates this engine.
    scikitplot.corpus._embeddings._multimodal_embedding.MultimodalEmbeddingEngine :
        Extends this engine with image, audio, and video modalities plus
        projection layer and LLM training export.

    Notes
    -----
    **Thread safety:** The internal model cache (``_embed_fn``) is
    initialised lazily and protected by a ``threading.Lock``.

    **Cache invalidation:** The cache key includes the source file's
    modification time. Any write to the source file (even a metadata
    update via ``touch``) invalidates the cache. If this is undesirable,
    pass a stable ``input_path`` (e.g. a logical identifier rather than
    the real path).

    **Normalisation:** When ``normalize=True``, zero-norm vectors (e.g.
    empty-string inputs) are left as zero vectors rather than producing
    NaN. The normalisation guard in :class:`~scikitplot.corpus.similarity.SimilarityIndex`
    will warn if any zero vectors are detected at search time.

    Examples
    --------
    Default usage (sentence_transformers):

    >>> engine = EmbeddingEngine()
    >>> texts = ["Hello world.", "Second sentence."]
    >>> vecs = engine.embed(texts)
    >>> vecs.shape
    (2, 768)

    Custom callable backend:

    >>> import numpy as np
    >>> engine = EmbeddingEngine(
    ...     backend="custom",
    ...     custom_fn=lambda texts: np.zeros((len(texts), 64), dtype=np.float32),
    ... )
    >>> engine.embed(["Hello."]).shape
    (1, 64)

    With source-file cache:

    >>> from pathlib import Path
    >>> vecs, from_cache = engine.embed_with_cache(
    ...     texts,
    ...     input_path=Path("corpus.txt"),
    ... )
    """

    VALID_BACKENDS: ClassVar[tuple[str, ...]] = (
        "sentence_transformers",
        "openai",
        "custom",
    )
    """Accepted ``backend`` values.

    For image/audio/video embeddings use
    :class:`~scikitplot.corpus._embeddings._multimodal_embedding.MultimodalEmbeddingEngine`
    which supports ``"clip"``, ``"whisper"``, ``"wav2vec"`` in addition.
    """

    model_name: str = field(default=DEFAULT_MODEL)
    backend: str = field(default="sentence_transformers")
    custom_fn: EmbedFn | None = field(default=None, repr=False)
    cache_dir: pathlib.Path | None = field(default=None)
    enable_cache: bool = field(default=True)
    batch_size: int = field(default=64)
    normalize: bool = field(default=True)
    dtype: Any = field(default=np.float32)
    show_progress_bar: bool = field(default=False)
    device: str | None = field(default=None)

    # Internal: lazily-initialised embed function + lock
    _embed_fn: EmbedFn | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate constructor fields after dataclass ``__init__``.

        Raises
        ------
        ValueError
            If ``backend`` is not in :attr:`VALID_BACKENDS`,
            ``backend="custom"`` without ``custom_fn``,
            ``batch_size <= 0``, or ``dtype`` is not a float dtype.
        """
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"EmbeddingEngine: backend must be one of"
                f" {self.VALID_BACKENDS}; got {self.backend!r}."
            )
        if self.backend == "custom" and self.custom_fn is None:
            raise ValueError(
                "EmbeddingEngine: backend='custom' requires a custom_fn"
                " callable. Pass custom_fn=your_function."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"EmbeddingEngine: batch_size must be > 0; got {self.batch_size!r}."
            )
        if self.dtype not in _FLOAT_DTYPES:
            raise ValueError(
                f"EmbeddingEngine: dtype must be one of"
                f" {[d.__name__ for d in _FLOAT_DTYPES]};"
                f" got {self.dtype!r}."
            )
        if self.cache_dir is None:
            object.__setattr__(self, "cache_dir", DEFAULT_CACHE_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
    ) -> npt.NDArray[np.float32]:
        """
        Compute embeddings for ``texts`` without caching.

        Parameters
        ----------
        texts : list of str
            Non-empty list of text strings. Empty strings produce zero
            vectors; they are not filtered here (filtering belongs in
            the pipeline).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(texts), dim)`` with dtype ``self.dtype``.

        Raises
        ------
        ValueError
            If ``texts`` is empty.
        ImportError
            If the required backend library is not installed.

        Examples
        --------
        >>> engine = EmbeddingEngine(
        ...     backend="custom",
        ...     custom_fn=lambda t: np.zeros((len(t), 32), dtype=np.float32),
        ... )
        >>> engine.embed(["hello"]).shape
        (1, 32)
        """
        if not texts:
            raise ValueError("EmbeddingEngine.embed: texts must be non-empty.")

        fn = self._get_embed_fn()
        start = timer()
        result = fn(texts)
        elapsed = timer() - start

        # Validate output
        if not isinstance(result, np.ndarray) or result.ndim != 2:  # noqa: PLR2004
            raise TypeError(
                f"EmbeddingEngine: backend returned"
                f" {type(result).__name__} with shape"
                f" {getattr(result, 'shape', '?')}."
                f" Expected 2-D numpy array."
            )
        if result.shape[0] != len(texts):
            raise ValueError(
                f"EmbeddingEngine: backend returned {result.shape[0]}"
                f" vectors for {len(texts)} input texts."
            )

        result = result.astype(self.dtype, copy=False)
        logger.info(
            "EmbeddingEngine: embedded %d texts → shape %s in %.1fs.",
            len(texts),
            result.shape,
            elapsed,
        )
        return result

    def embed_with_cache(
        self,
        texts: list[str],
        input_path: pathlib.Path,
    ) -> tuple[npt.NDArray[np.float32], bool]:
        """
        Compute embeddings with file caching keyed to ``input_path``.

        Parameters
        ----------
        texts : list of str
            Text strings to embed.
        input_path : pathlib.Path
            Path to the source file that generated ``texts``. Used to
            build the cache key (path + mtime + len(texts)).

        Returns
        -------
        embeddings : numpy.ndarray
            Array of shape ``(len(texts), dim)``.
        from_cache : bool
            ``True`` if the result was loaded from disk cache.

        Raises
        ------
        ValueError
            If ``texts`` is empty.
        OSError
            If the cache directory cannot be created.

        Examples
        --------
        >>> vecs, cached = engine.embed_with_cache(texts, Path("corpus.txt"))
        >>> cached  # True on second call with same inputs
        False
        """
        if not texts:
            raise ValueError(
                "EmbeddingEngine.embed_with_cache: texts must be non-empty."
            )

        if not self.enable_cache:
            return self.embed(texts), False

        # Build cache key
        try:
            mtime = input_path.stat().st_mtime
        except OSError:
            logger.warning(
                "EmbeddingEngine: cannot stat %s; caching disabled for this call.",
                input_path,
            )
            return self.embed(texts), False

        key = _make_cache_key(
            model_name=self.model_name,
            input_path=str(input_path.resolve()),
            source_mtime=mtime,
            n_texts=len(texts),
        )
        path = _cache_path(self.cache_dir, key)

        # Try cache load
        cached = _load_from_cache(path, expected_n=len(texts))
        if cached is not None:
            logger.info(
                "EmbeddingEngine: loaded %d embeddings from cache %s.",
                len(cached),
                path.name,
            )
            return cached.astype(self.dtype, copy=False), True

        # Compute and save
        embeddings = self.embed(texts)
        try:
            _save_to_cache(embeddings, path)
        except OSError as exc:
            logger.warning(
                "EmbeddingEngine: could not write cache to %s. %s", path, exc
            )

        return embeddings, False

    def embed_documents(
        self,
        documents: list[Any],
        input_path: pathlib.Path | None = None,
    ) -> list[Any]:
        """
        Embed a list of :class:`~scikitplot.corpus.schema.CorpusDocument`
        instances in-place (sets ``doc.embedding`` on each).

        Parameters
        ----------
        documents : list of CorpusDocument
            Documents to embed. Each must have a non-empty ``text`` field.
        input_path : pathlib.Path or None, optional
            Source path for cache key. ``None`` disables caching.

        Returns
        -------
        list of CorpusDocument
            The same list with ``embedding`` fields populated (via
            ``replace()`` — originals are not mutated).

        Examples
        --------
        >>> docs = list(reader.get_documents())
        >>> docs = engine.embed_documents(docs)
        >>> docs[0].has_embedding
        True
        """  # noqa: D205
        if not documents:
            return documents

        texts = [doc.normalized_text or doc.text for doc in documents]

        if input_path is not None and self.enable_cache:
            embeddings, _ = self.embed_with_cache(texts, input_path)
        else:
            embeddings = self.embed(texts)

        return [doc.replace(embedding=embeddings[i]) for i, doc in enumerate(documents)]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_embed_fn(self) -> EmbedFn:
        """
        Lazily initialise and return the backend embed function.

        Thread-safe via double-checked locking.

        Returns
        -------
        EmbedFn
            Callable accepting ``list[str]`` → ``np.ndarray``.
        """
        if self._embed_fn is not None:
            return self._embed_fn

        with self._lock:
            if self._embed_fn is not None:
                return self._embed_fn

            if self.backend == "sentence_transformers":
                fn = _make_sentence_transformers_fn(
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    normalize=self.normalize,
                    show_progress_bar=self.show_progress_bar,
                    device=self.device,
                )
            elif self.backend == "openai":
                fn = _make_openai_fn(
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                )
            else:  # custom
                fn = self.custom_fn  # type: ignore[assignment]

            object.__setattr__(self, "_embed_fn", fn)
            return fn

    def __repr__(self) -> str:
        """Return ``EmbeddingEngine(backend=..., model_name=..., ...)``."""
        return (
            f"EmbeddingEngine("
            f"backend={self.backend!r},"
            f" model_name={self.model_name!r},"
            f" batch_size={self.batch_size},"
            f" normalize={self.normalize},"
            f" cache={'enabled' if self.enable_cache else 'disabled'})"
        )
