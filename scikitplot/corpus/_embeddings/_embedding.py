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
files keyed by a content-addressed SHA-256 hash of the model identity,
backend/dtype/normalisation, and the full input text (see ``_make_cache_key``)
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
6. **Content-addressed cache key** — the key hashes the full input text
   plus model/backend/dtype/normalisation, so any change to text, model,
   revision, or output config invalidates stale caches (CORPUS-CACHE-001).
7. **No shape/dtype validation on cache load** — loaded arrays are
   validated for shape, dtype, and row count before use.

Python compatibility:

Python 3.8-3.15. ``numpy`` is required. ``sentence_transformers``,
``openai``, and ``tiktoken`` are optional; graceful ``ImportError`` at
call time when not installed.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import pathlib
import threading
import uuid
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

from .._atomic import atomic_write_path
from .._embedding_manifest import EmbeddingManifest
from ._cache_policy import (
    CachedEmbeddings,
    CachePolicy,
    CacheStatus,
    enforce_bounds,
    hash_fields,
    is_commit,
    touch_entry,
)

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


#: Bump to invalidate every cache entry when key semantics change. v3 added the
#: provider identity (a custom provider used to share one key with every other)
#: and keeps the full SHA-256 digest, the package rule for durable identities.
_CACHE_KEY_SCHEMA = "v3"


def _make_cache_key(
    model_name: str,
    texts: list[str],
    *,
    normalize: bool,
    dtype: str,
    backend: str = "",
    model_revision: str = "",
    provider_id: str = "",
) -> str:
    """
    Compute a content-addressed cache key.

    A cache hit under this key is semantically equivalent to recomputation:
    the key is a SHA-256 digest over everything that affects the output
    embeddings — the embedding model (``model_name`` + ``model_revision``),
    the ``backend`` and the provider identity (``provider_id``), the output
    ``dtype`` and ``normalize`` flag, and the **full text content** of every
    input (length-prefixed so that, e.g., ``["ab", "c"]`` and ``["a", "bc"]``
    never collide).

    Source path and modification time are deliberately *not* part of the key.
    They are a proxy for content that fails when the text changes without
    changing path/mtime/count — the CORPUS-CACHE-001 stale-hit defect. Because
    the key is content-addressed, identical texts embedded from different paths
    correctly share one cache entry.

    Parameters
    ----------
    model_name : str
        Name or identifier of the embedding model.
    texts : list of str
        The exact texts that will be embedded.
    normalize : bool
        Whether embeddings are L2-normalised (affects output values).
    dtype : str
        Canonical output dtype name (e.g. ``"float32"``).
    backend : str, optional
        Embedding backend identifier (e.g. ``"sentence_transformers"``).
    model_revision : str, optional
        Model revision/version pin, when known. Empty when unpinned.
    provider_id : str, optional
        Stable identity of a custom provider (``custom_fn_id``). Two different
        callables must never share a key: before this field existed, a second
        custom provider was served the first one's vectors and never called.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.

    Examples
    --------
    >>> a = _make_cache_key("m", ["hello"], normalize=True, dtype="float32")
    >>> b = _make_cache_key("m", ["world"], normalize=True, dtype="float32")
    >>> a != b  # different content -> different key
    True
    >>> len(a)
    64
    """
    fields = [
        _CACHE_KEY_SCHEMA,
        model_name,
        model_revision,
        backend,
        provider_id,
        "1" if normalize else "0",
        dtype,
        str(len(texts)),
        *texts,
    ]
    return hash_fields([value.encode("utf-8") for value in fields])


def _cache_path(cache_dir: pathlib.Path, key: str) -> pathlib.Path:
    """Return the ``.npy`` path for a given cache key.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory holding cached embedding arrays.
    key : str
        Hex cache key from :func:`_make_cache_key`.

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
    # Publish via the shared atomic primitive: unique staging temp + fsync +
    # atomic replace, so concurrent writers of the same content-addressed key
    # never clobber each other (CORPUS-TMP-001).
    atomic_write_path(
        path,
        lambda p: np.save(str(p), embeddings, allow_pickle=False),
        suffix=".npy",
    )
    logger.debug("EmbeddingEngine: cached %d embeddings to %s.", len(embeddings), path)


def _read_entry(path: pathlib.Path) -> npt.NDArray[Any] | None:
    """Read a cache entry, ``None`` when there is none to read.

    Parameters
    ----------
    path : pathlib.Path
        The entry's path.

    Returns
    -------
    numpy.ndarray or None
        The stored array; ``None`` when the entry is missing, empty, or
        unreadable.

    Notes
    -----
    **Developer.** One stat, no ``exists()`` first: a concurrent evictor may
    remove the entry at any moment (also between the stat and the load), and
    a vanished entry is a miss, logged at debug level. A path that cannot be
    read at all (``cache_dir`` is a file, permission denied) is a miss too;
    the store that follows reports its own failure. A corrupt entry is logged
    as a warning and regenerated.
    """
    try:
        if path.stat().st_size == 0:
            return None
        return np.load(str(path), allow_pickle=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("EmbeddingEngine: cannot read cache %s: %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "EmbeddingEngine: failed to load cache %s — will regenerate. %s",
            path,
            exc,
        )
        return None


def _load_from_cache(
    path: pathlib.Path,
    expected_n: int,
) -> npt.NDArray[np.floating] | None:
    """
    Load and validate a cached embedding array, in the dtype it was stored in.

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

    Notes
    -----
    **Developer.** The array is returned in its stored dtype. The cache key
    already includes the output dtype, so the stored dtype is the requested
    one; this used to force ``float32`` here, and a ``float64`` engine got
    ``float32``-rounded values back from every hit (``1 + 1e-12`` came back as
    ``1.0``) -- a hit that was not equivalent to recomputation.
    """
    arr = _read_entry(path)
    if arr is None:
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

    return arr


def _l2_normalise_rows(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Return ``matrix`` with every non-zero row scaled to unit L2 norm.

    Parameters
    ----------
    matrix : numpy.ndarray
        2-D float array.

    Returns
    -------
    numpy.ndarray
        A new ``float64`` array. Zero rows stay zero rather than NaN; rows
        with a non-finite value stay non-finite, so the index's finiteness
        check still refuses them.

    Notes
    -----
    **Developer.** The norm is taken in ``float64``, and the caller casts to
    its dtype afterwards. In ``float16`` the sum of squares overflows above
    65504 (a 768-d vector with components of 10 already does), and every
    such row came back as zeros; a NaN row also came back as zeros, which
    hid the provider error from the index's finiteness check.
    """
    work = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(work, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return work / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def _make_sentence_transformers_fn(
    model_name: str,
    batch_size: int,
    normalize: bool,
    show_progress_bar: bool,
    device: str | None,
    *,
    revision: str | None = None,
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
    revision : str or None, optional
        Model revision (commit, tag or branch) to load. ``None`` loads the
        hub's current version.

    Returns
    -------
    EmbedFn
        Callable accepting ``list[str]`` and returning ``np.ndarray``.

    Notes
    -----
    **Developer.** ``revision`` used to stop at the cache key and the
    manifest: both recorded a pinned revision while the model loaded was
    the hub's current one. It is now passed to ``SentenceTransformer``; an
    installed version that cannot take it raises, rather than loading another
    revision under the recorded one.
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
            if revision:
                kwargs["revision"] = revision
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
    keyed by a content-addressed hash of the texts and model params so that
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
        inner-product similarity search), for every backend. Default: ``True``.
    dtype : numpy.dtype, optional
        Output dtype. Default: ``numpy.float32``.
    show_progress_bar : bool, optional
        Show a tqdm progress bar during encoding (sentence_transformers
        only). Default: ``False``.
    device : str or None, optional
        PyTorch device for sentence_transformers (``"cpu"``, ``"cuda"``,
        ``"mps"``). ``None`` lets the library choose. Default: ``None``.
    model_revision : str, optional
        Model revision (commit, tag or branch). The sentence_transformers
        backend loads exactly this revision; every backend records it in the
        cache key and the embedding manifest. ``""`` (default) is unpinned:
        the hub's current version is loaded. Only a commit hash makes the disk
        cache exact for sentence_transformers; an empty value or a branch or
        tag (which can move) uses the cache only as ``cache_policy`` allows
        (by default it does not). For other backends the value is recorded,
        not sent: it is your statement of which model version the name stands
        for.
    custom_fn_id : str or None, optional
        Stable identity of ``custom_fn``, e.g. ``"my-embedder/v3"``; change it
        whenever the function's output changes. It is what the disk cache and
        the embedding manifest know the provider by. ``None`` (default) means
        the provider is unidentified: its vectors are not disk-cached, and the
        manifest it stamps is unique to this engine instance, so documents
        embedded by two engines cannot be indexed together by accident. Only
        valid with ``backend="custom"``.
    cache_policy : CachePolicy or None, optional
        How the disk cache treats an unpinned model and how big it may grow.
        ``None`` (default) is ``CachePolicy()``: an unpinned model's
        embeddings are computed, not cached, and the cache is unbounded.

    Attributes
    ----------
    VALID_BACKENDS : tuple of str
        Class variable. All accepted backend names.

    Raises
    ------
    ValueError
        If ``backend="custom"`` but ``custom_fn`` is ``None``, or
        ``custom_fn_id`` is given with another backend or is empty.
    ValueError
        If ``batch_size`` or ``dtype`` are invalid.
    TypeError
        If ``model_revision`` is not a str, or ``cache_policy`` is neither a
        :class:`CachePolicy` nor ``None``.
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

    **Cache identity:** The cache key is content-addressed: the full input
    text plus model, revision, backend, provider identity, dtype and
    normalisation (see ``_make_cache_key``). Source path and modification time
    are not part of it, so the same text is served from cache whether it came
    from a file or a URL. A custom provider is cached only when it carries a
    ``custom_fn_id``; without one, two different functions would share a key.

    **Provenance:** :meth:`embed_documents` stamps every document with the
    fingerprint of :meth:`embedding_manifest`, so an index refuses to rank
    vectors from two embedding generations against each other.

    **Normalisation:** When ``normalize=True``, every backend's vectors are
    L2-normalised; sentence-transformers does it while encoding, the engine
    does it for the others. Zero-norm vectors (e.g. empty-string inputs) are
    left as zero vectors rather than producing NaN. The normalisation guard in
    :class:`~scikitplot.corpus.similarity.RetrievalIndex` will warn if any zero
    vectors are detected at search time.

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

    With the disk cache (a custom provider needs an identity to be cached):

    >>> engine = EmbeddingEngine(
    ...     backend="custom",
    ...     custom_fn=lambda texts: np.ones((len(texts), 4), dtype=np.float32),
    ...     custom_fn_id="ones/v1",
    ... )
    >>> vecs, from_cache = engine.embed_with_cache(["Hello."])
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
    model_revision: str = field(default="")
    # Last, so that positional construction keeps its meaning.
    custom_fn_id: str | None = field(default=None)
    cache_policy: CachePolicy | None = field(default=None)

    # Internal: lazily-initialised embed function + lock
    _embed_fn: EmbedFn | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    # Internal: identity of an unidentified custom provider, unique to this
    # instance (see ``embedding_manifest``), the cache notices already logged
    # (once each), and the revision ``cache_policy`` resolved (``None``: not
    # yet asked; ``""``: resolution failed).
    _instance_token: str = field(
        default_factory=lambda: uuid.uuid4().hex, init=False, repr=False
    )
    _reported: set = field(default_factory=set, init=False, repr=False)
    _resolved_revision: str | None = field(default=None, init=False, repr=False)
    _revision_lock: threading.Lock = field(
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
        if self.custom_fn_id is not None:
            if self.backend != "custom":
                raise ValueError(
                    "EmbeddingEngine: custom_fn_id identifies a custom provider;"
                    f" it does not apply to backend={self.backend!r}."
                )
            if not isinstance(self.custom_fn_id, str) or not self.custom_fn_id.strip():
                raise ValueError(
                    "EmbeddingEngine: custom_fn_id must be a non-empty string;"
                    f" got {self.custom_fn_id!r}."
                )
        if self.batch_size <= 0:
            raise ValueError(
                f"EmbeddingEngine: batch_size must be > 0; got {self.batch_size!r}."
            )
        if not isinstance(self.model_revision, str):
            raise TypeError(
                'EmbeddingEngine: model_revision must be a str ("" when'
                f" unpinned); got {self.model_revision!r}."
            )
        if self.dtype not in _FLOAT_DTYPES:
            raise ValueError(
                f"EmbeddingEngine: dtype must be one of"
                f" {[d.__name__ for d in _FLOAT_DTYPES]};"
                f" got {self.dtype!r}."
            )
        if self.cache_dir is None:
            object.__setattr__(self, "cache_dir", DEFAULT_CACHE_DIR)
        if self.cache_policy is None:
            object.__setattr__(self, "cache_policy", CachePolicy())
        elif not isinstance(self.cache_policy, CachePolicy):
            raise TypeError(
                "EmbeddingEngine: cache_policy must be a CachePolicy or None;"
                f" got {type(self.cache_policy).__name__}."
            )

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

        if self.normalize and self.backend != "sentence_transformers":
            # sentence-transformers normalises while encoding; every other
            # backend is normalised here, so ``normalize=True`` -- which the
            # cache key and the manifest both record -- holds for all of them.
            # A custom provider's vectors used to come back unnormalised.
            # Normalise before the cast, in float64 (see _l2_normalise_rows).
            result = _l2_normalise_rows(result)
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
        input_path: pathlib.Path | None = None,
    ) -> tuple[npt.NDArray[np.float32], bool]:
        """
        Compute embeddings through the content-addressed disk cache.

        Parameters
        ----------
        texts : list of str
            Text strings to embed.
        input_path : pathlib.Path or None, optional
            Source of ``texts``, for diagnostics only: the cache key is
            content-addressed over ``texts`` and the model parameters, never the
            path or its mtime. ``None`` (a URL or in-memory source) is cached
            exactly like a file.

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

        See Also
        --------
        embed_cached : The same call, reporting what it did.

        Examples
        --------
        >>> engine = EmbeddingEngine(
        ...     backend="custom",
        ...     custom_fn=lambda t: np.ones((len(t), 4), dtype=np.float32),
        ...     custom_fn_id="ones/v1",
        ...     enable_cache=False,
        ... )
        >>> engine.embed_with_cache(["Hello."])[1]
        False
        """
        result = self.embed_cached(texts, input_path)
        return result.embeddings, result.status is CacheStatus.HIT

    def embed_cached(
        self,
        texts: list[str],
        input_path: pathlib.Path | None = None,
    ) -> CachedEmbeddings:
        """
        Compute embeddings through the disk cache, and say what happened.

        Parameters
        ----------
        texts : list of str
            Text strings to embed.
        input_path : pathlib.Path or None, optional
            Source of ``texts``, for diagnostics only.

        Returns
        -------
        CachedEmbeddings
            ``(embeddings, status, key, revision)``; ``status`` is a
            :class:`CacheStatus`.

        Raises
        ------
        ValueError
            If ``texts`` is empty.

        Notes
        -----
        **User.** The cache is consulted only when a hit is what this engine
        would compute: a sentence-transformers model pinned to a commit
        (``model_revision``, or resolved with ``CachePolicy(unpinned=
        "resolve")``), a custom provider's ``custom_fn_id``, the stated
        ``model_revision`` of another backend, or, by your choice,
        ``CachePolicy(unpinned="allow")``. Otherwise the embeddings are
        computed and the status says why the cache was bypassed (logged once
        per engine). A failed write is reported as ``STORE_FAILED``, never
        raised.

        **Developer.** A hit refreshes the entry's modification time; after a
        store, :func:`enforce_bounds` evicts least recently used entries.

        Examples
        --------
        >>> engine = EmbeddingEngine(
        ...     backend="custom",
        ...     custom_fn=lambda t: np.ones((len(t), 4), dtype=np.float32),
        ...     enable_cache=False,
        ... )
        >>> engine.embed_cached(["Hello."]).status.value
        'disabled'
        """
        if not texts:
            raise ValueError("EmbeddingEngine.embed_cached: texts must be non-empty.")
        if not self.enable_cache:
            return CachedEmbeddings(self.embed(texts), CacheStatus.DISABLED, None, None)
        revision, bypass = self._cache_identity()
        if bypass is not None:
            self._report_bypass(bypass)
            return CachedEmbeddings(self.embed(texts), bypass, None, None)

        # Content-addressed key: identity is the texts + model/backend/params,
        # NOT the source path or mtime. ``input_path`` is advisory (diagnostics
        # / provenance) and never affects cache identity.
        key = _make_cache_key(
            self.model_name,
            texts,
            normalize=self.normalize,
            dtype=np.dtype(self.dtype).name,
            backend=self.backend,
            model_revision=revision,
            provider_id=self.custom_fn_id or "",
        )
        path = _cache_path(self.cache_dir, key)
        logger.debug(
            "EmbeddingEngine: content-addressed cache key for %s (%d texts) -> %s",
            input_path,
            len(texts),
            path.name,
        )
        named = revision or None

        cached = _load_from_cache(path, expected_n=len(texts))
        if cached is not None:
            touch_entry(path)
            logger.info(
                "EmbeddingEngine: loaded %d embeddings from cache %s.",
                len(cached),
                path.name,
            )
            return CachedEmbeddings(
                cached.astype(self.dtype, copy=False), CacheStatus.HIT, key, named
            )

        embeddings = self.embed(texts)
        try:
            _save_to_cache(embeddings, path)
        except OSError as exc:
            logger.warning(
                "EmbeddingEngine: could not write cache to %s. %s", path, exc
            )
            return CachedEmbeddings(embeddings, CacheStatus.STORE_FAILED, key, named)
        if not enforce_bounds(self.cache_dir, self.cache_policy, stored=path):
            return CachedEmbeddings(embeddings, CacheStatus.TOO_LARGE, key, named)
        return CachedEmbeddings(embeddings, CacheStatus.STORED, key, named)

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
            Source of the documents, for diagnostics only. Caching follows
            ``enable_cache`` and ``cache_policy`` (see :meth:`embed_cached`);
            the key is content-addressed.

        Returns
        -------
        list of CorpusDocument
            The same list with ``embedding`` and ``embedding_manifest_id``
            populated (via ``replace()`` — originals are not mutated).

        Notes
        -----
        **Developer.** ``embedding_manifest_id`` is what lets an index refuse
        vectors from two generations (``RetrievalIndex._reject_mixed_manifests``).
        Documents used to leave this method untagged, and untagged vectors are
        never compared, so the guard never fired: two providers' vectors were
        built into one index and ranked against each other without error.

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

        embeddings = self.embed_cached(texts, input_path).embeddings

        manifest_id = self.embedding_manifest(int(embeddings.shape[1])).fingerprint
        return [
            doc.replace(embedding=embeddings[i], embedding_manifest_id=manifest_id)
            for i, doc in enumerate(documents)
        ]

    def embedding_manifest(self, dimension: int) -> EmbeddingManifest:
        """
        Describe the vectors this engine produces.

        Parameters
        ----------
        dimension : int
            Vector length, known once the first vectors exist.

        Returns
        -------
        EmbeddingManifest
            ``provider`` is the backend; ``model`` the model name, or for a
            custom provider its ``custom_fn_id``; ``revision`` the pinned
            revision or ``None``; plus dimension, dtype and normalisation.

        Raises
        ------
        ValueError
            If ``dimension`` is not a positive integer.

        Notes
        -----
        **User.** Give a custom provider a ``custom_fn_id``: an unidentified
        one is named after this engine *instance* (``custom:unidentified:...``),
        so documents embedded by two engines -- even two engines wrapping the
        same function -- are refused as different generations when indexed
        together. That is the safe reading of "no identity": nothing proves the
        two produced comparable vectors.
        """
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError(
                f"EmbeddingEngine.embedding_manifest: dimension must be a positive"
                f" integer; got {dimension!r}."
            )
        if self.backend == "custom":
            model = self.custom_fn_id or f"custom:unidentified:{self._instance_token}"
        else:
            model = self.model_name
        return EmbeddingManifest(
            provider=self.backend,
            model=model,
            revision=self._effective_revision() or None,
            dimension=dimension,
            dtype=np.dtype(self.dtype).name,
            normalization="l2" if self.normalize else "none",
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _effective_revision(self) -> str:
        """Return the pinned or resolved revision, ``""`` when there is none.

        Returns
        -------
        str
            ``model_revision`` when set; for sentence_transformers with
            ``CachePolicy(unpinned="resolve")`` the resolver's commit (asked
            once per engine, before the model is loaded, so the loaded model
            is that commit); else ``""``.
        """
        if self.model_revision or self.backend != "sentence_transformers":
            return self.model_revision
        if self.cache_policy.unpinned != "resolve":
            return ""
        if self._resolved_revision is None:
            # Its own lock: _get_embed_fn calls this while holding _lock.
            with self._revision_lock:
                if self._resolved_revision is None:
                    object.__setattr__(
                        self, "_resolved_revision", self._resolve_revision()
                    )
        return self._resolved_revision  # type: ignore[return-value]

    def _resolve_revision(self) -> str:
        """Ask ``cache_policy.revision_resolver``; ``""`` when it fails.

        Returns
        -------
        str
            The commit, or ``""`` (logged) when the resolver raised or did not
            return a commit hash (a branch or tag can move).
        """
        resolver = self.cache_policy.revision_resolver
        try:
            revision = resolver(self.model_name)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            # Any failure (offline, unknown id, a resolver bug) leaves the
            # model unpinned: it still loads, the cache is bypassed and says so.
            logger.warning(
                "EmbeddingEngine: could not resolve the revision of %r (%s: %s);"
                " the model loads unpinned and the disk cache is bypassed.",
                self.model_name,
                type(exc).__name__,
                exc,
            )
            return ""
        if not is_commit(revision):
            logger.warning(
                "EmbeddingEngine: revision_resolver returned %r for %r, not a"
                " commit hash; the model loads unpinned and the disk cache is"
                " bypassed.",
                revision,
                self.model_name,
            )
            return ""
        return revision

    def _cache_identity(self) -> tuple[str, CacheStatus | None]:
        """Return the revision the key names, or why the cache is bypassed.

        Returns
        -------
        revision : str
            The key's revision field (``""`` for an identified custom
            provider or an allowed unpinned model).
        bypass : CacheStatus or None
            ``None`` when a hit is provable.

        Notes
        -----
        **Developer.** For sentence_transformers only a commit hash is exact
        (a branch or tag such as ``"main"`` moves, so it follows
        ``unpinned`` like an empty revision, and is not resolved); other
        backends do not load a revision, so a non-empty ``model_revision`` is
        the caller's statement and is trusted as such.
        """
        if self.backend == "custom":
            if self.custom_fn_id is None:
                return "", CacheStatus.BYPASS_UNIDENTIFIED
            return self.model_revision, None
        revision = self._effective_revision()
        if self.backend != "sentence_transformers" and revision:
            # Recorded, not loaded: the caller's statement of the version.
            return revision, None
        if is_commit(revision):
            return revision, None
        # Unpinned, or a branch or tag (which can move upstream).
        if self.cache_policy.unpinned == "allow":
            return revision, None
        return revision, CacheStatus.BYPASS_UNPINNED

    def _report_bypass(self, status: CacheStatus) -> None:
        """Log, once per engine and status, why the cache was bypassed."""
        if status in self._reported:
            return
        self._reported.add(status)
        if status is CacheStatus.BYPASS_UNIDENTIFIED:
            logger.info(
                "EmbeddingEngine: custom provider has no custom_fn_id;"
                " its embeddings are computed, not disk-cached. Pass"
                " custom_fn_id='name/version' to cache them."
            )
        elif self.model_revision:
            logger.info(
                "EmbeddingEngine: model %r revision %r is not a commit (a branch"
                " or tag can move, and is not resolved); its embeddings are"
                " computed, not disk-cached. Pin model_revision to a lowercase"
                " commit hash, or pass cache_policy=CachePolicy(unpinned='allow').",
                self.model_name,
                self.model_revision,
            )
        else:
            logger.info(
                "EmbeddingEngine: model %r is not pinned to a commit; its"
                " embeddings are computed, not disk-cached (an entry could"
                " outlive an upstream change). Pin model_revision, or pass"
                " cache_policy=CachePolicy(unpinned='resolve',"
                " revision_resolver=resolve_hub_revision) or"
                " CachePolicy(unpinned='allow').",
                self.model_name,
            )

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
                    revision=self._effective_revision() or None,
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
