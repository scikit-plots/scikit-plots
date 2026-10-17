# scikitplot/corpus/_registry/_registry.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._registry._registry
======================================

Central component registry for the scikitplot corpus pipeline.

Design invariants:

* :class:`ComponentRegistry` is a simple, transparent dict wrapper.
  No metaclass magic, no global state beyond the module-level
  ``registry`` singleton.
* Registration is explicit: built-in components register themselves on
  import; third-party components call ``registry.register_*()``.
* All ``get_*`` methods fail fast with an actionable ``KeyError`` that
  lists all registered names.
* The registry is thread-safe for read access. Registrations should
  happen at import time (not in hot paths) so the lock is only used
  for writes.
* Built-in components are registered lazily (on first call to
  :meth:`ComponentRegistry.register_builtins`) to avoid circular
  imports during package initialisation.

Python compatibility:

Python 3.8-3.15. No external dependencies. ``from __future__ import
annotations`` throughout.
"""  # noqa: D205, D400

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import threading
from types import ModuleType
from typing import Any, Dict, List, Optional, Type  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "ComponentRegistry",
    "registry",
]


def _fqcn(cls: type) -> str:
    # __qualname__ or __name__ or f"{v.__module__}.{v.__name__}"
    return f"{cls.__module__}.{cls.__qualname__}"


def _load_class_from_fqcn(fqcn: str) -> type:
    """
    Load a class from its fully-qualified class name.

    Parameters
    ----------
    fqcn : str
        Fully-qualified class path, e.g.
        "package.module.ClassName" or
        "package.module.Outer.Inner".

    Returns
    -------
    type
        Loaded class object.

    Raises
    ------
    ValueError
        If fqcn is malformed.
    ImportError
        If module cannot be imported.
    AttributeError
        If attribute path is invalid.
    TypeError
        If resolved object is not a class.
    """
    if not isinstance(fqcn, str) or "." not in fqcn:
        raise ValueError(f"Invalid FQCN: {fqcn!r}")

    module_path, _, attr_path = fqcn.partition(".")
    # Need last module split, not first
    module_path, attr_path = fqcn.rsplit(".", 1)

    module: ModuleType = importlib.import_module(module_path)

    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)

    if not isinstance(obj, type):
        raise TypeError(f"Resolved object is not a class: {fqcn}")

    return obj


class ComponentRegistry:
    """
    Central look-up table for corpus pipeline components.

    Stores class references (not instances) for four component types:
    chunkers, filters, readers, and normalizers. Callers retrieve a class
    and instantiate it with their own parameters.

    Notes
    -----
    The module-level ``registry`` singleton is pre-populated with all
    built-in components via :meth:`register_builtins`. Third-party
    packages can register additional components after import.

    Examples
    --------
    >>> from scikitplot.corpus._registry import registry
    >>> registry.register_builtins()
    >>> chunker = registry.build_chunker("paragraph", min_length=20)
    >>> type(chunker).__name__, chunker.config.min_length
    ('ParagraphChunker', 20)
    """

    def __init__(self) -> None:
        self._chunkers: dict[str, type] = {}
        #: Config dataclass per chunker name (R3-1); a chunker registered
        #: without one is constructed from keyword arguments directly.
        self._chunker_configs: dict[str, type] = {}
        self._filters: dict[str, type] = {}
        self._readers: dict[str, type] = {}
        self._normalizers: dict[str, type] = {}
        #: Keys populated by :meth:`register_builtins`.  Overwriting one
        #: requires an explicit ``replace=True`` (P-I0-07 / F-R12-01).
        self._builtin_keys: dict[str, set[str]] = {
            "chunker": set(),
            "filter": set(),
            "reader": set(),
            "normalizer": set(),
        }
        self._lock: threading.Lock = threading.Lock()
        self._builtins_registered: bool = False

    # ------------------------------------------------------------------
    # Built-in registration (lazy, idempotent)
    # ------------------------------------------------------------------

    def register_builtins(self) -> None:
        """
        Register all built-in corpus pipeline components.

        Safe to call multiple times — subsequent calls are no-ops.
        Triggers the necessary imports to populate the
        :class:`~scikitplot.corpus._base.DocumentReader` registry as well.

        Notes
        -----
        Importing ``scikitplot.corpus._readers`` as a side effect here
        is intentional: it populates the ``DocumentReader._registry``
        extension map used by :meth:`~scikitplot.corpus._base.DocumentReader.create`.

        A name already registered is kept: a component you registered before
        this call is never replaced by the built-in of the same name (the
        builder calls this on the shared registry when a build starts).
        """
        with self._lock:
            if self._builtins_registered:
                return

            # --- Chunkers (R3-1: every built-in, each with its config) ---
            try:
                from .._chunkers import (  # noqa: PLC0415
                    FixedWindowChunker,
                    FixedWindowChunkerConfig,
                    ParagraphChunker,
                    ParagraphChunkerConfig,
                    ParentChildChunker,
                    ParentChildChunkerConfig,
                    RecursiveChunker,
                    RecursiveChunkerConfig,
                    SemanticChunker,
                    SemanticChunkerConfig,
                    SentenceChunker,
                    SentenceChunkerConfig,
                    StructureChunker,
                    StructureChunkerConfig,
                    WordChunker,
                    WordChunkerConfig,
                )

                for name, chunker_cls, config_cls in (
                    ("sentence", SentenceChunker, SentenceChunkerConfig),
                    ("paragraph", ParagraphChunker, ParagraphChunkerConfig),
                    ("fixed_window", FixedWindowChunker, FixedWindowChunkerConfig),
                    ("word", WordChunker, WordChunkerConfig),
                    ("semantic", SemanticChunker, SemanticChunkerConfig),
                    ("recursive", RecursiveChunker, RecursiveChunkerConfig),
                    ("parent_child", ParentChildChunker, ParentChildChunkerConfig),
                    ("structure", StructureChunker, StructureChunkerConfig),
                ):
                    if self._add_builtin("chunker", self._chunkers, name, chunker_cls):
                        self._chunker_configs[name] = config_cls
            except ImportError as exc:
                logger.warning("ComponentRegistry: could not import chunkers: %s.", exc)

            # --- Filters ---
            try:
                from .._base import DefaultFilter  # noqa: PLC0415

                self._add_builtin("filter", self._filters, "default", DefaultFilter)
            except ImportError as exc:
                logger.warning("ComponentRegistry: could not import filters: %s.", exc)

            # --- Readers (populates DocumentReader._registry as side effect) ---
            try:
                from .. import _readers  # noqa: F401, PLC0415
                from .._base import DocumentReader  # noqa: PLC0415

                for ext, cls in DocumentReader.subclass_by_type().items():
                    self._add_builtin("reader", self._readers, ext, cls)
            except ImportError as exc:
                logger.warning("ComponentRegistry: could not import readers: %s.", exc)

            # --- Normalizers ---
            try:
                from .._normalizers import (  # noqa: PLC0415
                    DedupLinesNormalizer,
                    HTMLStripNormalizer,
                    LanguageDetectionNormalizer,
                    LowercaseNormalizer,
                    UnicodeNormalizer,
                    WhitespaceNormalizer,
                )

                for name, normalizer_cls in (
                    ("unicode", UnicodeNormalizer),
                    ("whitespace", WhitespaceNormalizer),
                    ("html_strip", HTMLStripNormalizer),
                    ("lowercase", LowercaseNormalizer),
                    ("dedup_lines", DedupLinesNormalizer),
                    ("language_detect", LanguageDetectionNormalizer),
                ):
                    self._add_builtin(
                        "normalizer", self._normalizers, name, normalizer_cls
                    )
            except ImportError as exc:
                logger.warning(
                    "ComponentRegistry: could not import normalizers: %s.", exc
                )

            self._builtins_registered = True
            logger.debug(
                "ComponentRegistry: built-ins registered — "
                "%d chunkers, %d filters, %d readers, %d normalizers.",
                len(self._chunkers),
                len(self._filters),
                len(self._readers),
                len(self._normalizers),
            )

    def _add_builtin(
        self, kind: str, store: dict[str, type], name: str, cls: type
    ) -> bool:
        """Insert a built-in unless ``name`` is taken; return whether it was.

        Called under ``self._lock``. A registration made before
        :meth:`register_builtins` wins, and the built-in is recorded only when
        it is the one stored (review of R3-1, F1).
        """
        if name in store:
            logger.debug(
                "ComponentRegistry: %s %r already registered to %r; keeping it "
                "over the built-in %r.",
                kind,
                name,
                store[name].__name__,
                cls.__name__,
            )
            return False
        store[name] = cls
        self._builtin_keys[kind].add(name)
        return True

    # ------------------------------------------------------------------
    # Chunker registration / retrieval
    # ------------------------------------------------------------------

    def register_chunker(
        self,
        name: str,
        cls: type,
        *,
        config: type | None = None,
        replace: bool = False,
    ) -> None:
        """
        Register a chunker class under ``name``.

        Parameters
        ----------
        name : str
            Registry key (lowercase, underscore-separated). Must be
            non-empty.
        cls : type
            A class satisfying ``ChunkerProtocol`` (``chunk`` and
            ``chunk_batch``).
        config : type or None, optional
            The chunker's configuration dataclass. When given (or declared by
            the class as ``config_class``, as every built-in does),
            :meth:`build_chunker` (and ``BuilderConfig.chunker_kwargs``)
            build this config from keyword arguments and pass it as
            ``cls(config=...)``; unknown keys raise. When neither, keyword
            arguments go to ``cls`` directly. A class declaring
            ``config_class`` accepts only that class or a subclass.
        replace : bool, optional
            Replace an existing registration deliberately. Default ``False``.

        Raises
        ------
        ValueError
            If ``name`` is empty, or already registered and ``replace`` is
            ``False``.
        TypeError
            If ``cls`` is not a conforming class, its ``chunk`` cannot take
            ``extra_metadata``, or ``config`` is not a dataclass type (or not
            the class's declared ``config_class``).
        """
        self._validate_registration(name, cls, _lazy_protocol("ChunkerProtocol"))
        if not _accepts_extra_metadata(cls):
            raise TypeError(
                f"ComponentRegistry: {cls.__name__}.chunk must accept "
                "chunk(text, extra_metadata=...) (ChunkerProtocol); the pipeline "
                "calls it that way."
            )
        declared = getattr(cls, "config_class", None)
        if config is None:
            config = declared
        if config is not None and not (
            isinstance(config, type) and dataclasses.is_dataclass(config)
        ):
            raise TypeError(
                f"ComponentRegistry: config for chunker {name!r} must be a "
                f"dataclass type, got {config!r}."
            )
        if (
            config is not None
            and isinstance(declared, type)
            and not issubclass(config, declared)
        ):
            raise TypeError(
                f"ComponentRegistry: {cls.__name__} takes a "
                f"{declared.__name__}, not {config.__name__} (chunker {name!r})."
            )
        with self._lock:
            self._check_overwrite("chunker", name, self._chunkers, cls, replace)
            self._chunkers[name] = cls
            if config is None:
                self._chunker_configs.pop(name, None)
            else:
                self._chunker_configs[name] = config

    def get_chunker_config(self, name: str) -> type | None:
        """
        Return the configuration dataclass registered for chunker ``name``.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        type or None
            The config class, or ``None`` when the chunker takes keyword
            arguments directly.

        Raises
        ------
        KeyError
            If no chunker is registered under ``name``.
        """
        self.get_chunker(name)
        return self._chunker_configs.get(name)

    def get_chunker(self, name: str) -> type:
        """
        Return the chunker class registered under ``name``.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        type
            The registered chunker class.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        cls = self._chunkers.get(name)
        if cls is None:
            raise KeyError(
                f"ComponentRegistry: no chunker registered for {name!r}. "
                f"Available: {sorted(self._chunkers)}."
            )
        return cls

    def list_chunkers(self) -> list[str]:
        """Return sorted list of registered chunker names."""
        return sorted(self._chunkers)

    # ------------------------------------------------------------------
    # Filter registration / retrieval
    # ------------------------------------------------------------------

    def register_filter(  # noqa: D417
        self,
        name: str,
        cls: type,
        *,
        replace: bool = False,
    ) -> None:
        """
        Register a filter class under ``name``.

        Parameters
        ----------
        name : str
        cls : type
            Concrete class inheriting from
            :class:`~scikitplot.corpus._base.FilterBase`.
        """
        self._validate_registration(name, cls, None)
        with self._lock:
            self._check_overwrite("filter", name, self._filters, cls, replace)
            self._filters[name] = cls

    def get_filter(self, name: str) -> type:  # noqa: D417
        """
        Return the filter class registered under ``name``.

        Parameters
        ----------
        name : str

        Returns
        -------
        type

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        cls = self._filters.get(name)
        if cls is None:
            raise KeyError(
                f"ComponentRegistry: no filter registered for {name!r}. "
                f"Available: {sorted(self._filters)}."
            )
        return cls

    def list_filters(self) -> list[str]:
        """Return sorted list of registered filter names."""
        return sorted(self._filters)

    # ------------------------------------------------------------------
    # Reader registration / retrieval
    # ------------------------------------------------------------------

    def register_reader(
        self,
        name: str,
        cls: type,
        *,
        replace: bool = False,
    ) -> None:
        r"""
        Register a reader class under ``name`` (typically a file extension).

        Parameters
        ----------
        name : str
            File extension (e.g. ``\".txt\"``) or URL scheme key
            (e.g. ``\":url\"``).
        cls : type
            Concrete class inheriting from
            :class:`~scikitplot.corpus._base.DocumentReader`.
        replace : bool
            False
        """
        self._validate_registration(name, cls, None)
        with self._lock:
            self._check_overwrite("reader", name, self._readers, cls, replace)
            self._readers[name] = cls

    def get_reader(self, name: str) -> type:  # noqa: D417
        """
        Return the reader class registered under ``name``.

        Parameters
        ----------
        name : str

        Returns
        -------
        type

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        cls = self._readers.get(name)
        if cls is None:
            raise KeyError(
                f"ComponentRegistry: no reader registered for {name!r}. "
                f"Available: {sorted(self._readers)}."
            )
        return cls

    def list_readers(self) -> list[str]:
        """Return sorted list of registered reader names / extensions."""
        return sorted(self._readers)

    # ------------------------------------------------------------------
    # Normalizer registration / retrieval
    # ------------------------------------------------------------------

    def register_normalizer(
        self,
        name: str,
        cls: type,
        *,
        replace: bool = False,
    ) -> None:  # noqa: D417
        """
        Register a normalizer class under ``name``.

        Parameters
        ----------
        name : str
            a normalizer class.
        cls : type
            Concrete class inheriting from
            :class:`~scikitplot.corpus._normalizers.NormalizerBase`.
        replace : bool
            False

        Notes
        -----
        **Developer.** Checked against ``NormalizerBase``, the type
        ``NormalizationPipeline`` accepts. ``NormalizerProtocol``
        (``normalize(text)``) describes a text-level interface no built-in
        normalizer has, so the old check refused every built-in, and with it
        every snapshot round trip (R3-1).
        """
        self._validate_registration(name, cls, _normalizer_base())
        with self._lock:
            self._check_overwrite("normalizer", name, self._normalizers, cls, replace)
            self._normalizers[name] = cls

    def get_normalizer(self, name: str) -> type:  # noqa: D417
        """
        Return the normalizer class registered under ``name``.

        Parameters
        ----------
        name : str

        Returns
        -------
        type

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        cls = self._normalizers.get(name)
        if cls is None:
            raise KeyError(
                f"ComponentRegistry: no normalizer registered for {name!r}. "
                f"Available: {sorted(self._normalizers)}."
            )
        return cls

    def list_normalizers(self) -> list[str]:
        """Return sorted list of registered normalizer names."""
        return sorted(self._normalizers)

    # ------------------------------------------------------------------
    # Convenience: build an instance from registry
    # ------------------------------------------------------------------

    def build_chunker(self, name: str, **kwargs: Any) -> Any:
        """
        Instantiate the chunker registered under ``name``.

        Parameters
        ----------
        name : str
            Registry key.
        **kwargs
            For a chunker registered with a config class: the config's fields,
            plus any other keyword option the chunker's constructor takes
            (``gensim_dictionary`` for ``word``, ``adapter`` for
            ``semantic``); a name that is both goes to the config. Or
            ``config=`` (a ready config, ``None`` for the default) with only
            constructor options beside it. Without a config class: the
            constructor's keyword arguments.

        Returns
        -------
        object
            The chunker instance.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        TypeError
            If a keyword is neither a config field nor a constructor option
            (the message lists both), a field is given beside ``config=``, or
            ``config=`` is not that config type.

        Notes
        -----
        **Developer.** Raw keyword arguments used to reach chunkers whose
        constructor takes only ``config=``, so every named chunker raised
        ``TypeError`` for any option (R3-1).

        Examples
        --------
        >>> chunker = registry.build_chunker(
        ...     "fixed_window", window_size=50, step_size=25
        ... )
        >>> chunker.config.window_size, chunker.config.step_size
        (50, 25)
        """
        cls = self.get_chunker(name)
        config_cls = self._chunker_configs.get(name)
        if config_cls is None:
            return cls(**kwargs)
        options = dict(kwargs)
        constructor = _constructor_options(cls)
        fields = sorted(f.name for f in dataclasses.fields(config_cls) if f.init)
        if "config" in options:
            config = options.pop("config")
            if config is None:
                config = config_cls()
            elif not isinstance(config, config_cls):
                raise TypeError(
                    f"chunker {name!r}: config must be a {config_cls.__name__}, "
                    f"got {type(config).__name__}."
                )
            stray = sorted(
                key
                for key in options
                if constructor is not None and key not in constructor
            )
            if stray:
                raise TypeError(
                    f"chunker {name!r}: with config=, only the constructor's own "
                    f"options {sorted(constructor or ())} may be given; got {stray} "
                    f"(set {config_cls.__name__} fields on the config itself)."
                )
            return cls(config=config, **options)
        field_options = {k: v for k, v in options.items() if k in fields}
        rest = {k: v for k, v in options.items() if k not in fields}
        unknown = sorted(
            key for key in rest if constructor is not None and key not in constructor
        )
        if unknown:
            raise TypeError(
                f"chunker {name!r}: unknown option(s) {unknown}; "
                f"{config_cls.__name__} accepts {fields}"
                + (
                    f" and {cls.__name__} also takes {sorted(constructor)}."
                    if constructor
                    else "."
                )
            )
        return cls(config=config_cls(**field_options), **rest)

    def build_filter(self, name: str, **kwargs: Any) -> Any:  # noqa: D417
        """
        Instantiate the filter registered under ``name``.

        Parameters
        ----------
        name : str
        **kwargs

        Returns
        -------
        FilterBase instance
        """
        return self.get_filter(name)(**kwargs)

    def build_normalizer(self, name: str, **kwargs: Any) -> Any:  # noqa: D417
        """
        Instantiate the normalizer registered under ``name``.

        Parameters
        ----------
        name : str
        **kwargs

        Returns
        -------
        NormalizerBase instance
        """
        return self.get_normalizer(name)(**kwargs)

    # ------------------------------------------------------------------
    # Snapshot / inspection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, str]]:
        """
        Return a JSON-safe snapshot of all registered components.

        Returns
        -------
        dict[str, dict[str, str]]
            Keys: ``"chunkers"``, ``"chunker_configs"``, ``"filters"``,
            ``"readers"``, ``"normalizers"``. Values: dicts mapping name to
            the fully qualified class name.
        """
        with self._lock:
            # class = fqcn.split('.')[-1]
            # module = fqcn.rsplit('.', 1)[0]
            # qualname = fqcn[len(module)+1:]
            return {
                "chunkers": {k: _fqcn(v) for k, v in self._chunkers.items()},
                "chunker_configs": {
                    k: _fqcn(v) for k, v in self._chunker_configs.items()
                },
                "filters": {k: _fqcn(v) for k, v in self._filters.items()},
                "readers": {k: _fqcn(v) for k, v in self._readers.items()},
                "normalizers": {k: _fqcn(v) for k, v in self._normalizers.items()},
            }

    @classmethod
    def load_from_snapshot(  # noqa: PLR0912
        cls,
        snapshot: dict[str, dict[str, str]],
        *,
        allowed_module_prefixes: list[str] | None = "scikitplot.",
    ) -> ComponentRegistry:
        """
        Reconstruct a registry from a snapshot.

        Parameters
        ----------
        snapshot : dict
            Snapshot created by `snapshot()`.
        allowed_module_prefixes : str | list[str] | None, default="scikitplot."
            If provided, only classes whose module starts with one of these
            prefixes are allowed. Recommended for security.

            .. caution::
                * ⚠: Loading arbitrary FQCN from untrusted JSON is remote code
                  execution risk.

        Returns
        -------
        ComponentRegistry
            New registry populated from snapshot.

        Raises
        ------
        ValueError
            If snapshot structure is invalid.
        TypeError
            If resolved class does not match expected base type.
        """
        required_keys = {"chunkers", "filters", "readers", "normalizers"}
        # Optional since R3-1: snapshots written before have no config section.
        optional_keys = {"chunker_configs"}

        if isinstance(allowed_module_prefixes, str):
            allowed_module_prefixes = [allowed_module_prefixes]

        if not isinstance(snapshot, dict):
            raise TypeError("Snapshot must be a dict.")

        keys = set(snapshot.keys())
        if not required_keys <= keys or keys - required_keys - optional_keys:
            raise ValueError(
                f"Snapshot keys must be {sorted(required_keys)} plus optionally "
                f"{sorted(optional_keys)}, got {sorted(keys)}."
            )

        for section in sorted(keys):
            if not isinstance(snapshot[section], dict):
                raise TypeError(f"Snapshot[{section!r}] must be a dict.")

            for k, v in snapshot[section].items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError(f"Invalid registration name in {section}: {k!r}")
                if not isinstance(v, str) or "." not in v:
                    raise ValueError(f"Invalid FQCN in {section} for {k!r}: {v!r}")

        registry = cls()

        def _safe_load(fqcn: str) -> type:
            cls_obj = _load_class_from_fqcn(fqcn)

            if allowed_module_prefixes is not None:  # noqa: SIM102
                if not any(
                    cls_obj.__module__.startswith(prefix)
                    for prefix in allowed_module_prefixes
                ):
                    raise ValueError(f"Module {cls_obj.__module__} not allowed.")

            return cls_obj

        # Load and validate each component type
        from .._base import DocumentReader, FilterBase  # noqa: PLC0415
        from .._normalizers import NormalizerBase  # noqa: PLC0415

        configs = snapshot.get("chunker_configs", {})
        unknown_configs = sorted(set(configs) - set(snapshot["chunkers"]))
        if unknown_configs:
            raise ValueError(
                f"chunker_configs names chunkers the snapshot does not register: "
                f"{unknown_configs}."
            )
        for name, fqcn in snapshot["chunkers"].items():
            cls_obj = _safe_load(fqcn)
            # Structural, as at registration: the built-in chunkers satisfy
            # ChunkerProtocol without inheriting ChunkerBase, so a nominal
            # check refused every snapshot of the built-ins (R3-1).
            config_cls = _safe_load(configs[name]) if name in configs else None
            registry.register_chunker(name, cls_obj, config=config_cls)

        for name, fqcn in snapshot["filters"].items():
            cls_obj = _safe_load(fqcn)
            if not issubclass(cls_obj, FilterBase):
                raise TypeError(f"{fqcn} is not a FilterBase.")
            registry.register_filter(name, cls_obj)

        for name, fqcn in snapshot["readers"].items():
            cls_obj = _safe_load(fqcn)
            if not issubclass(cls_obj, DocumentReader):
                raise TypeError(f"{fqcn} is not a DocumentReader.")
            registry.register_reader(name, cls_obj)

        for name, fqcn in snapshot["normalizers"].items():
            cls_obj = _safe_load(fqcn)
            if not issubclass(cls_obj, NormalizerBase):
                raise TypeError(f"{fqcn} is not a NormalizerBase.")
            registry.register_normalizer(name, cls_obj)

        return registry

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ComponentRegistry("
            f"chunkers={len(self._chunkers)}, "
            f"filters={len(self._filters)}, "
            f"readers={len(self._readers)}, "
            f"normalizers={len(self._normalizers)})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_registration(
        name: str, cls: type, expected_base: type | None = None
    ) -> None:
        """Validate name, class and protocol conformance before insertion.

        Parameters
        ----------
        name : str
            Registry key. Must be a non-empty, non-whitespace string.
        cls : type
            Candidate class.
        expected_base : type or None, optional
            Runtime-checkable Protocol the candidate must satisfy structurally.
            ``None`` skips the conformance check, which is used only where no
            base is importable without a circular import.

        Raises
        ------
        ValueError
            If ``name`` is empty or whitespace-only.
        TypeError
            If ``cls`` is not a class, or does not satisfy ``expected_base``.

        Notes
        -----
        **Developer.**  The conformance check exists so a non-conforming class
        fails *at the registration site* rather than at an unrelated later call
        site (finding F-R12-01 / P3).  Before this check, ``register_chunker
        ("bogus", object)`` succeeded and the resulting ``AttributeError``
        surfaced during chunking, far from its cause.
        """
        if not name or not name.strip():
            raise ValueError(
                "ComponentRegistry: registration name must be a non-empty string."
            )
        if not isinstance(cls, type):
            raise TypeError(
                f"ComponentRegistry: expected a class (type), "
                f"got {type(cls).__name__} for name {name!r}."
            )
        if expected_base is not None:
            try:
                conforms = issubclass(cls, expected_base)
            except TypeError:  # non-method members make issubclass unusable
                conforms = True
        else:
            conforms = True
        if not conforms:
            raise TypeError(
                f"ComponentRegistry: {cls.__name__!r} registered under {name!r} "
                f"does not satisfy {expected_base.__name__!r}. "
                f"Registering a non-conforming class would defer the failure to "
                f"an unrelated call site."
            )

    def _check_overwrite(
        self, kind: str, name: str, store: dict[str, type], cls: type, replace: bool
    ) -> None:
        """Refuse to silently overwrite an existing registration.

        Raises
        ------
        ValueError
            If ``name`` is already registered and ``replace`` is ``False``.

        Notes
        -----
        **User.**  Pass ``replace=True`` to intentionally substitute a component.

        **Developer.**  Finding F-R12-01 (P2) recorded that a built-in could be
        displaced silently -- ``register_chunker("sentence", Impostor)`` returned
        normally and every later chunk used the impostor.  Silent replacement is
        the "plausible success after altered execution" shape this codebase has
        already been bitten by three times; it is now an error by default.
        """
        if name not in store or replace:
            return
        builtin = name in self._builtin_keys.get(kind, ())
        raise ValueError(
            f"ComponentRegistry: {kind} {name!r} is already registered to "
            f"{store[name].__name__!r}"
            + (" (a built-in)" if builtin else "")
            + f"; refusing to replace it with {cls.__name__!r} silently. "
            f"Pass replace=True to substitute it deliberately."
        )


def _constructor_options(cls: type) -> frozenset[str] | None:
    """Return the keyword options ``cls(...)`` takes besides ``config``.

    ``None`` means the constructor takes ``**kwargs`` (anything goes to it).
    """
    try:
        parameters = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover - builtins, C classes
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return None
    return frozenset(
        p.name
        for p in parameters
        if p.name != "config"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )


def _accepts_extra_metadata(cls: type) -> bool:
    """Whether ``cls.chunk`` binds ``(text, extra_metadata=...)`` (static look-up)."""
    chunk = inspect.getattr_static(cls, "chunk", None)
    if chunk is None:
        return False
    if isinstance(chunk, (staticmethod, classmethod)):
        chunk = chunk.__func__
    try:
        inspect.signature(chunk).bind(None, "text", extra_metadata=None)
    except TypeError:
        return False
    except ValueError:  # pragma: no cover - no signature available
        return True
    return True


def _normalizer_base() -> type | None:
    """Resolve ``NormalizerBase`` lazily (see the header note on imports)."""
    try:
        from .._normalizers import NormalizerBase  # noqa: PLC0415
    except Exception:  # ruff: ignore[blind-except]  # pragma: no cover
        return None
    return NormalizerBase


def _lazy_protocol(name: str) -> type | None:
    """Resolve a runtime-checkable Protocol from ``_types`` lazily.

    Notes
    -----
    **Developer.**  Conformance is checked *structurally* against a
    ``runtime_checkable`` Protocol rather than nominally against a base class.
    A plugin that implements ``chunk``/``chunk_batch`` is a valid chunker
    whether or not it happens to inherit :class:`ChunkerBase`; requiring
    inheritance would reject duck-typed third-party components, which is the
    opposite of what a plugin registry is for.

    ``_registry`` is imported during package initialisation, so ``_types``
    cannot be imported at module scope (see the header note on circular
    imports).  Failure to resolve degrades to *no* check rather than to a
    spurious rejection.
    """
    try:
        from .. import _types  # ruff: ignore[import-outside-top-level]
    # defensive, see module header
    except Exception:  # ruff: ignore[blind-except]  # pragma: no cover
        return None
    obj = getattr(_types, name, None)
    return obj if isinstance(obj, type) else None


# ===========================================================================
# Module-level singleton
# ===========================================================================

#: Global registry singleton. Call :meth:`ComponentRegistry.register_builtins`
#: once to populate built-in components.
registry: ComponentRegistry = ComponentRegistry()
