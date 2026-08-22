# scikitplot/corpus/_registry/_registry.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._registry._registry
=========================================
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

import importlib
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
    >>> cls = registry.get_chunker("paragraph")
    >>> chunker = cls(min_chars=20)
    """

    def __init__(self) -> None:
        self._chunkers: dict[str, type] = {}
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
        """
        with self._lock:
            if self._builtins_registered:
                return

            # --- Chunkers ---
            try:
                from .._chunkers import (  # noqa: PLC0415
                    FixedWindowChunker,
                    ParagraphChunker,
                    SentenceChunker,
                )

                self._chunkers["sentence"] = SentenceChunker
                self._chunkers["paragraph"] = ParagraphChunker
                self._chunkers["fixed_window"] = FixedWindowChunker
            except ImportError as exc:
                logger.warning("ComponentRegistry: could not import chunkers: %s.", exc)

            # --- Filters ---
            try:
                from .._base import DefaultFilter  # noqa: PLC0415

                self._filters["default"] = DefaultFilter
            except ImportError as exc:
                logger.warning("ComponentRegistry: could not import filters: %s.", exc)

            # --- Readers (populates DocumentReader._registry as side effect) ---
            try:
                from .. import _readers  # noqa: F401, PLC0415
                from .._base import DocumentReader  # noqa: PLC0415

                for ext, cls in DocumentReader.subclass_by_type().items():
                    self._readers[ext] = cls
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

                self._normalizers["unicode"] = UnicodeNormalizer
                self._normalizers["whitespace"] = WhitespaceNormalizer
                self._normalizers["html_strip"] = HTMLStripNormalizer
                self._normalizers["lowercase"] = LowercaseNormalizer
                self._normalizers["dedup_lines"] = DedupLinesNormalizer
                self._normalizers["language_detect"] = LanguageDetectionNormalizer
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

    # ------------------------------------------------------------------
    # Chunker registration / retrieval
    # ------------------------------------------------------------------

    def register_chunker(
        self,
        name: str,
        cls: type,
        *,
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
            Concrete class inheriting from
            :class:`~scikitplot.corpus._base.ChunkerBase`.
        replace : bool
            False

        Raises
        ------
        ValueError
            If ``name`` is empty.
        TypeError
            If ``cls`` is not a type.
        """
        self._validate_registration(name, cls, _lazy_protocol("ChunkerProtocol"))
        with self._lock:
            self._check_overwrite("chunker", name, self._chunkers, cls, replace)
            self._chunkers[name] = cls

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
        """
        self._validate_registration(name, cls, _lazy_protocol("NormalizerProtocol"))
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
            Constructor keyword arguments.

        Returns
        -------
        ChunkerBase instance

        Raises
        ------
        KeyError
            If ``name`` is not registered.

        Examples
        --------
        >>> chunker = registry.build_chunker("paragraph", min_chars=20)
        """
        return self.get_chunker(name)(**kwargs)

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
            Keys: ``"chunkers"``, ``"filters"``, ``"readers"``,
            ``"normalizers"``. Values: dicts mapping name → class qualname.
        """
        with self._lock:
            # class = fqcn.split('.')[-1]
            # module = fqcn.rsplit('.', 1)[0]
            # qualname = fqcn[len(module)+1:]
            return {
                "chunkers": {k: _fqcn(v) for k, v in self._chunkers.items()},
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

        if isinstance(allowed_module_prefixes, str):
            allowed_module_prefixes = [allowed_module_prefixes]

        if not isinstance(snapshot, dict):
            raise TypeError("Snapshot must be a dict.")

        if set(snapshot.keys()) != required_keys:
            raise ValueError(
                f"Snapshot keys must be exactly {required_keys}, "
                f"got {set(snapshot.keys())}."
            )

        for section in required_keys:
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
        from .._base import ChunkerBase, DocumentReader, FilterBase  # noqa: PLC0415
        from .._normalizers import NormalizerBase  # noqa: PLC0415

        for name, fqcn in snapshot["chunkers"].items():
            cls_obj = _safe_load(fqcn)
            if not issubclass(cls_obj, ChunkerBase):
                raise TypeError(f"{fqcn} is not a ChunkerBase.")
            registry.register_chunker(name, cls_obj)

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
