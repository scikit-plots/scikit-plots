# scikitplot/annoy/_mixins/_meta.py
"""
Metadata helpers and (optional) scikit-learn integration.

This mixin layer provides deterministic export/import of configuration metadata
for Annoy-backed indices. The metadata is designed to be:

- **explicit**: no implicit guessing (missing required fields are errors)
- **stable**: schema-versioned
- **minimal**: focused on configuration and (optionally) diagnostic summary

The concrete wrapper (or its low-level backend) is expected to provide
scikit-learn compatible methods such as ``get_params`` and ``set_params``.

See Also
--------
scikitplot.annoy._mixins._io.IndexIOMixin
scikitplot.annoy._mixins._pickle.PickleMixin
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from typing import Any, TypedDict, cast

from typing_extensions import Self

from .._utils import atomic_write_text, backend_for, lock_for, read_text

__all__ = [
    "IndexMetadata",
    "MetaMixin",
    "MetadataRoutingMixin",
]


def _require_schema_version(obj: object) -> int:
    """Return the class schema version or raise a clear error."""
    ver = getattr(type(obj), "_META_SCHEMA_VERSION", None)
    if ver is None:
        raise RuntimeError(
            "MetaMixin requires the concrete class to define _META_SCHEMA_VERSION"
        )
    return int(ver)


class IndexMetadata(TypedDict, total=True):
    """Serializable metadata for an Annoy-backed index."""

    index_schema_version: int | None
    params: dict[str, Any] | None
    info: dict[str, Any] | None
    persistence: dict[str, Any] | None


class MetaMixin:
    r"""
    Mixin that exports and restores index metadata.

    The concrete class (or its backend) must implement:

    - ``get_params(deep: bool = ...) -> Mapping[str, Any]``
    - ``set_params(**params) -> Self``

    Optional backend methods:

    - ``info() -> Mapping[str, Any] | None``

    The concrete class must define:

    - ``_META_SCHEMA_VERSION`` (int)

    Notes
    -----
    ``from_metadata(..., load=True)`` optionally ``load``\s the on-disk index when
    the ``params`` mapping contains ``on_disk_path``. This is a deterministic
    behavior controlled only by explicit fields.
    """

    _META_SCHEMA_VERSION: int

    def to_metadata(  # noqa: PLR0912
        self, *, include_info: bool = True, strict: bool = True
    ) -> IndexMetadata:
        """
        Export a serializable metadata payload.

        Parameters
        ----------
        include_info
            If True, include an ``info()`` mapping when available.
        strict
            If True, failures in optional ``info()`` propagation raise.

        Returns
        -------
        metadata
            A JSON/YAML-serializable mapping containing configuration parameters
            and optional info.

        Raises
        ------
        RuntimeError
            If ``_META_SCHEMA_VERSION`` is missing on the concrete class.
        TypeError
            If ``get_params`` does not return a mapping.
        AttributeError
            If neither the instance nor the backend implements ``get_params``.
        """
        schema = _require_schema_version(self)

        backend = backend_for(self)

        # Prefer wrapper methods when present; fall back to the backend.
        get_params = getattr(self, "get_params", None)
        get_params_owner: object = self
        if not callable(get_params):
            get_params = getattr(backend, "get_params", None)
            get_params_owner = backend
        if not callable(get_params):
            raise AttributeError("Missing get_params(deep=...) on instance/backend")

        info_fn = getattr(self, "info", None)
        info_owner: object = self
        if not callable(info_fn):
            info_fn = getattr(backend, "info", None)
            info_owner = backend

        # Acquire the instance lock only when calling into the low-level backend.
        params_obj = None
        if get_params_owner is backend:
            with lock_for(self):
                params_obj = get_params(deep=False)
        else:
            params_obj = get_params(deep=False)

        if not isinstance(params_obj, Mapping):
            raise TypeError("get_params(deep=False) must return a mapping")
        params = dict(params_obj)

        info_payload: dict[str, Any] | None = None
        if include_info and callable(info_fn):
            try:
                if info_owner is backend:
                    with lock_for(self):
                        info_obj = info_fn()
                else:
                    info_obj = info_fn()
            except Exception as e:
                if strict:
                    raise RuntimeError("info() failed while exporting metadata") from e
                info_obj = None

            if info_obj is not None:
                if not isinstance(info_obj, Mapping):
                    if strict:
                        raise TypeError("info() must return a mapping")
                else:
                    info_payload = dict(info_obj)

        persistence: dict[str, Any] = {}
        for key in ("pickle_mode", "compress_mode"):
            with contextlib.suppress(Exception):
                value = getattr(self, key)
                if value is not None:
                    persistence[key] = value

        return {
            "index_schema_version": int(schema),
            "params": params,
            "info": info_payload,
            "persistence": persistence or None,
        }

    def to_json(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        indent: int = 2,
        sort_keys: bool = True,
        ensure_ascii: bool = False,
        include_info: bool = True,
        strict: bool = True,
    ) -> str:
        """
        Serialize :meth:`to_metadata` to JSON.

        Parameters
        ----------
        path
            If provided, write the JSON to this path atomically.
        indent
            Indentation level passed to :func:`json.dumps`.
        sort_keys
            If True, sort keys for stable output.
        ensure_ascii
            If True, escape non-ASCII characters.
        include_info, strict
            Forwarded to :meth:`to_metadata`.

        Returns
        -------
        json_str
            JSON representation of the metadata.

        See Also
        --------
        from_json
        to_metadata
        """
        metadata = self.to_metadata(include_info=include_info, strict=strict)
        s = json.dumps(
            metadata, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii
        )
        if path is not None:
            atomic_write_text(path, s)
        return s

    @classmethod
    def from_metadata(  # noqa: PLR0912
        cls: type[Self],
        metadata: Mapping[str, Any],
        *,
        load: bool = True,
    ) -> Self:
        """
        Construct an index from a metadata payload.

        Parameters
        ----------
        metadata : Mapping[str, Any]
            Payload as produced by :meth:`to_metadata`.
        load : bool, default=True
            If True and ``params['on_disk_path']`` is present, attempt to load the
            index into the returned object via backend ``load``.

        Returns
        -------
        index : Self
            Newly constructed index.

        Raises
        ------
        TypeError
            If input types are invalid.
        ValueError
            If required fields are missing or invalid.
        AttributeError
            If backend ``set_params``/``load`` are missing when required.
        """
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        schema_obj = metadata.get("index_schema_version", None)

        expected_schema = getattr(cls, "_META_SCHEMA_VERSION", None)
        if expected_schema is None:
            raise RuntimeError("MetaMixin requires cls._META_SCHEMA_VERSION")
        expected = int(expected_schema)
        if schema_obj is not None and int(schema_obj) != expected:
            raise ValueError(
                f"Unsupported index_schema_version {int(schema_obj)}; expected {expected}."
            )

        params_obj = metadata.get("params", None)
        if not isinstance(params_obj, Mapping):
            raise TypeError("metadata['params'] must be a mapping")
        params = dict(params_obj)

        if "f" not in params or "metric" not in params:
            raise ValueError("metadata params must contain 'f' and 'metric'")

        f = int(params["f"])
        metric = params["metric"]
        if not isinstance(metric, str) or not metric:
            raise ValueError("metadata['params']['metric'] must be a non-empty string")

        obj = cls(f, metric)

        # Apply remaining params via strict backend API.
        rest = {k: v for k, v in params.items() if k not in {"f", "metric"}}
        if rest:
            backend = backend_for(obj)
            set_params = getattr(obj, "set_params", None)
            set_params_owner: object = obj
            if not callable(set_params):
                set_params = getattr(backend, "set_params", None)
                set_params_owner = backend
            if not callable(set_params):
                raise AttributeError("Missing set_params(**params) on instance/backend")

            if set_params_owner is backend:
                with lock_for(obj):
                    set_params(**rest)
            else:
                set_params(**rest)

        # Apply optional persistence knobs when supported by the instance.
        persistence = metadata.get("persistence", None)
        if isinstance(persistence, Mapping):
            for key in ("pickle_mode", "compress_mode"):
                if hasattr(obj, key) and key in persistence:
                    with contextlib.suppress(Exception):
                        setattr(obj, key, persistence[key])

        # Optionally load the on-disk index if requested and available.
        on_disk_path = params.get("on_disk_path", None)  # noqa: SIM910
        if load and on_disk_path is not None:
            # Prefer the explicit IO mixin helper when present; otherwise fall back
            # to the backend load. This keeps mixins independent and avoids
            # reimplementing persistence logic.
            load_index = getattr(obj, "load_index", None)
            if callable(load_index):
                prefault = params.get("prefault", None)  # noqa: SIM910
                load_index(os.fspath(on_disk_path), prefault=prefault)
            else:
                backend = backend_for(obj)
                load_fn = getattr(backend, "load", None)
                if callable(load_fn):
                    prefault = params.get("prefault", None)  # noqa: SIM910
                    if prefault is None:
                        with lock_for(obj):
                            load_fn(os.fspath(on_disk_path))
                    else:
                        with lock_for(obj):
                            load_fn(os.fspath(on_disk_path), prefault=bool(prefault))

        return cast(Self, obj)

    @classmethod
    def from_json(
        cls: type[Self], path: str | os.PathLike[str], *, load: bool = True
    ) -> Self:
        """Load metadata from JSON and construct an index."""
        obj = json.loads(read_text(path))
        if not isinstance(obj, Mapping):
            raise TypeError("JSON root must be a mapping")
        return cls.from_metadata(cast(Mapping[str, Any], obj), load=load)

    def to_yaml(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        include_info: bool = True,
        strict: bool = True,
    ) -> str:
        """Serialize :meth:`to_metadata` to YAML (requires PyYAML)."""
        try:
            import yaml  # type: ignore[import-not-found]  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for YAML export. Install `pyyaml`."
            ) from e

        metadata = self.to_metadata(include_info=include_info, strict=strict)
        s = yaml.safe_dump(metadata, sort_keys=True)
        if path is not None:
            atomic_write_text(path, s)
        return s

    @classmethod
    def from_yaml(
        cls: type[Self], path: str | os.PathLike[str], *, load: bool = True
    ) -> Self:
        """Load metadata from YAML and construct an index (requires PyYAML)."""
        try:
            import yaml  # type: ignore[import-not-found]  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for YAML import. Install `pyyaml`."
            ) from e

        obj = yaml.safe_load(read_text(path))
        if not isinstance(obj, Mapping):
            raise TypeError("YAML root must be a mapping")
        return cls.from_metadata(cast(Mapping[str, Any], obj), load=load)


class MetadataRoutingMixin:
    """
    Optional scikit-learn metadata routing integration.

    This mixin is only needed if you integrate the index into scikit-learn
    pipelines that make use of *metadata routing*.

    Notes
    -----
    scikit-learn is imported lazily and only when these methods are called.
    """

    def get_metadata_routing(self) -> Any:
        """Return a scikit-learn ``MetadataRequest`` describing consumed metadata."""
        try:
            from sklearn.utils.metadata_routing import MetadataRequest  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for metadata routing. Install `scikit-learn`."
            ) from e

        # Minimal explicit metadata request: accept no routed metadata by default.
        return MetadataRequest(owner=type(self))

    @staticmethod
    def build_metadata_router(*, owner: object) -> Any:
        """Build a scikit-learn ``MetadataRouter`` for a given owner."""
        try:
            from sklearn.utils.metadata_routing import MetadataRouter  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for metadata routing. Install `scikit-learn`."
            ) from e

        try:
            return MetadataRouter(owner=owner)
        except TypeError:  # pragma: no cover
            return MetadataRouter(
                owner=getattr(owner, "__class__", type(owner)).__name__
            )
