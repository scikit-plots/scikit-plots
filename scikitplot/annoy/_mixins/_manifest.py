# scikitplot/annoy/_mixins/_manifest.py
"""
Manifest (metadata) import/export helpers.

This module defines :class:`~.ManifestMixin`, a small mixin that exports and
restores a *manifest* describing an Annoy-style index.

The Annoy C-extension already exposes a structured :meth:`info` summary with a
stable key order and no side effects. The high-level API needs one additional
layer on top:

- **A manifest *format* version**, so we can evolve the manifest schema without
  colliding with Annoy's own ``schema_version`` attribute (which is a stored
  marker on the low-level object).
- Optional high-level persistence knobs that are not part of Annoy itself
  (e.g., pickle/compress preferences).
- **Small**: metadata only (never serializes the full forest).
- **Stable**: the *manifest schema* is versioned for forward-compatible changes.
- **Deterministic**: exporting a manifest has no side effects and yields stable
  JSON/YAML for the same inputs.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Mapping, MutableMapping, TypedDict, cast

from typing_extensions import Self

# NOTE:
# - Annoy's low-level object exposes `schema_version` (a stored marker).
# - This mixin needs a *separate* key for the manifest format version.
_MANIFEST_SCHEMA_VERSION_KEY = "manifest_schema_version"

__all__: tuple[str, ...] = ("Manifest", "ManifestMixin")


class Manifest(TypedDict, total=False):
    """
    Typed representation of a manifest payload.

    The schema is intentionally permissive (``total=False``) to allow forward
    additions while keeping backward compatibility.

    Required keys for reconstruction (by convention):

    - ``f`` (int)

    and treat other keys as optional.

    Notes
    -----
    The key ``schema_version`` in this manifest refers to **Annoy's stored marker**
    (as reported by :meth:`~scikitplot.cexternals._annoy.annoylib.Annoy.info`).
    The manifest *format* version is stored separately under
    ``manifest_schema_version``.
    """

    # Manifest format version (high-level; this mixin).
    manifest_schema_version: int

    # Core Annoy configuration (low-level).
    f: int | None
    metric: str | None
    on_disk_path: str | None
    prefault: bool | None
    schema_version: int | None
    seed: int | None
    verbose: int | None

    # Optional keys (from Annoy.info include_* flags)
    n_items: int
    n_trees: int
    memory_usage_byte: int
    memory_usage_mib: float

    # Optional high-level persistence knobs (if exposed by the concrete class).
    pickle_mode: str | None
    compress_mode: str | None


class ManifestMixin:
    """
    Export/import a small *manifest* describing an index.

    This mixin is designed to layer cleanly on top of the low-level Annoy wrapper.

    Notes
    -----
    - :meth:`to_manifest` never serializes the full in-memory forest; it exports
      metadata only.
    - :meth:`from_manifest` can *optionally* load (mmap) an index from
      ``on_disk_path`` when ``load=True`` and the concrete class provides
      :meth:`load`.

    See Also
    --------
    scikitplot.cexternals._annoy.annoylib.Annoy.info
        Structured, JSON-like summary used as the primary source of truth.
    """

    # Manifest format version for forward compatibility (high-level).
    _MANIFEST_FORMAT_VERSION: int = 0

    # This mixin supports both inheritance-style (Index subclasses Annoy)
    # and composition-style (Index wraps a low-level Annoy instance).
    #
    # Concrete classes may optionally provide ``self._annoy``; if absent,
    # the low-level object is assumed to be ``self``.

    def _low_level(self) -> Any:
        """
        Return the low-level Annoy object.

        Preference order is explicit and deterministic:

        1) ``object._annoy`` when present (composition style)
        2) ``self`` (inheritance style)

        Notes
        -----
        This helper uses :func:`object.__getattribute__` to avoid triggering custom
        ``__getattr__`` / ``__getattribute__`` side effects during low-level access.
        """
        # ll = getattr(self, "_annoy", None)
        # return ll if ll is not None else obj
        try:
            return object.__getattribute__(self, "_annoy")
        except AttributeError:
            return self

    def _manifest_source_info(self) -> Mapping[str, Any] | None:
        """
        Return the low-level :meth:`info` mapping if available.

        Notes
        -----
        The lookup is performed on :meth:`_low_level` to support both inheritance
        and composition styles.
        """
        # Prefer the C-extension's structured info() if present.
        # if hasattr(self, "info") and callable(getattr(self, "info", None)):
        #     info = getattr(self, "info", None)()
        #     if isinstance(info, Mapping):
        #         payload.update(dict(info))  # type: ignore[arg-type]
        ll = self._low_level()
        info = getattr(ll, "info", None)
        if callable(info):
            out = info()
            if isinstance(out, Mapping):
                return out
        return None

    def to_manifest(self) -> Manifest:  # noqa: PLR0912
        """
        Return a JSON-serializable manifest describing this index.

        Returns
        -------
        manifest : Manifest
            JSON-serializable dictionary describing configuration and optional
            persistence knobs.

        Raises
        ------
        ValueError
            If the index dimension (``f``) cannot be determined.

        Notes
        -----
        The manifest includes:

        - **Annoy configuration**: ``f``, ``metric``, ``on_disk_path``, ``prefault``,
          ``schema_version``, ``seed``, ``verbose``.
        - **Manifest format**: ``manifest_schema_version`` (this mixin).
        - Optional summary fields from :meth:`info` (e.g., ``n_items``, ``n_trees``,
          memory usage), plus optional high-level knobs if exposed by the concrete
          class.

        The manifest does **not** include the full index data.
        """
        payload: MutableMapping[str, Any] = {}

        ll = self._low_level()

        # 1) Prefer the C-extension's structured info() (stable order, no effects).
        info = self._manifest_source_info()
        if info is not None:
            payload.update(dict(info))

        # 2) Fill core fields from public attributes if missing.
        if payload.get("f") is None and hasattr(ll, "f"):
            f_val = getattr(ll, "f", None)
            if f_val is not None:
                payload["f"] = int(f_val)

        if "metric" not in payload and hasattr(ll, "metric"):
            metric_val = getattr(ll, "metric", None)
            payload["metric"] = None if metric_val is None else str(metric_val)

        if "on_disk_path" not in payload:
            if hasattr(ll, "on_disk_path"):
                payload["on_disk_path"] = getattr(ll, "on_disk_path", None)
            else:
                payload["on_disk_path"] = getattr(ll, "_on_disk_path", None)

        if "prefault" not in payload:
            if hasattr(ll, "prefault"):
                v = getattr(ll, "prefault", None)
                if v is not None and not callable(v):
                    payload["prefault"] = bool(v)
            elif hasattr(ll, "_prefault"):
                payload["prefault"] = bool(getattr(ll, "_prefault", False))

        if "schema_version" not in payload and hasattr(ll, "schema_version"):
            v = getattr(ll, "schema_version", None)
            if v is not None and not callable(v):
                payload["schema_version"] = int(v)

        if "seed" not in payload and hasattr(ll, "seed"):
            v = getattr(ll, "seed", None)
            if v is not None and not callable(v):
                payload["seed"] = None if v is None else int(v)

        if "verbose" not in payload and hasattr(ll, "verbose"):
            v = getattr(ll, "verbose", None)
            # If .verbose is a setter method, do not call it here.
            if v is not None and not callable(v):
                payload["verbose"] = None if v is None else int(v)

        # 3) Optional high-level persistence knobs (only if exposed).
        if hasattr(self, "pickle_mode"):
            payload["pickle_mode"] = getattr(self, "pickle_mode", None)
        elif hasattr(self, "_pickle_mode"):
            payload["pickle_mode"] = getattr(self, "_pickle_mode", None)

        if hasattr(self, "compress_mode"):
            payload["compress_mode"] = getattr(self, "compress_mode", None)
        elif hasattr(self, "_compress_mode"):
            payload["compress_mode"] = getattr(self, "_compress_mode", None)

        # 4) Add manifest format version under a dedicated key (no collision).
        payload[_MANIFEST_SCHEMA_VERSION_KEY] = int(self._MANIFEST_FORMAT_VERSION)

        # 5) A manifest must always include `f` to allow reconstruction.
        if payload.get("f") is None:
            raise ValueError("Cannot export manifest: index dimension `f` is unknown.")

        return cast(Manifest, dict(payload))

    def to_json(self, path: str | os.PathLike[str] | None = None) -> str:
        """
        Serialize the manifest to JSON.

        Parameters
        ----------
        path : str or os.PathLike or None, default=None
            If provided, write the JSON payload to this path.

        Returns
        -------
        json_str : str
            JSON representation of the manifest.
        """
        payload = json.dumps(self.to_manifest(), ensure_ascii=False, indent=2)
        if path is not None:
            with open(os.fspath(path), "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    @classmethod
    def _constructor_accepts_kwargs(cls, **kwargs: Any) -> dict[str, Any]:
        """
        Filter kwargs to those accepted by calling ``cls(...)``.

        This is deterministic (signature-based) and avoids passing unknown keys
        to custom wrappers.

        Notes
        -----
        For some C-extension types, :func:`inspect.signature` may fail; in that case
        this function returns an empty dict and configuration is applied after
        construction via attribute setters / methods.
        """
        try:
            sig = inspect.signature(cls)  # type: ignore[misc]
        except Exception:
            return {}

        params = sig.parameters
        out: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in params:
                out[k] = v
        return out

    @classmethod
    def from_manifest(  # noqa: PLR0912
        cls: type[Self],
        manifest: Mapping[str, Any],
        *,
        load: bool = True,
    ) -> Self:
        """
        Construct an instance from a manifest.

        Parameters
        ----------
        manifest : mapping
            A manifest previously produced by :meth:`to_manifest`.
        load : bool, default=True
            If True and the manifest contains ``on_disk_path``, attempt to call
            :meth:`load` on the created object (when available).

        Returns
        -------
        obj
            A new instance of ``cls`` configured according to the manifest.

        Raises
        ------
        TypeError
            If ``manifest`` is not a mapping.
        ValueError
            If required keys are missing or the manifest format version is unsupported.

        Notes
        -----
        The manifest format version is stored under ``manifest_schema_version``.
        Older manifests that lack this key are treated as version 0.
        """
        if not isinstance(manifest, Mapping):
            raise TypeError("`manifest` must be a mapping.")

        # Manifest *format* version gate (separate from Annoy's schema_version).
        fmt_version = manifest.get(_MANIFEST_SCHEMA_VERSION_KEY, 0)
        try:
            fmt_version_i = int(fmt_version)
        except Exception as e:
            raise ValueError(
                f"Invalid manifest_schema_version={fmt_version!r}; expected an integer."
            ) from e

        if fmt_version_i > cls._MANIFEST_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported manifest_schema_version={fmt_version_i!r}. "
                f"This version supports up to {cls._MANIFEST_FORMAT_VERSION!r}."
            )

        if "f" not in manifest:
            raise ValueError("Manifest is missing required key 'f'.")

        f = int(manifest["f"])
        metric = manifest.get("metric")
        metric_str = None if metric is None else str(metric)

        # Constructor kwargs if supported by the concrete class.
        ctor_kwargs: dict[str, Any] = {}

        path = manifest.get("on_disk_path")
        on_disk_path = None if path in (None, "") else str(path)
        if on_disk_path is not None:
            ctor_kwargs["on_disk_path"] = on_disk_path

        if "prefault" in manifest and manifest["prefault"] is not None:
            ctor_kwargs["prefault"] = bool(manifest["prefault"])
        if "schema_version" in manifest and manifest["schema_version"] is not None:
            ctor_kwargs["schema_version"] = int(manifest["schema_version"])
        if "seed" in manifest:
            seed = manifest.get("seed")
            ctor_kwargs["seed"] = None if seed is None else int(seed)
        if "verbose" in manifest:
            vb = manifest.get("verbose")
            ctor_kwargs["verbose"] = None if vb is None else int(vb)

        accepted = cls._constructor_accepts_kwargs(**ctor_kwargs)

        # Annoy-style constructors accept (f, metric, *, ...). If metric is None,
        # we still pass it positionally to preserve explicitness.
        obj = cls(f, metric_str, **accepted)

        if (
            on_disk_path is not None
            and hasattr(obj, "on_disk_path")
            and "on_disk_path" not in accepted
        ):
            obj.on_disk_path = on_disk_path

        # Apply configuration that wasn't accepted by the constructor.
        if (
            "prefault" in ctor_kwargs
            and hasattr(obj, "prefault")
            and "prefault" not in accepted
        ):
            obj.prefault = bool(ctor_kwargs["prefault"])

        if (
            "schema_version" in ctor_kwargs
            and hasattr(obj, "schema_version")
            and "schema_version" not in accepted
        ):
            obj.schema_version = int(ctor_kwargs["schema_version"])

        # seed / verbose: prefer methods when present (Annoy uses methods).
        if (
            "seed" in ctor_kwargs
            and ctor_kwargs["seed"] is not None
            and "seed" not in accepted
        ):
            set_seed = getattr(obj, "set_seed", None)
            if callable(set_seed):
                set_seed(int(ctor_kwargs["seed"]))
            elif hasattr(obj, "seed"):
                obj.seed = int(ctor_kwargs["seed"])

        if (
            "verbose" in ctor_kwargs
            and ctor_kwargs["verbose"] is not None
            and "verbose" not in accepted
        ):
            set_verbose = getattr(obj, "verbose", None)
            if callable(set_verbose):
                set_verbose(int(ctor_kwargs["verbose"]))
            elif hasattr(obj, "verbose"):
                obj.verbose = int(ctor_kwargs["verbose"])

        # Restore optional high-level knobs if they exist on the concrete class.
        if "pickle_mode" in manifest and hasattr(obj, "pickle_mode"):
            pm = manifest.get("pickle_mode")
            obj.pickle_mode = None if pm is None else str(pm)

        if "compress_mode" in manifest and hasattr(obj, "compress_mode"):
            cm = manifest.get("compress_mode")
            obj.compress_mode = None if cm is None else str(cm)

        # Optionally mmap-load the index.
        if load and on_disk_path is not None and hasattr(obj, "load"):
            load_fn = getattr(obj, "load", None)
            if callable(load_fn):
                # Pass `prefault` only if explicitly present in the manifest; otherwise
                # leave it to the implementation's default.
                if "prefault" in manifest and manifest["prefault"] is not None:
                    load_fn(on_disk_path, prefault=bool(manifest["prefault"]))
                else:
                    load_fn(on_disk_path)

        return obj

    @classmethod
    def from_json(
        cls: type[Self], path: str | os.PathLike[str], *, load: bool = True
    ) -> Self:
        """Load a manifest from a JSON file and construct an instance."""
        with open(os.fspath(path), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, Mapping):
            raise ValueError("JSON manifest must be a mapping/dict.")
        return cls.from_manifest(manifest, load=load)

    def to_yaml(self, path: str | os.PathLike[str]) -> None:
        """
        Write the manifest as YAML.

        Parameters
        ----------
        path : str or os.PathLike
            Output YAML file path.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore[import-not-found]  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for `to_yaml`. Install `pyyaml`."
            ) from e

        with open(os.fspath(path), "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_manifest(), f, sort_keys=False)

    @classmethod
    def from_yaml(
        cls: type[Self], path: str | os.PathLike[str], *, load: bool = True
    ) -> Self:
        """
        Load a manifest from a YAML file and construct an instance.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore[import-not-found]  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for `from_yaml`. Install `pyyaml`."
            ) from e

        with open(os.fspath(path), "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        if not isinstance(manifest, Mapping):
            raise ValueError("YAML manifest must be a mapping/dict.")

        return cls.from_manifest(manifest, load=load)
