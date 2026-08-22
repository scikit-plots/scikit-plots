# scikitplot/corpus/_catalog.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""One read-only view over every component registry in Corpus.

Corpus registers components in four different places, in three different shapes:
``ComponentRegistry`` instance methods, module-level singleton registries in
``_chunkers._custom_tokenizer``, a class-keyed bridge registry, and a plain
``_BACKENDS`` dict.  They differ in lookup key, in whether registration is a
function or a dict literal, and in what (if anything) validates a registration.

Notes
-----
**User-focused.**  There is now one place to ask *what components exist*::

    from scikitplot.corpus import component_catalog

    catalog = component_catalog()
    catalog.categories()  # 'chunker', 'reader', 'index', ...
    catalog.list("index")  # every vector-index backend
    catalog.get("index", "annoy").status  # why it is or is not usable

**Developer-focused.**  This is a **read-only aggregating view**.  The four
registries are *not* rewritten: they publish into the catalog and keep their own
shapes (decision DEC-108).  Merging registries with different lookup keys and
error behaviours in one change would be a large blast radius for no gain, and
each has tests depending on its current shape.

Fields the underlying registry cannot supply default to
:attr:`~scikitplot.corpus._capabilities.CapabilityStatus.UNKNOWN` and ``None``
-- **never to a guess**.  Finding F-R13-03 measured that the registry recorded
3 of the 10 properties a useful catalog needs, and that several of the missing
seven already existed in fragmented form: ``is_available()`` is an availability
probe, ``capability_snapshot()`` reports optional dependencies, and the ANN
backends declare their own metric and score semantics.  Nothing collected them.

See Also
--------
scikitplot.corpus._capabilities.capability_snapshot : the lightweight dependency probe.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ._capabilities import CapabilityStatus, probe_backend

__all__: list[str] = [
    "ComponentCatalog",
    "ComponentSpec",
    "component_catalog",
]


@dataclasses.dataclass(frozen=True)
class ComponentSpec:
    """Everything the catalog knows about one component.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"sentence"`` or ``"annoy"``.
    category : str
        Component category, e.g. ``"chunker"``, ``"index"``, ``"reader"``.
    implementation : str
        Fully-qualified class path, for provenance.
    status : CapabilityStatus, optional
        Whether the component is usable. ``UNKNOWN`` when the registry offers no
        probe -- which is honest, and different from claiming availability.
    reason_code : str or None, optional
        Stable machine-readable reason, ``None`` when ``AVAILABLE``.
    aliases : tuple of str, optional
        Alternative names accepted at call sites but not separate entries.
    optional_dependencies : tuple of str, optional
        Distributions this component needs.
    capabilities : dict, optional
        Declared properties, e.g. an index backend's ``metric`` and
        ``score_semantics``.
    config_type : str or None, optional
        Name of the configuration class this component accepts.
    version : str or None, optional
        Version of the providing distribution.

    Notes
    -----
    **Developer.**  ``frozen=True``: a spec describes a registration as observed
    at snapshot time.  Mutating it would let a catalog disagree with the
    registry it was built from.
    """

    name: str
    category: str
    implementation: str
    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    reason_code: str | None = None
    aliases: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    capabilities: dict[str, Any] = dataclasses.field(default_factory=dict)
    config_type: str | None = None
    version: str | None = None

    @property
    def available(self) -> bool:
        """Whether the component is usable now.

        Notes
        -----
        Derived from :attr:`status`, which is the authority.  ``UNKNOWN`` is
        **not** available: an unprobed component must not be reported as usable.
        """
        return self.status is CapabilityStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        data = dataclasses.asdict(self)
        data["status"] = self.status.value
        data["aliases"] = list(self.aliases)
        data["optional_dependencies"] = list(self.optional_dependencies)
        data["available"] = self.available
        return data


class ComponentCatalog:
    """An immutable snapshot of every registered component.

    Parameters
    ----------
    specs : iterable of ComponentSpec
        The collected specs.

    Notes
    -----
    A snapshot, not a live proxy: building it walks the registries once.  Call
    :func:`component_catalog` again after registering something new.
    """

    def __init__(self, specs: Any = ()) -> None:
        self._specs: tuple[ComponentSpec, ...] = tuple(specs)

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._specs)

    def __iter__(self):
        """Iterate every spec."""
        return iter(self._specs)

    def categories(self) -> list[str]:
        """Return the sorted category names present."""
        return sorted({spec.category for spec in self._specs})

    def list(self, category: str | None = None) -> list[ComponentSpec]:
        """Return specs, optionally filtered to one category."""
        specs = [
            spec
            for spec in self._specs
            if category is None or spec.category == category
        ]
        return sorted(specs, key=lambda s: (s.category, s.name))

    def get(self, category: str, name: str) -> ComponentSpec | None:
        """Return one spec, or ``None`` if it is not registered."""
        for spec in self._specs:
            if spec.category == category and (
                spec.name == name or name in spec.aliases
            ):
                return spec
        return None

    def available(self, category: str | None = None) -> list[ComponentSpec]:
        """Return only components that are usable now."""
        return [spec for spec in self.list(category) if spec.available]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping grouped by category."""
        grouped: dict[str, Any] = {}
        for spec in self.list():
            grouped.setdefault(spec.category, []).append(spec.to_dict())
        return grouped


def _fqcn(cls: Any) -> str:
    """Return a fully-qualified class path for provenance."""
    module = getattr(cls, "__module__", "?")
    name = getattr(cls, "__qualname__", getattr(cls, "__name__", "?"))
    return f"{module}.{name}"


def _index_specs() -> list[ComponentSpec]:
    """Collect vector-index backends, which do expose an availability probe."""
    try:
        from ._similarity._backends import (  # noqa: PLC0415
            _BACKENDS,
            backend_aliases,
        )
    except Exception:  # noqa: BLE001 - the catalog must never fail on imports
        return []

    specs = []
    for name, cls in sorted(_BACKENDS.items()):
        status, reason = probe_backend(cls)
        capabilities = {}
        for attr in ("metric", "score_semantics", "supports_persistence"):
            value = getattr(cls, attr, None)
            if value is not None and not callable(value):
                capabilities[attr] = value
        specs.append(
            ComponentSpec(
                name=name,
                category="index",
                implementation=_fqcn(cls),
                status=status,
                reason_code=reason,
                aliases=tuple(backend_aliases(name)),
                optional_dependencies=(name,) if name != "bruteforce" else (),
                capabilities=capabilities,
            )
        )
    return specs


def _registry_specs() -> list[ComponentSpec]:
    """Collect chunkers, filters, readers and normalizers.

    Notes
    -----
    These registries expose no availability probe, so status is ``UNKNOWN``
    rather than an assumed ``AVAILABLE``.  A registered class is *present*,
    which is not the same as *usable* -- it may still need an optional
    dependency at construction time.
    """
    try:
        from ._registry._registry import registry  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return []

    specs = []
    stores = (
        ("chunker", "_chunkers"),
        ("filter", "_filters"),
        ("reader", "_readers"),
        ("normalizer", "_normalizers"),
    )
    for category, attribute in stores:
        for name, cls in sorted(getattr(registry, attribute, {}).items()):
            specs.append(
                ComponentSpec(
                    name=name,
                    category=category,
                    implementation=_fqcn(cls),
                    status=CapabilityStatus.UNKNOWN,
                    reason_code="not_probed",
                )
            )
    return specs


def component_catalog() -> ComponentCatalog:
    """Build a snapshot of every registered component.

    Returns
    -------
    ComponentCatalog

    Notes
    -----
    Never raises.  A registry that cannot be imported contributes nothing rather
    than failing the whole catalog, so a partially-installed environment can
    still ask what it has.

    Examples
    --------
    >>> catalog = component_catalog()
    >>> "index" in catalog.categories()
    True
    """
    return ComponentCatalog([*_index_specs(), *_registry_specs()])
