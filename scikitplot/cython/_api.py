# scikitplot/cython/_api.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
API stability tiers for the ``scikitplot.cython`` facade (CYTHON-API-002).

The package historically re-exported high-level entry points *and*
implementation-level primitives (cache, lock, loader, registry, builder
internals) at the top level with no way to tell which are safe to depend on.
Rather than remove exports (a breaking change that needs a deprecation cycle),
this module attaches a machine-readable **stability tier** to every public
symbol so callers and tools can reason about drift:

- :attr:`Stability.STABLE` — the supported high-level facade; changes follow a
  deprecation process.
- :attr:`Stability.ADVANCED` — power-user / introspection primitives; usable but
  lower-level and more likely to change.
- :attr:`Stability.EXPERIMENTAL` — recently introduced; may change without a
  deprecation cycle.

Use :func:`api_stability` to query a symbol and :func:`list_api` to list a tier.

Notes
-----
This is additive: no symbol is removed here.  It provides the tier *contract*
the review asks for; actually narrowing the surface (moving primitives behind an
``advanced`` namespace) is a follow-up that must go through deprecation.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "API_STABILITY",
    "Stability",
    "api_stability",
    "list_api",
]


class Stability(str, Enum):
    """Stability tier of a public API symbol."""

    STABLE = "stable"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


# --- Tier assignments -------------------------------------------------------
# STABLE: the supported high-level facade most users should use.
_STABLE: frozenset[str] = frozenset(
    {
        # High-level compile/import facade
        "compile_and_load",
        "compile_and_load_result",
        "cython_import",
        "cython_import_all",
        "cython_import_result",
        "build_package_from_code",
        "build_package_from_code_result",
        "build_package_from_paths",
        "build_package_from_paths_result",
        "import_artifact_bytes",
        "import_artifact_path",
        "import_pinned",
        "import_pinned_result",
        "import_cached",
        "import_cached_by_name",
        "import_cached_result",
        "import_cached_package",
        "import_cached_package_result",
        "list_cached",
        "list_cached_packages",
        "get_cache_dir",
        "check_build_prereqs",
        # Pins
        "pin",
        "unpin",
        "list_pins",
        "resolve_pinned_key",
        "PinRegistryError",
        # Result / value types
        "BuildResult",
        "PackageBuildResult",
        "CacheStats",
        "CacheGCResult",
        # Security policy (public, documented contract)
        "SecurityPolicy",
        "SecurityError",
        "DEFAULT_SECURITY_POLICY",
        "RELAXED_SECURITY_POLICY",
        # High-level builder entry points
        "build_extension_module",
        "build_extension_module_result",
        "build_extension_package_from_code_result",
        "build_extension_package_from_paths_result",
    }
)

# EXPERIMENTAL: introduced during the hardening effort; may still change.
_EXPERIMENTAL: frozenset[str] = frozenset(
    {
        "PlatformCapabilities",
        "platform_capabilities",
        "verify_template_assets",
        "CompilerCapabilities",
        "compiler_capabilities",
        "COMPILER_SPEC_VERSION",
    }
)

# Everything else that is public is ADVANCED (power-user / introspection /
# implementation primitives).  Computed against the live package __all__ in
# :func:`_build_registry` so the mapping can never silently drift out of date.


def _build_registry() -> dict[str, Stability]:
    """
    Build the name→tier mapping against the live package ``__all__``.

    Any public symbol not explicitly STABLE or EXPERIMENTAL is ADVANCED.  This
    guarantees total coverage: a newly added export is ADVANCED by default until
    deliberately promoted.
    """
    from . import __all__ as public_names  # noqa: PLC0415

    registry: dict[str, Stability] = {}
    for name in public_names:
        if name in _STABLE:
            registry[name] = Stability.STABLE
        elif name in _EXPERIMENTAL:
            registry[name] = Stability.EXPERIMENTAL
        else:
            registry[name] = Stability.ADVANCED
    return registry


class _LazyRegistry(dict):
    """
    A dict that populates itself from the package ``__all__`` on first use.

    Deferred so importing ``_api`` does not trigger a circular import of the
    package ``__init__`` before its ``__all__`` is assembled.
    """

    _populated: bool = False

    def _ensure(self) -> None:
        if not self._populated:
            self.update(_build_registry())
            self._populated = True

    def __getitem__(self, key: str) -> Stability:  # type: ignore[override]
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure()
        return super().__len__()

    def items(self):
        self._ensure()
        return super().items()


#: Mapping of public symbol name → :class:`Stability`.  Lazily populated.
API_STABILITY: _LazyRegistry = _LazyRegistry()


def api_stability(name: str) -> Stability:
    """
    Return the :class:`Stability` tier of a public API symbol.

    Parameters
    ----------
    name : str
        A public symbol name (must be in the package ``__all__``).

    Returns
    -------
    Stability
        The symbol's stability tier.

    Raises
    ------
    KeyError
        If ``name`` is not a public symbol of ``scikitplot.cython``.
    """
    try:
        return API_STABILITY[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a public symbol of scikitplot.cython"
        ) from None


def list_api(tier: Stability | str) -> list[str]:
    """
    Return the sorted public symbol names in a given stability ``tier``.

    Parameters
    ----------
    tier : Stability or str
        The tier to list (``"stable"``, ``"advanced"``, ``"experimental"``).

    Returns
    -------
    list of str
        Sorted symbol names in that tier.
    """
    t = Stability(tier)
    return sorted(name for name, s in API_STABILITY.items() if s == t)
