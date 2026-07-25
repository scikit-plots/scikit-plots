# scikitplot/cython/_operations_examples.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Executable operational examples for :mod:`scikitplot.cython`.

The prose operational guide lives in ``OPERATIONS.md``.  This module holds the
*executable* counterpart: every operational claim that can be checked without a
C compiler is expressed here as a doctest, so the documentation cannot silently
drift from the implementation (CYTHON-DOC-001).  Run with::

    python -m doctest scikitplot/cython/_operations_examples.py -v

These are examples/tests, not public API; nothing here is re-exported.
"""

from __future__ import annotations

__all__: list[str] = []


def security_trust_model() -> None:
    """Security: the default policy is strict; ``strict=`` is operative.

    The default :class:`~scikitplot.cython.SecurityPolicy` rejects dangerous
    inputs; ``strict=False`` relaxes the unset guards, and an explicit per-flag
    value always overrides the master switch (CYTHON-SEC-002).

    >>> from scikitplot.cython import SecurityPolicy
    >>> SecurityPolicy().strict
    True
    >>> SecurityPolicy().allow_shell_metacharacters  # strict default
    False
    >>> SecurityPolicy(strict=False).allow_shell_metacharacters
    True
    >>> SecurityPolicy(
    ...     strict=False, allow_shell_metacharacters=False
    ... ).allow_shell_metacharacters  # explicit override
    False
    """


def cache_recovery_protocol() -> None:
    """Cache: entries are schema-versioned; incompatible entries are rebuilt.

    ``meta.json`` carries a schema version (CYTHON-SCH-001).  Legacy
    (unversioned) and newer-unknown entries are treated as incompatible, so they
    are rebuilt rather than misread.

    >>> from scikitplot.cython import (
    ...     CACHE_SCHEMA_VERSION,
    ...     is_meta_schema_compatible,
    ...     meta_schema_version,
    ... )
    >>> CACHE_SCHEMA_VERSION >= 1
    True
    >>> is_meta_schema_compatible({"meta_schema_version": CACHE_SCHEMA_VERSION})
    True
    >>> is_meta_schema_compatible({"kind": "module"})  # legacy, no version
    False
    >>> meta_schema_version({"kind": "module"})
    0
    """


def concurrency_model() -> None:
    """Concurrency: builds are serialised per cache key by a file lock.

    ``build_lock`` provides advisory, owner-tokened exclusion (CYTHON-CON-001),
    and the setuptools singleton + compiler registry are thread-safe
    (CYTHON-CON-002).  The lock is a context manager keyed by a directory.

    >>> from scikitplot.cython import build_lock
    >>> callable(build_lock)
    True
    """


def recovery_and_batch() -> None:
    """Recovery: batch builds report committed work and a resume token.

    A batch that fails part-way raises :class:`~scikitplot.cython.BatchBuildError`
    whose ``result`` lists the committed items and whose ``resume_token`` names
    the items still to attempt (CYTHON-BATCH-001).

    >>> from scikitplot.cython import BatchBuildError, BatchBuildResult
    >>> issubclass(BatchBuildError, RuntimeError)
    True
    >>> sorted(f for f in BatchBuildResult.__dataclass_fields__)
    ['committed', 'failures', 'policy', 'successes']
    """


def unsupported_capability_behavior() -> None:
    """Platform: unsupported capabilities are introspectable, not surprising.

    ``platform_capabilities()`` reports whether runtime compilation is available
    (e.g. it is not under browser WASM), so callers can branch deterministically
    instead of hitting an opaque failure (CYTHON-WASM-001).

    >>> from scikitplot.cython import platform_capabilities
    >>> caps = platform_capabilities()
    >>> isinstance(caps.can_compile_at_runtime, bool)
    True
    >>> isinstance(caps.is_browser_wasm, bool)
    True
    """


def api_stability_contract() -> None:
    """
    Stable API: every public symbol has a stability tier (CYTHON-API-002).

    >>> import scikitplot.cython as skc
    >>> from scikitplot.cython import api_stability, Stability
    >>> api_stability("compile_and_load") is Stability.STABLE
    True
    >>> all(api_stability(n) in Stability for n in skc.__all__)
    True
    """
