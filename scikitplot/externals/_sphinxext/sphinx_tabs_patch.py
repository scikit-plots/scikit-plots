# scikitplot/externals/_sphinxext/sphinx_tabs_patch.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Sphinx compatibility shim for docutils ``backrefs``.

This extension prevents a hard failure in ``sphinx-tabs`` (sphinx_tabs-3.4.7)
that manifests as ``KeyError: 'backrefs'`` with Docutils 0.22 during HTML builds.

The failure is triggered by ``sphinx-tabs`` attempting to remove ``backrefs``
from an attribute mapping using ``pop('backrefs')`` without providing a
default. With recent docutils versions, some element nodes may not have the
``backrefs`` list attribute present in their ``.attributes`` mapping at that
moment, so the unconditional pop raises.

This module fixes the problem *without modifying docutils* by ensuring that
all docutils element nodes in the doctree have a well-formed ``backrefs``
entry (a list) before writers/visitors run.

Notes
-----
- This is a compatibility workaround. The ideal long-term fix is in
  ``sphinx-tabs``: replace ``attrs.pop('backrefs')`` with
  ``attrs.pop('backrefs', None)`` (or remove local attributes generically).
- This shim is safe because it only adds a missing key or normalizes an
  invalid value for a docutils-internal attribute.
- https://github.com/executablebooks/sphinx-tabs/pull/207

Examples
--------
1) Place this file at ``docs/source/_sphinx_ext/sphinx_tabs_patch.py``.

2) In ``conf.py``:

>>> import os
>>> import sys
>>> import scikitplot
>>> sys.path.insert(0, os.path.abspath('_sphinx_ext'))
>>> extensions = [
...     # 'scikitplot.externals._sphinxext.sphinx_tabs_patch',
...     'sphinx_tabs_patch',   # <-- must come BEFORE sphinx_tabs.tabs
...     'sphinx_tabs.tabs',
...     # ... your other extensions
... ]

3) Rebuild:

$ make clean html
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, MutableMapping, Optional


_BACKREFS_KEY = "backrefs"


def _is_mutable_mapping(obj: Any) -> bool:
    """
    Return True if *obj* looks like a mutable mapping.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True if the object behaves like a mutable mapping (supports ``__setitem__``).

    Notes
    -----
    This is intentionally conservative: we only accept true mutable mappings.
    """

    return isinstance(obj, MutableMapping)


def _ensure_backrefs_list(attributes: MutableMapping[str, Any]) -> None:
    """
    Ensure ``attributes['backrefs']`` exists and is a list.

    Parameters
    ----------
    attributes : MutableMapping[str, Any]
        The node attributes mapping to normalize.

    Returns
    -------
    None

    Notes
    -----
    - If the key is missing, it is created with a new empty list.
    - If the value is ``None``, it is replaced with a new empty list.
    - If the value is a tuple, it is converted to a list.
    - If the value is any other non-list iterable, it is converted to a list.

    This normalization is deterministic and does not infer meaning; it only
    ensures the value type is consistent with docutils' expectation that
    ``backrefs`` is list-like.
    """

    if _BACKREFS_KEY not in attributes or attributes[_BACKREFS_KEY] is None:
        attributes[_BACKREFS_KEY] = []
        return

    value = attributes[_BACKREFS_KEY]
    if isinstance(value, list):
        return

    # Convert common iterable containers deterministically.
    if isinstance(value, tuple):
        attributes[_BACKREFS_KEY] = list(value)
        return

    # Avoid treating strings/bytes as iterables of characters.
    if isinstance(value, (str, bytes, bytearray)):
        attributes[_BACKREFS_KEY] = [value]
        return

    try:
        attributes[_BACKREFS_KEY] = list(value)  # type: ignore[arg-type]
    except TypeError:
        # Non-iterable: wrap as a single value.
        attributes[_BACKREFS_KEY] = [value]


def _iter_element_nodes(doctree: Any) -> Iterable[Any]:
    """
    Yield docutils element nodes from a doctree.

    Parameters
    ----------
    doctree : Any
        A docutils document tree.

    Yields
    ------
    Any
        Nodes that are instances of ``docutils.nodes.Element``.

    Raises
    ------
    ImportError
        If docutils is not importable.

    Notes
    -----
    We import docutils lazily to keep this extension importable even in
    non-doc-build contexts.
    """

    from docutils import nodes

    # ``traverse`` is the stable API across docutils versions.
    for node in doctree.traverse(nodes.Element):
        yield node


def _normalize_doctree_backrefs(doctree: Any) -> None:
    """
    Normalize ``backrefs`` attributes throughout a doctree.

    Parameters
    ----------
    doctree : Any
        The docutils document tree.

    Returns
    -------
    None
    """

    for node in _iter_element_nodes(doctree):
        attrs = getattr(node, "attributes", None)
        if not _is_mutable_mapping(attrs):
            continue
        _ensure_backrefs_list(attrs)


def _on_doctree_read(app: Any, doctree: Any) -> None:
    """
    Sphinx event handler for ``doctree-read``.

    Parameters
    ----------
    app : Any
        The Sphinx application.
    doctree : Any
        The parsed doctree for a single document.

    Returns
    -------
    None

    Notes
    -----
    ``doctree-read`` runs after parsing and initial transforms. Normalizing
    here ensures the visitor stage does not hit ``KeyError``.
    """

    _normalize_doctree_backrefs(doctree)


def _on_doctree_resolved(app: Any, doctree: Any, docname: str) -> None:
    """
    Sphinx event handler for ``doctree-resolved``.

    Parameters
    ----------
    app : Any
        The Sphinx application.
    doctree : Any
        The resolved doctree.
    docname : str
        Document name.

    Returns
    -------
    None

    Notes
    -----
    Some extensions/transforms can mutate attributes after ``doctree-read``.
    Normalizing again at ``doctree-resolved`` provides an extra safety net.
    """

    _normalize_doctree_backrefs(doctree)


def setup(app: Any) -> Mapping[str, Any]:
    """
    Register the extension with Sphinx.

    Parameters
    ----------
    app : Any
        The Sphinx application instance.

    Returns
    -------
    dict
        Sphinx extension metadata.

    See Also
    --------
    sphinx.application.Sphinx.connect
        Sphinx API used to register event callbacks.

    Notes
    -----
    ``parallel_read_safe`` and ``parallel_write_safe`` are set to True because
    this extension only mutates the current document's in-memory doctree.
    """

    app.connect("doctree-read", _on_doctree_read)
    app.connect("doctree-resolved", _on_doctree_resolved)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
