"""
Guard shared by every extension test suite under ``_sphinx_ext``.

A test that runs a Sphinx application must leave docutils' process-wide
tables as it found them. Sphinx registers nodes, directives and roles in those
tables (``sphinx.util.docutils.register_node``, ``docutils.parsers.rst``), and
only two things put them back: ``sphinx.util.docutils.docutils_namespace()``,
which ``sphinx-build`` wraps around every application, and
``sphinx.testing.util.SphinxTestApp.cleanup()``. A bare
``sphinx.application.Sphinx(...)`` does neither.

Notes
-----
**User.** If a test fails at teardown with "left N Sphinx node class(es)
registered", that test is the culprit: build inside
``with docutils_namespace():`` or use ``SphinxTestApp`` and call
``cleanup()``. Before this guard, the failure appeared in an unrelated test
much later: the next real build in the process warned "node class 'toctree' is
already registered" (round 32: a feedback-site build broke the gallery-grid
suite on every CI leg).

**Developer.** The signal is ``sphinx.util.docutils.additional_nodes``: only
``app.add_node`` fills it and both clean-up paths empty it, so a set larger
after a test than before is a leak, with no false positives from docutils
caching its own directives lazily (``directives._directives`` grows by design
when a document uses ``.. note::``). The module is read from ``sys.modules``,
never imported, so a test that does not use Sphinx pays nothing and a
distribution without Sphinx still collects. After reporting, the guard
restores what ``docutils_namespace`` would have restored, so one leak is
reported once and does not cascade into the tests that follow.

See Also
--------
sphinx.util.docutils.docutils_namespace
sphinx.testing.util.SphinxTestApp.cleanup
"""

from __future__ import annotations

import sys

import pytest

_SPHINX_DOCUTILS = "sphinx.util.docutils"
_RST_DIRECTIVES = "docutils.parsers.rst.directives"
_RST_ROLES = "docutils.parsers.rst.roles"


def _registered_nodes() -> frozenset:
    """
    Return the node classes Sphinx has registered with docutils.

    Returns
    -------
    frozenset of type
        Empty when ``sphinx.util.docutils`` was never imported in this process.
    """
    module = sys.modules.get(_SPHINX_DOCUTILS)
    if module is None:
        return frozenset()
    return frozenset(module.additional_nodes)


def _registry_snapshot(module_name: str, attribute: str) -> dict | None:
    """
    Return a copy of one docutils registry, or ``None`` if it is not loaded.

    Parameters
    ----------
    module_name : str
        Dotted name of the docutils module that owns the registry.
    attribute : str
        Name of the registry mapping in that module.

    Returns
    -------
    dict or None
        A shallow copy, as ``docutils_namespace`` takes.
    """
    module = sys.modules.get(module_name)
    return None if module is None else dict(getattr(module, attribute))


@pytest.fixture(autouse=True)
def _sphinx_registries_restored_after_each_test():
    """
    Fail the test that leaves Sphinx's docutils registrations behind.

    Yields
    ------
    None

    Raises
    ------
    pytest.fail.Exception
        At teardown, when the test left node classes registered.
    """
    nodes_before = _registered_nodes()
    directives_before = _registry_snapshot(_RST_DIRECTIVES, "_directives")
    roles_before = _registry_snapshot(_RST_ROLES, "_roles")
    yield
    leaked = _registered_nodes() - nodes_before
    if not leaked:
        return
    # The same steps, in the same order, as ``docutils_namespace``'s exit;
    # ``unregister_node`` alone leaves the class in ``additional_nodes``.
    docutils_support = sys.modules[_SPHINX_DOCUTILS]
    for node in leaked:
        docutils_support.unregister_node(node)
        docutils_support.additional_nodes.discard(node)
    if directives_before is not None:
        sys.modules[_RST_DIRECTIVES]._directives = directives_before
    if roles_before is not None:
        sys.modules[_RST_ROLES]._roles = roles_before
    names = ", ".join(sorted(node.__name__ for node in leaked)[:5])
    pytest.fail(
        f"this test left {len(leaked)} Sphinx node class(es) registered with "
        f"docutils ({names}, ...): it ran a Sphinx application without "
        "restoring process-wide state, so the next build in this process would "
        "warn 'already registered'. Build inside "
        "`with sphinx.util.docutils.docutils_namespace():`, or use "
        "`SphinxTestApp` and call `cleanup()`.",
        pytrace=False,
    )
