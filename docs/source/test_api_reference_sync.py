"""
Consistency check between ``apis_reference.py`` and the live package.

Background
----------
``doc/apis_reference.py`` (``APIS_REFERENCE``) is a hand-maintained index of
every public name Sphinx's ``autosummary`` should document for each module.
Nothing enforces that it stays in sync with the code: when a class or
function is renamed, split, or removed, the code side updates immediately,
but the doc side only breaks silently, surfacing (if at all) as a
``WARNING: [autosummary] failed to import ...`` line buried in a Sphinx
build log -- easy to miss, and each occurrence has historically had to be
tracked down by hand after the fact (see e.g. the 2026-08 incident where
``SimilarityIndex``, ``CustomSimilarityIndex``, ``SearchConfig``,
``SearchResult``, and ``NormalizerConfig`` lingered in
``scikitplot.corpus``'s entry after the corpus "Retrieval" rename).

This module closes that gap: it resolves every ``autosummary``/``classes``
entry, for every module listed in ``APIS_REFERENCE``, against the real,
importable namespace, so a stale entry fails CI the moment it's introduced
instead of waiting to be noticed in a doc build.
"""

from __future__ import annotations

import importlib
import types

import pytest

from apis_reference import APIS_REFERENCE


def _iter_reference_names(module_entry: dict) -> list[tuple[str, str, str]]:
    """
    Flatten one module's ``APIS_REFERENCE`` entry into name references.

    Parameters
    ----------
    module_entry : dict
        The value for one module key in :data:`APIS_REFERENCE`, i.e. the
        dict containing a ``"sections"`` list.

    Returns
    -------
    list of (str, str, str)
        ``(section_title, list_kind, name)`` triples, one per documented
        name, where ``list_kind`` is ``"autosummary"`` or ``"classes"``.

    Notes
    -----
    Pure and side-effect free so it can be reused or unit-tested on its
    own, independent of import machinery.
    """
    references: list[tuple[str, str, str]] = []
    for section in module_entry.get("sections", []):
        title = section.get("title", "<untitled section>")
        for list_kind in ("autosummary", "classes"):
            for name in section.get(list_kind, []):
                references.append((title, list_kind, name))
    return references


@pytest.mark.parametrize("module_path", sorted(APIS_REFERENCE))
def test_api_reference_names_resolve(module_path: str) -> None:
    """
    Every name documented for ``module_path`` must exist on the live module.

    Parameters
    ----------
    module_path : str
        Dotted module path (an ``APIS_REFERENCE`` key), e.g.
        ``"scikitplot.corpus"``. Supplied by ``pytest.mark.parametrize``,
        one test instance per module -- a broken module is reported by
        name, not lost inside one monolithic assertion.

    Raises
    ------
    AssertionError
        If one or more documented names are missing from the live module.

    Notes
    -----
    An import failure for `module_path` itself is treated as a skip, not
    a failure: modules gated behind optional extras (e.g. an ``mlflow`` or
    ``annoy`` integration) are expected to be unimportable in a minimal
    environment, and that is a separate concern from doc/code drift. A
    module that is always expected to import in CI will still fail loudly
    here if it doesn't -- the skip only fires on `ImportError`.

    Examples
    --------
    Run just this check::

        $ pytest tests/test_api_reference_sync.py -v
    """
    try:
        module: types.ModuleType = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(f"{module_path} not importable in this environment: {exc}")

    missing = [
        f"{title!r} [{list_kind}]: {name!r}"
        for title, list_kind, name in _iter_reference_names(APIS_REFERENCE[module_path])
        if not hasattr(module, name)
    ]
    assert not missing, (
        f"apis_reference.py lists {len(missing)} name(s) not found on the "
        f"live '{module_path}' module (renamed, split, or removed in code "
        f"without updating the doc reference):\n  " + "\n  ".join(missing)
    )
