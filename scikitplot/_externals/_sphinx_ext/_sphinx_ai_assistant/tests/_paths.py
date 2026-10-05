"""
Stable path authorities for the mirrored test tree.

Tests may move within ``tests/`` without changing how they locate runtime or
maintenance files.  Never derive runtime ownership from a test file's parent
count; that makes directory refactors silently change what a test exercises.
"""
from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = TESTS_ROOT.parent

# The private extension stack (``_sphinx_ext``) this runtime belongs to.
STACK_ROOT = RUNTIME_ROOT.parent

# Where the stack sits inside a documentation source directory.
_STACK_IN_DOCS_SOURCE = ("scikitplot", "_externals", "_sphinx_ext")


def _find_docs_source_root() -> Path | None:
    """
    Return the documentation source directory that owns this stack, if any.

    The same tree is deployed in two checkouts. In a documentation checkout it
    lives at ``<docs source>/scikitplot/_externals/_sphinx_ext`` beside the
    site's ``conf.py``; in the library checkout it lives at
    ``<repository>/scikitplot/_externals/_sphinx_ext`` and there is no site.
    An ancestor is the docs source only if both facts hold - it has a
    ``conf.py`` and its stack directory is this one - so the answer comes from
    what is on disk, never from how many directories up it is.
    """
    for candidate in STACK_ROOT.parents:
        if not (candidate / "conf.py").is_file():
            continue
        if candidate.joinpath(*_STACK_IN_DOCS_SOURCE).resolve() == STACK_ROOT:
            return candidate
    return None


#: ``None`` in the library checkout. Tests of the site's own configuration
#: skip there, with that reason, and run in the documentation checkout.
DOCS_SOURCE_ROOT = _find_docs_source_root()


# Repository root is the nearest ancestor that owns both runtime and
# maintenance planes.  The extracted maintenance workspace uses this layout too.
def _find_repository_root() -> Path:
    for candidate in (RUNTIME_ROOT, *RUNTIME_ROOT.parents):
        if (candidate / "scikitplot").is_dir() and (candidate / "maintenances").is_dir():
            return candidate
    raise RuntimeError(
        "AI-assistant tests require a repository/workspace root containing both "
        "'scikitplot/' and 'maintenances/'; do not infer ownership from parent depth."
    )


def _maintenance_root() -> Path:
    return (
        __getattr__("REPOSITORY_ROOT")
        / "maintenances"
        / "_externals"
        / "_sphinx_ext"
        / "_sphinx_ai_assistant"
    )


def _skill_root() -> Path:
    return (
        __getattr__("REPOSITORY_ROOT")
        / "skills"
        / "_externals"
        / "_sphinx_ext"
        / "_sphinx_ai_assistant"
    )


# The repository-plane authorities are resolved on first use rather than at
# import. A documentation checkout has no ``maintenances/`` plane; resolving
# eagerly made every test module that imports this file fail there, including
# the ones that only need ``RUNTIME_ROOT``. A test that does need the
# maintenance plane still gets the same RuntimeError, at the point it asks.
_LAZY = {
    "REPOSITORY_ROOT": _find_repository_root,
    "MAINTENANCE_ROOT": _maintenance_root,
    "SKILL_ROOT": _skill_root,
}

# Declare the lazily resolved exports for static analyzers such as Flake8.
REPOSITORY_ROOT: Path
MAINTENANCE_ROOT: Path
SKILL_ROOT: Path


def __getattr__(name: str) -> Path:
    try:
        resolve = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = resolve()
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DOCS_SOURCE_ROOT",
    "MAINTENANCE_ROOT",
    "REPOSITORY_ROOT",
    "RUNTIME_ROOT",
    "SKILL_ROOT",
    "STACK_ROOT",
    "TESTS_ROOT",
]
