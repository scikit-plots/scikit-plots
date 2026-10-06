# scikitplot/_distributions.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Distribution map: which installable package ships which part of ``scikitplot``.

``scikitplot`` is published two ways from one source tree:

* the **full** distribution, ``scikit-plots``, which contains everything and
  needs a compiler toolchain to build; and
* a family of **partial** distributions under ``libs/`` (``scikit-plots-skinny``,
  ``scikit-plots-rank-bm25``, ``scikit-plots-corpus``, ...), each of which ships
  one focused part of the same ``scikitplot`` import package.

This module is the single source of truth for *who ships what*. Three consumers
read it and nothing else decides the answer:

1. the build tooling in ``libs/_tools`` (which files go into which wheel),
2. the root package and the CLI loader (which package to name when a part is
   missing), and
3. ``scikitplot doctor`` (what is installed, and whether it is coherent).

Notes
-----
**User.** Install only what you need; the parts compose into one package::

    pip install scikit-plots-rank-bm25                 # just BM25 ranking
    pip install scikit-plots-corpus scikit-plots-mcp   # two parts, one package

Every spelling pip accepts for a name is the same project, so
``scikit_plots_rank_bm25`` and ``scikit-plots-rank_bm25`` both resolve to
``scikit-plots-rank-bm25`` (see :func:`canonicalize_name`).

**Developer.** The invariant that makes mixed installs safe is *one owner per
file*: no path under ``scikitplot/`` is claimed by two partial distributions.
``scikit-plots-skinny`` (:data:`CORE`) owns the root ``__init__.py`` and the
shared infrastructure; every other partial distribution depends on it and owns
only its own subtree. Uninstalling one therefore never removes a file another
one needs. ``libs/_tools`` asserts the invariant statically and against the
built wheels.

This module is standard-library only, imports nothing from ``scikitplot`` and
has no import-time side effects, so the build tooling can load it by file path
without importing the package.

Examples
--------
>>> from scikitplot._distributions import canonicalize_name, provider_of
>>> canonicalize_name("scikit_plots_rank_bm25")
'scikit-plots-rank-bm25'
>>> provider_of("scikitplot.rank_bm25")
'scikit-plots-rank-bm25'
>>> provider_of("cexternals._annoy.annoylib")
'scikit-plots-annoy'
"""

from __future__ import annotations

import re
from typing import Mapping, NamedTuple

__all__ = [
    "CORE",
    "DISTRIBUTIONS",
    "FLAVORS",
    "FULL",
    "IMPORT_NAME",
    "Distribution",
    "canonicalize_name",
    "flavor",
    "get",
    "install_hint",
    "installed",
    "provider_of",
    "report",
]

#: The import package every distribution contributes to.
IMPORT_NAME = "scikitplot"

#: The full distribution: the whole package, compiled extensions included.
FULL = "scikit-plots"

#: The core partial distribution. It owns the root ``__init__.py`` and the
#: shared infrastructure, and every other partial distribution depends on it.
CORE = "scikit-plots-skinny"

#: The values :func:`flavor` can return.
FLAVORS = ("full", "partial", "source")

_NAME_SEPARATORS = re.compile(r"[-_.]+")


class Distribution(NamedTuple):
    """
    One partial distribution and the part of ``scikitplot`` it owns.

    Parameters
    ----------
    name : str
        Canonical project name (PEP 503 normalised), e.g.
        ``"scikit-plots-rank-bm25"``.
    trees : tuple of str
        Directories owned recursively, as POSIX paths relative to the
        ``scikitplot`` package directory, e.g. ``("rank_bm25",)`` or
        ``("annoy", "cexternals/_annoy")``.
    files : tuple of str
        Individual files owned, as POSIX paths relative to the ``scikitplot``
        package directory. Used for files whose directory is shared, such as
        the root ``__init__.py``.
    summary : str
        One-line description, used for the package metadata and in reports.

    Notes
    -----
    **Developer.** ``trees`` and ``files`` are the whole ownership statement.
    A tree owns every file beneath it, so a file must never be listed when its
    directory is already a tree, and no path may fall under two distributions.
    """

    name: str
    trees: tuple[str, ...]
    files: tuple[str, ...]
    summary: str


#: Every partial distribution, core first. Order is the order they are
#: generated, built and reported in.
DISTRIBUTIONS: tuple[Distribution, ...] = (
    Distribution(
        name=CORE,
        trees=("_cli", "logging"),
        files=(
            "__init__.py",
            "__init__.pyi",
            "__main__.py",
            "_distributions.py",
            "environment_variables.py",
            "exceptions.py",
            "exceptions.pyi",
            "py.typed",
        ),
        summary=(
            "Dependency-free core of scikit-plots: the root package, logging "
            "and the scikitplot command line."
        ),
    ),
    Distribution(
        name="scikit-plots-rank-bm25",
        trees=("rank_bm25",),
        files=(),
        summary="BM25 ranking algorithms (Okapi BM25, BM25L, BM25+) from scikit-plots.",
    ),
    Distribution(
        name="scikit-plots-corpus",
        trees=("corpus",),
        files=(),
        summary=(
            "Document ingestion, chunking, embedding and retrieval pipeline "
            "from scikit-plots."
        ),
    ),
    Distribution(
        name="scikit-plots-annoy",
        trees=("annoy", "cexternals/_annoy"),
        files=("cexternals/__init__.py",),
        summary=(
            "Approximate nearest-neighbour index (Annoy) with the scikit-plots "
            "high-level wrapper; the one partial distribution that is compiled."
        ),
    ),
    Distribution(
        name="scikit-plots-sphinx-ext",
        trees=("_externals/_sphinx_ext",),
        files=("_externals/__init__.py",),
        summary=(
            "The Sphinx extensions of the scikit-plots documentation, "
            "installable on their own for any Sphinx project."
        ),
    ),
    Distribution(
        name="scikit-plots-mcp",
        trees=("mcp",),
        files=(),
        summary=(
            "Documentation retrieval for Model Context Protocol servers "
            "from scikit-plots."
        ),
    ),
    Distribution(
        name="scikit-plots-cleanprompt",
        trees=("cleanprompt",),
        files=(),
        summary=(
            "Redact sensitive values from a prompt before it is sent to an "
            "LLM, from scikit-plots."
        ),
    ),
    Distribution(
        name="scikit-plots-cython",
        trees=("cython",),
        files=(),
        summary="Runtime Cython and pybind11 build helpers from scikit-plots.",
    ),
    Distribution(
        name="scikit-plots-mlflow",
        trees=("mlflow",),
        files=(),
        summary="Project-level MLflow configuration and workflow helpers from scikit-plots.",
    ),
)


def canonicalize_name(name: str) -> str:
    """
    Return the canonical (PEP 503) form of a project name.

    Parameters
    ----------
    name : str
        A project name in any spelling an installer accepts.

    Returns
    -------
    str
        Lower-cased name with every run of ``-``, ``_`` and ``.`` replaced by a
        single ``-``.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` is empty after normalisation.

    Notes
    -----
    **User.** Installers compare names in this form, so every spelling of a
    partial distribution installs the same project.

    Examples
    --------
    >>> canonicalize_name("scikit_plots_rank_bm25")
    'scikit-plots-rank-bm25'
    >>> canonicalize_name("Scikit.Plots-rank_bm25")
    'scikit-plots-rank-bm25'
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__!r}")
    canonical = _NAME_SEPARATORS.sub("-", name.strip()).lower().strip("-")
    if not canonical:
        raise ValueError(f"name {name!r} is empty after normalisation")
    return canonical


def get(name: str) -> Distribution:
    """
    Return the partial distribution called ``name``.

    Parameters
    ----------
    name : str
        Project name in any spelling, e.g. ``"scikit_plots_rank_bm25"``.

    Returns
    -------
    Distribution
        The matching entry of :data:`DISTRIBUTIONS`.

    Raises
    ------
    KeyError
        If no partial distribution has that name. The message lists the
        known names.
    """
    canonical = canonicalize_name(name)
    for dist in DISTRIBUTIONS:
        if dist.name == canonical:
            return dist
    known = ", ".join(dist.name for dist in DISTRIBUTIONS)
    raise KeyError(f"unknown partial distribution {name!r}; known: {known}")


def _relative_module(module: str) -> str:
    """Strip a leading ``scikitplot.`` so both spellings name the same module."""
    if not isinstance(module, str):
        raise TypeError(f"module must be a str, got {type(module).__name__!r}")
    if module == IMPORT_NAME:
        return ""
    prefix = IMPORT_NAME + "."
    return module[len(prefix) :] if module.startswith(prefix) else module


def provider_of(module: str) -> str | None:
    """
    Return the partial distribution that ships a ``scikitplot`` module.

    Parameters
    ----------
    module : str
        Dotted module name, absolute (``"scikitplot.corpus._registry"``) or
        relative to the package (``"corpus._registry"``).

    Returns
    -------
    str or None
        Canonical name of the owning partial distribution, or ``None`` when no
        partial distribution ships it (it is then part of the full
        distribution only).

    Notes
    -----
    **Developer.** Ownership is decided on path components, never on string
    prefixes, so ``"corpus_extra"`` is not mistaken for ``"corpus"``. The root
    package itself (``"scikitplot"``) belongs to :data:`CORE`.

    Examples
    --------
    >>> provider_of("scikitplot.mcp._server")
    'scikit-plots-mcp'
    >>> provider_of("scikitplot.cexternals._annoy")
    'scikit-plots-annoy'
    >>> provider_of("scikitplot.cexternals._astropy") is None
    True
    """
    relative = _relative_module(module)
    if not relative:
        return CORE
    parts = tuple(relative.split("."))
    for dist in DISTRIBUTIONS:
        for tree in dist.trees:
            tree_parts = tuple(tree.split("/"))
            if parts[: len(tree_parts)] == tree_parts:
                return dist.name
        for path in dist.files:
            if not path.endswith(".py"):
                continue
            file_parts = tuple(path[: -len(".py")].split("/"))
            if file_parts[-1] == "__init__":
                file_parts = file_parts[:-1]
            if file_parts and parts == file_parts:
                return dist.name
    return None


def install_hint(module: str) -> str:
    """
    Return the command that makes a missing ``scikitplot`` module available.

    Parameters
    ----------
    module : str
        Dotted module name, absolute or relative to the package.

    Returns
    -------
    str
        ``"pip install <distribution>"`` for the smallest distribution that
        ships the module: its partial distribution when one exists, otherwise
        the full distribution.

    Examples
    --------
    >>> install_hint("scikitplot.corpus")
    'pip install scikit-plots-corpus'
    >>> install_hint("scikitplot.utils")
    'pip install scikit-plots'
    """
    return f"pip install {provider_of(module) or FULL}"


def _installed_version(name: str) -> str | None:
    """Return the installed version of a distribution, or ``None`` if it is absent."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def installed() -> dict[str, str]:
    """
    Return the installed ``scikitplot`` distributions and their versions.

    Returns
    -------
    dict of str to str
        Canonical project name to installed version, for the full
        distribution and every partial distribution that is installed.
        Distributions that are not installed are absent.

    Notes
    -----
    **Developer.** Reads installed package metadata only; nothing is imported
    from ``scikitplot``. The metadata import is deferred so that importing this
    module stays free of side effects.
    """
    names = (FULL, *(dist.name for dist in DISTRIBUTIONS))
    versions = {name: _installed_version(name) for name in names}
    return {name: version for name, version in versions.items() if version is not None}


def flavor(found: Mapping[str, str] | None = None) -> str:
    """
    Classify how ``scikitplot`` is installed.

    Parameters
    ----------
    found : mapping of str to str, optional
        Result of :func:`installed`. Queried when omitted.

    Returns
    -------
    {"full", "partial", "source"}
        ``"full"`` when the full distribution is installed (alone or beside
        partial ones); ``"partial"`` when only partial distributions are
        installed; ``"source"`` when neither is, which is what an import from
        an unbuilt source tree looks like.

    Notes
    -----
    **Developer.** The answer comes from installed metadata, which is the only
    place that records which distribution was installed. The files on disk
    cannot tell a partial install from a source checkout, because both lack
    the compiled extensions.
    """
    found = installed() if found is None else found
    if FULL in found:
        return "full"
    if CORE in found:
        return "partial"
    return "source"


def _provided_modules(dist: Distribution) -> list[str]:
    """Return the importable top-level parts of ``scikitplot`` that ``dist`` owns."""
    provided = [tree.replace("/", ".") for tree in dist.trees]
    provided += [
        path[: -len(".py")]
        for path in dist.files
        if path.endswith(".py") and "/" not in path and path != "__init__.py"
    ]
    return sorted(provided)


def _is_present(module: str) -> bool:
    """Return whether ``scikitplot.<module>`` can be located, without importing it."""
    from importlib import util  # noqa: PLC0415

    # Only the first component is probed: locating a deeper name would import
    # its parent package, and a report must not run a subpackage's code.
    top = module.split(".", 1)[0]
    try:
        return util.find_spec(f"{IMPORT_NAME}.{top}") is not None
    except (ImportError, ValueError):
        return False


def report() -> dict[str, object]:
    """
    Describe the installed distributions and whether they are coherent.

    Returns
    -------
    dict
        ``flavor``
            See :func:`flavor`.
        ``installed``
            Project name to version, for every installed distribution.
        ``available``
            Project name to the parts of ``scikitplot`` it would add, for
            every partial distribution that is not installed. Empty when the
            full distribution is installed, because it already contains them.
        ``problems``
            Actionable messages; empty when the installation is coherent.

    Notes
    -----
    **User.** Three situations are reported as problems, each with the command
    that fixes it:

    * the full distribution installed beside partial ones, where both own the
      same files and uninstalling either removes files the other needs;
    * partial distributions of different versions, which were not built from
      the same source tree; and
    * a partial distribution that is installed but whose files are gone.

    **Developer.** Whether a part is on disk is checked only for distributions
    that are installed. For one that is not, its part may still be present
    because the full distribution provides it, which is not a fault.
    """
    found = installed()
    partial_names = [dist.name for dist in DISTRIBUTIONS if dist.name in found]
    problems: list[str] = []

    for dist in DISTRIBUTIONS:
        if dist.name not in found:
            continue
        if not all(_is_present(module) for module in _provided_modules(dist)):
            problems.append(
                f"{dist.name} is installed but some of its files are missing. "
                f"Fix: pip install --force-reinstall --no-deps {dist.name}"
            )

    if FULL in found and partial_names:
        problems.append(
            f"{FULL} is installed together with {', '.join(partial_names)}. "
            f"{FULL} already contains them and both own the same files, so "
            "uninstalling either removes files the other needs. "
            f"Fix: pip uninstall {' '.join(partial_names)} && "
            f"pip install --force-reinstall --no-deps {FULL}"
        )

    if len({found[name] for name in partial_names}) > 1:
        detail = ", ".join(f"{name} {found[name]}" for name in partial_names)
        problems.append(
            f"Partial distributions have different versions ({detail}); they "
            "were not built from the same source tree. "
            f"Fix: pip install --upgrade {' '.join(partial_names)}"
        )

    available: dict[str, list[str]] = {}
    if FULL not in found:
        available = {
            dist.name: _provided_modules(dist)
            for dist in DISTRIBUTIONS
            if dist.name not in found
        }
    return {
        "flavor": flavor(found),
        "installed": dict(found),
        "available": available,
        "problems": problems,
    }
