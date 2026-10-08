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
    "CORE_API",
    "CORE_API_HISTORY",
    "DISTRIBUTIONS",
    "FLAVORS",
    "FULL",
    "IMPORT_NAME",
    "PARTS_GROUP",
    "Distribution",
    "canonicalize_name",
    "declared_core_api",
    "flavor",
    "get",
    "install_hint",
    "installed",
    "log_report",
    "parts_entry_points",
    "parts_group",
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

#: Number of the contract between the core and the other partial distributions.
#:
#: Every partial distribution is built from one source tree at one version,
#: but a user can upgrade one of them and not the rest. Whether such a mix
#: works does not depend on the version numbers being equal; it depends on
#: whether the *contract* between the core and a part changed in between. This
#: number names that contract. A part records the number it was built for
#: (see :data:`PARTS_GROUP`); it works with a core that provides the same one.
#:
#: The contract is:
#:
#: 1. what a part may use from the core: the public names of
#:    ``scikitplot.logging``, ``scikitplot.exceptions`` and
#:    ``scikitplot.environment_variables``;
#: 2. how the core reaches a part: as the module ``scikitplot.<part>``, and for
#:    a command of the ``scikitplot`` CLI through the handler that
#:    ``scikitplot/_cli/registry.py`` names, called the way ``_cli/loader.py``
#:    calls it;
#: 3. the ownership map below: which distribution ships which path.
#:
#: Raise the number, by one, in the same change that *breaks* any of the
#: three: a name removed or given another meaning, a handler renamed or called
#: differently, a path moved from one distribution to another. Do not raise it
#: for an addition (a new name, a new command, a new distribution): an older
#: part does not use what it does not know. Add a line to
#: :data:`CORE_API_HISTORY` every time.
CORE_API = 1

#: Each value of :data:`CORE_API`, the first version of the core that provides
#: it, and what changed. A record for people; nothing computes with it.
CORE_API_HISTORY: tuple[tuple[int, str, str], ...] = (
    (1, "0.5.0", "first numbered contract"),
)

#: Prefix of the entry-point group in which a partial distribution states the
#: core API it was built for: the group is ``scikitplot.parts.api<N>`` and
#: holds one entry per part the distribution ships (``annoy =
#: scikitplot.annoy``). Entry points are part of the installed metadata, so the
#: statement is read without importing the part, and it is written by the
#: build tooling from :data:`CORE_API`, so it cannot be forgotten.
PARTS_GROUP = "scikitplot.parts.api"

#: Combinations of third-party packages that are known not to work together,
#: as measured (see ``_third_party_findings``): the first package at or above
#: its release together with the second package below its release.
_INCOMPATIBLE_PAIRS: tuple[
    tuple[str, tuple[int, ...], str, tuple[int, ...], str], ...
] = (
    (
        "numpy",
        (2,),
        "scikit-learn",
        (1, 4, 2),
        (
            "scikit-learn before 1.4.2 is built against NumPy 1 and cannot be "
            'imported with NumPy 2 ("numpy.dtype size changed" or a failed '
            "import from numpy.core)."
        ),
    ),
)

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


def parts_group(api: int = CORE_API) -> str:
    """
    Return the entry-point group that states "built for core API ``api``".

    Parameters
    ----------
    api : int, optional
        A core API number; the one this core provides when omitted.

    Returns
    -------
    str
        ``"scikitplot.parts.api<api>"``.

    Examples
    --------
    >>> parts_group(1)
    'scikitplot.parts.api1'
    """
    return f"{PARTS_GROUP}{int(api)}"


def parts_entry_points(name: str) -> dict[str, str]:
    """
    Return the entry points with which a distribution states its core API.

    Parameters
    ----------
    name : str
        Project name of a partial distribution, in any spelling.

    Returns
    -------
    dict of str to str
        Entry-point name to module, one for each tree the distribution owns
        (``{"annoy": "scikitplot.annoy", ...}``), to be written in the group
        :func:`parts_group`. Empty for the core, which provides the contract
        and does not consume it.

    Raises
    ------
    KeyError
        If ``name`` is not a partial distribution.

    Notes
    -----
    **Developer.** The build tooling writes exactly this mapping into each
    generated ``pyproject.toml``; :func:`declared_core_api` reads it back from
    installed metadata. Keeping both ends in this module means the statement
    and its reader cannot drift apart.

    Examples
    --------
    >>> parts_entry_points("scikit-plots-rank-bm25")
    {'rank_bm25': 'scikitplot.rank_bm25'}
    >>> parts_entry_points("scikit-plots-skinny")
    {}
    """
    dist = get(name)
    if dist.name == CORE:
        return {}
    return {
        module: f"{IMPORT_NAME}.{module}"
        for module in sorted(tree.replace("/", ".") for tree in dist.trees)
    }


def declared_core_api(name: str) -> int | None:
    """
    Return the core API an installed partial distribution was built for.

    Parameters
    ----------
    name : str
        Project name of a partial distribution, in any spelling.

    Returns
    -------
    int or None
        The number in the distribution's ``scikitplot.parts.api<N>``
        entry-point group; ``None`` when the distribution is not installed or
        declares none (the core itself, the full distribution, and a part
        built before the number existed).

    Raises
    ------
    ValueError
        If the distribution declares more than one number, or a group whose
        suffix is not a number: its metadata is not something this tooling
        wrote.

    Notes
    -----
    **Developer.** Only installed metadata is read; the part is not imported.
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        entry_points = metadata.distribution(canonicalize_name(name)).entry_points
    except metadata.PackageNotFoundError:
        return None
    suffixes = sorted(
        {
            entry.group[len(PARTS_GROUP) :]
            for entry in entry_points
            if entry.group.startswith(PARTS_GROUP)
        }
    )
    if not suffixes:
        return None
    if len(suffixes) > 1 or not suffixes[0].isdigit():
        raise ValueError(
            f"{name}: expected one entry-point group '{PARTS_GROUP}<number>', "
            f"found {[PARTS_GROUP + suffix for suffix in suffixes]}"
        )
    return int(suffixes[0])


def _release(version: str) -> tuple[int, ...]:
    """
    Return the leading release numbers of a version string.

    ``"1.3.0rc1"`` is ``(1, 3, 0)`` and ``"2.5.3"`` is ``(2, 5, 3)``; a string
    that does not start with a number is ``()``. Enough to tell which side of
    a release a version is on; pre-release ordering is not needed for that.
    """
    match = re.match(r"\d+(?:\.\d+)*", version.strip())
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def _compatibility_findings(found: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """
    Judge whether the installed partial distributions work with the core.

    Parameters
    ----------
    found : mapping of str to str
        Result of :func:`installed`.

    Returns
    -------
    problems : list of str
        One message for each part that cannot be relied on with this core.
    notes : list of str
        Observations that need no action.

    Notes
    -----
    **Developer.** The rule, for each installed part other than the core:

    ======================  ====================  ==========================
    declares a core API     version equals core   result
    ======================  ====================  ==========================
    yes, equal to ours      (any)                 compatible; a note if the
                                                  versions differ
    yes, different          (any)                 problem
    no                      yes                   compatible (same build)
    no                      no                    problem: cannot be judged
    ======================  ====================  ==========================

    Equal versions were the only test before the number existed, and every
    mix of versions was reported as a problem. That refused combinations that
    work (a part left behind by an upgrade of another) and could not tell
    them from the ones that do not.
    """
    problems: list[str] = []
    notes: list[str] = []
    if CORE not in found:
        return problems, notes
    core_version = found[CORE]
    behind: list[str] = []
    for dist in DISTRIBUTIONS:
        if dist.name == CORE or dist.name not in found:
            continue
        version = found[dist.name]
        try:
            declared = declared_core_api(dist.name)
        except ValueError as error:
            problems.append(
                f"{error}. Its metadata was not written by this project's build "
                f"tooling. Fix: pip install --force-reinstall --no-deps {dist.name}"
            )
            continue
        if declared is None:
            if version != core_version:
                problems.append(
                    f"{dist.name} {version} does not state which core API it was "
                    f"built for, and {CORE} is {core_version}: whether they work "
                    "together cannot be judged. "
                    f"Fix: pip install --upgrade {dist.name} {CORE}"
                )
        elif declared != CORE_API:
            newer = dist.name if declared > CORE_API else CORE
            problems.append(
                f"{dist.name} {version} was built for core API {declared}, and "
                f"{CORE} {core_version} provides core API {CORE_API}; the two "
                "disagree about how the core and a part work together, so the "
                f"part may fail in ways that look unrelated. {newer} is the "
                "newer side. "
                f"Fix: pip install --upgrade {dist.name} {CORE}"
            )
        elif version != core_version:
            behind.append(f"{dist.name} {version}")
    if behind:
        notes.append(
            f"Mixed versions: {', '.join(behind)} with {CORE} {core_version}. "
            f"They state the same core API ({CORE_API}), so they are expected to "
            "work together; nothing needs to change. To align them anyway: "
            f"pip install --upgrade {CORE} "
            + " ".join(entry.split(" ")[0] for entry in behind)
        )
    return problems, notes


def _third_party_findings() -> tuple[list[str], list[str]]:
    """
    Report third-party versions that matter for how the parts are supported.

    Returns
    -------
    problems : list of str
        Installed combinations known not to work, each with a fix.
    notes : list of str
        Installed versions that are supported although an installer may say
        otherwise, with the reason.

    Notes
    -----
    **User.** The project *asks* installers for NumPy 2 on Python 3.9 and
    newer, and for scikit-learn 1.3 or newer. NumPy 1.x installed afterwards
    (``pip install "numpy<2"``) keeps working; ``pip check`` will name the
    declared requirement, which is expected.

    **Developer.** Versions are read from installed metadata; neither package
    is imported, so this is safe to call where they are broken. The pairs in
    ``_INCOMPATIBLE_PAIRS`` were measured on Python 3.10 with NumPy 2.0.0:
    scikit-learn 1.3.0rc1 and 1.3.0 install and fail on import, 1.3.2 to 1.4.1
    are refused by the installer, 1.4.2 is the first that imports.
    """
    problems: list[str] = []
    notes: list[str] = []
    versions = {name: _installed_version(name) for name in ("numpy", "scikit-learn")}
    for first, at_least, second, below, reason in _INCOMPATIBLE_PAIRS:
        one, two = versions.get(first), versions.get(second)
        if one is None or two is None:
            continue
        if _release(one) >= at_least and _release(two) < below:
            problems.append(
                f"{first} {one} with {second} {two}: {reason} "
                f'Fix: pip install --upgrade "{second}>='
                f'{".".join(str(part) for part in below)}"  (or, to stay on '
                f'{second} {two}: pip install "{first}<{at_least[0]}")'
            )
    numpy_version = versions["numpy"]
    if numpy_version is not None and _release(numpy_version) < (2,):
        notes.append(
            f"numpy {numpy_version}: NumPy 1.x is supported. The project asks "
            "installers for NumPy 2 on Python 3.9 and newer, so `pip check` "
            "reports that requirement; it is a preference for new "
            "environments, not a limit of the code."
        )
    return problems, notes


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
        ``core_api``
            :data:`CORE_API`, the contract number this core provides.
        ``problems``
            Actionable messages; empty when the installation is coherent.
        ``notes``
            Observations that need no action: a mix of versions that is
            compatible, a third-party version that is supported although an
            installer reports a conflict.

    Notes
    -----
    **User.** Four situations are reported as problems, each with the command
    that fixes it:

    * the full distribution installed beside partial ones, where both own the
      same files and uninstalling either removes files the other needs;
    * a partial distribution built for another core API than the installed
      core provides (different *versions* alone are a note, not a problem:
      see ``_compatibility_findings``);
    * third-party versions that are known not to work together; and
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

    notes: list[str] = []
    for finder in (lambda: _compatibility_findings(found), _third_party_findings):
        more_problems, more_notes = finder()
        problems += more_problems
        notes += more_notes

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
        "core_api": CORE_API,
        "problems": problems,
        "notes": notes,
    }


def log_report(logger: object = None) -> dict[str, object]:
    """
    Write the installation report to a logger and return it.

    Parameters
    ----------
    logger : logging.Logger, optional
        Where to write. The ``"scikitplot"`` logger of the standard library
        when omitted.

    Returns
    -------
    dict
        The result of :func:`report`.

    See Also
    --------
    report : The same information as data.

    Notes
    -----
    **User.** Call this when something fails in a way that looks unrelated to
    what you did (an ``ImportError`` from NumPy, a missing attribute in a part
    you just upgraded). Each *problem* is logged at ``WARNING`` and ends with
    the command that fixes it; each *note* is logged at ``INFO`` and needs no
    action. ``scikitplot doctor`` prints the same report from a shell.

    How conflicts arise, and how to avoid them:

    * Install the full distribution **or** partial ones in an environment,
      not both: they own the same files.
    * Upgrade the parts together (``pip install --upgrade scikit-plots-skinny
      scikit-plots-annoy ...``). A mix of versions is accepted while the core
      API number is the same, and is reported as a note; it becomes a problem
      only when the number differs.
    * NumPy 1.x installed after the fact keeps working. NumPy 2 needs
      scikit-learn 1.4.2 or newer; with an older scikit-learn, upgrade it or
      stay on NumPy 1.

    **Developer.** Nothing is logged for a coherent installation except one
    ``DEBUG`` line, so this is safe to call from application start-up. The
    logger is looked up by name with the standard library because this module
    must stay importable by file path, outside the package.

    Examples
    --------
    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)
    >>> result = log_report()
    >>> sorted(result)
    ['available', 'core_api', 'flavor', 'installed', 'notes', 'problems']
    """
    import logging  # noqa: PLC0415

    target = logging.getLogger(IMPORT_NAME) if logger is None else logger
    result = report()
    for problem in result["problems"]:
        target.warning("scikitplot installation problem: %s", problem)
    for note in result["notes"]:
        target.info("scikitplot installation note: %s", note)
    target.debug(
        "scikitplot installation: flavor=%s core_api=%s installed=%s",
        result["flavor"],
        result["core_api"],
        result["installed"],
    )
    return result
