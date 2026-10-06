# libs/_tools/staging.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Owned-file selection and build-time staging for the partial distributions.

A partial distribution (``libs/<name>``) ships a subset of the one
``scikitplot`` source tree at the repository root. This module answers the
only two questions the build needs answered, and answers them in one place:

1. *Which files does a distribution own?* (:func:`iter_owned_files`,
   :func:`check_ownership`)
2. *How do those files get in front of the build backend?* (:func:`stage`,
   :func:`unstage`)

Notes
-----
**User.** You never call this directly. ``pip install ./libs/<name>`` and
``python -m build libs/<name>`` run the generated ``setup.py``, which stages
the files, builds, and removes them again.

**Developer.** Staging copies files; it does not link them. A symbolic link to
the repository's ``scikitplot`` directory (the approach ``mlflow-skinny`` uses)
exposes the *whole* package to the build, so every distribution would need its
own filter to keep the others' files out, and on Windows git materialises the
link as a text file unless Developer Mode is on. A staged copy contains
exactly the owned files and nothing else, so the build configuration needs no
filter at all, and there is nothing platform specific in it.

Staging is ephemeral. ``setup.py`` stages before the build and unstages when
the interpreter exits, so a lib directory holds no copy of the source between
builds: nothing can go stale, and nothing is duplicated in a checkout, an
archive or a search.

This module is standard-library only and has no sibling imports, because the
generated ``setup.py`` loads it by file path, outside any package.
"""

from __future__ import annotations

import functools
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Iterable, Iterator, Sequence

__all__ = [
    "EXCLUDED_DIR_NAMES",
    "EXCLUDED_SUFFIXES",
    "LICENSE_NAME",
    "PACKAGE_NAME",
    "RESIDUE_DIR_NAMES",
    "check_ownership",
    "iter_owned_files",
    "load_distributions",
    "repo_root",
    "stage",
    "unstage",
]

#: The import package whose files are staged.
PACKAGE_NAME = "scikitplot"

#: The repository's licence file, staged beside each lib's ``pyproject.toml``.
LICENSE_NAME = "LICENSE.txt"

#: Directory names never shipped: interpreter caches.
EXCLUDED_DIR_NAMES = frozenset({"__pycache__"})

#: File suffixes never shipped: byte-code caches and native build outputs.
#: A native output left in the source tree by an in-place build belongs to the
#: interpreter and platform that built it, not to the wheel being built now.
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".so", ".pyd", ".dylib", ".o", ".obj")

#: Directories a build leaves in a lib directory. They are removed on stage and
#: on unstage: ``build/`` in particular keeps files from earlier builds, and
#: the wheel is assembled from it, so a stale ``build/`` ships stale files.
RESIDUE_DIR_NAMES = ("build",)

_RESIDUE_DIR_SUFFIX = ".egg-info"


def repo_root() -> Path:
    """
    Return the repository root this file belongs to.

    Returns
    -------
    pathlib.Path
        The directory that contains ``libs/`` and ``scikitplot/``.

    Raises
    ------
    FileNotFoundError
        If this file is not at ``<root>/libs/_tools/staging.py`` beside a
        ``<root>/scikitplot`` package, which means it is being used outside
        the repository.
    """
    root = Path(__file__).resolve().parents[2]
    marker = root / PACKAGE_NAME / "_distributions.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"{marker} not found: libs/_tools must be used from inside the "
            "scikit-plots repository."
        )
    return root


def load_distributions(root: Path | None = None) -> ModuleType:
    """
    Load ``scikitplot/_distributions.py`` by path, without importing ``scikitplot``.

    Parameters
    ----------
    root : pathlib.Path, optional
        Repository root. Defaults to :func:`repo_root`.

    Returns
    -------
    module
        The distribution map module.

    Raises
    ------
    ImportError
        If the file cannot be loaded as a module.

    Notes
    -----
    **Developer.** Importing ``scikitplot`` would run the package's
    ``__init__`` (and, in a full checkout, try to load compiled extensions).
    The map is plain data in a dependency-free file precisely so that it can
    be read this way.
    """
    root = repo_root() if root is None else Path(root)
    return _load_distributions(root.resolve())


@functools.lru_cache(maxsize=None)
def _load_distributions(root: Path) -> ModuleType:
    """Load the distribution map of ``root`` once per process."""
    path = root / PACKAGE_NAME / "_distributions.py"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImportError(f"cannot load the distribution map from {path}") from exc
    # Compiled and executed directly rather than imported through the import
    # system, which would write a byte-code cache into the source tree being
    # packaged.
    module = ModuleType("_skplt_distributions")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def _is_excluded(relative: PurePosixPath) -> bool:
    """Return whether a path is a cache or a native build output."""
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts):
        return True
    return relative.name.endswith(EXCLUDED_SUFFIXES)


def iter_owned_files(dist, package_dir: Path) -> Iterator[PurePosixPath]:
    """
    Yield every file a distribution owns, in a stable order.

    Parameters
    ----------
    dist : Distribution
        An entry of ``scikitplot._distributions.DISTRIBUTIONS``.
    package_dir : pathlib.Path
        The ``scikitplot`` package directory to read from.

    Yields
    ------
    pathlib.PurePosixPath
        File path relative to ``package_dir``, sorted, each exactly once.

    Raises
    ------
    FileNotFoundError
        If an owned tree or file does not exist. A map that names something
        absent is a map that is wrong, and a wheel silently missing a part is
        worse than a build that stops.
    ValueError
        If an owned file is also inside one of the distribution's own trees.
    """
    package_dir = Path(package_dir)
    seen: dict[PurePosixPath, None] = {}
    for tree in dist.trees:
        base = package_dir / tree
        if not base.is_dir():
            raise FileNotFoundError(
                f"{dist.name}: owned tree {PACKAGE_NAME}/{tree} does not exist"
            )
        for current, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIR_NAMES)
            for filename in sorted(filenames):
                relative = PurePosixPath(
                    (Path(current) / filename).relative_to(package_dir).as_posix()
                )
                if not _is_excluded(relative):
                    seen[relative] = None
    for name in dist.files:
        relative = PurePosixPath(name)
        if not (package_dir / name).is_file():
            raise FileNotFoundError(
                f"{dist.name}: owned file {PACKAGE_NAME}/{name} does not exist"
            )
        if relative in seen:
            raise ValueError(
                f"{dist.name}: {PACKAGE_NAME}/{name} is listed as a file but is "
                "already inside one of its trees"
            )
        seen[relative] = None
    yield from sorted(seen)


def check_ownership(distributions: Sequence, package_dir: Path) -> dict[str, list[str]]:
    """
    Assert that no file is owned by two distributions.

    Parameters
    ----------
    distributions : sequence of Distribution
        The distributions to check together.
    package_dir : pathlib.Path
        The ``scikitplot`` package directory to read from.

    Returns
    -------
    dict of str to list of str
        Distribution name to its owned files (POSIX paths relative to
        ``package_dir``).

    Raises
    ------
    ValueError
        If two distributions own the same file, or a name is repeated. The
        message names the file and both owners.

    Notes
    -----
    **Developer.** This is the invariant that makes mixed installs safe: an
    installer removes every file a distribution recorded when that
    distribution is uninstalled, so a shared file disappears from the
    distributions that still need it.
    """
    owners: dict[str, str] = {}
    owned: dict[str, list[str]] = {}
    for dist in distributions:
        if dist.name in owned:
            raise ValueError(f"distribution {dist.name} is declared twice")
        files = [path.as_posix() for path in iter_owned_files(dist, package_dir)]
        for path in files:
            if path in owners:
                raise ValueError(
                    f"{PACKAGE_NAME}/{path} is owned by both {owners[path]} "
                    f"and {dist.name}; a file must have exactly one owner"
                )
            owners[path] = dist.name
        owned[dist.name] = files
    return owned


def _is_link(path: Path) -> bool:
    """Return whether ``path`` is a symbolic link or a Windows junction."""
    if os.path.islink(path):
        return True
    try:
        attributes = os.lstat(path).st_file_attributes  # Windows only
    except (AttributeError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _remove(path: Path) -> None:
    """
    Remove a file, a directory tree, or a link, never following a link.

    Notes
    -----
    **Developer.** An earlier layout linked ``libs/<name>/scikitplot`` to the
    repository's own ``scikitplot`` directory. Deleting through such a link
    would delete the source tree, so a link is always removed as a link.
    """
    if _is_link(path):
        try:
            os.unlink(path)
        except OSError:
            os.rmdir(path)  # a directory link on Windows
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _validate_lib_dir(lib_dir: Path, root: Path) -> Path:
    """Return ``lib_dir`` resolved, after checking it is ``<root>/libs/<name>``."""
    lib_dir = Path(lib_dir).resolve()
    if lib_dir.parent != (root / "libs").resolve():
        raise ValueError(
            f"{lib_dir} is not a direct child of {root / 'libs'}; refusing to "
            "stage into, or remove files from, any other location"
        )
    if not (lib_dir / "pyproject.toml").is_file():
        raise ValueError(f"{lib_dir} has no pyproject.toml; it is not a lib directory")
    return lib_dir


def _residue(lib_dir: Path) -> Iterable[Path]:
    """Yield everything staging or a build may have left in ``lib_dir``."""
    yield lib_dir / PACKAGE_NAME
    yield lib_dir / LICENSE_NAME
    for name in RESIDUE_DIR_NAMES:
        yield lib_dir / name
    yield from sorted(lib_dir.glob("*" + _RESIDUE_DIR_SUFFIX))


def unstage(lib_dir: Path, root: Path | None = None) -> None:
    """
    Remove everything :func:`stage` and a build left in a lib directory.

    Parameters
    ----------
    lib_dir : pathlib.Path
        The ``libs/<name>`` directory.
    root : pathlib.Path, optional
        Repository root. Defaults to :func:`repo_root`.

    Raises
    ------
    ValueError
        If ``lib_dir`` is not ``<root>/libs/<name>``.

    Notes
    -----
    **Developer.** Idempotent: calling it on a clean directory does nothing.
    Built artefacts in ``dist/`` are kept; they are the product, not residue.
    """
    root = repo_root() if root is None else Path(root)
    lib_dir = _validate_lib_dir(lib_dir, root)
    for path in _residue(lib_dir):
        _remove(path)


def stage(name: str, lib_dir: Path, root: Path | None = None) -> list[Path]:
    """
    Copy a distribution's owned files, and the licence, into its lib directory.

    Parameters
    ----------
    name : str
        Distribution name in any spelling, e.g. ``"scikit-plots-rank-bm25"``.
    lib_dir : pathlib.Path
        The ``libs/<name>`` directory to stage into.
    root : pathlib.Path, optional
        Repository root. Defaults to :func:`repo_root`.

    Returns
    -------
    list of pathlib.Path
        The staged files, in a stable order.

    Raises
    ------
    KeyError
        If ``name`` is not a partial distribution.
    FileNotFoundError
        If an owned path, or the repository's ``LICENSE.txt``, is missing.
    ValueError
        If ``lib_dir`` is not ``<root>/libs/<name>``.

    Notes
    -----
    **Developer.** Idempotent and total: the previous stage and any build
    residue are removed first, so the result depends only on the source tree
    and the distribution map, never on what an earlier run left behind.
    """
    root = repo_root() if root is None else Path(root)
    lib_dir = _validate_lib_dir(lib_dir, root)
    dist = load_distributions(root).get(name)
    package_dir = root / PACKAGE_NAME
    license_file = root / LICENSE_NAME
    if not license_file.is_file():
        raise FileNotFoundError(
            f"{license_file} not found; every distribution must ship the licence"
        )
    # Resolve the whole file list before touching the disk, so a wrong map
    # stops the build without leaving a half-staged directory.
    relatives = list(iter_owned_files(dist, package_dir))

    unstage(lib_dir, root)
    staged: list[Path] = []
    for relative in relatives:
        target = lib_dir / PACKAGE_NAME / Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_dir / Path(*relative.parts), target)
        staged.append(target)
    shutil.copy2(license_file, lib_dir / LICENSE_NAME)
    staged.append(lib_dir / LICENSE_NAME)
    return staged
