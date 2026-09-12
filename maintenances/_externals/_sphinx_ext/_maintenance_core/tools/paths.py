"""Portable path discovery for wide-repository and standalone ``_sphinx_ext`` layouts."""
from __future__ import annotations

from pathlib import Path

_RUNTIME_MARKERS = frozenset({
    "_extension_setup.py",
    "_sphinx_collection",
    "_sphinx_gallery_grid",
    "_sphinx_youtube_gallery",
    "_sphinxcontrib_youtube",
})

def _as_dir(start: str | Path) -> Path:
    path = Path(start).resolve()
    return path if path.is_dir() else path.parent

def is_runtime_sphinx_ext(path: str | Path) -> bool:
    """Return whether *path* is a real bundled runtime namespace, not a namesake."""
    path = Path(path)
    if path.name != "_sphinx_ext" or not (path / "__init__.py").is_file():
        return False
    present = sum((path / marker).exists() for marker in _RUNTIME_MARKERS)
    extension_children = sum(
        child.is_dir() and child.name.startswith(("_sphinx", "_pydata"))
        for child in path.iterdir()
    )
    return present >= 2 or ((path / "_extension_setup.py").is_file() and extension_children >= 1)

def discover_repository_root(start: str | Path) -> Path | None:
    """Find a wide scikit-plots root containing both runtime and maintenance planes."""
    base = _as_dir(start)
    for candidate in (base, *base.parents):
        runtime = candidate / "scikitplot" / "_externals" / "_sphinx_ext"
        maintenance = candidate / "maintenances" / "_externals" / "_sphinx_ext"
        if is_runtime_sphinx_ext(runtime) and maintenance.is_dir():
            return candidate
    return None

def discover_runtime_sphinx_ext(start: str | Path) -> Path:
    """Find the runtime ``_sphinx_ext`` tree in a wide repo or standalone archive."""
    repo = discover_repository_root(start)
    if repo is not None:
        return repo / "scikitplot" / "_externals" / "_sphinx_ext"
    base = _as_dir(start)
    for candidate in (base, *base.parents):
        if is_runtime_sphinx_ext(candidate):
            return candidate
    raise FileNotFoundError("could not locate a marked runtime _sphinx_ext tree")

def discover_maintenance_family_root(start: str | Path) -> Path:
    """Find the family maintenance root without confusing runtime with maintenance."""
    repo = discover_repository_root(start)
    if repo is not None:
        return repo / "maintenances" / "_externals" / "_sphinx_ext"
    base = _as_dir(start)
    for candidate in (base, *base.parents):
        if candidate.name == "_sphinx_ext" and (candidate / "_maintenance_core").is_dir():
            return candidate
    raise FileNotFoundError("could not locate the _sphinx_ext maintenance family root")
