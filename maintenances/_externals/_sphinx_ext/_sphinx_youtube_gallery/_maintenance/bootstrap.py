"""Activate the runtime namespace for maintenance tests in any supported layout."""
from __future__ import annotations
from pathlib import Path
import sys

def _family_root(start: str | Path) -> Path | None:
    base = Path(start).resolve()
    base = base if base.is_dir() else base.parent
    for candidate in (base, *base.parents):
        if candidate.name == "_sphinx_ext" and (candidate / "_maintenance_core" / "tools" / "paths.py").is_file():
            return candidate
    return None

def runtime_sphinx_ext(start: str | Path) -> Path:
    family = _family_root(start)
    if family is not None:
        tools = family / "_maintenance_core" / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        from paths import discover_runtime_sphinx_ext
        return discover_runtime_sphinx_ext(start)
    # Backward-compatible fallback for a standalone historical archive where
    # maintenance lived directly inside the runtime _sphinx_ext tree.
    base = Path(start).resolve()
    base = base if base.is_dir() else base.parent
    for candidate in (base, *base.parents):
        if (candidate.name == "_sphinx_ext" and (candidate / "__init__.py").is_file()
                and (candidate / "_sphinx_youtube_gallery").is_dir()
                and (candidate / "_sphinx_collection").is_dir()):
            return candidate
    raise FileNotFoundError("could not discover runtime _sphinx_ext for YouTube maintenance")

def activate(start: str | Path) -> Path:
    root = runtime_sphinx_ext(start)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return root
