"""plugins.py."""

import importlib as _importlib
import sys as _sys


def _get_entry_points(group: str) -> list[_importlib.metadata.EntryPoint]:
    if _sys.version_info >= (3, 10):
        return _importlib.metadata.entry_points(group=group)

    entrypoints = _importlib.metadata.entry_points()
    try:
        return entrypoints.get(group, [])
    except AttributeError:
        return entrypoints.select(group=group)


def get_entry_points(group: str) -> list[_importlib.metadata.EntryPoint]:
    """get_entry_points."""
    return _get_entry_points(group)
