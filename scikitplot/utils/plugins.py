"""plugins.py."""

import sys as _sys

try:
    from importlib.metadata import EntryPoint, entry_points
except ImportError:
    from importlib_metadata import EntryPoint, entry_points  # Backport for <3.8 or 3.10


def _get_entry_points(group: str) -> list[EntryPoint]:
    if _sys.version_info >= (3, 10):
        return entry_points(group=group)

    eps = entry_points()
    try:
        return eps.get(group, [])
    except AttributeError:
        return eps.select(group=group)


def get_entry_points(group: str) -> list[EntryPoint]:
    """Return a list of registered entry points under the given group."""
    return _get_entry_points(group)
