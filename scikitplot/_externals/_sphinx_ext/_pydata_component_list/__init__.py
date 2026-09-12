"""
PyData Sphinx Theme component inventory extension.

The public ``component-list`` directive is intentionally kept separate from the
generic gallery engine because it reads PyData Sphinx Theme component templates
and links to that project's repository.
"""

from __future__ import annotations

__all__ = [  # ruff: ignore[undefined-export]
    "ComponentListDirective",
    "setup",
]


def __getattr__(name: str):
    if name in __all__:
        from .directive import (  # ruff: ignore[import-outside-top-level]
            ComponentListDirective,
            setup,
        )

        return {"ComponentListDirective": ComponentListDirective, "setup": setup}[name]
    raise AttributeError(name)
