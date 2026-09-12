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


REPOSITORY_ROOT = _find_repository_root()
MAINTENANCE_ROOT = (
    REPOSITORY_ROOT
    / "maintenances"
    / "_externals"
    / "_sphinx_ext"
    / "_sphinx_ai_assistant"
)
SKILL_ROOT = (
    REPOSITORY_ROOT
    / "skills"
    / "_externals"
    / "_sphinx_ext"
    / "_sphinx_ai_assistant"
)

__all__ = [
    "MAINTENANCE_ROOT",
    "REPOSITORY_ROOT",
    "RUNTIME_ROOT",
    "SKILL_ROOT",
    "TESTS_ROOT",
]
