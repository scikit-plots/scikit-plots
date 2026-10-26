# scikitplot/tests/test_usage_skills.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The usage skills of the ``scikitplot`` plugin match the installed package (Rule 165).

``plugins/scikitplot/skills/`` is generated from each listed module's
``__all__`` and docstrings by ``tools/maint_tools/generate_usage_skills.py``.
A change to a public API or a docstring makes this test fail until the skills
are regenerated with ``python tools/maint_tools/generate_usage_skills.py sync``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "tools/maint_tools/generate_usage_skills.py"

pytestmark = pytest.mark.skipif(
    not GENERATOR.is_file() or not (REPO / "plugins/scikitplot").is_dir(),
    reason="not a scikit-plots checkout (the generator and the plugin live there)",
)


def _generator():
    spec = importlib.util.spec_from_file_location("_generate_usage_skills", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_usage_skills_are_current():
    generator = _generator()
    try:
        wanted = generator.build(REPO)
    except generator.SkillError as exc:
        if "does not import" in str(exc) or "do not resolve here" in str(exc):
            pytest.skip(f"an optional dependency is missing here: {exc}")
        raise
    have = generator.current(REPO)
    stale = sorted(p for p in set(wanted) | set(have) if wanted.get(p) != have.get(p))
    assert stale == [], (
        "regenerate with `python tools/maint_tools/generate_usage_skills.py sync`: "
        f"{stale}"
    )


def test_every_public_subpackage_is_listed_or_excluded():
    """A new public subpackage gets a usage skill or a stated reason (Rule 165)."""
    generator = _generator()
    policy = generator.load_policy(REPO)
    known = set(policy["usage_modules"]) | set(policy["usage_excluded"])
    package = REPO / "scikitplot"
    public = {
        f"scikitplot.{p.name}"
        for p in package.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "__init__.py").is_file()
        and p.name not in ("tests",)
    }
    assert sorted(public - known) == []
