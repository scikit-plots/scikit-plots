# libs/_tools/tests/conftest.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures: a miniature repository the tooling can be pointed at."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

#: A distribution map with the same shape as the real one, small enough to
#: reason about: a core that owns the root, and two parts.
MAP = textwrap.dedent(
    '''
    """Miniature distribution map for the tooling tests."""
    import re
    from typing import NamedTuple, Tuple

    IMPORT_NAME = "scikitplot"
    FULL = "scikit-plots"
    CORE = "scikit-plots-skinny"


    class Distribution(NamedTuple):
        name: str
        trees: Tuple[str, ...]
        files: Tuple[str, ...]
        summary: str


    DISTRIBUTIONS = (
        Distribution(CORE, ("logging",), ("__init__.py", "py.typed"), "The core."),
        Distribution("scikit-plots-alpha", ("alpha",), (), "Part alpha."),
        Distribution("scikit-plots-beta", ("beta", "shared/beta_only"), ("shared/__init__.py",), "Part beta."),
    )


    def canonicalize_name(name):
        return re.sub(r"[-_.]+", "-", name.strip()).lower()


    def get(name):
        for dist in DISTRIBUTIONS:
            if dist.name == canonicalize_name(name):
                return dist
        raise KeyError(f"unknown partial distribution {name!r}")
    '''
)

_FILES = {
    "LICENSE.txt": "Miniature licence.\n",
    "scikitplot/__init__.py": '__version__ = "1.2.dev3"\n',
    "scikitplot/py.typed": "",
    "scikitplot/conftest.py": "# owned by no partial distribution\n",
    "scikitplot/logging/__init__.py": "",
    "scikitplot/logging/_logging.py": "X = 1\n",
    "scikitplot/alpha/__init__.py": "",
    "scikitplot/alpha/_core.py": "A = 1\n",
    "scikitplot/alpha/data/table.json": "{}\n",
    "scikitplot/alpha/.hidden/config.json": "{}\n",
    "scikitplot/alpha/tests/__init__.py": "",
    "scikitplot/alpha/tests/test__core.py": "def test_a(): pass\n",
    "scikitplot/beta/__init__.py": "",
    "scikitplot/shared/__init__.py": "",
    "scikitplot/shared/beta_only/__init__.py": "",
    "scikitplot/shared/beta_only/native.cc": "// source\n",
    "scikitplot/shared/other/__init__.py": "# owned by no partial distribution\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return the root of a miniature repository with three lib directories."""
    for relative, content in _FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (tmp_path / "scikitplot" / "_distributions.py").write_text(MAP, encoding="utf-8")
    for directory in ("skinny", "alpha", "beta"):
        lib = tmp_path / "libs" / directory
        lib.mkdir(parents=True)
        (lib / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path
