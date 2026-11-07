"""
The numpy include path ``meson.build`` computes is valid from the project root.

``include_directories()`` reads a relative path against the directory of the
``meson.build`` that calls it (the project root). The snippet that computes the
path is extracted from ``meson.build`` (between its BEGIN/END markers) and run
the way Meson runs it -- ``python -c <code> <project root>`` -- with a stand-in
``numpy`` whose ``get_include()`` is chosen by the test. Nothing is imported
from ``scikitplot`` and no Meson is needed.

Notes
-----
**Developer notes.** The earlier snippet changed into ``tools/`` and printed a
path relative to it, which is one ``..`` short from the project root: Meson
reported ``Include dir ../../../venv/... does not exist`` for an environment
outside the tree whenever the build directory was not a direct child of the
root (pip's ``build/cp3XY``). Live evidence, Meson 1.12.1 and a numpy outside
the tree, build directory ``build/cp312``: the old snippet fails at
``include_directories``; the new one configures and compiles a file that
includes ``numpy/_numpyconfig.h``; a virtualenv inside the tree gives the
relative ``.venv/lib/...`` path (an absolute one is refused by Meson).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
MESON = REPO / "meson.build"
_BLOCK = re.compile(
    r"# BEGIN numpy-include-dir snippet.*?\['-c', '''\\?\n(?P<code>.*?)''',.*?"
    r"# END numpy-include-dir snippet",
    re.DOTALL,
)


def _snippet():
    if not MESON.is_file():  # pragma: no cover - not a source checkout
        pytest.skip("meson.build is not available")
    match = _BLOCK.search(MESON.read_text(encoding="utf-8"))
    assert match, "the numpy-include-dir snippet markers are missing from meson.build"
    return match.group("code")


def _fake_numpy(directory, include):
    package = directory / "numpy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"def get_include():\n    return {str(include)!r}\n", encoding="utf-8"
    )
    return directory


def _run(code, root, *, pythonpath, cwd):
    env = dict(os.environ, PYTHONPATH=str(pythonpath))
    done = subprocess.run(  # noqa: S603 - fixed argv: this interpreter, the snippet, a temp path
        [sys.executable, "-c", code, str(root)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "work" / "deep" / "checkout"
    (root / "tools").mkdir(parents=True)
    return root


def test_numpy_outside_the_tree_is_given_as_an_absolute_path(tmp_path, tree):
    include = tmp_path / "venv" / "lib" / "numpy" / "_core" / "include"
    include.mkdir(parents=True)
    site = _fake_numpy(tmp_path / "venv" / "site", include)
    printed = _run(_snippet(), tree, pythonpath=site, cwd=tree)
    assert printed == os.path.realpath(include)


def test_numpy_inside_the_tree_is_relative_to_the_project_root(tree):
    include = tree / ".venv" / "lib" / "numpy" / "_core" / "include"
    include.mkdir(parents=True)
    site = _fake_numpy(tree / ".venv" / "site", include)
    printed = _run(_snippet(), tree, pythonpath=site, cwd=tree / "tools")
    assert not os.path.isabs(printed)
    # Resolved the way include_directories() resolves it: from the root.
    assert (tree / printed).resolve() == include.resolve()


def test_the_result_does_not_depend_on_the_working_directory(tmp_path, tree):
    include = tree / ".venv" / "inc"
    include.mkdir(parents=True)
    site = _fake_numpy(tmp_path / "site", include)
    printed = {
        _run(_snippet(), tree, pythonpath=site, cwd=where)
        for where in (tree, tree / "tools", tmp_path)
    }
    assert printed == {os.path.join(".venv", "inc")}


def test_a_symlinked_root_still_gives_a_relative_path(tmp_path, tree):
    include = tree / ".venv" / "inc"
    include.mkdir(parents=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(tree, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - no symlinks
        pytest.skip("symlinks are not available")
    site = _fake_numpy(tmp_path / "site", link / ".venv" / "inc")
    printed = _run(_snippet(), link, pythonpath=site, cwd=tmp_path)
    assert printed == os.path.join(".venv", "inc")


def test_modules_in_the_working_directory_cannot_shadow_numpy_or_the_stdlib(
    tmp_path, tree
):
    include = tmp_path / "real" / "include"
    include.mkdir(parents=True)
    site = _fake_numpy(tmp_path / "site", include)
    cwd = tmp_path / "cwd"
    _fake_numpy(cwd, tmp_path / "WRONG")  # a numpy package in the cwd
    (cwd / "signal.py").write_text("raise ImportError('shadowed')\n", encoding="utf-8")
    printed = _run(_snippet(), tree, pythonpath=site, cwd=cwd)
    assert printed == os.path.realpath(include)
