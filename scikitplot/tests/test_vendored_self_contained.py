"""
The package never imports ``pip``, and its vendored code imports itself.

Notes
-----
**User notes.** ``import scikitplot.datasets`` must work in an environment
that has no ``pip`` (one made by ``uv venv``, a slim container image, an
embedded interpreter), and must not change how the running process resolves
``distutils``.

**Developer notes.** ``pip`` is an installer with no importable API. Its
vendored packages (``pip._vendor``) are private, change with every release,
and are absent wherever pip is. Importing ``pip`` also has a process-wide
side effect on Python older than 3.12 when setuptools is installed: the
setuptools import hook removes every loaded ``distutils`` module so that pip
gets the standard-library one, and warns ``Setuptools is replacing
distutils``. Under this project's ``filterwarnings = error`` that warning
fails whichever test happens to trigger the import first, so the failure
moves with test order.

A copy of ``platformdirs`` taken from pip's tree carries imports of
``pip._vendor.platformdirs`` instead of imports of itself. The first test
rejects that for every source file of the package, at any depth, so a later
re-vendoring cannot bring it back unnoticed. The second runs the vendored
copy in a fresh interpreter under the conditions that expose it. Neither
depends on what the current session has already imported.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import textwrap

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
PLATFORMDIRS = PACKAGE / "externals" / "_platformdirs"

#: Top-level modules the package must not import, each with the reason.
FORBIDDEN = {
    "pip": "pip has no importable API; run it as a subprocess (python -m pip)",
}


def _imports(tree):
    """Yield ``(line, module)`` for every absolute import in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.lineno, node.module


def _forbidden_imports(source):
    """Return ``(line, module)`` for each forbidden import in ``source``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # A file Python cannot parse cannot be imported, so it imports nothing.
        return []
    return [
        (line, module)
        for line, module in _imports(tree)
        if module.split(".", 1)[0] in FORBIDDEN
    ]


def test_the_check_recognises_every_form_of_the_import():
    source = textwrap.dedent(
        """
        import pip
        import os, pip._internal.cli as cli
        from pip._vendor.platformdirs.unix import Unix
        def lazy():
            try:
                from pip import main
            except ImportError:
                pass
        from . import pip
        from .pip import x
        import pipeline
        from pip_tools import y
        text = "import pip"
        """
    )
    assert _forbidden_imports(source) == [
        (2, "pip"),
        (3, "pip._internal.cli"),
        (4, "pip._vendor.platformdirs.unix"),
        (7, "pip"),
    ]
    assert _forbidden_imports("def broken(:\n") == []


def test_no_source_file_imports_pip():
    assert PACKAGE.is_dir()
    found = []
    scanned = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        scanned += 1
        source = path.read_text(encoding="utf-8", errors="replace")
        if "pip" not in source:
            continue
        found.extend(
            f"{path.relative_to(PACKAGE.parent).as_posix()}:{line}: imports {module}"
            for line, module in _forbidden_imports(source)
        )
    # The scan must have looked at the package, vendored code included.
    assert scanned > 1
    assert (PLATFORMDIRS / "__init__.py").is_file()
    assert not found, "\n".join(
        [*found, *(f"{name}: {why}" for name, why in FORBIDDEN.items())]
    )


def test_vendored_platformdirs_runs_on_its_own_modules():
    # A fresh interpreter, warnings as errors, and ``distutils`` loaded first
    # where it exists: the state in which an import of pip fails loudly.
    program = textwrap.dedent(
        """
        import importlib.util, sys, warnings
        with warnings.catch_warnings():
            # Loading distutils may itself warn (deprecated since 3.10);
            # only what the vendored package does is under test.
            warnings.simplefilter("ignore")
            try:
                import distutils  # noqa: F401
            except ImportError:
                pass
        directory = sys.argv[1]
        name = "_vendored_platformdirs_under_test"
        spec = importlib.util.spec_from_file_location(
            name, directory + "/__init__.py", submodule_search_locations=[directory]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert module.PlatformDirs.__module__.startswith(name + "."), module.PlatformDirs
        assert module.AppDirs is module.PlatformDirs
        assert issubclass(module.PlatformDirs, module.PlatformDirsABC)
        cache = module.user_cache_dir("scikit-plots", appauthor=False)
        assert isinstance(cache, str) and "scikit-plots" in cache, cache
        assert "pip" not in sys.modules, "pip was imported"
        print("ok")
        """
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error", "-c", program, str(PLATFORMDIRS)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok", done.stdout
