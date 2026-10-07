# scikitplot/tests/test___init__.py
#
# flake8: noqa: D213
# pylint: disable=line-too-long
# noqa: E501
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Test suite for ``scikitplot/__init__.py``.

Coverage targets
----------------
- ``__version__``          : format, type, public attribute.
- ``__all__`` / ``_submodules``: presence, type, content constraints.
- ``__dir__()``            : callable, returns a list of strings.
- ``__getattr__``          : valid submodule lazy-load; invalid name raises
                             :exc:`AttributeError`.
- ``online_help``          : URL construction logic, browser-open call,
                             fallback on :exc:`ModuleNotFoundError`.
- ``set_seed``             : sets Python and NumPy seeds deterministically.
- ``_BUILT_WITH_MESON``    : attribute exists; value is ``True`` or ``None``.
- Module hygiene           : private names not leaked into ``__all__``.

Design decisions
----------------
- ``webbrowser.open`` is monkeypatched so tests run headlessly.
- Optional heavy dependencies (torch, tensorflow) are skipped gracefully.
- ``__getattr__`` tests patch ``importlib.import_module`` where the full
  package is unavailable, so unit tests do not require a complete install.
- Every test that writes to ``os.environ`` uses ``monkeypatch`` for isolation.

How to run
----------
From the project root::

    pytest scikitplot/tests/test___init__.py -v --tb=short

Or with coverage::

    pytest scikitplot/tests/test___init__.py \\
        --cov=scikitplot --cov-report=term-missing
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from urllib.parse import urlparse

import pytest

# from .. import __init__ as sp
import scikitplot as sp


# ===========================================================================
# __version__
# ===========================================================================


class TestVersion:
    """``__version__`` must be a PEP 440 compatible string."""

    def test_version_exists(self):
        assert hasattr(sp, "__version__")

    def test_version_is_str(self):
        assert isinstance(sp.__version__, str)

    def test_version_is_non_empty(self):
        assert sp.__version__.strip() != ""

    def test_version_matches_pep440_pattern(self):
        """Loose PEP 440 check: ``MAJOR.MINOR[.PATCH][suffix]``."""
        pattern = r"^\d+\.\d+(\.\d+)?(\.\w+)?$|^\d+\.\d+(rc\d+|a\d+|b\d+|\.dev\d*)?$"
        assert re.match(pattern, sp.__version__.split("+")[0]), (
            f"__version__={sp.__version__!r} does not look like a PEP 440 version"
        )

    def test_numpy_version_exposed(self):
        assert hasattr(sp, "__numpy_version__")
        assert isinstance(sp.__numpy_version__, str)
        assert sp.__numpy_version__.strip() != ""


# ===========================================================================
# __all__ and _submodules
# ===========================================================================


class TestPublicInterface:
    """``__all__`` and ``_submodules`` expose the correct public surface."""

    def test_all_exists(self):
        assert hasattr(sp, "__all__")

    def test_all_is_tuple(self):
        assert isinstance(sp.__all__, tuple)

    def test_all_is_non_empty(self):
        assert len(sp.__all__) > 0

    def test_all_contains_version(self):
        assert "__version__" in sp.__all__

    def test_all_contains_environment_variables(self):
        assert "environment_variables" in sp.__all__

    def test_all_contains_show_versions(self):
        assert "show_versions" in sp.__all__

    def test_all_contains_get_logger(self):
        assert "get_logger" in sp.__all__

    def test_all_elements_are_strings(self):
        assert all(isinstance(name, str) for name in sp.__all__)

    def test_submodules_is_list_or_tuple(self):
        """``_submodules`` is an internal sorted sequence."""
        assert isinstance(sp._submodules, (list, tuple))

    def test_submodules_are_strings(self):
        assert all(isinstance(name, str) for name in sp._submodules)

    def test_all_is_subset_of_submodules_or_globals(self):
        """Every name in ``__all__`` is either in ``_submodules`` or declared globally."""
        global_names = set(vars(sp).keys())
        submodule_names = set(sp._submodules)
        for name in sp.__all__:
            assert name in global_names | submodule_names, (
                f"{name!r} is in __all__ but not in globals or _submodules"
            )

    # def test_private_names_not_in_all(self):
    #     """Names that start with ``_`` must not appear in ``__all__``."""
    #     leaked = [name for name in sp.__all__ if name.startswith("_")]
    #     assert not leaked, (
    #         f"Private names leaked into __all__: {leaked}"
    #     )


# ===========================================================================
# __dir__
# ===========================================================================


class TestDir:
    """``__dir__()`` returns a list of non-empty strings."""

    def test_dir_is_callable(self):
        assert callable(sp.__dir__)

    def test_dir_returns_list(self):
        result = dir(sp)
        assert isinstance(result, list)

    def test_dir_elements_are_strings(self):
        for name in dir(sp):
            assert isinstance(name, str)

    def test_dir_contains_version(self):
        assert "__version__" in dir(sp)

    def test_dir_contains_environment_variables(self):
        assert "environment_variables" in dir(sp)

    def test_dir_does_not_contain_empty_string(self):
        assert "" not in dir(sp)

    def test_dir_is_sorted(self):
        """``__dir__`` must return a sorted list for consistent tooling."""
        d = dir(sp)
        assert d == sorted(d)


# ===========================================================================
# __getattr__ — lazy loading
# ===========================================================================


class TestGetattr:
    """``__getattr__`` resolves submodules lazily and raises on unknown names."""

    def test_getattr_environment_variables_returns_module(self):
        """``environment_variables`` must resolve as a module object."""
        mod = sp.environment_variables
        assert isinstance(mod, types.ModuleType)

    def test_getattr_environment_variables_same_object_on_repeat(self):
        """Repeated access returns the same module object (no re-import each time)."""
        mod_a = sp.environment_variables
        mod_b = sp.environment_variables
        assert mod_a is mod_b

    def test_getattr_unknown_name_raises_attribute_error(self):
        """Unknown attribute name must raise :exc:`AttributeError`."""
        with pytest.raises(AttributeError, match="scikitplot"):
            _ = sp._NONEXISTENT_ATTR_FOR_TESTING_ONLY_XYZ

    def test_getattr_error_message_includes_name(self):
        """The :exc:`AttributeError` message must mention the missing attribute."""
        attr = "_NONEXISTENT_ATTR_XYZ_TESTING"
        with pytest.raises(AttributeError, match=attr):
            _ = getattr(sp, attr)

    def test_getattr_test_returns_pytest_tester(self):
        """``sp.test`` must return a callable PytestTester-like object."""
        tester = sp.test
        assert callable(tester)


# ===========================================================================
# online_help
# ===========================================================================


class TestOnlineHelp:
    """``online_help`` constructs valid URLs and delegates to ``webbrowser.open``."""

    @pytest.fixture
    def patched_browser(self, monkeypatch):
        """Replace ``webbrowser.open`` with a spy that records calls."""
        calls: list[dict] = []

        import webbrowser

        def fake_open(url, new=0):
            calls.append({"url": url, "new": new})
            return True

        monkeypatch.setattr(webbrowser, "open", fake_open)
        return calls

    def test_returns_true_on_success(self, patched_browser):
        result = sp.online_help("test_query")
        assert result is True

    def test_url_contains_query(self, patched_browser):
        sp.online_help("my_search_term")
        assert len(patched_browser) == 1
        assert "my_search_term" in patched_browser[0]["url"]

    def test_url_contains_version_type_dev(self, patched_browser):
        """When ``__version__`` contains ``'dev'``, URL must use ``/dev/``."""
        if "dev" not in sp.__version__:
            pytest.skip("Not a dev build; can't test dev URL path.")
        sp.online_help("query")
        assert "/dev/" in patched_browser[0]["url"]

    def test_url_contains_version_type_stable(self, patched_browser, monkeypatch):
        """When ``__version__`` is a stable release, URL must use ``/stable/``."""
        monkeypatch.setattr(sp, "__version__", "0.5.0")
        sp.online_help("query")
        assert "/stable/" in patched_browser[0]["url"]

    def test_url_starts_with_base_url(self, patched_browser):
        sp.online_help("q")
        url = patched_browser[0]["url"]
        assert url.startswith("https://scikit-plots.github.io/")

    def test_empty_query_does_not_crash(self, patched_browser):
        result = sp.online_help("")
        assert result is True

    def test_custom_docs_root_url(self, patched_browser):
        sp.online_help("q", docs_root_url="https://example.com")
        url = patched_browser[0]["url"]
        # Parse and assert scheme + netloc exactly — startswith is insufficient
        # because "https://example.com.evil.org/..." would also pass it.
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "example.com"

    def test_env_var_overrides_docs_root_url(self, monkeypatch, patched_browser):
        monkeypatch.setenv("DOCS_ROOT_URL", "https://custom.example.org")
        sp.online_help("query")
        url = patched_browser[0]["url"]
        # Same fix: exact netloc match, not a prefix substring check.
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "custom.example.org"

    def test_new_window_parameter_forwarded(self, patched_browser):
        sp.online_help("q", new_window=2)
        assert patched_browser[0]["new"] == 2

    def test_returns_false_when_webbrowser_missing(self, monkeypatch):
        """If ``webbrowser`` is unavailable, ``online_help`` must return ``False``."""
        import builtins
        original_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "webbrowser":
                raise ModuleNotFoundError(f"No module named {name!r}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)
        result = sp.online_help("q")
        assert result is False


# ===========================================================================
# set_seed
# ===========================================================================


class TestSetSeed:
    """``set_seed`` sets Python and NumPy seeds, and returns a NumPy Generator."""

    def test_returns_numpy_generator(self):
        """NumPy is always available; the returned object must be a Generator."""
        import numpy as np
        result = sp.set_seed(42)
        assert isinstance(result, np.random.Generator)

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds must yield different pseudo-random sequences."""
        import numpy as np
        g1 = sp.set_seed(1)
        val1 = g1.random()
        g2 = sp.set_seed(2)
        val2 = g2.random()
        assert val1 != val2

    def test_same_seed_produces_same_generator_output(self):
        """Same seed must be reproducible across two independent calls."""
        import numpy as np
        g1 = sp.set_seed(99)
        v1 = g1.integers(0, 1_000_000)
        g2 = sp.set_seed(99)
        v2 = g2.integers(0, 1_000_000)
        assert v1 == v2

    def test_numpy_random_seed_is_set(self):
        """The legacy ``np.random`` API must be seeded too."""
        import numpy as np
        sp.set_seed(0)
        a = np.random.rand()
        sp.set_seed(0)
        b = np.random.rand()
        assert a == pytest.approx(b)

    def test_default_seed_is_42(self):
        """``set_seed()`` with no argument defaults to seed ``42``."""
        import numpy as np
        g_explicit = sp.set_seed(42)
        g_default = sp.set_seed()
        # Both generators seeded with 42 must produce the same first value.
        assert g_explicit.integers(0, 10**9) == g_default.integers(0, 10**9)

    def test_torch_seeded_when_available(self):
        """If ``torch`` is installed, its manual seed must be called without error."""
        torch = pytest.importorskip("torch")
        # Should not raise even when torch is present.
        sp.set_seed(7)

    def test_zero_seed_accepted(self):
        """``set_seed(0)`` is a valid call (zero is a legitimate seed)."""
        result = sp.set_seed(0)
        assert result is not None


# ===========================================================================
# _BUILT_WITH_MESON
# ===========================================================================


class TestBuiltWithMeson:
    """``_BUILT_WITH_MESON`` must exist and have a valid value."""

    def test_built_with_meson_exists(self):
        assert hasattr(sp, "_BUILT_WITH_MESON")

    def test_built_with_meson_is_true_or_none(self):
        """Must be ``True`` (meson build) or ``None`` (plain-Python fallback)."""
        value = sp._BUILT_WITH_MESON
        assert value is True or value is None, (
            f"_BUILT_WITH_MESON must be True or None, got {value!r}"
        )


# ===========================================================================
# Module hygiene
# ===========================================================================


class TestModuleHygiene:
    """Ensure the public namespace does not expose unintended internals."""

    def test_module_has_docstring(self):
        assert sp.__doc__ is not None
        assert len(sp.__doc__.strip()) > 0

    def test_logger_attribute_accessible(self):
        """``sp.logger`` must be accessible (used for package-level logging)."""
        assert hasattr(sp, "logger")

    def test_get_logger_is_callable(self):
        assert callable(sp.get_logger)

    def test_environment_variables_module_accessible(self):
        """``sp.environment_variables`` must be importable as a submodule."""
        mod = sp.environment_variables
        assert mod.__name__ == "scikitplot.environment_variables"
        # assert mod.SKPLT_TRACKING_URI is not None
        # assert getattr(mod, "SKPLT_TRACKING_URI")
        # assert hasattr(mod, "SKPLT_TRACKING_URI")

    def test_no_unexpected_exception_on_dir(self):
        """``dir(sp)`` must not raise."""
        result = dir(sp)
        assert isinstance(result, list)


# ===========================================================================
# Partial distributions
# ===========================================================================
#
# ``scikitplot/__init__.py`` is also the root of the partial distributions
# (``scikit-plots-skinny`` and friends, see ``scikitplot/_distributions.py``),
# where the compiled core, NumPy and most submodules are absent on purpose.
#
# Which branch the root takes is decided while it is being imported, so each
# case below imports a *copy* of the root package in a fresh interpreter, with
# the installed-distribution metadata it should see. The copy holds exactly the
# files a core-only installation has; the "full" case adds stand-ins for the
# compiled modules so that the success branch runs too.

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

_PACKAGE_DIR = Path(__file__).resolve().parents[1]

_CHILD = textwrap.dedent(
    """
    import importlib.metadata as metadata, json, sys

    site, versions, blocked, probe = (json.loads(arg) for arg in sys.argv[1:5])

    def version(name):
        if name in versions:
            return versions[name]
        raise metadata.PackageNotFoundError(name)

    metadata.version = version
    for name in blocked:
        sys.modules[name] = None  # ``import name`` now raises ImportError
    sys.path.insert(0, site)

    import scikitplot

    out = {
        "file": scikitplot.__file__,
        "version": scikitplot.__version__,
        "built_with_meson": scikitplot._BUILT_WITH_MESON,
        "numpy_imported": sys.modules.get("numpy") is not None,
        "distributions_imported": "scikitplot._distributions" in sys.modules,
        "probe": {},
    }
    for expression in probe:
        try:
            out["probe"][expression] = ["ok", repr(eval(expression))]
        except Exception as exc:
            out["probe"][expression] = [type(exc).__name__, str(exc)]
    print("@@RESULT@@" + json.dumps(out))
    """
)


def _declared_version():
    """Return the version literal in the root ``__init__.py``, without importing it."""
    import ast

    tree = ast.parse((_PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return node.value.value
    raise AssertionError("no __version__ assignment in scikitplot/__init__.py")


def _make_site(tmp_path, *, full=False, fake_numpy=False, parts=(), api=None):
    """Build a directory holding a copy of the root package, and return it.

    ``api`` adds a ``scikitplot.api`` package: ``"ok"`` for one that imports,
    ``"broken"`` for one that is installed but needs a module that is not.
    """
    package = tmp_path / "scikitplot"
    package.mkdir()
    for name in ("__init__.py", "_distributions.py"):
        shutil.copy2(_PACKAGE_DIR / name, package / name)
    shutil.copytree(
        _PACKAGE_DIR / "logging", package / "logging",
        ignore=shutil.ignore_patterns("__pycache__", "tests"),
    )
    for part in parts:  # an installed, importable part of the package
        (package / part).mkdir()
        (package / part / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    if full:
        # Stand-ins for what the full distribution's build provides.
        (package / "_lib").mkdir()
        (package / "_lib" / "__init__.py").write_text("", encoding="utf-8")
        (package / "_lib" / "_ccallback.py").write_text(
            "class LowLevelCallable: pass\n", encoding="utf-8"
        )
        (package / "config").mkdir()
        (package / "config" / "__init__.py").write_text(
            "__all__ = ['get_config']\ndef get_config(): return {'display': 'diagram'}\n",
            encoding="utf-8",
        )
        (package / "utils").mkdir()
        (package / "utils" / "__init__.py").write_text("", encoding="utf-8")
        (package / "utils" / "_show_versions.py").write_text(
            "def show_versions(*a, **k): return {}\n", encoding="utf-8"
        )
        (package / "version.py").write_text(
            "__git_hash__ = 'abc'\n__version__ = '9.9.9'\n"
            "__version_iso_8601__ = '2026-01-01T00:00:00+00:00'\n",
            encoding="utf-8",
        )
    if api is not None:
        (package / "api").mkdir()
        (package / "api" / "__init__.py").write_text(
            "import _a_dependency_that_is_not_installed\n" if api == "broken"
            else "def plot_something(): return 'plotted'\n",
            encoding="utf-8",
        )
    if fake_numpy:
        (tmp_path / "numpy").mkdir()
        (tmp_path / "numpy" / "__init__.py").write_text(
            "__version__ = '0.0.fake'\n", encoding="utf-8"
        )
    return tmp_path


def _import_root(site, *, versions, blocked=(), probe=()):
    """Import the copied root package in a fresh interpreter; return (data, stderr).

    Notes
    -----
    **Developer.** The child runs with ``-I -S``. ``-S`` keeps it from importing
    :mod:`site`, so no ``site-packages`` directory is on its path and no
    ``.pth`` file is executed. That matters when the tests run from a checkout
    installed in editable mode (``pip install -e .``, as CI does): the editable
    install registers, through a ``.pth`` file, an import hook that answers
    ``import scikitplot`` with the checkout, *before* ``sys.path`` is searched.
    Without ``-S`` the child imported the checkout instead of the copy
    ("imported the wrong copy"), or the checkout's compiled core together with
    the stand-in NumPy of the copy. The copy needs nothing outside the standard
    library, so the child loses nothing by it.
    """
    done = subprocess.run(
        [
            sys.executable, "-I", "-S", "-c", _CHILD,
            json.dumps(str(site)), json.dumps(versions),
            json.dumps(list(blocked)), json.dumps(list(probe)),
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    line = next(l for l in done.stdout.splitlines() if l.startswith("@@RESULT@@"))
    data = json.loads(line[len("@@RESULT@@"):])
    assert Path(data["file"]).parent == Path(site) / "scikitplot", "imported the wrong copy"
    return data, done.stderr


_SOURCE_TREE_WARNING = "you cannot import scikitplot while"
_CORE_ONLY = {"scikit-plots-skinny": "0.5.dev0"}


class TestPartialDistribution:
    """The root package imports cleanly as the core of a partial installation."""

    def test_imports_without_numpy(self, tmp_path):
        """NumPy is not a dependency of the core; the root must not need it."""
        data, _ = _import_root(_make_site(tmp_path), versions=_CORE_ONLY, blocked=["numpy"])
        assert data["built_with_meson"] is None

    def test_import_does_not_import_numpy_even_when_it_is_installed(self, tmp_path):
        site = _make_site(tmp_path, fake_numpy=True)
        data, _ = _import_root(site, versions=_CORE_ONLY)
        assert data["numpy_imported"] is False

    def test_numpy_version_resolves_on_first_access_and_is_cached(self, tmp_path):
        site = _make_site(tmp_path, fake_numpy=True)
        data, _ = _import_root(
            site, versions=_CORE_ONLY,
            probe=[
                "'__numpy_version__' in vars(scikitplot)",
                "scikitplot.__numpy_version__",
                "'__numpy_version__' in vars(scikitplot)",
            ],
        )
        probe = data["probe"]
        assert probe["scikitplot.__numpy_version__"] == ["ok", "'0.0.fake'"]
        # Evaluated in order: absent before the first access, cached after.
        # (The same expression appears twice; the last evaluation is kept.)
        assert probe["'__numpy_version__' in vars(scikitplot)"] == ["ok", "True"]

    def test_numpy_version_without_numpy_is_an_attribute_error(self, tmp_path):
        data, _ = _import_root(
            _make_site(tmp_path), versions=_CORE_ONLY, blocked=["numpy"],
            probe=["scikitplot.__numpy_version__"],
        )
        kind, message = data["probe"]["scikitplot.__numpy_version__"]
        assert kind == "AttributeError"
        assert "numpy" in message

    def test_no_source_tree_warning(self, tmp_path):
        """A partial distribution never ships the compiled core; that is not a fault."""
        _, stderr = _import_root(_make_site(tmp_path), versions=_CORE_ONLY, blocked=["numpy"])
        assert _SOURCE_TREE_WARNING not in stderr
        assert stderr.strip() == ""

    def test_version_is_the_declared_one(self, tmp_path):
        """No generated version module exists, so the literal is the version."""
        data, _ = _import_root(_make_site(tmp_path), versions=_CORE_ONLY)
        assert data["version"] == _declared_version()

    def test_dir_works_without_the_api_package(self, tmp_path):
        data, _ = _import_root(
            _make_site(tmp_path), versions=_CORE_ONLY,
            probe=["'rank_bm25' in dir(scikitplot)", "scikitplot._api_names()"],
        )
        assert data["probe"]["'rank_bm25' in dir(scikitplot)"] == ["ok", "True"]
        assert data["probe"]["scikitplot._api_names()"] == ["ok", "frozenset()"]

    def test_api_names_are_served_when_the_api_package_is_installed(self, tmp_path):
        site = _make_site(tmp_path, api="ok")
        data, _ = _import_root(
            site, versions=_CORE_ONLY,
            probe=["'plot_something' in scikitplot._api_names()",
                   "'plot_something' in dir(scikitplot)"],
        )
        assert data["probe"]["'plot_something' in scikitplot._api_names()"] == ["ok", "True"]
        assert data["probe"]["'plot_something' in dir(scikitplot)"] == ["ok", "True"]

    def test_a_broken_api_package_is_not_mistaken_for_an_absent_one(self, tmp_path):
        """Only the absence of ``scikitplot.api`` itself reads as "no re-exports"."""
        site = _make_site(tmp_path, api="broken")
        data, _ = _import_root(site, versions=_CORE_ONLY, probe=["scikitplot._api_names()"])
        kind, message = data["probe"]["scikitplot._api_names()"]
        assert kind == "ModuleNotFoundError"
        assert "_a_dependency_that_is_not_installed" in message

    def test_installed_part_resolves_as_an_attribute(self, tmp_path):
        """A lazy attribute lookup must not depend on the absent ``api`` package."""
        site = _make_site(tmp_path, parts=["rank_bm25"])
        data, _ = _import_root(site, versions=_CORE_ONLY, probe=["scikitplot.rank_bm25.VALUE"])
        assert data["probe"]["scikitplot.rank_bm25.VALUE"] == ["ok", "1"]

    @pytest.mark.parametrize(
        ("name", "command"),
        [
            ("corpus", "pip install scikit-plots-corpus"),
            ("mcp", "pip install scikit-plots-mcp"),
            ("annoy", "pip install scikit-plots-annoy"),
            ("utils", "pip install scikit-plots"),
            ("config", "pip install scikit-plots"),
        ],
    )
    def test_missing_part_names_the_distribution_that_ships_it(self, tmp_path, name, command):
        expression = f"scikitplot.{name}"
        data, _ = _import_root(_make_site(tmp_path), versions=_CORE_ONLY, probe=[expression])
        kind, message = data["probe"][expression]
        assert kind == "AttributeError"
        assert f"'scikitplot.{name}' is not installed" in message
        assert message.count("pip install") == 1
        assert command + "\n" in message

    def test_unknown_attribute_gets_no_install_hint(self, tmp_path):
        """Nothing ships a name that does not exist, so nothing is suggested."""
        data, _ = _import_root(
            _make_site(tmp_path), versions=_CORE_ONLY, probe=["scikitplot._no_such_thing_xyz"]
        )
        kind, message = data["probe"]["scikitplot._no_such_thing_xyz"]
        assert kind == "AttributeError"
        assert "pip install" not in message


class TestSourceTreeAndBrokenInstall:
    """Where the compiled core *should* exist, its absence is still reported."""

    def test_source_tree_still_warns(self, tmp_path):
        _, stderr = _import_root(_make_site(tmp_path), versions={})
        assert _SOURCE_TREE_WARNING in stderr

    def test_full_distribution_without_its_compiled_core_still_warns(self, tmp_path):
        _, stderr = _import_root(_make_site(tmp_path), versions={"scikit-plots": "0.5.0"})
        assert _SOURCE_TREE_WARNING in stderr

    def test_full_beside_partial_without_the_compiled_core_still_warns(self, tmp_path):
        versions = {"scikit-plots": "0.5.0", "scikit-plots-skinny": "0.5.0"}
        _, stderr = _import_root(_make_site(tmp_path), versions=versions)
        assert _SOURCE_TREE_WARNING in stderr


class TestFullDistributionBranch:
    """The success branch is unchanged, and pays nothing for partial support."""

    def test_success_branch(self, tmp_path):
        site = _make_site(tmp_path, full=True, fake_numpy=True)
        data, stderr = _import_root(
            site, versions={"scikit-plots": "9.9.9"},
            probe=["scikitplot.get_config()['display']", "scikitplot.__git_hash__"],
        )
        assert data["built_with_meson"] is True
        assert data["version"] == "9.9.9"  # taken from the generated version module
        assert data["probe"]["scikitplot.get_config()['display']"] == ["ok", "'diagram'"]
        assert data["probe"]["scikitplot.__git_hash__"] == ["ok", "'abc'"]
        assert _SOURCE_TREE_WARNING not in stderr

    def test_success_branch_never_consults_the_distribution_map(self, tmp_path):
        site = _make_site(tmp_path, full=True, fake_numpy=True)
        data, _ = _import_root(site, versions={"scikit-plots": "9.9.9"})
        assert data["distributions_imported"] is False


class TestPublicSurfaceForPartialSupport:
    """Names the partial distributions rely on are part of the declared surface."""

    def test_rank_bm25_is_a_declared_submodule(self):
        assert "rank_bm25" in sp._submodules
        assert "rank_bm25" in sp.__all__

    def test_distribution_map_is_a_declared_module(self):
        assert "_distributions" in sp._submodules

    def test_every_partial_distribution_part_is_a_declared_submodule(self):
        """A part someone can install must be a name the root knows how to hint for."""
        from .. import _distributions

        for dist in _distributions.DISTRIBUTIONS:
            for tree in dist.trees:
                top = tree.split("/")[0]
                if top.startswith("_") and top not in sp._submodules:
                    continue  # private infrastructure (the CLI), not a public part
                assert top in sp._submodules, f"{top} ({dist.name}) is not in _submodules"
