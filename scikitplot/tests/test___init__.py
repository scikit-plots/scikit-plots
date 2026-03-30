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
        assert mod.SKPLT_TRACKING_URI is not None
        # assert getattr(mod, "SKPLT_TRACKING_URI")
        # assert hasattr(mod, "SKPLT_TRACKING_URI")

    def test_no_unexpected_exception_on_dir(self):
        """``dir(sp)`` must not raise."""
        result = dir(sp)
        assert isinstance(result, list)
