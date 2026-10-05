"""Tests for :mod:`_extension_setup`: one import root per Sphinx application."""

from __future__ import annotations

import types

import pytest
from sphinx.errors import ExtensionError

from .._extension_setup import check_namespace

INSTALLED = "scikitplot._externals._sphinx_ext"
LOCAL = "_sphinx_ext"


def _app(configured=(), loaded=(), previous=None):
    app = types.SimpleNamespace(
        config=types.SimpleNamespace(extensions=list(configured)),
        extensions={name: object() for name in loaded},
    )
    if previous is not None:
        app._scikitplot_sphinx_extension_root = previous
    return app


class TestOneNamespace:
    @pytest.mark.parametrize("root", [INSTALLED, LOCAL])
    def test_a_single_root_is_accepted_and_recorded(self, root):
        app = _app([root + "._sphinx_gallery_grid", "sphinx.ext.autodoc"])
        check_namespace(app, root)
        assert app._scikitplot_sphinx_extension_root == root

    def test_unrelated_extensions_do_not_count_as_a_root(self):
        app = _app(["sphinx.ext.autodoc", "myst_parser"])
        check_namespace(app, LOCAL)
        assert app._scikitplot_sphinx_extension_root == LOCAL

    def test_a_second_call_with_the_same_root_is_idempotent(self):
        app = _app([LOCAL + "._sphinx_collection"])
        check_namespace(app, LOCAL)
        check_namespace(app, LOCAL)
        assert app._scikitplot_sphinx_extension_root == LOCAL


class TestMixedNamespaces:
    def test_configured_extensions_from_two_roots_are_refused(self):
        app = _app([INSTALLED + "._sphinx_gallery_grid", LOCAL + "._sphinx_collection"])
        with pytest.raises(ExtensionError, match="Mixed scikit-plots extension namespaces"):
            check_namespace(app, INSTALLED)

    def test_the_message_names_both_roots(self):
        app = _app([LOCAL + "._sphinx_collection"])
        with pytest.raises(ExtensionError) as caught:
            check_namespace(app, INSTALLED)
        assert INSTALLED in str(caught.value) and LOCAL in str(caught.value)

    def test_an_already_loaded_extension_counts(self):
        app = _app(loaded=[LOCAL + "._sphinx_collection"])
        with pytest.raises(ExtensionError, match="Mixed"):
            check_namespace(app, INSTALLED)

    def test_a_root_recorded_by_an_earlier_extension_counts(self):
        app = _app(previous=LOCAL)
        with pytest.raises(ExtensionError, match="Mixed"):
            check_namespace(app, INSTALLED)

    def test_a_refused_call_does_not_record_its_root(self):
        app = _app(previous=LOCAL)
        with pytest.raises(ExtensionError):
            check_namespace(app, INSTALLED)
        assert app._scikitplot_sphinx_extension_root == LOCAL


class TestRetiredNames:
    @pytest.mark.parametrize(
        ("name", "replacement"),
        [
            (LOCAL + ".youtube_catalog", "_sphinx_youtube_gallery"),
            (INSTALLED + ".youtube_catalog", "_sphinx_youtube_gallery"),
            (LOCAL + ".collection", "_sphinx_collection"),
            (LOCAL + "._pydata_sphinx_theme", "_sphinx_gallery_grid"),
            (LOCAL + "._pydata_sphinx_theme.gallery_grid", "_pydata_component_list"),
        ],
    )
    def test_a_retired_entry_is_refused_with_its_replacement(self, name, replacement):
        with pytest.raises(ExtensionError) as caught:
            check_namespace(_app([name]), LOCAL)
        assert replacement in str(caught.value)
        assert "rebuild with -E" in str(caught.value)

    def test_the_current_collection_name_is_not_mistaken_for_the_retired_one(self):
        app = _app([LOCAL + "._sphinx_collection"])
        check_namespace(app, LOCAL)
        assert app._scikitplot_sphinx_extension_root == LOCAL
