"""
Tests for the ``component-list`` directive.

Notes
-----
**Developer notes.** The directive reads the installed PyData Sphinx Theme's
component templates. The tests supply a directory of their own in place of
the package resource, so they check the directive's behaviour for templates
with a description, without one, with none at all and with an unreadable
directory, and do not depend on which theme release is installed.
"""

from __future__ import annotations

import types

import pytest
from docutils import nodes

from .. import directive as module
from ..directive import ComponentListDirective


class _Reporter:
    def __init__(self):
        self.errors = []

    def error(self, message, line=None):
        self.errors.append((message, line))
        return nodes.system_message(message, level=3, type="ERROR")


def _run(monkeypatch, components=None, *, error=None):
    def files(name):
        assert name == "pydata_sphinx_theme"
        if error is not None:
            raise error
        return components

    monkeypatch.setattr(module.resources, "files", files)
    directive = ComponentListDirective.__new__(ComponentListDirective)
    reporter = _Reporter()
    directive.state_machine = types.SimpleNamespace(reporter=reporter)
    directive.lineno = 7
    return directive.run(), reporter


@pytest.fixture
def theme(tmp_path):
    root = tmp_path / "theme" / "pydata_sphinx_theme" / "components"
    root.mkdir(parents=True)
    return tmp_path, root


class TestInventory:
    def test_one_item_per_template_in_name_order(self, monkeypatch, theme):
        base, root = theme
        (root / "search-button.html").write_text("{# Opens the search dialog. #}<button/>", encoding="utf-8")
        (root / "copyright.html").write_text("{#\n  Site copyright.\n#}\n<p/>", encoding="utf-8")
        (root / "notes.txt").write_text("not a template", encoding="utf-8")
        (result,), reporter = _run(monkeypatch, base)
        assert isinstance(result, nodes.bullet_list) and reporter.errors == []
        assert [item.astext() for item in result] == [
            "copyright: Site copyright.",
            "search-button: Opens the search dialog.",
        ]

    def test_each_item_links_to_the_upstream_template(self, monkeypatch, theme):
        base, root = theme
        (root / "icon-links.html").write_text("{# Icons. #}", encoding="utf-8")
        (result,), _ = _run(monkeypatch, base)
        (reference,) = result.findall(nodes.reference)
        assert reference["refuri"].endswith("/components/icon-links.html")
        assert reference["refuri"].startswith("https://github.com/pydata/pydata-sphinx-theme/")
        assert reference["internal"] is False and reference.astext() == "icon-links"

    def test_a_template_without_a_comment_says_so(self, monkeypatch, theme):
        base, root = theme
        (root / "bare.html").write_text("<div/>", encoding="utf-8")
        (result,), _ = _run(monkeypatch, base)
        assert result.astext() == "bare: No description available."

    def test_only_the_first_comment_describes_the_template(self, monkeypatch, theme):
        base, root = theme
        (root / "two.html").write_text("{# First. #}<p/>{# Second. #}", encoding="utf-8")
        (result,), _ = _run(monkeypatch, base)
        assert result.astext() == "two: First."


class TestErrors:
    def test_no_templates_is_a_located_error(self, monkeypatch, theme):
        base, _ = theme
        (result,), reporter = _run(monkeypatch, base)
        assert isinstance(result, nodes.system_message)
        assert reporter.errors == [
            ("component-list: the installed pydata_sphinx_theme package contains no component templates.", 7)
        ]

    @pytest.mark.parametrize(
        "error",
        [ModuleNotFoundError("No module named 'pydata_sphinx_theme'"), FileNotFoundError("gone"), OSError("denied")],
    )
    def test_an_unavailable_theme_is_a_located_error(self, monkeypatch, error):
        (result,), reporter = _run(monkeypatch, error=error)
        assert isinstance(result, nodes.system_message)
        ((message, line),) = reporter.errors
        assert "could not read the installed" in message and str(error) in message and line == 7

    def test_a_missing_components_directory_is_a_located_error(self, monkeypatch, tmp_path):
        (result,), reporter = _run(monkeypatch, tmp_path)
        assert isinstance(result, nodes.system_message) and len(reporter.errors) == 1

    def test_an_undecodable_template_is_a_located_error(self, monkeypatch, theme):
        base, root = theme
        (root / "broken.html").write_bytes(b"\xff\xfe\x00bad")
        (result,), reporter = _run(monkeypatch, base)
        ((message, _),) = reporter.errors
        assert isinstance(result, nodes.system_message) and "'broken.html'" in message


class TestSetup:
    def _app(self, extensions=()):
        added = []
        app = types.SimpleNamespace(
            config=types.SimpleNamespace(extensions=list(extensions)),
            extensions={},
            add_directive=lambda name, cls: added.append((name, cls)),
        )
        return app, added

    def test_registers_the_directive_and_declares_parallel_safety(self):
        app, added = self._app()
        assert module.setup(app) == {"parallel_read_safe": True, "parallel_write_safe": True}
        assert added == [("component-list", ComponentListDirective)]

    def test_a_mixed_namespace_is_refused_before_registration(self):
        from sphinx.errors import ExtensionError

        app, added = self._app(["some.other.root._sphinx_ext._sphinx_collection"])
        with pytest.raises(ExtensionError, match="Mixed"):
            module.setup(app)
        assert added == []

    def test_the_package_exposes_the_directive_lazily(self):
        from .. import ComponentListDirective as exported

        assert exported is ComponentListDirective
