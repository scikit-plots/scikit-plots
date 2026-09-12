"""
PyData Sphinx Theme ``component-list`` directive.

The inventory is intentionally separate from the generic gallery engine: it
reads PyData Sphinx Theme's installed component templates and links each item to
its upstream source file.  Template discovery uses package resources, not a
checkout-relative ``src/`` path, so the directive works when vendored into
another project as well as inside the PyData repository itself.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


class ComponentListDirective(SphinxDirective):
    """Generate a list of PyData Sphinx Theme component templates."""

    name = "component-list"
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        """Create the component inventory or a located error node."""
        try:
            component_dir = (
                resources.files("pydata_sphinx_theme")
                / "theme"
                / "pydata_sphinx_theme"
                / "components"
            )
            components = sorted(
                (
                    entry
                    for entry in component_dir.iterdir()
                    if entry.name.endswith(".html")
                ),
                key=lambda entry: entry.name,
            )
        except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            return [
                self.state_machine.reporter.error(
                    "component-list: could not read the installed "
                    f"pydata_sphinx_theme component templates ({exc}).",
                    line=self.lineno,
                )
            ]

        if not components:
            return [
                self.state_machine.reporter.error(
                    "component-list: the installed pydata_sphinx_theme package "
                    "contains no component templates.",
                    line=self.lineno,
                )
            ]

        pattern = re.compile(r"(?<={#).*?(?=#})", flags=re.DOTALL)
        items: list[nodes.list_item] = []
        upstream = (
            "https://github.com/pydata/pydata-sphinx-theme/blob/main/"
            "src/pydata_sphinx_theme/theme/pydata_sphinx_theme/components"
        )
        for component in components:
            try:
                text = component.read_text(encoding="utf-8")
            except (UnicodeError, OSError) as exc:
                return [
                    self.state_machine.reporter.error(
                        f"component-list: could not read {component.name!r} ({exc}).",
                        line=self.lineno,
                    )
                ]
            comments = pattern.findall(text)
            description = (
                comments[0].strip() if comments else "No description available."
            )
            name = component.name.removesuffix(".html")
            url = f"{upstream}/{component.name}"
            items.append(
                nodes.list_item(
                    "",
                    nodes.paragraph(
                        "",
                        "",
                        nodes.reference("", name, internal=False, refuri=url),
                        nodes.Text(f": {description}"),
                    ),
                )
            )

        return [nodes.bullet_list("", *items)]


def setup(app: Sphinx) -> dict[str, Any]:
    """Register ``component-list`` with namespace validation."""
    from .._extension_setup import (  # ruff: ignore[import-outside-top-level]
        check_namespace,
    )

    check_namespace(app, __package__.rsplit(".", 1)[0])
    app.add_directive("component-list", ComponentListDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
