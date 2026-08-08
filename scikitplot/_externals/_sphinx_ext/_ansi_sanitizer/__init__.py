"""
Sanitize terminal control sequences for non-terminal builders.

extensions = [
    # ...
    "scikitplot._externals._sphinx_ext.ansi_sanitizer",
]
"""

from __future__ import annotations

import re

from docutils import nodes

_ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _sanitize_latex_text(app, doctree, docname):
    if app.builder.format != "latex":
        return

    for node in list(doctree.findall(nodes.Text)):
        original = node.astext()

        cleaned = _ANSI_ESCAPE_RE.sub("", original)
        cleaned = _CONTROL_CHAR_RE.sub("", cleaned)

        if original != cleaned:
            node.parent.replace(
                node,
                nodes.Text(cleaned),
            )


def setup(app):
    app.connect(
        "doctree-resolved",
        _sanitize_latex_text,
    )

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
