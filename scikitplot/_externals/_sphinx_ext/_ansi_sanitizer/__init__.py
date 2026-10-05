"""
Sanitize terminal control sequences for non-terminal builders.

Notes
-----
**User notes.** Captured program output often carries terminal escape
sequences: colours, cursor movement, window titles, hyperlinks. A LaTeX build
cannot typeset them and may stop on the control bytes. Enable the extension
and they are removed from every text node of a LaTeX build; other builders
are left alone::

    extensions = [
        # ...
        "scikitplot._externals._sphinx_ext._ansi_sanitizer",
    ]

:func:`strip_terminal_controls` is the same operation on a plain string.

**Developer notes.** The expression follows the ECMA-48 grammar instead of
one catch-all pattern. A single pattern of the form *ESC, one byte, parameter
bytes, a final byte* is right for a control sequence (``ESC [ 3 1 m``) and
wrong for everything else, in the direction that loses text:

- a two-byte escape has no final byte, so the pattern consumed the next
  character of the output (``ESC M`` followed by ``Hello`` became ``ello``);
- an operating-system command (``ESC ] 0 ; title BEL``) is a string up to a
  terminator, so the pattern removed its first bytes and left the rest of the
  title in the document;
- an escape with intermediate bytes (``ESC ( B``) was not matched and its
  printable bytes stayed behind.

Each form now has its own alternative. A string sequence is removed only when
its terminator is on the same line; without one, only the introducer is
removed, so no visible text is lost to a truncated capture.
"""

from __future__ import annotations

import re
from typing import Any

from docutils import nodes

__all__ = ["setup", "strip_terminal_controls"]

_ANSI_ESCAPE_RE = re.compile(
    # Control sequence: CSI, parameter bytes, intermediate bytes, final byte.
    r"(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~]"
    # String sequences (OSC, DCS, SOS, PM, APC) up to BEL or ST on this line.
    r"|(?:\x1B[\]PX^_]|[\x90\x98\x9D\x9E\x9F])[^\x07\x1B\x9C\n]*(?:\x07|\x1B\\|\x9C)"
    # Escape with intermediate bytes, for example a character-set selection.
    r"|\x1B[ -/]+[0-~]"
    # Any other two-byte escape, including an unterminated string introducer.
    r"|\x1B[0-~]"
)

#: C0 controls except tab, line feed and carriage return; DEL; and the C1 set.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def strip_terminal_controls(text: str) -> str:
    r"""
    Return ``text`` without terminal escape sequences and control characters.

    Parameters
    ----------
    text : str
        Any text, typically captured program output.

    Returns
    -------
    str
        ``text`` with escape sequences and control characters removed. Tab,
        line feed and carriage return are kept, as is every printable
        character.

    Raises
    ------
    TypeError
        If ``text`` is not a string.

    Examples
    --------
    >>> strip_terminal_controls("\x1b[31mred\x1b[0m")
    'red'
    >>> strip_terminal_controls("\x1bMHello")
    'Hello'
    >>> strip_terminal_controls("\x1b]0;window title\x07body")
    'body'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}")
    return _CONTROL_CHAR_RE.sub("", _ANSI_ESCAPE_RE.sub("", text))


def _sanitize_latex_text(app: Any, doctree: nodes.document, docname: str) -> None:
    """Replace each text node of a LaTeX build that carries control sequences."""
    if app.builder.format != "latex":
        return

    for node in list(doctree.findall(nodes.Text)):
        original = node.astext()
        cleaned = strip_terminal_controls(original)
        if original != cleaned:
            node.parent.replace(node, nodes.Text(cleaned))


def setup(app: Any) -> dict[str, Any]:
    """
    Register the sanitizer on ``doctree-resolved``.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application being configured.

    Returns
    -------
    dict
        Sphinx extension metadata; the extension is safe for parallel reading
        and writing because it keeps no state.
    """
    app.connect("doctree-resolved", _sanitize_latex_text)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
