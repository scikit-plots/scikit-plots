r"""
Presentation helpers — strictly separate from the data they describe.

Notes
-----
**User notes.** These functions return decorated copies for display. They never
change a :class:`~scikitplot.cleanprompt._types.RedactionResult` or a
:class:`~scikitplot.cleanprompt._vault.Vault`.

**Developer notes.** This module exists because upstream mixed the two:
``revert_text(..., color=True)`` returned ``'hi \x1b[92mAda\x1b[0m'`` — ANSI
escape sequences embedded in the value a caller would then store, compare,
re-redact or send onward. Colour is a property of a terminal, not of text.

Colour is emitted only when the destination is a terminal and the environment
does not ask otherwise. Three signals are honoured, in order: an explicit
``color`` argument, the ``NO_COLOR`` convention, and
:meth:`io.IOBase.isatty`. Nothing here writes to a stream; the caller decides
where the string goes.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ._policy import DEFAULT_POLICY, RedactionPolicy
from ._types import RedactionResult, RestorationResult

__all__ = [
    "ANSI",
    "highlight_placeholders",
    "restoration_note",
    "should_colorize",
    "summary_table",
]


class ANSI:
    """
    Terminal escape sequences used by :func:`highlight_placeholders`.

    Attributes
    ----------
    GREEN, CYAN, MAGENTA, BOLD, RESET : str
        Select Graphic Rendition sequences.
    """

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def should_colorize(
    stream: Any | None = None,
    color: bool | None = None,
) -> bool:
    """
    Decide whether to emit terminal colour.

    Parameters
    ----------
    stream : file-like, optional
        Destination. Defaults to :data:`sys.stdout`.
    color : bool, optional
        Explicit override. ``True`` or ``False`` decides immediately.

    Returns
    -------
    bool
        Whether colour should be emitted.

    Notes
    -----
    **Developer notes.** ``NO_COLOR`` is honoured whenever it is set to any
    value, per the convention at <https://no-color.org>. A stream without
    :meth:`isatty` — a pipe, a file, a captured buffer in a test — is treated as
    not a terminal.

    Examples
    --------
    >>> should_colorize(color=False)
    False
    """
    if color is not None:
        return color
    if os.environ.get("NO_COLOR") is not None:
        return False
    target = stream if stream is not None else sys.stdout
    try:
        return bool(target.isatty())
    except Exception:  # noqa: BLE001 - a stream without isatty is not a terminal
        return False


def highlight_placeholders(
    text: str,
    policy: RedactionPolicy = DEFAULT_POLICY,
    color: bool | None = None,
    stream: Any | None = None,
) -> str:
    """
    Return a copy of ``text`` with placeholders wrapped in colour.

    Parameters
    ----------
    text : str
        Text containing placeholders.
    policy : RedactionPolicy, default=DEFAULT_POLICY
        Supplies the placeholder grammar.
    color : bool, optional
        Explicit override; see :func:`should_colorize`.
    stream : file-like, optional
        Destination used to detect a terminal.

    Returns
    -------
    str
        A decorated copy, or ``text`` unchanged when colour is disabled.

    Examples
    --------
    >>> highlight_placeholders("a [EMAIL-1] b", color=False)
    'a [EMAIL-1] b'
    """
    if not should_colorize(stream=stream, color=color):
        return text
    pattern = policy.tag_style.pattern()
    return pattern.sub(
        lambda match: f"{ANSI.GREEN}{match.group()}{ANSI.RESET}",
        text,
    )


def summary_table(
    result: RedactionResult,
    reveal: bool = False,
    color: bool | None = None,
    stream: Any | None = None,
) -> str:
    """
    Render a table of what was redacted.

    Parameters
    ----------
    result : RedactionResult
        The pass to describe.
    reveal : bool, default=False
        When ``True``, include the original values. Off by default: a summary is
        the thing most likely to be pasted into a ticket or a log.
    color : bool, optional
        Explicit override; see :func:`should_colorize`.
    stream : file-like, optional
        Destination used to detect a terminal.

    Returns
    -------
    str
        A plain-text table, or a note when nothing was redacted.

    Notes
    -----
    **Developer notes.** ``reveal`` defaults to ``False`` because upstream
    printed every secret on one line unconditionally
    (``"Removed sensitive information: " + ", ".join(...)``), which puts personal
    data into terminal scrollback, shell transcripts and any captured log.

    Examples
    --------
    >>> from ._engine import Redactor
    >>> print(summary_table(Redactor().redact("no secrets here"), color=False))
    no sensitive values detected
    """
    if not result.entries:
        return "no sensitive values detected"
    tint = should_colorize(stream=stream, color=color)
    headers = ["placeholder", "kind", "count", "detector"]
    if reveal:
        headers.append("original")
    rows: list[list[str]] = []
    for entry in result.entries:
        row = [entry.label, entry.kind, str(entry.count), entry.detector]
        if reveal:
            row.append(entry.original)
        rows.append(row)
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        rendered = "  ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        )
        lines.append(f"{ANSI.CYAN}{rendered}{ANSI.RESET}" if tint else rendered)
    return "\n".join(lines)


def restoration_note(
    result: RestorationResult,
) -> str:
    """
    Return a one-line, secret-free description of a restoration.

    Parameters
    ----------
    result : RestorationResult
        The pass to describe.

    Returns
    -------
    str
        A summary naming counts, and any unknown labels. Labels are not
        secrets, so naming them is safe and actionable.

    Examples
    --------
    >>> restoration_note(RestorationResult("x", ("[EMAIL-1]",)))
    'restored 1 placeholder(s)'
    """
    note = f"restored {len(result.restored)} placeholder(s)"
    if result.repaired:
        note += "; {} repaired: {}".format(
            len(result.repaired),
            ", ".join(
                f"{_compact(found)} -> {label}" for found, label in result.repaired
            ),
        )
    if result.unknown:
        note += "; {} unknown: {}".format(
            len(result.unknown), ", ".join(result.unknown)
        )
    if result.unused:
        note += f"; {len(result.unused)} vault entr(y/ies) unused"
    if result.restored == () and result.unused:
        # Nothing came back and the vault is not empty. Either the reply never
        # mentioned a placeholder, or the model rewrote them past recognition.
        # Saying so is the difference between a user noticing and a user
        # pasting a half-restored answer onward.
        note += (
            "\n  nothing was restored: check the reply still contains the "
            "placeholders as they were issued"
        )
    return note


def _compact(found: str) -> str:
    """
    Return a label as the reply spelled it, on one line.

    Notes
    -----
    **Developer notes.** A placeholder wrapped across a line arrives with a
    newline inside it, and a diagnostic that breaks its own line in the middle
    of a token is harder to read than the defect it reports.
    """
    return " ".join(found.split())
