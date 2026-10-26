# scikitplot/_ipymagic/_completion.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tab completion of ``%scikitplot``, read from the CLI's command model.

Notes
-----
**User notes.** ``%scikitplot <TAB>`` lists the commands and their aliases;
after a native command (``info``, ``doctor`` ...) ``<TAB>`` lists its options
and, after an option with fixed choices (``--format``), the choices.
Delegated commands (``env``, ``mcp``, ``cleanprompt``) parse their own
arguments; ``%scikitplot env --help`` lists them.

**Developer notes.** Everything comes from :py:mod:`scikitplot._cli.registry`
(``CommandSpec`` and ``Param``), the same model the argparse and click
frontends project, so completion cannot drift from the CLI.
"""

from __future__ import annotations

import shlex

__all__ = ["complete"]

_ROOT_OPTIONS = ("--help", "--quiet", "--verbose", "--version")


def _words(line: str) -> tuple[list[str], str]:
    """
    Return the complete words of ``line`` and the word being typed.

    Parameters
    ----------
    line : str
        The line typed so far.

    Returns
    -------
    words : list of str
        The finished words.
    current : str
        The word being typed (empty after a space; a NUL character on an open quote).
    """
    try:
        words = shlex.split(line, posix=True)
    except ValueError:  # an open quote: complete nothing rather than guess
        return [], "\0"
    if line.endswith((" ", "\t")) or not words:
        return words, ""
    return words[:-1], words[-1]


def complete(line: str) -> list[str]:
    """
    Return the completions of the last word of a ``%scikitplot`` line.

    Parameters
    ----------
    line : str
        The line typed so far, magic name included or not.

    Returns
    -------
    list of str
        Sorted candidates that start with the word being typed.

    Examples
    --------
    >>> from scikitplot._ipymagic import complete
    >>> "env" in complete("%scikitplot e")
    True
    >>> complete("%scikitplot info --format ")
    ['json', 'text', 'toml', 'yaml']
    """
    from .._cli import registry  # noqa: PLC0415

    words, current = _words(line)
    if current == "\0":
        return []
    if words and words[0].lstrip("%") == "scikitplot":
        words = words[1:]
    command = next((w for w in words if not w.startswith("-")), None)
    if command is None:
        names = [n for n, s in registry.BY_NAME.items() if not s.hidden]
        pool = [*names, *registry.BY_ALIAS, *_ROOT_OPTIONS]
        return sorted(c for c in set(pool) if c.startswith(current))
    spec = registry.resolve(command)
    if spec is None or spec.delegate is not None:
        return []
    previous = words[-1] if words and words[-1] != command else None
    for prm in spec.params:
        if previous in prm.flags and prm.choices:
            return sorted(c for c in prm.choices if c.startswith(current))
    flags = [f for prm in spec.params for f in prm.flags]
    flags += [
        "--no-" + f[2:]
        for prm in spec.params
        if prm.negatable
        for f in prm.flags
        if f.startswith("--")
    ]
    flags += ["--help", "--quiet", "--verbose"]
    return sorted(f for f in set(flags) if f.startswith(current))
