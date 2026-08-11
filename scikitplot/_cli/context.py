# scikitplot/_cli/context.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Neutral per-invocation runtime context.

Handlers receive a :class:`Context` and write results to ``ctx.stdout`` and
diagnostics to ``ctx.stderr``. Streams are bound at construction so tests can
redirect them before a frontend runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, TextIO

Format = Literal["text", "json", "yaml", "toml"]


@dataclass(slots=True)
class Context:
    """State shared with a command handler for one invocation.

    Parameters
    ----------
    stdout, stderr : text stream
        Result and diagnostic channels (invariant: stdout is results only).
    fmt : {"text", "json", "yaml"}
        Requested output format.
    color : bool
        Whether colored human output is permitted.
    verbosity : int
        Net verbosity (``-v`` increases, ``-q`` decreases).
    """

    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    fmt: Format = "text"
    color: bool = True
    verbosity: int = 0


__all__ = ["Context", "Format"]
