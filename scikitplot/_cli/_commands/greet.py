# scikitplot/_cli/_commands/greet.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot greet [NAME]` - a minimal example command."""

from __future__ import annotations

from ..context import Context


def run(ctx: Context, *, name: str = "World", emoji: bool = True) -> int:
    """Greet ``name``, optionally with an emoji."""
    if isinstance(name, (list, tuple)):  # argparse nargs / click tuple safety
        name = name[0] if name else "World"
    message = f"Hello, {name}!"
    if emoji:
        message += " \N{WAVING HAND SIGN}"
    ctx.stdout.write(message + "\n")
    return 0
