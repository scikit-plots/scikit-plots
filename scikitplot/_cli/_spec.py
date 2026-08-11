# scikitplot/_cli/_spec.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-neutral command intermediate representation.

``Param`` and ``CommandSpec`` are the single source of truth that both the
argparse and click frontends consume. The IR expresses only the *intersection*
of what both frontends can render faithfully; frontend-specific styling is
presentation and lives in :mod:`scikitplot._cli.output`, never here.

Notes
-----
This module is stdlib-only and MUST stay importable without ``click``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

ParamKind = Literal["flag", "option", "argument"]


@dataclass(frozen=True, slots=True)
class Param:
    """A single command-line parameter, expressed once for all frontends.

    Parameters
    ----------
    dest : str
        Canonical snake_case name; the handler receives it as a keyword argument.
    flags : tuple of str
        Shell spellings (hyphenated), e.g. ``("--mask-envs",)`` or
        ``("-v", "--verbose")``. Empty only for ``kind="argument"``.
    kind : {"flag", "option", "argument"}
        ``flag`` is a boolean switch; ``option`` takes a value; ``argument`` is
        positional.
    help : str
        One-line help text.
    type : callable, optional
        Value converter for options (e.g. :class:`int`, :class:`float`,
        :class:`pathlib.Path`).
    default : any
        Default value.
    required : bool
        Whether an option/argument must be supplied.
    multiple : bool
        Collect repeated values (argparse ``append`` / click ``multiple``).
    count : bool
        Count occurrences (``-v -vv``); options only.
    negatable : bool
        Provide a ``--no-<flag>`` negation; flags only.
    choices : tuple of str, optional
        Restrict values to a fixed set.
    metavar : str, optional
        Display name for the value in help.

    Raises
    ------
    ValueError
        If the field combination is invalid (validated in ``__post_init__``).
    """

    dest: str
    flags: tuple[str, ...] = ()
    kind: ParamKind = "option"
    help: str = ""
    type: Callable[[str], Any] | None = None
    default: Any = None
    required: bool = False
    multiple: bool = False
    count: bool = False
    negatable: bool = False
    choices: tuple[str, ...] | None = None
    metavar: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "argument" and self.flags:
            raise ValueError(f"argument {self.dest!r} must not declare flags")
        if self.kind != "argument" and not self.flags:
            raise ValueError(
                f"{self.kind} {self.dest!r} must declare at least one flag"
            )
        if self.count and self.kind != "option":
            raise ValueError(f"count param {self.dest!r} must be kind='option'")
        if self.negatable and self.kind != "flag":
            raise ValueError(f"negatable param {self.dest!r} must be kind='flag'")
        # Shell-convention spelling: long flags use hyphens, never underscores
        # (FINDINGS CLI-FE-007). Short flags (-v) are exempt.
        for flag in self.flags:
            if flag.startswith("--") and "_" in flag:
                raise ValueError(f"flag {flag!r} must use hyphens, not underscores")


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Metadata describing one CLI command.

    Parameters
    ----------
    name : str
        Public command name (hyphenated).
    summary : str
        One-line description used in help.
    handler : str
        Lazy import target ``"package.module:attribute"``. The attribute must be
        a callable ``run(ctx, **params) -> int``.
    params : tuple of Param
        Command parameters, in declaration order.
    aliases : tuple of str
        Alternate names.
    hidden : bool
        Hide from help listings.
    deprecated : bool
        Mark as deprecated (still runnable; emits a stderr notice).
    capabilities : tuple of str
        Capability keys this command requires (checked before dispatch).
    """

    name: str
    summary: str
    handler: str
    params: tuple[Param, ...] = ()
    aliases: tuple[str, ...] = ()
    hidden: bool = False
    deprecated: bool = False
    capabilities: tuple[str, ...] = ()


__all__ = ["CommandSpec", "Param", "ParamKind"]
