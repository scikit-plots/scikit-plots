"""
Framework-neutral description of the command-line surface.

:class:`Param` and :class:`Command` are the single source of truth that both the
argparse and the click frontend render from.

Notes
-----
**Developer notes.** This mirrors ``scikitplot._cli._spec`` deliberately, and
carries it **by value** rather than importing it, because this submodule imports
no sibling package. Keeping the shape identical means a maintainer who knows the
project-wide IR already knows this one; inventing a second vocabulary for the
same idea would cost more than the duplication saves.

The IR expresses only the *intersection* of what both frontends render
faithfully. Anything one can do and the other cannot has no place here: a flag
that behaves differently depending on which frontend is installed is worse than
a flag that does not exist, because the difference only shows up on someone
else's machine.

This module is standard library only and must stay importable without ``click``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

__all__ = [
    "Command",
    "Param",
    "ParamKind",
]

ParamKind = Literal["flag", "option", "argument"]


@dataclass(frozen=True)
class Param:
    """
    One command-line parameter, expressed once for every frontend.

    Parameters
    ----------
    dest : str
        Canonical snake_case name. The handler receives it under this name.
    flags : tuple of str
        Shell spellings, hyphenated. Empty only for ``kind="argument"``.
    kind : {"flag", "option", "argument"}
        ``flag`` is a boolean switch, ``option`` takes a value, ``argument`` is
        positional.
    help : str
        One-line help text.
    type : callable, optional
        Value converter for options.
    default : object
        Default value.
    choices : tuple of str, optional
        Restrict values to a fixed set.
    multiple : bool, default=False
        Collect repeated values into a list.
    metavar : str, optional
        Display name for the value in help.
    required : bool, default=False
        Whether the parameter must be supplied.

    Raises
    ------
    ValueError
        If the field combination is invalid.

    Notes
    -----
    **Developer notes.** Long flags must use hyphens. A ``--word_boundary``
    would work in argparse and read as wrong to every shell user, and the two
    frontends would disagree about the resulting ``dest``.
    """

    dest: str
    flags: tuple[str, ...] = ()
    kind: ParamKind = "option"
    help: str = ""
    type: Callable[[str], Any] | None = None
    default: Any = None
    choices: tuple[str, ...] | None = None
    multiple: bool = False
    metavar: str | None = None
    required: bool = False

    def __post_init__(self) -> None:
        if self.kind == "argument" and self.flags:
            raise ValueError(f"argument {self.dest!r} must not declare flags")
        if self.kind != "argument" and not self.flags:
            raise ValueError(
                f"{self.kind} {self.dest!r} must declare at least one flag"
            )
        if self.kind == "flag" and self.choices:
            raise ValueError(f"flag {self.dest!r} cannot declare choices")
        for flag in self.flags:
            if flag.startswith("--") and "_" in flag:
                raise ValueError(f"flag {flag!r} must use hyphens, not underscores")


@dataclass(frozen=True)
class Command:
    """
    One subcommand.

    Parameters
    ----------
    name : str
        Public command name.
    summary : str
        One-line description used in help listings.
    handler : str
        Name of the handler in :mod:`scikitplot.cleanprompt._cli`. Resolved by
        name rather than held as a reference, so this module stays free of
        import cycles and remains pure data.
    params : tuple of Param
        Parameters, in declaration order.
    aliases : tuple of str
        Alternate names.
    description : str
        Longer help shown on ``<command> --help``. Falls back to ``summary``.

    Raises
    ------
    ValueError
        If two parameters share a ``dest``.
    """

    name: str
    summary: str
    handler: str
    params: tuple[Param, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for param in self.params:
            if param.dest in seen:
                raise ValueError(f"command {self.name!r} declares {param.dest!r} twice")
            seen.add(param.dest)

    @property
    def help_text(self) -> str:
        """str: The longer description, falling back to the summary."""
        return self.description or self.summary

    def defaults(self) -> dict[str, Any]:
        """
        Return every parameter's default, keyed by ``dest``.

        Returns
        -------
        dict
            Defaults for this command.

        Notes
        -----
        **Developer notes.** Used to fill a namespace for a frontend that only
        reports the parameters it actually saw, so a handler always receives
        every field it reads. Without this a handler would need
        ``getattr(args, "x", default)`` at each use, and the default would live
        in two places.
        """
        return {param.dest: param.default for param in self.params}
