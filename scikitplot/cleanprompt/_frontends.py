"""
Argparse and click renderers for the command surface.

Both render from :mod:`scikitplot.cleanprompt._spec`, so the two frontends
cannot drift apart.

Notes
-----
**User notes.** The frontend is chosen automatically: click when it is
installed, argparse otherwise. Force one with the ``CLEANPROMPT_CLI_FRONTEND``
environment variable, or globally with ``SCIKITPLOT_CLI_FRONTEND``::

    CLEANPROMPT_CLI_FRONTEND=argparse python -m scikitplot.cleanprompt doctor

Either way the commands, options and output are the same. If they ever differ,
that is a bug, and ``test__frontends.py`` is where it is caught.

**Developer notes — why two frontends at all.**

The project-wide CLI renders the same commands through argparse and click, and
a submodule that supported only one would behave differently from every other
command under ``scikitplot``. Click gives better help, completion and error
messages where it is installed; argparse guarantees the tool works on a bare
standard library, which is the whole premise of this submodule's base tier.

**The rule that keeps them honest.** Neither frontend owns any behaviour. They
parse, and hand a plain :class:`argparse.Namespace` to the same handler. A
handler cannot tell which frontend called it, so there is nowhere for a
behavioural difference to hide.

Selecting a frontend imports only that frontend. ``click`` is never imported to
answer the question of whether ``click`` is available.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import IO, Any, Sequence

from ._spec import Command, Param

__all__ = [
    "FRONTENDS",
    "build_argparse",
    "is_click_available",
    "load_runner",
    "namespace_for",
    "run_argparse",
    "run_click",
    "select_frontend",
]

#: Frontend names this submodule can render.
FRONTENDS = ("argparse", "click")

#: Environment variables that pin the frontend, most specific first.
_ENV_VARS = ("CLEANPROMPT_CLI_FRONTEND", "SCIKITPLOT_CLI_FRONTEND")


def is_click_available() -> bool:
    """
    Return whether ``click`` can be imported, without importing it.

    Returns
    -------
    bool
        ``True`` when a ``click`` module is on the path.

    Notes
    -----
    **Developer notes.** :func:`importlib.util.find_spec` is the right tool
    *here* and the wrong one in
    :mod:`scikitplot.cleanprompt._capabilities`. The questions differ: this one
    is "can I import this module at all", which is what ``find_spec`` answers,
    while capability probing asks "is this distribution installed at a version
    I support", which needs metadata. Using ``find_spec`` for the second is the
    defect that module's notes describe.
    """
    return importlib.util.find_spec("click") is not None


def select_frontend(requested: str | None = None) -> str:
    """
    Decide which frontend to use.

    Parameters
    ----------
    requested : str, optional
        An explicit choice, which wins over everything else.

    Returns
    -------
    str
        ``"click"`` or ``"argparse"``.

    Raises
    ------
    ValueError
        If an explicit choice names an unknown frontend.

    Notes
    -----
    **Developer notes.** Precedence is explicit argument, then
    ``CLEANPROMPT_CLI_FRONTEND``, then ``SCIKITPLOT_CLI_FRONTEND``, then
    availability. An environment variable naming an *unavailable* frontend
    falls back rather than failing: the variable is usually set once for a
    whole shell, and refusing to run a working command because of an inherited
    preference would be obstructive.

    Examples
    --------
    >>> select_frontend("argparse")
    'argparse'
    """
    if requested is not None:
        if requested not in FRONTENDS:
            msg = "unknown frontend {!r}; choose from {}".format(
                requested,
                ", ".join(FRONTENDS),
            )
            raise ValueError(msg)
        return requested

    for name in _ENV_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value in FRONTENDS:
            if value == "click" and not is_click_available():
                break  # asked for click, do not have it: fall through
            return value

    return "click" if is_click_available() else "argparse"


def namespace_for(command: Command, values: dict[str, Any]) -> argparse.Namespace:
    """
    Build a handler namespace from a command's defaults plus ``values``.

    Parameters
    ----------
    command : Command
        The command being invoked.
    values : dict
        Parsed values, keyed by ``dest``. ``None`` entries are treated as
        "not supplied" and fall back to the declared default.

    Returns
    -------
    argparse.Namespace
        Every declared ``dest``, plus ``command``.

    Notes
    -----
    **Developer notes.** This is what makes the frontends interchangeable.
    Click reports unsupplied options as ``None`` and argparse applies defaults
    itself; normalising both here means a handler sees one shape and can read
    ``args.x`` directly.
    """
    declared = {param.dest: param for param in command.params}
    merged = command.defaults()
    for key, value in values.items():
        if value is None:
            continue
        param = declared.get(key)
        if param is not None and param.multiple:
            # Click yields () for an unused repeatable option and argparse
            # yields None. Collapse both to the declared default so the two
            # frontends hand a handler the identical namespace.
            value = list(value)  # ruff: ignore[redefined-loop-name]
            if not value:
                continue
        merged[key] = value
    merged["command"] = command.name
    return argparse.Namespace(**merged)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_argparse_param(parser: argparse.ArgumentParser, param: Param) -> None:
    """Render one :class:`Param` onto an argparse parser."""
    if param.kind == "argument":
        parser.add_argument(
            param.dest,
            nargs="*" if param.multiple else None,
            default=param.default,
            help=param.help,
            metavar=param.metavar,
        )
        return

    if param.kind == "flag":
        parser.add_argument(
            *param.flags,
            dest=param.dest,
            action="store_true",
            default=param.default,
            help=param.help,
        )
        return

    # ``append`` rather than ``nargs="+"``. Click renders a repeatable option
    # as "--hide A --hide B" and has no way to express "--hide A B" for an
    # option, so argparse conforms to click rather than the reverse. The IR
    # promises only what BOTH frontends render faithfully, and a flag that
    # accepts two spellings depending on which library happens to be installed
    # is precisely the divergence this design exists to prevent.
    parser.add_argument(
        *param.flags,
        dest=param.dest,
        action="append" if param.multiple else None,
        type=param.type,
        default=None if param.multiple else param.default,
        choices=list(param.choices) if param.choices else None,
        required=param.required,
        help=param.help,
        metavar=param.metavar,
    )


class _Parser(argparse.ArgumentParser):
    """
    An argparse parser whose errors name the form that would have worked.

    Notes
    -----
    **Developer notes.** One message is rewritten. ``--hide -secret`` fails in
    argparse with "argument --hide: expected one argument", because argparse
    reads ``-secret`` as an option rather than as the value. The message is
    true and useless: it describes what the parser wanted without saying how to
    give it.

    Click accepts the same command line, which makes this a divergence as well
    as a bad message. Conforming the two by parsing is not available — argparse
    decides an option-looking token is an option inside ``_parse_optional``,
    and overriding that reaches into internals that move between Python
    versions, against a dependency policy that requires this code to work
    across the whole supported range. So the portable form is documented
    instead: ``--hide=-secret`` is accepted by both, and is the POSIX-preferred
    spelling regardless.

    The rewrite is narrow on purpose. It fires only on argparse's own
    "expected one argument" text, and it appends rather than replaces, so a
    message this code has not anticipated still reaches the user intact.
    """

    def error(self, message: str) -> None:  # noqa: D102 - argparse's own contract
        if "expected one argument" in message:
            option = message.split(":", maxsplit=1)[0].replace("argument ", "").strip()
            message = (
                f"{message}\n"
                "  if the value begins with a dash, attach it with '=' so it is "
                "not read as an option:\n"
                f"      {option}=-your-value\n"
                "  to end the options entirely and pass the rest as text, use "
                "'--':\n"
                f"      {self.prog} ... -- '-your text'"
            )
        super().error(message)


def build_argparse(
    commands: Sequence[Command], prog: str, description: str, epilog: str = ""
) -> argparse.ArgumentParser:
    """
    Render the command surface as an argparse parser.

    Parameters
    ----------
    commands : sequence of Command
        The commands to render.
    prog : str
        Program name shown in help.
    description : str
        Top-level description.
    epilog : str, default=""
        Trailing help text.

    Returns
    -------
    argparse.ArgumentParser
        A parser whose namespace carries ``command`` and every declared
        ``dest``.
    """
    parser = _Parser(
        prog=prog,
        description=description,
        epilog=epilog or None,
        # argparse accepts any unambiguous prefix of a long option by default,
        # so ``--form json`` reaches ``--format``. Click has no such feature and
        # rejects it. That makes the same command line succeed or fail
        # depending on which library happens to be installed — the CP-021 class
        # of divergence, and the reason a script written on a machine without
        # click dies the day a colleague installs it.
        #
        # argparse is the one that conforms, as it did for CP-021, because the
        # stricter behaviour is the one that can be relied on: an abbreviation
        # that works today also silently changes meaning the moment a new
        # option shares its prefix.
        allow_abbrev=False,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="print the submodule version and exit",
    )
    parser.add_argument(
        "--frontend",
        choices=list(FRONTENDS),
        default=None,
        help="force a command-line frontend (default: click when installed)",
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        parser_class=_Parser,
    )
    for command in commands:
        child = sub.add_parser(
            command.name,
            help=command.summary,
            description=command.help_text,
            aliases=list(command.aliases),
            # Subparsers do not inherit it; every declared option lives here,
            # so this is the one that actually matters.
            allow_abbrev=False,
        )
        for param in command.params:
            _add_argparse_param(child, param)
    return parser


def _reject_stray_options(
    parser: argparse.ArgumentParser,
    tokens: Sequence[str],
    args: argparse.Namespace,
) -> None:
    """
    Refuse an option-looking positional that no ``--`` authorised.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser, used for its usage message and exit behaviour.
    tokens : sequence of str
        The raw argument list, needed because the namespace cannot say whether
        a ``--`` delimiter was present.
    args : argparse.Namespace
        The parsed namespace.

    Notes
    -----
    **Developer notes.** argparse treats any token containing a space as a
    positional, however it begins. So ``inspect "--secret is a@b.co"`` is
    accepted as text and silently redacted, while click rejects the same
    command line as an unknown option. The divergence is the CP-021 class
    again, and here argparse holds the dangerous side: a mistyped option is
    quietly swallowed as if it were the prompt, and a tool whose job is to
    decide what leaves the machine must not quietly do something other than
    what was asked.

    argparse decides this inside ``_parse_optional``, which is private and has
    moved between releases, so the conforming is done here instead: parse
    normally, then refuse a dash-leading positional that no ``--`` authorised.
    After a ``--`` the user has said explicitly that what follows is text, so
    nothing is checked past that point — which is exactly the guarantee the
    delimiter is for.

    The message names the delimiter, because a user who meant the text has a
    one-token fix and a user who meant the option has a typo to find.
    """
    if "--" in tokens:
        return  # the delimiter was given: what follows is text, by request
    for value in getattr(args, "text_args", None) or ():
        if isinstance(value, str) and value.startswith("-") and value != "-":
            parser.error(
                f"unrecognised option {value!r}\n"
                "  if that is your text and not an option, end the options "
                "with '--' first:\n"
                f"      {parser.prog} ... -- {value!r}"
            )


def run_argparse(  # ruff: ignore[too-many-positional-arguments]
    commands: Sequence[Command],
    argv: Sequence[str] | None,
    prog: str,
    description: str,
    epilog: str,
    stdout: IO[str],
) -> tuple[str | None, argparse.Namespace]:
    """
    Parse ``argv`` with argparse.

    Returns
    -------
    command : str or None
        The resolved command name, or ``None`` when none was given.
    args : argparse.Namespace
        The parsed namespace.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parser = build_argparse(commands, prog, description, epilog)
    args = parser.parse_args(tokens)
    _reject_stray_options(parser, tokens, args)
    if getattr(args, "version", False):
        return "--version", args
    if not args.command:
        parser.print_help(stdout)
        return None, args
    by_name = _index(commands)
    command = by_name[args.command]
    values = {
        param.dest: getattr(args, param.dest, param.default) for param in command.params
    }
    return command.name, namespace_for(command, values)


# ---------------------------------------------------------------------------
# click
# ---------------------------------------------------------------------------


def _click_decorate(function, param: Param):
    """Apply one :class:`Param` to a click callback."""
    import click  # noqa: PLC0415 - only reached when click was selected

    if param.kind == "argument":
        return click.argument(
            param.dest,
            nargs=-1 if param.multiple else 1,
            required=param.required,
            type=param.type or str,
        )(function)

    if param.kind == "flag":
        return click.option(
            *param.flags,
            param.dest,
            is_flag=True,
            default=param.default,
            help=param.help,
        )(function)

    return click.option(
        *param.flags,
        param.dest,
        multiple=param.multiple,
        type=(
            click.Choice(list(param.choices)) if param.choices else (param.type or str)
        ),
        default=None if param.multiple else param.default,
        required=param.required,
        help=param.help,
        metavar=param.metavar,
    )(function)


def run_click(  # ruff: ignore[too-many-positional-arguments]
    commands: Sequence[Command],
    argv: Sequence[str] | None,
    prog: str,
    description: str,
    epilog: str,
    stdout: IO[str],
) -> tuple[str | None, argparse.Namespace]:
    """
    Parse ``argv`` with click.

    Returns
    -------
    command : str or None
        The resolved command name, ``"--version"``, or ``None``.
    args : argparse.Namespace
        The parsed namespace, in the same shape argparse would produce.

    Raises
    ------
    SystemExit
        Propagated from click for ``--help`` and usage errors, so the caller
        converts it to an exit code exactly as it does for argparse.

    Notes
    -----
    **Developer notes.** Click's group is built here and invoked with
    ``standalone_mode=False`` so that it returns rather than calling
    :func:`sys.exit`. The parsed result is captured into the same
    :class:`argparse.Namespace` the argparse path produces, and the *handler*
    runs afterwards in the caller. Dispatching inside the click callback would
    put the two frontends on different code paths, which is the one thing this
    design exists to prevent.
    """
    import click  # noqa: PLC0415 - only reached when click was selected

    captured: dict[str, Any] = {}

    @click.group(
        name=prog,
        help=description,
        epilog=epilog or None,
        invoke_without_command=True,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    @click.option(
        "-V",
        "--version",
        is_flag=True,
        help="print the submodule version and exit",
    )
    @click.option(
        "--frontend",
        type=click.Choice(list(FRONTENDS)),
        default=None,
        help="force a command-line frontend (default: click when installed)",
    )
    @click.pass_context
    def root(context: click.Context, version: bool, frontend: str | None) -> None:
        del frontend  # handled before parsing; accepted here for parity
        if version:
            captured["__command__"] = "--version"
            return
        if context.invoked_subcommand is None:
            stdout.write(context.get_help() + "\n")

    for command in commands:
        _attach_click_command(root, command, captured)

    try:
        result = root.main(
            args=list(argv) if argv is not None else None,
            prog_name=prog,
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:  # pragma: no cover - see the note below
        raise SystemExit(exc.exit_code) from exc
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(2) from exc
    except click.exceptions.Abort as exc:  # pragma: no cover - Ctrl-C
        raise SystemExit(130) from exc

    name = captured.pop("__command__", None)

    # In standalone_mode=False click *returns* the exit code for --help rather
    # than raising Exit, so the exception handler above almost never fires.
    # Without this, `--help` would land in the "no command given" branch and
    # report a usage error, which is what the parity test caught.
    if name is None and isinstance(result, int):
        raise SystemExit(result)

    if name in (None, "--version"):
        return name, argparse.Namespace(command=None, version=name == "--version")
    command = _index(commands)[name]
    return name, namespace_for(command, captured)


def _attach_click_command(group, command: Command, captured: dict[str, Any]) -> None:
    """Register one command on a click group, capturing its parsed values."""
    import click  # noqa: PLC0415

    def callback(**values: Any) -> None:
        captured["__command__"] = command.name
        captured.update(values)

    callback.__name__ = command.name.replace("-", "_")
    decorated = callback
    # Applied in reverse so that help lists them in declaration order.
    for param in reversed(command.params):
        decorated = _click_decorate(decorated, param)
    click_command = click.command(
        name=command.name, help=command.help_text, short_help=command.summary
    )(decorated)
    group.add_command(click_command)
    for alias in command.aliases:
        group.add_command(click_command, name=alias)


def _index(commands: Sequence[Command]) -> dict[str, Command]:
    """Return commands keyed by name and by alias."""
    index: dict[str, Command] = {}
    for command in commands:
        index[command.name] = command
        for alias in command.aliases:
            index[alias] = command
    return index


def load_runner(preferred: str | None = None):
    """
    Return ``(runner, frontend_name)``, falling back when click will not load.

    Parameters
    ----------
    preferred : str, optional
        An explicit choice, passed through to :func:`select_frontend`.

    Returns
    -------
    runner : callable
        :func:`run_click` or :func:`run_argparse`.
    name : str
        The frontend actually selected, which may differ from the preference.

    Raises
    ------
    ValueError
        If ``preferred`` names an unknown frontend.

    Notes
    -----
    **Developer notes.** :func:`select_frontend` answers "which frontend is
    wanted"; this answers "which one will actually work". The two differ when
    click is *present but unimportable* — a half-removed install, a broken
    dependency of click itself, a shadowing stub, an import blocker in a test
    harness.

    This is the same distinction the capability vocabulary draws between
    ``BROKEN`` and ``ABSENT``, and getting it wrong here had the same
    consequence: :func:`is_click_available` uses
    :func:`importlib.util.find_spec`, which reported click as present, the
    click runner was selected, and the whole command-line interface then died
    on the import — including ``--help``, which needs no third-party package at
    all.

    A tool whose base tier is the standard library must not be taken down by an
    optional convenience. Falling back is silent because there is nothing for
    the user to act on: the argparse frontend renders the same commands and
    produces the same output, which is what the parity tests exist to
    guarantee. ``doctor`` reports the frontend in use for anyone who wants to
    know.
    """
    name = select_frontend(preferred)
    if name == "click":
        try:
            import click  # noqa: F401, PLC0415 - probing that it really loads
        except Exception:  # noqa: BLE001 - any import failure means unusable
            return run_argparse, "argparse"
        return run_click, "click"
    return run_argparse, "argparse"
