"""
Interactive terminal session: paste, send, paste back, restore.

Notes
-----
**User notes.** ``python -m scikitplot.cleanprompt cli`` opens a session::

    1. paste your text, then press Ctrl-D (Ctrl-Z then Enter on Windows)
    2. the redacted copy is printed — send that to the model
    3. paste the reply, press Ctrl-D again
    4. your values come back

Between steps you can type a command instead of pasting: ``:hide TERM`` to hide
something the detectors missed, ``:allow TERM`` to stop hiding something,
``:suggest`` to see candidates, ``:why`` to see what is and is not being
detected, ``:again`` to re-run, and ``:quit`` to leave. Nothing is written to
disk unless you pass ``--save-vault``.

**Developer notes.** Upstream's entry point was this workflow and only this
workflow, driven by typing the word ``END`` on its own line. Two things changed.

*End-of-input is the terminal's job.* Ctrl-D is how a terminal says "that is all
the input", it composes with pipes and heredocs, and it cannot collide with the
text being pasted. A sentinel word can: a document containing a line reading
``END`` silently truncates, and that document is exactly the kind of thing
someone pastes into a redaction tool.

*The session is a loop over a state machine, not a straight line.* Upstream ran
one fixed sequence and exited, so a user who noticed a missed name had to start
over and paste everything again. Here the text stays in the session and every
command re-runs the pipeline over the **original** text, so corrections compose
instead of stacking.

Nothing here imports an optional dependency. The session is base tier.
"""

from __future__ import annotations

import argparse
from typing import IO

from ._diagnostics import describe_outcome, diagnose, suggest_terms
from ._engine import Redactor, restore
from ._exceptions import CleanPromptError
from ._render import ANSI, highlight_placeholders, should_colorize, summary_table

__all__ = ["run_session"]

_BANNER = """\
CleanPrompt interactive session
  Paste your text, then press Ctrl-D on a blank line to submit.
  Commands: :hide TERM   :allow TERM   :suggest   :why   :again   :quit
"""

_COMMAND_HELP = """\
  :hide TERM [TERM ...]   hide these exact strings as well
  :allow TERM [TERM ...]  stop hiding these
  :suggest                list capitalised candidates still in the clear
  :why                    what is being detected, and what is not
  :show                   print the redacted text again
  :again                  re-run detection over the original text
  :reset                  forget the current text and start over
  :quit                   leave the session
"""


def _paint(text: str, color: bool, code: str) -> str:
    """Wrap ``text`` in an ANSI colour when colour is enabled."""
    return f"{code}{text}{ANSI.RESET}" if color else text


def _read_block(
    stdin: IO[str],
    stdout: IO[str],
    prompt: str,
    color: bool,
) -> str | None:
    """
    Read a block of text until end-of-input.

    Returns
    -------
    str or None
        The text, or ``None`` when the stream closed with nothing pending.

    Notes
    -----
    **Developer notes.** ``EOFError`` from :func:`input` and an empty read both
    mean "the user is done". They are handled identically so that a piped
    heredoc and an interactive Ctrl-D behave the same, which is what makes the
    session testable without a pseudo-terminal.
    """
    stdout.write(_paint(prompt, color, ANSI.BOLD + ANSI.CYAN))
    stdout.flush()
    lines: list[str] = []
    while True:
        line = stdin.readline()
        if line == "":
            break
        lines.append(line)
    text = "".join(lines)
    return text if text.strip() else None


def _read_line(stdin: IO[str], stdout: IO[str], prompt: str, color: bool) -> str | None:
    """Read a single line, or ``None`` at end of input."""
    stdout.write(_paint(prompt, color, ANSI.BOLD + ANSI.CYAN))
    stdout.flush()
    line = stdin.readline()
    if line == "":
        return None
    return line.rstrip("\n")


def run_session(  # ruff: ignore[too-many-branches]
    args: argparse.Namespace,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """
    Run the interactive session.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from the ``cli`` subcommand.
    stdin, stdout, stderr : file-like
        Streams to use.

    Returns
    -------
    int
        Process exit status.

    Notes
    -----
    **Developer notes.** The session holds the **original** text and re-runs the
    whole pipeline whenever the term lists change. It never redacts an
    already-redacted string, which would be the staged-rewriting defect the
    engine exists to prevent.
    """
    from ._cli import (  # ruff: ignore[import-outside-top-level]
        EXIT_OK,
        _redactor_for,
        _write_vault,
    )

    color = should_colorize(
        stream=stdout,
        color={"always": True, "never": False}.get(getattr(args, "color", "auto")),
    )
    redactor, policy, settings = _redactor_for(args)
    hide = list(settings["hide"])
    allow = list(policy.allow)

    stdout.write(_paint(_BANNER, color, ANSI.BOLD))
    report = diagnose(policy, redactor.registry)
    stdout.write("  " + report.headline() + "\n")
    for spot in report.blind_spots:
        if spot.severity == "high":
            stdout.write(
                _paint(f"  ! {spot.category}\n", color, ANSI.MAGENTA),
            )
            stdout.write(f"    fix: {spot.remedy}\n")
    stdout.write("\n")

    original: str | None = None
    result = None

    def rebuild() -> None:
        """Re-run the pipeline over the original text."""
        nonlocal redactor, result
        current = policy.evolve(allow=tuple(allow)) if allow else policy
        redactor = Redactor(policy=current, registry=redactor.registry)
        result = redactor.redact(
            original or "",
            extra_terms=tuple(hide) or None,
            word_boundary=settings["word_boundary"],
        )

    def present() -> None:
        """Print the redacted text and its explanation."""
        assert result is not None  # ruff: ignore[assert]
        outcome = describe_outcome(
            result,
            diagnose(redactor.policy, redactor.registry),
            suggest_terms(original or "", result),
        )
        stdout.write("\n" + _paint("--- send this ---", color, ANSI.BOLD) + "\n")
        stdout.write(highlight_placeholders(result.text, redactor.policy, color=color))
        stdout.write("\n\n")
        marker = {"ok": ANSI.GREEN, "warning": ANSI.MAGENTA, "alert": ANSI.MAGENTA}[
            outcome["level"]
        ]
        stdout.write(_paint(outcome["headline"], color, marker) + "\n")
        if outcome["level"] != "ok":
            stdout.write("  " + outcome["detail"] + "\n")
        for action in outcome["actions"]:
            stdout.write(f"  -> {action}\n")
        if result.entries:
            stdout.write(summary_table(result, color=color, stream=stdout) + "\n")
        stdout.write("\n")

    while True:
        if original is None:
            text = _read_block(
                stdin,
                stdout,
                "Paste your text, then Ctrl-D:\n",
                color,
            )
            if text is None:
                stdout.write("\nnothing to do\n")
                return EXIT_OK
            original = text
            rebuild()
            present()

        command = _read_line(stdin, stdout, "cleanprompt> ", color)
        if command is None or command.strip() in (":quit", ":q"):
            break
        command = command.strip()
        if not command:
            continue

        try:
            if command in (":help", ":h", "?"):
                stdout.write(_COMMAND_HELP)
            elif command.startswith(":hide"):
                terms = command.split()[1:]
                if not terms:
                    stdout.write("usage: :hide TERM [TERM ...]\n")
                else:
                    hide.extend(terms)
                    rebuild()
                    present()
            elif command.startswith(":allow"):
                terms = command.split()[1:]
                if not terms:
                    stdout.write("usage: :allow TERM [TERM ...]\n")
                else:
                    allow.extend(terms)
                    hide[:] = [term for term in hide if term not in terms]
                    rebuild()
                    present()
            elif command == ":suggest":
                candidates = suggest_terms(original, result)
                if not candidates:
                    stdout.write("no further candidates\n")
                for item in candidates:
                    stdout.write(
                        f"  {item.text:<30} x{item.count}\n",
                    )
                stdout.write("\nhide any with:  :hide TERM\n")
            elif command == ":why":
                current = diagnose(redactor.policy, redactor.registry)
                stdout.write(current.headline() + "\n")
                stdout.write(
                    "  active: {}\n".format(", ".join(current.active_kinds)),
                )
                for spot in current.blind_spots:
                    stdout.write(
                        f"  [{spot.severity}] {spot.category}\n      fix: {spot.remedy}\n"
                    )
            elif command in (":show", ":again"):
                rebuild()
                present()
            elif command == ":reset":
                original = None
                result = None
                hide[:] = list(settings["hide"])
                allow[:] = list(policy.allow)
                continue
            elif command == ":restore":
                reply = _read_block(
                    stdin, stdout, "Paste the model's reply, then Ctrl-D:\n", color
                )
                if reply is None:
                    stdout.write("nothing pasted\n")
                    continue
                assert result is not None  # ruff: ignore[assert]
                back = restore(reply, result.vault, policy=redactor.policy)
                stdout.write(
                    "\n" + _paint("--- restored ---", color, ANSI.BOLD) + "\n",
                )
                stdout.write(back.text + "\n\n")
                if back.unknown:
                    stdout.write(
                        "note: {} placeholder(s) were not in the vault: {}\n".format(
                            len(back.unknown),
                            ", ".join(back.unknown),
                        )
                    )
            else:
                stdout.write(
                    f"unknown command {command!r}; try :help\n",
                )
        except CleanPromptError as exc:
            stderr.write(f"error: {exc}\n")

    if getattr(args, "save_vault", None) and result is not None:
        _write_vault(args.save_vault, result.vault, redactor.policy)
        stdout.write(f"vault written to {args.save_vault}\n")
    stdout.write("bye\n")
    return EXIT_OK
