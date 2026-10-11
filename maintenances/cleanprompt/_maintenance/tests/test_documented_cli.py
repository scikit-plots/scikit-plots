"""
Every command line the documentation shows must use options that exist.

Notes
-----
**User notes.** Scans the package README, every module docstring, the agent
skill files, the user guide and the gallery for ``cleanprompt <command> ...``
lines, and checks each ``--option`` against what that command declares in
``_cli.COMMANDS``.

**Developer notes — why (``CP-101``).** ``_custom.py``'s docstring told users
to run ``cleanprompt encode --pack-file hr_pack.yaml --pack hr``; ``encode``
has never accepted ``--pack-file``, and the line failed with "No such option".
The user guide had copied it. An instruction that cannot be followed is worse
than none, because it reads as reassurance (``CP-039``). Options are checked,
not prose: a line that mentions an option in passing, or demonstrates one
being refused, is listed in :data:`INTENDED` with the reason.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[4]
PACKAGE = CHECKOUT / "scikitplot" / "cleanprompt"

#: ``(file, command, option)`` that are shown on purpose, with why.
INTENDED = {
    # The README demonstrates that a mistyped option is refused, not redacted.
    ("scikitplot/cleanprompt/README.md", "inspect", "--secret"),
    # Prose listing plan's own options and ``--plan FILE`` on other commands.
    ("galleries/examples/cleanprompt/README.txt", "plan", "--plan"),
}

#: Options every command accepts through the root parser.
GLOBAL = frozenset({"--help", "--version", "--frontend"})

_CALL = re.compile(
    r"(?:\bcleanprompt|python -m scikitplot\.cleanprompt|scikitplot cleanprompt)"
    r"\s+([a-z][a-z-]*)([^\n]*)"
)
_OPTION = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def _commands():
    """Return ``{name or alias: declared long options}`` from the real table."""
    if "scikitplot" not in sys.modules:
        # A stand-in parent: the claims are about this submodule, whatever the
        # real parent package imports (the CP-085 lesson).
        stand_in = types.ModuleType("scikitplot")
        stand_in.__path__ = [str(CHECKOUT / "scikitplot")]
        sys.modules["scikitplot"] = stand_in
    from scikitplot.cleanprompt._cli import COMMANDS  # noqa: PLC0415 - after the stand-in parent

    table = {}
    for command in COMMANDS:
        options = {
            flag
            for param in command.params
            for flag in (param.flags or ())
            if flag.startswith("--")
        }
        for name in (command.name, *command.aliases):
            table[name] = options
    return table


def _documents():
    """Return the files whose command lines are checked."""
    if not PACKAGE.is_dir():
        pytest.skip(f"package not present in this checkout: {PACKAGE}")
    found = [PACKAGE / "README.md", PACKAGE / "_config" / "agent" / "SKILL.md"]
    found += sorted(PACKAGE.glob("*.py"))
    found += [CHECKOUT / "skills" / "cleanprompt" / "SKILL.md"]
    found += sorted((CHECKOUT / "docs" / "source" / "user_guide" / "cleanprompt").glob("*.rst"))
    gallery = CHECKOUT / "galleries" / "examples" / "cleanprompt"
    found += sorted(gallery.glob("*.py")) + sorted(gallery.glob("*.txt"))
    return [path for path in found if path.is_file()]


def _undeclared():
    """Return every documented option its command does not declare."""
    table = _commands()
    offenders = []
    for path in _documents():
        text = path.read_text(encoding="utf-8").replace("\\\n", " ")
        relative = path.relative_to(CHECKOUT).as_posix()
        for match in _CALL.finditer(text):
            command, rest = match.group(1), match.group(2)
            if command not in table:
                continue
            rest = rest.split(" -- ")[0]  # past the delimiter it is text
            for option in _OPTION.findall(rest):
                if option in table[command] | GLOBAL:
                    continue
                if (relative, command, option) in INTENDED:
                    continue
                offenders.append((relative, command, option))
    return offenders


def test_every_documented_option_exists():
    offenders = _undeclared()
    assert not offenders, (
        "documented command lines use options their command does not declare: "
        f"{offenders}"
    )


def test_the_check_can_fail():
    """A confirmation must be able to fail when the claim is false (lessons, rule 34)."""
    table = _commands()
    assert "--pack-file" not in table["encode"]
    assert "--pack-file" in table["batch"]


def test_every_exemption_is_still_needed():
    """An exemption that no longer matches anything is stale and hides nothing."""
    table = _commands()
    still = set()
    for path in _documents():
        text = path.read_text(encoding="utf-8").replace("\\\n", " ")
        relative = path.relative_to(CHECKOUT).as_posix()
        for match in _CALL.finditer(text):
            command = match.group(1)
            if command not in table:
                continue
            for option in _OPTION.findall(match.group(2).split(" -- ")[0]):
                still.add((relative, command, option))
    stale = INTENDED - still
    assert not stale, f"stale exemptions: {sorted(stale)}"
