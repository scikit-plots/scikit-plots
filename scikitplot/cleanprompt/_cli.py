"""
Command-line interface for :mod:`scikitplot.cleanprompt`.

Notes
-----
**User notes.** The pair that does the work::

    python -m scikitplot.cleanprompt encode "Mail ada@example.com"   # -> paste this
    python -m scikitplot.cleanprompt decode "I mailed [EMAIL-1]."    # <- paste the answer

``encode`` puts the redacted text on standard output and nothing else, so it
can be piped or copied as it stands. ``decode`` reads the model's answer and
puts the values back. Neither needs a path: they share a vault at a default
location, in append mode, so a placeholder means the same thing in every turn
of one conversation. ``forget`` deletes that vault when the conversation ends.

The rest, grouped by what you are trying to do::

    # see both halves at once, with no vault file at all
    python -m scikitplot.cleanprompt roundtrip "Mail ada@example.com"

    # look before you leap
    python -m scikitplot.cleanprompt doctor --format json
    python -m scikitplot.cleanprompt inspect "your text here"
    python -m scikitplot.cleanprompt kinds

    # scripted and explicit: a named vault, a report of what was removed
    python -m scikitplot.cleanprompt redact --in prompt.txt --vault v.json
    python -m scikitplot.cleanprompt decode --in reply.txt  --vault v.json
    python -m scikitplot.cleanprompt scan   --in prompt.txt     # CI gate
    python -m scikitplot.cleanprompt forget --force             # delete the vault

    # interactive
    python -m scikitplot.cleanprompt cli                        # paste-and-go
    python -m scikitplot.cleanprompt flask                      # web interface
    python -m scikitplot.cleanprompt docker                     # container files

``doctor`` is the one to run first. It says which detectors are live, which are
installable, and what the current configuration is blind to.

Every command that takes text accepts it directly as an argument, from a file
with ``--in``, or on standard input. For prose, prefer a heredoc: a shell
consumes ``(``, ``)``, ``$`` and friends before this program runs, and
``<<'END'`` with a quoted delimiter disables every substitution::

    python -m scikitplot.cleanprompt roundtrip --ner <<'END'
    ... paste anything at all ...
    END

**Developer notes — the shape of this surface.**

*Why* ``doctor`` *and not* ``capabilities``. ``doctor`` is the name the rest of
this project already uses for "diagnose this installation", and a submodule that
invents its own word for an existing concept makes the user learn which one
applies where. The payload follows the same shape too: a mapping ending in a
``status`` key, rendered through one ``--format`` option.

*Why every command takes* ``--format``. A CLI whose output is only human-shaped
cannot be used by anything else. ``json`` is the contract for scripts, ``text``
for people, and ``yaml``/``toml`` are offered because the rest of the project
offers them. The format vocabulary and the exit-code vocabulary are both
consumed **by value** — copied, not imported — because this submodule must stay
independently importable.

*Stream discipline.* Results go to standard output, diagnostics to standard
error. That is what lets ``redact`` be used in a pipe.

*Why this is not only an interactive prompt loop.* Upstream's entry point was a
fixed conversation: paste, type ``END``, paste the reply, type ``END``. That
cannot be scripted, cannot be tested without driving standard input, and loses
everything if the terminal closes. The interactive workflow is still here as
``cli`` — it is genuinely the nicest way to use the tool by hand — but it is one
subcommand among several rather than the only way in.

The vault file is created with mode ``0600`` where the platform supports it. It
holds the removed values in clear text unless ``--encrypt`` is given, which uses
the ``crypto`` tier and a key *you* supply; a key written next to the file it
protects would be decoration.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from typing import IO, Any, Callable, Sequence

from ._artifacts import ArtifactPlan, encode_artifact
from ._detectors import default_registry
from ._diagnostics import describe_outcome, diagnose, suggest_terms
from ._documents import detect_format
from ._engine import Redactor, restore
from ._engines import DEFAULT_ENGINE, ENGINE_MODES, build_detectors, describe_engines
from ._exceptions import CapabilityError, CleanPromptError, PolicyError
from ._files import atomic_write, locked
from ._frontends import build_argparse, load_runner
from ._languages import (
    DEFAULT_LANGUAGE,
    MODEL_SIZES,
    language_report,
)
from ._logging import configure_logging, get_logger, log_level_from_env
from ._patterns import PATTERNS
from ._plan import CleanPlan, FluentCleanPrompt
from ._policy import (
    DEFAULT_POLICY,
    PROFILES,
    OverlapStrategy,
    RedactionPolicy,
    profile,
)
from ._render import highlight_placeholders, restoration_note, summary_table
from ._schema import ROLES
from ._spec import Command, Param
from ._vault import Vault

__all__ = ["build_parser", "main"]

#: Version of the on-disk vault document this build writes.
#:
#: Format 2 adds an ``index`` array carrying each placeholder's category and
#: ordinal, which ``--vault-mode append`` needs in order to reuse the labels a
#: previous run issued.
VAULT_FORMAT = 2

#: Versions this build can read. Format 1 has no ``index``, so it restores
#: normally and cannot be appended to; see :func:`_seed_from_vault`.
VAULT_FORMATS_READ = (1, 2)

#: Output formats, matching the project-wide vocabulary.
FORMATS = ("text", "json", "yaml", "toml")

# Exit codes, consumed by value from the project's vocabulary rather than
# imported, because this submodule imports no sibling package. Keep the numbers
# aligned with ``scikitplot._cli.exit_codes``; they are sysexits conventions.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_FOUND = 3  # scan: sensitive values were present
EXIT_UNAVAILABLE = 69
EXIT_INTERRUPTED = 130

#: Status for "the reader closed the pipe". ``128 + SIGPIPE`` is what a shell
#: reports for a process the kernel killed with that signal, so a pipeline sees
#: the same number whether the reader closed early or this process noticed the
#: closed descriptor first. Windows has no ``SIGPIPE``; there the honest answer
#: is the ordinary error status.
_SIGPIPE = getattr(signal, "SIGPIPE", None)
EXIT_BROKEN_PIPE = EXIT_ERROR if _SIGPIPE is None else 128 + int(_SIGPIPE)

#: Default config file names, searched upward from the working directory.
CONFIG_NAMES = (".cleanprompt.toml", "cleanprompt.toml")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _emit(data: Any, fmt: str, stream: IO[str]) -> None:
    """
    Render ``data`` in ``fmt`` to ``stream``.

    Parameters
    ----------
    data : object
        A JSON-safe mapping or value.
    fmt : str
        One of :data:`FORMATS`.
    stream : file-like
        Destination.

    Raises
    ------
    CapabilityError
        If ``fmt`` needs a writer that is not installed.

    Notes
    -----
    **Developer notes.** ``json`` does not sort keys: the caller's ordering is
    information (a report reads top-down), and sorting would scramble it.
    """
    if fmt == "json":
        json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=False)
        stream.write("\n")
        return
    if fmt == "yaml":
        try:
            import yaml  # noqa: PLC0415 - optional writer, deferred
        except ImportError as exc:
            raise CapabilityError(
                "YAML output needs PyYAML; install it with: pip install pyyaml",
                tier="yaml",
                status="ABSENT",
                install_hint="pip install pyyaml",
            ) from exc
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
        return
    if fmt == "toml":
        rendered = _toml_writer()(_toml_safe(data))
        stream.write(rendered)
        if not rendered.endswith("\n"):
            stream.write("\n")
        return
    _render_text(data, stream)


def _toml_writer() -> Callable[[Any], str]:
    """
    Return a TOML ``dumps`` callable.

    Raises
    ------
    CapabilityError
        If no TOML writer is installed.

    Notes
    -----
    **Developer notes.** The standard library's :mod:`tomllib` is read-only, so
    it is never a writer here however tempting its presence looks.
    """
    from importlib import import_module  # ruff: ignore[import-outside-top-level]

    # Ordered providers. tomllib is absent on purpose: the standard library's
    # TOML support is read-only, however tempting its presence looks.
    for module_name, attribute in (("tomli_w", "dumps"), ("toml", "dumps")):
        try:
            module = import_module(module_name)
        except ImportError:
            continue  # try the next provider; exhaustion is reported below
        return getattr(module, attribute)

    raise CapabilityError(
        "TOML output needs a writer; install one with: pip install tomli-w",
        tier="toml",
        status="ABSENT",
        install_hint="pip install tomli-w",
    )


def _toml_safe(data: Any) -> Any:
    """
    Return ``data`` with ``None`` values dropped, for TOML.

    Notes
    -----
    **Developer notes.** TOML has no null. A writer handed ``None`` either
    raises or invents a representation, and both are worse than the honest
    answer: a key whose value is unknown is simply absent, which TOML expresses
    natively. This is the documented lossy edge of the ``toml`` format, and it
    is why ``json`` is the format to script against.
    """
    if isinstance(data, dict):
        return {
            key: _toml_safe(value) for key, value in data.items() if value is not None
        }
    if isinstance(data, list):
        return [_toml_safe(item) for item in data if item is not None]
    return data


def _render_text(data: Any, stream: IO[str], indent: int = 0) -> None:
    """Render a mapping as indented key/value text."""
    pad = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                stream.write(f"{pad}{key}:\n")
                _render_text(value, stream, indent + 1)
            else:
                stream.write(f"{pad}{key}: {_scalar(value)}\n")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _render_text(item, stream, indent)
                stream.write("\n")
            else:
                stream.write(f"{pad}- {_scalar(item)}\n")
    else:
        stream.write(f"{pad}{_scalar(data)}\n")


def _scalar(value: Any) -> str:
    """Render one scalar for text output."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


# ---------------------------------------------------------------------------
# shared options
# ---------------------------------------------------------------------------


# -- neutral command declarations -------------------------------------------
#
# Declared once, rendered by both frontends. A subcommand added here appears in
# argparse and in click with no further work, and cannot appear in one and not
# the other.

FORMAT = Param(
    dest="fmt",
    flags=("-f", "--format"),
    choices=FORMATS,
    default="text",
    help="Output format (text, json, yaml, toml).",
)

IN = Param(
    dest="input_path",
    flags=("-i", "--in"),
    metavar="PATH",
    help=(
        "Read the text from this file. Omit it to pass the text directly as "
        "an argument, or to read standard input."
    ),
)

#: The text itself, given directly on the command line.
#:
#: Variadic so that an unquoted phrase still arrives intact rather than being
#: rejected as an unexpected extra argument. A shell splits ``Ada Lovelace``
#: into two words before this program ever sees it, and refusing the second
#: one would be a distinction the user never made.
#:
#: This does not rescue text containing shell metacharacters — no program can,
#: because the shell consumes them first. It does mean that the obvious thing
#: to type now works, and that the failure mode for the rest is the shell's own
#: error rather than ours.
TEXT = Param(
    dest="text_args",
    kind="argument",
    multiple=True,
    metavar="TEXT",
    help=(
        "The text to process, given directly. Omit it to use --in, or to read "
        "standard input."
    ),
)

OUT = Param(
    dest="output_path",
    flags=("-o", "--out"),
    metavar="PATH",
    help="Output file; omit or use '-' to write standard output.",
)


def _detection_params() -> tuple[Param, ...]:
    """
    Return the detection options shared by every command that redacts.

    Notes
    -----
    **Developer notes.** One definition, so ``redact``, ``inspect``, ``scan``,
    ``cli`` and ``flask`` cannot drift apart. A flag that works on ``inspect``
    but silently does nothing on ``redact`` would make the dry run a lie.
    """
    return (
        Param(
            dest="profile",
            flags=("--profile",),
            choices=tuple(sorted(PROFILES)),
            help="Named policy bundle (default: balanced).",
        ),
        Param(
            dest="config",
            flags=("--config",),
            metavar="PATH",
            help="Config file; by default {} is searched upward.".format(
                " or ".join(CONFIG_NAMES)
            ),
        ),
        Param(
            dest="kinds",
            flags=("-k", "--kinds"),
            multiple=True,
            metavar="KIND",
            help="Detection kinds to enable; default is every built-in pattern.",
        ),
        Param(
            dest="hide",
            flags=("--hide",),
            multiple=True,
            metavar="TERM",
            help="Additional exact strings to hide.",
        ),
        Param(
            dest="allow",
            flags=("--allow",),
            multiple=True,
            metavar="TERM",
            help="Surfaces that must never be redacted.",
        ),
        Param(
            dest="word_boundary",
            flags=("--word-boundary",),
            kind="flag",
            default=False,
            help="Match --hide terms only at word boundaries.",
        ),
        Param(
            dest="ignore_case",
            flags=("--ignore-case",),
            kind="flag",
            default=False,
            help="Treat values differing only in case as one entity.",
        ),
        Param(
            dest="ner",
            flags=("--ner",),
            kind="flag",
            default=False,
            help="Also run named-entity detection (requires the 'ner' tier).",
        ),
        Param(
            dest="ner_engine",
            flags=("--ner-engine",),
            choices=ENGINE_MODES,
            default=DEFAULT_ENGINE,
            help=(
                "Which entity engine: auto (spaCy, else NLTK), spacy, nltk, "
                "both (union: higher recall, more noise) or none."
            ),
        ),
        Param(
            dest="language",
            flags=("--lang",),
            default=DEFAULT_LANGUAGE,
            metavar="CODE",
            help=(
                "Language for entity detection, e.g. en, de, fr, es, ja. "
                "Unsupported codes fall back to the multilingual model."
            ),
        ),
        Param(
            dest="model_size",
            flags=("--model-size",),
            choices=MODEL_SIZES,
            default="sm",
            help="Preferred spaCy model size (sm, md, lg, trf).",
        ),
        Param(
            dest="ner_model",
            flags=("--ner-model",),
            default=None,
            metavar="NAME",
            help="Explicit spaCy model, overriding --lang and --model-size.",
        ),
        Param(
            dest="overlap",
            flags=("--overlap",),
            choices=tuple(strategy.value for strategy in OverlapStrategy),
            help="How to arbitrate overlapping detections.",
        ),
    )


LOG_LEVEL = Param(
    dest="log_level",
    flags=("--log-level",),
    choices=("critical", "error", "warning", "info", "debug"),
    default=None,
    help="Log to standard error at this level (default: warning).",
)

LOG_FORMAT = Param(
    dest="log_format",
    flags=("--log-format",),
    choices=("text", "json"),
    default="text",
    help="Log record format.",
)

COLOR = Param(
    dest="color",
    flags=("--color",),
    choices=("auto", "always", "never"),
    default="auto",
    help="Colourise placeholders in terminal output.",
)

#: Artefact handling. These five turn ``encode`` from a prose command into one
#: that understands a notebook or a module.
#:
#: They are shared Params rather than per-command definitions for the reason
#: ``CP-039`` exists: an option present on one command and missing from its
#: pair is a gap a user falls into. The command IR also refuses a duplicate, so
#: sharing is what keeps the two frontends agreeing about them.
AS_FORMAT = Param(
    dest="as_format",
    flags=("--as",),
    choices=("auto", "text", "python", "notebook"),
    default="auto",
    help=(
        "How to read the input: auto (from the file name), text, python or "
        "notebook. A notebook keeps its structure; only the strings change."
    ),
)

#: Whether a column's role may be guessed from its name.
#:
#: Off by default, and that is the module's claim to being deductive rather
#: than a word list: a name is not evidence of a type. With it on, the report
#: says a role was inferred, so a reader can tell a fact from a suggestion.
INFER_ROLES = Param(
    dest="infer_roles",
    flags=("--infer-roles",),
    kind="flag",
    default=False,
    help=(
        "Guess each column's role from its name, so a stand-in reads as "
        "amount_1 rather than field_1. Reported as inferred."
    ),
)

#: Roles the user states outright, which nothing else may override.
COLUMNS = Param(
    dest="columns",
    flags=("--columns",),
    metavar="NAME:ROLE,...",
    help=(
        "State a column's role, e.g. 'acct_balance_usd:amount,churned:target'. "
        "Wins over anything discovered or inferred."
    ),
)

#: Columns discovery could not establish, named by hand.
ADD_COLUMN = Param(
    dest="add_columns",
    flags=("--add-column",),
    metavar="NAME",
    multiple=True,
    help=(
        "Also hide this column. Use it to answer a 'suggested column list' "
        "line in the report."
    ),
)

#: Parts of an artefact that are removed unless asked for.
#:
#: Each is data rather than description: an output is the rows themselves, a
#: figure can show what the table above it no longer says, and a path names the
#: organisation, the environment and often the analyst. Keeping one is a
#: decision, so it is spelled as one.
KEEP = Param(
    dest="keep",
    flags=("--keep",),
    metavar="WHAT",
    multiple=True,
    choices=("outputs", "figures", "paths"),
    help=(
        "Send parts that are removed by default: outputs (rendered tables and "
        "printed results), figures (embedded images), paths (file names and "
        "URIs)."
    ),
)

QUIET = Param(
    dest="quiet",
    flags=("-q", "--quiet"),
    kind="flag",
    default=False,
    help="Suppress the summary on standard error.",
)

#: Whether a run starts a fresh vault or continues the previous one.
VAULT_MODE = Param(
    dest="vault_mode",
    flags=("--vault-mode",),
    choices=("overwrite", "append"),
    default="overwrite",
    help=(
        "overwrite: start a fresh vault (default). append: continue the "
        "existing one, so a value keeps the placeholder it already had."
    ),
)

#: What a removed value is replaced with.
STYLE = Param(
    dest="style",
    flags=("--style",),
    choices=("placeholder", "surrogate"),
    default="placeholder",
    help=(
        "placeholder: [PERSON-1] (default, unmistakable). surrogate: an "
        "invented but ordinary-looking value, which reads better and which a "
        "model will not rewrite. Credentials keep placeholders either way."
    ),
)

#: Encrypt the vault's values at rest.
ENCRYPT = Param(
    dest="encrypt",
    flags=("--encrypt",),
    kind="flag",
    default=False,
    help=(
        "Encrypt the removed values with a passphrase. Needs nothing "
        "installed; you are asked for the passphrase, or set "
        "CLEANPROMPT_VAULT_KEY."
    ),
)

#: Which construction encrypts the vault.
CIPHER = Param(
    dest="cipher",
    flags=("--cipher",),
    choices=("portable", "auto", "fernet"),
    default="portable",
    help=(
        "portable: standard library only, opens on any machine (default). "
        "fernet: the audited AES backend, needs the 'crypto' tier. auto: "
        "fernet when available, else portable."
    ),
)

REVEAL = Param(
    dest="reveal",
    flags=("--reveal",),
    kind="flag",
    default=False,
    help="Include the removed values in the report.",
)


def _pack_params() -> tuple[Param, ...]:
    """
    Return the pack selection options shared by ``packs`` and ``batch``.

    Notes
    -----
    **Developer notes.** One definition, so the command that lists packs and
    the command that uses them accept the same custom files the same way.
    """
    return (
        Param(
            dest="packs",
            flags=("--pack",),
            multiple=True,
            metavar="NAME",
            help=(
                "Packs to use: names, or all, auto or none (default: auto, "
                "each file's format chooses)."
            ),
        ),
        Param(
            dest="pack_files",
            flags=("--pack-file",),
            multiple=True,
            metavar="PATH",
            help="Your own pack or format definitions (.yaml, .yml or .json).",
        ),
        Param(
            dest="replace_builtins",
            flags=("--replace-builtins",),
            kind="flag",
            default=False,
            help="Let a --pack-file definition replace a built-in of the same name.",
        ),
    )


COMMANDS: tuple[Command, ...] = (
    Command(
        name="redact",
        summary="Redact a text and write a vault, with a report of what was removed.",
        handler="_cmd_redact",
        description=(
            "Replace sensitive values with placeholders and write the vault "
            "needed to restore them."
        ),
        params=(
            TEXT,
            IN,
            OUT,
            Param(
                dest="vault",
                flags=("--vault",),
                metavar="PATH",
                help=(
                    "Where to write the vault (holds the removed values). "
                    "Defaults to the state-directory vault; 'doctor' prints "
                    "the resolved path."
                ),
            ),
            VAULT_MODE,
            STYLE,
            ENCRYPT,
            CIPHER,
            *_detection_params(),
            REVEAL,
            QUIET,
            COLOR,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="decode",
        summary="Put the values back into the model's answer. Pairs with 'encode'.",
        handler="_cmd_restore",
        aliases=("restore",),
        description=(
            "The other half of the pair. Paste in what the model replied and "
            "every placeholder becomes the value it stood for. With no "
            "--vault it reads the same default vault 'encode' writes, so an "
            "ordinary conversation needs no paths at all."
        ),
        params=(
            TEXT,
            IN,
            OUT,
            Param(
                dest="vault",
                flags=("--vault",),
                metavar="PATH",
                help=(
                    "Vault written by 'redact' or 'clean'. Defaults to the "
                    "same state-directory vault those commands write."
                ),
            ),
            Param(
                dest="strict",
                flags=("--strict",),
                kind="flag",
                default=False,
                help=(
                    "Fail when the text contains a placeholder the vault does not hold."
                ),
            ),
            Param(
                dest="exact",
                flags=("--exact",),
                kind="flag",
                default=False,
                help=(
                    "Match only placeholders spelled exactly as issued. By "
                    "default one the model rewrote ([EMAIL_1] for [EMAIL-1]) "
                    "is matched too, and every such repair is reported."
                ),
            ),
            QUIET,
        ),
    ),
    Command(
        name="inspect",
        summary="Dry run: show what would be redacted, and what would not.",
        handler="_cmd_inspect",
        description=(
            "Report what a redaction would do to a text without producing one. "
            "Includes suggested terms the active detectors did not claim."
        ),
        params=(
            TEXT,
            IN,
            AS_FORMAT,
            INFER_ROLES,
            COLUMNS,
            ADD_COLUMN,
            KEEP,
            *_detection_params(),
            Param(
                dest="suggest_min_tokens",
                flags=("--suggest-min-tokens",),
                type=int,
                default=1,
                metavar="N",
                help="Least capitalised words a suggestion must have (default: 1).",
            ),
            Param(
                dest="no_suggest",
                flags=("--no-suggest",),
                kind="flag",
                default=False,
                help="Skip the suggestion pass.",
            ),
            REVEAL,
            FORMAT,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="encode",
        summary="Text in, a prompt you can paste into any chat out. Pairs with 'decode'.",
        handler="_cmd_clean",
        aliases=("clean", "prompt"),
        description=(
            "Half of the pair. 'encode' replaces the sensitive values and puts "
            "the result on standard output alone, ready to paste into any "
            "chat; 'decode' takes the answer that comes back and puts the "
            "values into it. Neither needs a path: the vault goes to the "
            "default location in append mode, so a placeholder means the same "
            "thing in every turn of one conversation."
        ),
        params=(
            TEXT,
            IN,
            OUT,
            Param(
                dest="vault",
                flags=("--vault",),
                metavar="PATH",
                help=(
                    "Where to keep the removed values. Defaults to the "
                    "state-directory vault; 'doctor' prints the resolved path."
                ),
            ),
            Param(
                dest="vault_mode",
                flags=("--vault-mode",),
                choices=("append", "overwrite"),
                default="append",
                help=(
                    "append: keep the placeholders a value already had "
                    "(default, and what a conversation wants). overwrite: "
                    "start a fresh vault."
                ),
            ),
            STYLE,
            ENCRYPT,
            CIPHER,
            AS_FORMAT,
            INFER_ROLES,
            COLUMNS,
            ADD_COLUMN,
            KEEP,
            *_detection_params(),
            QUIET,
            COLOR,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="forget",
        summary="Delete the vault, ending the conversation it held.",
        handler="_cmd_forget",
        aliases=("clear-vault",),
        description=(
            "'encode' keeps the removed values in a vault so 'decode' can put "
            "them back. When the conversation is finished, this deletes it. "
            "Prints what it would remove and stops; pass --force to go "
            "through with it."
        ),
        params=(
            Param(
                dest="vault",
                flags=("--vault",),
                metavar="PATH",
                help="Which vault to delete. Defaults to the one 'encode' writes.",
            ),
            Param(
                dest="force",
                flags=("--force",),
                kind="flag",
                default=False,
                help="Actually delete it. Without this, nothing is removed.",
            ),
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="roundtrip",
        summary="Show the whole loop in one shot: redact, send, restore.",
        handler="_cmd_roundtrip",
        aliases=("demo",),
        description=(
            "Redact a text, show what you would send, show a reply coming "
            "back, and restore the values into it — in one command, with no "
            "vault file to manage. Use --reply to feed in a real model's "
            "answer; without it a stand-in reply is shown and labelled as one."
        ),
        params=(
            TEXT,
            IN,
            Param(
                dest="reply",
                flags=("--reply",),
                metavar="TEXT",
                help="The model's reply, given directly.",
            ),
            Param(
                dest="reply_in",
                flags=("--reply-in",),
                metavar="PATH",
                help="Read the model's reply from this file.",
            ),
            STYLE,
            *_detection_params(),
            REVEAL,
            FORMAT,
            COLOR,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="scan",
        summary="Exit non-zero when sensitive values are present (CI gate).",
        handler="_cmd_scan",
        description=(
            "Report whether a text contains sensitive values. Exit status 0 "
            "when clean, 3 when something was found, so it can gate a commit "
            "or a pipeline."
        ),
        params=(
            TEXT,
            IN,
            *_detection_params(),
            Param(
                dest="max_findings",
                flags=("--max-findings",),
                type=int,
                default=0,
                metavar="N",
                help="Tolerate up to N findings before failing (default: 0).",
            ),
            FORMAT,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="doctor",
        summary="Diagnose this installation: what is active, what is blind.",
        handler="_cmd_doctor",
        aliases=("capabilities",),
        description=(
            "Report optional tier status, active detectors and blind spots, "
            "with the exact remedy for each gap. Pass --in to diagnose a text "
            "as well."
        ),
        params=(
            *_detection_params(),
            IN,
            Param(
                dest="new_key",
                flags=("--new-key",),
                kind="flag",
                default=False,
                help=(
                    "Print a fresh vault passphrase and exit. With --cipher "
                    "fernet, a Fernet key instead."
                ),
            ),
            CIPHER,
            FORMAT,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="kinds",
        summary="List the built-in detection kinds.",
        handler="_cmd_kinds",
        params=(FORMAT,),
    ),
    Command(
        name="cli",
        summary="Interactive paste-and-go session in the terminal.",
        handler="_cmd_session",
        aliases=("session", "repl"),
        description=(
            "Paste text, get a redacted copy to send, paste the reply back, "
            "get your values restored. Everything stays in this process."
        ),
        params=(
            *_detection_params(),
            COLOR,
            Param(
                dest="save_vault",
                flags=("--save-vault",),
                metavar="PATH",
                help="Also write the vault to a file when the session ends.",
            ),
        ),
    ),
    Command(
        name="flask",
        summary="Start the local web interface.",
        handler="_cmd_flask",
        aliases=("web", "serve"),
        description="Start the Flask web interface.",
        params=(
            *_detection_params(),
            Param(
                dest="host",
                flags=("--host",),
                metavar="ADDR",
                help="Bind address (default: 127.0.0.1, or 0.0.0.0 with --docker).",
            ),
            Param(
                dest="port",
                flags=("--port",),
                type=int,
                default=5000,
                help="Bind port.",
            ),
            Param(
                dest="docker",
                flags=("--docker",),
                kind="flag",
                default=False,
                help=(
                    "Container mode: bind 0.0.0.0 so the host can reach the "
                    "app. This exposes it to every network the container sees."
                ),
            ),
            Param(
                dest="allow_remote",
                flags=("--allow-remote",),
                kind="flag",
                default=False,
                help="Acknowledge binding to a non-loopback address.",
            ),
            Param(
                dest="debug",
                flags=("--debug",),
                kind="flag",
                default=False,
                help="Enable Flask debug mode.",
            ),
            Param(
                dest="open",
                flags=("--open",),
                kind="flag",
                default=False,
                help="Open a browser once the server is up.",
            ),
        ),
    ),
    Command(
        name="packs",
        summary="List, show or check the packs and formats (YAML definitions).",
        handler="_cmd_packs",
        description=(
            "List every pack (what to hide) and format (how a file is read), "
            "show one in full, or check that definitions load. Add --pack-file "
            "to include your own YAML or JSON definitions."
        ),
        params=(
            Param(
                dest="show",
                flags=("--show",),
                metavar="NAME",
                help="Show one pack or format in full.",
            ),
            Param(
                dest="check",
                flags=("--check",),
                kind="flag",
                default=False,
                help=(
                    "Check that every definition loads and that the compiled "
                    "catalogue matches the YAML (exit 1 on any problem)."
                ),
            ),
            Param(
                dest="compile",
                flags=("--compile",),
                kind="flag",
                default=False,
                help=(
                    "Maintainers: rebuild the compiled catalogue from the YAML "
                    "in the installed package (needs PyYAML)."
                ),
            ),
            *_pack_params(),
            FORMAT,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="batch",
        summary="Encode a folder or zip into a safe copy, or decode one back.",
        handler="_cmd_batch",
        description=(
            "Encode every file a selected format reads, under one vault, into "
            "a mirror folder or a new zip. Files no format reads are skipped "
            "and files that cannot be read safely are refused; neither is "
            "written. With --decode, restore an encoded folder or zip."
        ),
        params=(
            Param(
                dest="source",
                kind="argument",
                metavar="SOURCE",
                help="A folder, or a .zip archive.",
            ),
            Param(
                dest="target",
                flags=("-o", "--out"),
                metavar="PATH",
                help="The folder (or .zip, for a zip source) to write. Required unless --dry-run.",
            ),
            Param(
                dest="dry_run",
                flags=("--dry-run",),
                kind="flag",
                default=False,
                help="Report what would be encoded, skipped or refused, by kind; write nothing.",
            ),
            Param(
                dest="plan",
                flags=("--plan",),
                metavar="FILE",
                help=(
                    "Use a saved plan file (see 'plan --write'); refused if the packs "
                    "it relies on changed since it was saved."
                ),
            ),
            Param(
                dest="decode",
                flags=("--decode",),
                kind="flag",
                default=False,
                help="Restore SOURCE with the vault instead of encoding it.",
            ),
            *_pack_params(),
            Param(
                dest="file_formats",
                flags=("--file-format",),
                multiple=True,
                metavar="NAME",
                help="Read only these formats, by name or extension (default: all).",
            ),
            Param(
                dest="profile",
                flags=("--profile",),
                choices=tuple(sorted(PROFILES)),
                help="Named policy bundle for the core patterns (default: balanced).",
            ),
            Param(
                dest="no_core",
                flags=("--no-core",),
                kind="flag",
                default=False,
                help="Run only the packs: switch the built-in patterns off.",
            ),
            Param(
                dest="hide",
                flags=("--hide",),
                multiple=True,
                metavar="TERM",
                help="Additional exact strings to hide.",
            ),
            Param(
                dest="allow",
                flags=("--allow",),
                multiple=True,
                metavar="TERM",
                help="Surfaces that must never be redacted.",
            ),
            Param(
                dest="ner",
                flags=("--ner",),
                kind="flag",
                default=False,
                help="Also run named-entity detection (requires the 'ner' tier).",
            ),
            Param(
                dest="no_remember",
                flags=("--no-remember",),
                kind="flag",
                default=False,
                help=(
                    "Do not hide a value where it recurs unless a rule finds it "
                    "again (by default a value hidden once is hidden everywhere)."
                ),
            ),
            STYLE,
            KEEP,
            INFER_ROLES,
            Param(
                dest="vault",
                flags=("--vault",),
                metavar="PATH",
                help=(
                    "Where the vault is written (encode) or read (decode). "
                    "Defaults to the state-directory vault 'decode' reads."
                ),
            ),
            VAULT_MODE,
            ENCRYPT,
            CIPHER,
            FORMAT,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="ask",
        summary="Send one guarded prompt through any command-line model.",
        handler="_cmd_ask",
        description=(
            "Encode the text, check that no removed value remains, pipe it to "
            "the model command on its standard input, and decode the answer as "
            "it streams back. The values stay in memory and are never written "
            "to disk. The command runs without a shell."
        ),
        params=(
            TEXT,
            IN,
            Param(
                dest="via",
                flags=("--via",),
                metavar="COMMAND",
                required=True,
                help="The model command, e.g. 'ollama run llama3' or 'llm -m gpt-4o'.",
            ),
            Param(
                dest="plan",
                flags=("--plan",),
                metavar="FILE",
                help=(
                    "Use a saved plan file (see 'plan --write'); refused if the packs "
                    "it relies on changed since it was saved."
                ),
            ),
            Param(
                dest="read_as",
                flags=("--read-as",),
                metavar="FORMAT",
                default="text",
                help="How the prompt is read: text (default), markdown, json, python, ...",
            ),
            *_pack_params(),
            Param(
                dest="profile",
                flags=("--profile",),
                choices=tuple(sorted(PROFILES)),
                help="Named policy bundle for the core patterns (default: balanced).",
            ),
            Param(
                dest="hide",
                flags=("--hide",),
                multiple=True,
                metavar="TERM",
                help="Additional exact strings to hide.",
            ),
            Param(
                dest="allow",
                flags=("--allow",),
                multiple=True,
                metavar="TERM",
                help="Surfaces that must never be redacted.",
            ),
            Param(
                dest="ner",
                flags=("--ner",),
                kind="flag",
                default=False,
                help="Also run named-entity detection (requires the 'ner' tier).",
            ),
            STYLE,
            Param(
                dest="no_remember",
                flags=("--no-remember",),
                kind="flag",
                default=False,
                help=(
                    "Do not hide a value where it recurs unless a rule finds it "
                    "again (by default a value hidden once is hidden everywhere)."
                ),
            ),
            Param(
                dest="show_sent",
                flags=("--show-sent",),
                kind="flag",
                default=False,
                help="Also print exactly what was sent, on standard error.",
            ),
            Param(
                dest="timeout",
                flags=("--timeout",),
                type=float,
                metavar="SECONDS",
                help="Stop the command after this many seconds.",
            ),
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="mcp",
        summary="Serve cleanprompt to any MCP agent over standard input and output.",
        handler="_cmd_mcp",
        description=(
            "Run a Model Context Protocol server (JSON-RPC over stdio, standard "
            "library only). Agents read files through it as encoded text and "
            "write model output back with values restored locally; no tool ever "
            "returns a removed value. Files are confined to the --root folders."
        ),
        params=(
            Param(
                dest="plan",
                flags=("--plan",),
                metavar="FILE",
                help=(
                    "Use a saved plan file (see 'plan --write'); refused if the packs "
                    "it relies on changed since it was saved."
                ),
            ),
            Param(
                dest="roots",
                flags=("--root",),
                multiple=True,
                metavar="DIR",
                help="A folder the tools may read and write under (default: the current folder).",
            ),
            *_pack_params(),
            Param(
                dest="profile",
                flags=("--profile",),
                choices=tuple(sorted(PROFILES)),
                help="Named policy bundle for the core patterns (default: balanced).",
            ),
            Param(
                dest="no_remember",
                flags=("--no-remember",),
                kind="flag",
                default=False,
                help="Do not hide a value where it recurs unless a rule finds it again.",
            ),
            STYLE,
            LOG_LEVEL,
            LOG_FORMAT,
        ),
    ),
    Command(
        name="plan",
        summary="Save, show or verify a plan file a team can share and pin.",
        handler="_cmd_plan",
        description=(
            "A plan file records which packs, formats and rules to use, with a "
            "fingerprint of the definitions they resolve to. Commit it; every "
            "'--plan FILE' run then uses exactly that, and is refused if a pack "
            "it relies on has changed until the plan is saved again."
        ),
        params=(
            Param(
                dest="write",
                flags=("--write",),
                metavar="FILE",
                help="Save the plan built from the options below to FILE.",
            ),
            Param(
                dest="check",
                flags=("--check",),
                metavar="FILE",
                help="Verify FILE: valid, and its fingerprint still matches (exit 1 if not).",
            ),
            *_pack_params(),
            Param(
                dest="file_formats",
                flags=("--file-format",),
                multiple=True,
                metavar="NAME",
                help="Read only these formats, by name or extension (default: all).",
            ),
            Param(
                dest="profile",
                flags=("--profile",),
                choices=tuple(sorted(PROFILES)),
                help="Named policy bundle for the core patterns (default: balanced).",
            ),
            Param(
                dest="no_core",
                flags=("--no-core",),
                kind="flag",
                default=False,
                help="Run only the packs: switch the built-in patterns off.",
            ),
            Param(
                dest="hide",
                flags=("--hide",),
                multiple=True,
                metavar="TERM",
                help="Additional exact strings to hide.",
            ),
            Param(
                dest="allow",
                flags=("--allow",),
                multiple=True,
                metavar="TERM",
                help="Surfaces that must never be redacted.",
            ),
            Param(
                dest="no_remember",
                flags=("--no-remember",),
                kind="flag",
                default=False,
                help="Do not hide a value where it recurs unless a rule finds it again.",
            ),
            STYLE,
            KEEP,
            INFER_ROLES,
            FORMAT,
        ),
    ),
    Command(
        name="skill",
        summary="Print, or install, the agent skill that makes an AI assistant use cleanprompt.",
        handler="_cmd_skill",
        description=(
            "An instruction file (SKILL.md) for AI agents and assistants: "
            "encode before anything leaves the machine, decode what comes "
            "back, stop on a refusal. Printed by default; --write DIR installs "
            "it as DIR/cleanprompt-guard/SKILL.md."
        ),
        params=(
            Param(
                dest="write",
                flags=("--write",),
                metavar="DIR",
                help="Install into this skills directory instead of printing.",
            ),
            Param(
                dest="force",
                flags=("--force",),
                kind="flag",
                default=False,
                help="Replace an existing SKILL.md.",
            ),
        ),
    ),
    Command(
        name="docker",
        summary="Write a Dockerfile and compose file for the web interface.",
        handler="_cmd_docker",
        description="Emit container files for the web interface.",
        params=(
            Param(
                dest="write",
                flags=("--write",),
                metavar="DIR",
                help="Write the files into this directory instead of stdout.",
            ),
            Param(
                dest="port",
                flags=("--port",),
                type=int,
                default=5000,
                help="Published port.",
            ),
            Param(
                dest="with_ner",
                flags=("--with-ner",),
                kind="flag",
                default=False,
                help="Include spaCy and a model in the image.",
            ),
        ),
    ),
)

PROG = "python -m scikitplot.cleanprompt"
DESCRIPTION = (
    "Replace sensitive values in a prompt with stable placeholders before "
    "sending it to a language model, and restore them afterwards."
)
EPILOG = "Start with: doctor (what is active) or inspect (what would go)."


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argparse parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser, rendered from :data:`COMMANDS`.

    Notes
    -----
    **Developer notes.** Kept as a named function because it is the documented
    way to introspect the surface, and because the tests build a fresh parser
    per case. Both frontends render from :data:`COMMANDS`, so this is one view
    of the surface rather than the definition of it.
    """
    return build_argparse(COMMANDS, PROG, DESCRIPTION, EPILOG)


# ---------------------------------------------------------------------------
# configuration assembly
# ---------------------------------------------------------------------------


def _find_config(explicit: str | None) -> str | None:
    """Return the config file path to use, or ``None``."""
    if explicit:
        return explicit
    here = os.path.abspath(os.getcwd())
    while True:
        for name in CONFIG_NAMES:
            candidate = os.path.join(here, name)
            if os.path.isfile(candidate):
                return candidate
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def _load_config(path: str | None) -> dict[str, Any]:
    """
    Read a ``[cleanprompt]`` table from a TOML file.

    Raises
    ------
    CleanPromptError
        If the file cannot be parsed, or no TOML reader is available.

    Notes
    -----
    **Developer notes.** Reading uses the standard library's :mod:`tomllib` on
    Python 3.11 and newer and falls back to ``tomli``. A config file is opt-in:
    when none is found this returns an empty mapping and nothing is logged, so
    the common case costs nothing.
    """
    if not path:
        return {}
    try:
        import tomllib as reader  # noqa: PLC0415
    except ImportError:
        try:
            import tomli as reader  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError as exc:
            raise CleanPromptError(
                f"reading {path} needs a TOML reader on Python < 3.11; install "
                "one with: pip install tomli"
            ) from exc
    try:
        with open(path, "rb") as handle:
            document = reader.load(handle)
    except OSError as exc:
        raise CleanPromptError(f"cannot read {path}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - reported, not suppressed
        raise CleanPromptError(f"{path} is not valid TOML: {exc}") from exc
    section = document.get("cleanprompt", document)
    if not isinstance(section, dict):
        raise CleanPromptError(f"{path}: the [cleanprompt] table must be a table")
    return section


def _policy_from(args: argparse.Namespace) -> tuple[RedactionPolicy, dict[str, Any]]:
    """
    Assemble the policy from config file, profile and flags.

    Returns
    -------
    policy : RedactionPolicy
        The assembled policy.
    settings : dict
        Resolved extras the policy does not carry (``hide``, ``ner``, ...).

    Notes
    -----
    **Developer notes.** Precedence is config file, then profile, then explicit
    flags — least specific to most specific, so a flag on the command line
    always wins. Every layer is reported in ``doctor`` output, because a policy
    assembled from three places is otherwise impossible to debug.
    """
    config_path = _find_config(getattr(args, "config", None))
    config = _load_config(config_path)

    name = getattr(args, "profile", None) or config.get("profile")
    base = profile(name) if name else DEFAULT_POLICY

    changes: dict[str, Any] = {}
    if config.get("kinds") is not None:
        changes["kinds"] = tuple(config["kinds"])
    if config.get("allow") is not None:
        changes["allow"] = tuple(config["allow"])
    if config.get("ignore_case") is not None:
        changes["case_insensitive"] = bool(config["ignore_case"])
    if config.get("overlap") is not None:
        changes["overlap"] = OverlapStrategy(config["overlap"])

    if getattr(args, "kinds", None):
        changes["kinds"] = tuple(args.kinds)
    if getattr(args, "allow", None):
        changes["allow"] = tuple(args.allow)
    if getattr(args, "ignore_case", False):
        changes["case_insensitive"] = True
    if getattr(args, "overlap", None):
        changes["overlap"] = OverlapStrategy(args.overlap)

    style = getattr(args, "style", None) or config.get("style")
    if style and style != base.tag_style.style:
        # The style belongs to the grammar, so it travels with the tag style
        # and lands in the fingerprint a vault is checked against.
        from dataclasses import replace  # ruff: ignore[import-outside-top-level]

        changes["tag_style"] = replace(base.tag_style, style=style)

    policy = base.evolve(**changes) if changes else base

    hide = list(config.get("hide", ()))
    hide.extend(getattr(args, "hide", None) or ())
    settings = {
        "config_path": config_path,
        "profile": name,
        "hide": tuple(hide),
        "word_boundary": bool(
            getattr(args, "word_boundary", False) or config.get("word_boundary", False)
        ),
        "ner": bool(getattr(args, "ner", False) or config.get("ner", False)),
        "ner_model": getattr(args, "ner_model", None) or config.get("ner_model"),
        "ner_engine": (
            getattr(args, "ner_engine", None)
            or config.get("ner_engine")
            or DEFAULT_ENGINE
        ),
        "language": (
            getattr(args, "language", None)
            or config.get("language")
            or DEFAULT_LANGUAGE
        ),
        "model_size": (
            getattr(args, "model_size", None) or config.get("model_size", "sm")
        ),
    }
    return policy, settings


def _registry_for(policy: RedactionPolicy, settings: dict[str, Any]):
    """Build the detector registry for a policy."""
    registry = default_registry(kinds=policy.kinds)
    if settings["ner"]:
        for detector in build_detectors(
            mode=settings["ner_engine"],
            language=settings["language"],
            model=settings["ner_model"],
            size=settings["model_size"],
            # --ner is a request, not a default: if it cannot be met, say so
            # rather than scanning for nothing and reporting success.
            required=True,
        ):
            registry.add(detector)
    return registry


def _redactor_for(args: argparse.Namespace):
    """Return ``(redactor, policy, settings)`` for a parsed namespace."""
    policy, settings = _policy_from(args)
    registry = _registry_for(policy, settings)
    return Redactor(policy=policy, registry=registry), policy, settings


# ---------------------------------------------------------------------------
# file helpers
# ---------------------------------------------------------------------------


def _read_text(path: str | None, stdin: IO[str]) -> str:
    """Read the input text from a path or standard input."""
    if path is None or path == "-":
        return stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


#: Shown when a command is about to block on an interactive terminal.
#:
#: Printed to standard error, so it never contaminates the redacted text on
#: standard output when the command is used in a pipe.
_STDIN_HINT = (
    "reading from standard input — paste your text, then press Ctrl-D on a "
    "blank line.\n"
    "  other ways to pass text:\n"
    '    {prog} {command} "your text here"      # short text, one line\n'
    "    {prog} {command} --in notes.txt        # from a file\n"
    "    {prog} {command} <<'END'               # paste anything, no escaping\n"
    "    ... your text ...\n"
    "    END\n"
)


def _command_name(args: argparse.Namespace) -> str:
    """
    Return the subcommand the user typed, for use in messages.

    Notes
    -----
    **Developer notes.** The hint is meant to be pasted, so it must name the
    command the person actually ran rather than a representative one. Both
    frontends record it, under different attribute names; a missing value falls
    back to ``redact`` rather than raising, because a wrong word in a hint is a
    smaller failure than a crash while producing one.
    """
    for attribute in ("command", "_command", "subcommand"):
        value = getattr(args, attribute, None)
        if isinstance(value, str) and value:
            return value
    return "redact"


def _resolve_text(
    args: argparse.Namespace,
    stdin: IO[str],
    stderr: IO[str],
    command: str = "redact",
) -> str:
    """
    Return the input text from the positional argument, a file or stdin.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed namespace. ``text_args`` and ``input_path`` are consulted.
    stdin : file-like
        Standard input, read when no other source was given.
    stderr : file-like
        Where the interactive hint is written.
    command : str, default='redact'
        The subcommand name, used only to make the hint copy-pasteable.

    Returns
    -------
    str
        The text to process.

    Raises
    ------
    PolicyError
        If text was given both positionally and with ``--in``. Guessing which
        one was meant would silently process one and ignore the other, and the
        ignored one is exactly the text the person cared about.

    Notes
    -----
    **Developer notes.** Three sources, resolved in one place so that every
    command behaves the same way. The order is positional, then ``--in``, then
    standard input.

    The interactive hint exists because of a reported failure. A first-time
    user ran ``cleanprompt inspect --ner`` with their text pasted after it,
    the shell ate the parentheses, and the retries that quoted the text were
    rejected with "Got unexpected extra argument" — because there was no
    positional argument at all. The version that *would* have worked,
    ``--in`` or a pipe, was documented in one line of ``--help`` and never
    seen. When the command now falls through to standard input on a terminal
    it says so, and says what else it accepts, rather than appearing to hang.

    The check is ``stdin.isatty()``, guarded: a :class:`io.StringIO` in a test
    and a pipe in a script both answer ``False`` or raise, and neither should
    print anything.
    """
    positional = getattr(args, "text_args", None) or ()
    if isinstance(positional, str):  # a single-value argument, not variadic
        positional = (positional,)
    path = getattr(args, "input_path", None)

    if positional and path:
        raise PolicyError(
            f"text was given both on the command line and with --in {path!r}; "
            "pass one or the other"
        )

    if positional:
        return " ".join(positional)

    if path is not None and path != "-":
        return _read_text(path, stdin)

    try:
        interactive = stdin.isatty()
    except Exception:  # noqa: BLE001 - a stream without isatty is not a terminal
        interactive = False
    if interactive:
        stderr.write(_STDIN_HINT.format(prog="scikitplot cleanprompt", command=command))
        stderr.flush()
    return stdin.read()


def _write_text(path: str | None, text: str, stdout: IO[str]) -> None:
    """Write the output text to a path or standard output."""
    if path is None or path == "-":
        stdout.write(text)
        if text and not text.endswith("\n"):
            stdout.write("\n")
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


#: Environment variable holding the vault passphrase or key.
VAULT_KEY_ENV = "CLEANPROMPT_VAULT_KEY"


def _vault_key(cipher: str = "portable", prompt: bool = True) -> bytes:
    """
    Return the vault passphrase, from the environment or the terminal.

    Parameters
    ----------
    cipher : str, default='portable'
        Which cipher is being used, which decides what the message says.
    prompt : bool, default=True
        Whether to ask at an interactive terminal when the variable is unset.

    Returns
    -------
    bytes
        The passphrase as the user supplied it.

    Raises
    ------
    CleanPromptError
        If no passphrase is available.

    Notes
    -----
    **Developer notes.** The environment variable comes first, because that is
    what a script sets. Falling back to a terminal prompt matters more than it
    looks: without it the only way to encrypt is to put the passphrase in an
    environment variable, which on most shells means it lands in the shell
    history file — a plain-text copy of the key, next to the machine holding
    the vault it protects.

    :func:`getpass.getpass` reads without echoing and prefers the terminal
    device over standard input, so it still works when text is being piped in,
    which is the ordinary way this tool is used. It is skipped when there is no
    terminal, because a prompt in a pipeline is a hang.
    """
    raw = os.environ.get(VAULT_KEY_ENV)
    if raw:
        return raw.encode("utf-8")

    if prompt and sys.stdin.isatty():
        import getpass  # ruff: ignore[import-outside-top-level]

        try:
            entered = getpass.getpass("vault passphrase: ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise CleanPromptError("no passphrase given") from exc
        if entered:
            return entered.encode("utf-8")

    if cipher == "fernet":
        raise CleanPromptError(
            f"--cipher fernet needs a Fernet key in {VAULT_KEY_ENV}. Generate one with:"
            "\n  python -m scikitplot.cleanprompt doctor --new-key --cipher fernet"
            "\nOr use --cipher portable, which takes any passphrase and needs "
            "nothing installed."
        )
    raise CleanPromptError(
        f"--encrypt needs a passphrase. Set {VAULT_KEY_ENV}, or run it at a terminal and "
        "you will be asked for one. Generate a strong one with:"
        "\n  python -m scikitplot.cleanprompt doctor --new-key"
    )


def _decrypt_entries(path: str, document: dict[str, Any], entries: Any) -> Any:
    """
    Decrypt a vault document's entries with the cipher that wrote them.

    Raises
    ------
    CleanPromptError
        If the cipher is unknown, or is unavailable on this machine.

    Notes
    -----
    **Developer notes.** The document names its cipher, so this dispatches
    rather than guesses. A vault written before the field existed can only have
    been Fernet, since that was the only backend, so an absent field means
    Fernet — stated here rather than left to a default, because the two ciphers
    fail differently and a wrong guess would report "the passphrase is wrong"
    to someone whose passphrase is fine.
    """
    cipher = document.get("cipher") or "fernet"

    if cipher == "fernet":
        from ._capabilities import probe  # ruff: ignore[import-outside-top-level]

        report = probe("crypto")
        if not report.available:
            # The report says which of "not installed" and "installed but
            # refused" this is; the two need different actions.
            raise CleanPromptError(
                f"vault {_display_path(path)} was encrypted with Fernet, which needs the 'crypto' "
                f"tier ({report.detail}): {report.install_hint}. A vault written "
                "with --cipher portable needs nothing installed and would open "
                "here."
            )
        # deferred: the tier may be absent
        from ._crypto import (  # ruff: ignore[import-outside-top-level]
            decrypt_mapping,
        )

        return decrypt_mapping(entries, _vault_key("fernet"))

    from ._vaultcrypt import (  # ruff: ignore[import-outside-top-level]
        PORTABLE_CIPHER,
    )

    if cipher != PORTABLE_CIPHER:
        raise CleanPromptError(
            f"vault {_display_path(path)} names an unknown cipher {cipher!r}; this build knows "
            f"{PORTABLE_CIPHER!r} and 'fernet'"
        )

    from ._vaultcrypt import (  # ruff: ignore[import-outside-top-level]
        decrypt_mapping as portable_decrypt,
    )

    kdf = document.get("kdf")
    if not isinstance(kdf, dict):
        raise CleanPromptError(
            f"vault {_display_path(path)} is encrypted but records no key-derivation parameters, "
            "so its key cannot be reproduced"
        )
    return portable_decrypt(entries, _vault_key("portable"), kdf)


#: Environment variable naming the vault, overriding the default location.
VAULT_ENV = "CLEANPROMPT_VAULT"

#: File name used inside the state directory.
VAULT_FILENAME = "vault.json"


def default_vault_path() -> str:
    r"""
    Return the vault path used when ``--vault`` is not given.

    Returns
    -------
    str
        An absolute path. The file need not exist yet.

    Notes
    -----
    **User notes.** Override it for one run with ``--vault``, or for a whole
    shell with ``CLEANPROMPT_VAULT``. ``doctor`` prints the resolved path.

    **Developer notes — why not the working directory.**

    A vault holds the removed values in clear text. Defaulting it to
    ``./cleanprompt-vault.json`` would put a plain-text secrets file in
    whatever directory the command was run from, and this tool is used inside
    checkouts: the reported session that prompted the default ran from
    ``/work/.git_clones/learn/docs`` on a checked-out branch. One ``git add .``
    later, the values are in a commit, which is the single worst outcome this
    submodule can produce.

    So the default goes to the platform's state directory — the place meant for
    data a program keeps between runs that the user does not edit by hand:
    ``$XDG_STATE_HOME/cleanprompt/`` on Linux and macOS, falling back to
    ``~/.local/state/cleanprompt/``, and ``%LOCALAPPDATA%\cleanprompt\`` on
    Windows. It is outside every repository, it survives a ``cd``, and it is
    one fixed place to find or delete.

    The directory is created with ``0700`` and the file with ``0600``; see
    :func:`_write_vault`.

    Examples
    --------
    >>> default_vault_path().endswith("vault.json")
    True
    """
    override = os.environ.get(VAULT_ENV, "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))

    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
    else:
        base = os.environ.get("XDG_STATE_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "state"
        )
    return os.path.join(os.path.abspath(base), "cleanprompt", VAULT_FILENAME)


def resolve_vault_path(explicit: str | None) -> str:
    """
    Return the vault path to use, preferring an explicit one.

    Parameters
    ----------
    explicit : str or None
        The value of ``--vault``, if given.

    Returns
    -------
    str
        The path to read or write.
    """
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return default_vault_path()


def _display_path(path: str) -> str:
    """Return ``path`` with the home directory abbreviated, for messages."""
    home = os.path.expanduser("~")
    if home and path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _vault_index(entries: Sequence[Any]) -> list[dict[str, Any]]:
    """Return the JSON-safe label index for a vault document."""
    return [
        {"label": entry.label, "kind": entry.kind, "ordinal": entry.ordinal}
        for entry in entries
    ]


def _vault_lock(path: str, stderr: IO[str]):
    """
    Hold the vault's inter-process lock, saying so if another run has it.

    Notes
    -----
    **Developer notes.** Every command that reads a vault and writes it back
    holds this from the read to the write (``CP-068``): two runs seeding from
    the same vault issued the same label to two values. ``forget`` holds it
    too, or an ``encode`` that read the vault before the deletion would write
    the forgotten values back after it.
    """

    def waiting() -> None:
        stderr.write(
            f"waiting for another cleanprompt run using {_display_path(path)}\n"
        )

    return locked(path, waiting=waiting)


def _write_vault(  # ruff: ignore[too-many-positional-arguments]
    path: str,
    vault: Vault,
    policy: RedactionPolicy,
    encrypt: bool = False,
    index: Sequence[Any] = (),
    cipher: str = "portable",
) -> None:
    """
    Write a vault document, restricting its permissions where possible.

    Notes
    -----
    **Developer notes.** Written by :func:`~scikitplot.cleanprompt._files.atomic_write`:
    created ``0600`` rather than created and then chmod-ed, so there is no
    window in which the secrets are world-readable, and renamed into place so
    there is no window in which the vault is empty (``CP-069``). Callers that
    read the vault first hold :func:`_vault_lock` across both (``CP-068``).
    """
    entries: Any = vault.export()
    encrypted = False
    cipher_used: str | None = None
    kdf: dict[str, Any] | None = None
    if encrypt:
        from ._vaultcrypt import (  # ruff: ignore[import-outside-top-level]
            resolve_cipher,
        )

        resolved = resolve_cipher(cipher)
        if resolved == "fernet":
            # deferred: the tier may be absent
            from ._crypto import (  # ruff: ignore[import-outside-top-level]
                encrypt_mapping,
            )

            entries = encrypt_mapping(entries, _vault_key(resolved))
            cipher_used = "fernet"
        else:
            from ._vaultcrypt import (  # ruff: ignore[import-outside-top-level]
                PORTABLE_CIPHER,
            )
            from ._vaultcrypt import (  # ruff: ignore[import-outside-top-level]
                encrypt_mapping as portable_encrypt,
            )

            entries, kdf = portable_encrypt(entries, _vault_key(resolved))
            # The *versioned construction* name, not the ``--cipher`` word the
            # user typed. "portable" is a choice on a command line and may one
            # day resolve to a different construction; a document must say
            # which one actually wrote it, or a future change makes today's
            # vaults unreadable.
            cipher_used = PORTABLE_CIPHER
        encrypted = True

    document = {
        "format": VAULT_FORMAT,
        "encrypted": encrypted,
        # Which cipher wrote this, and how its key was derived. A document that
        # does not say how it was encrypted can only be decrypted by guessing,
        # and a wrong guess fails much later and less clearly than a refusal.
        "cipher": cipher_used,
        "kdf": kdf,
        "grammar_fingerprint": policy.tag_style.fingerprint,
        "policy_fingerprint": policy.fingerprint,
        "tag_style": policy.tag_style.as_dict(),
        "entries": entries,
        # The index carries no secrets: a label and its category are already
        # in the text that was sent, and the value stays in "entries" where
        # encryption covers it. It exists so that --vault-mode append can seed
        # the next run with real Entry objects. Recovering the category and
        # ordinal by slicing the label back apart was tried and rejected in the
        # engine for being fragile under a customised tag style; writing them
        # down is the version that cannot drift.
        "index": _vault_index(index),
    }
    # One step, 0600 from creation: a reader sees the old vault or the new
    # one, never an empty file, and a crash cannot destroy it (CP-069).
    atomic_write(
        path,
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_vault(path: str) -> Vault:
    """
    Read a vault document written by :func:`_write_vault`.

    Raises
    ------
    CleanPromptError
        If the document is malformed or of an unsupported format version.
    """
    with open(path, "r", encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except ValueError as exc:
            raise CleanPromptError(f"vault {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CleanPromptError(f"vault {path!r} is not a JSON object")
    version = document.get("format")
    if version not in VAULT_FORMATS_READ:
        msg = "vault {!r} has format {!r}; this build reads {}".format(
            path, version, " or ".join(str(v) for v in VAULT_FORMATS_READ)
        )
        raise CleanPromptError(msg)
    entries = document.get("entries")
    if document.get("encrypted"):
        entries = _decrypt_entries(path, document, entries)
    if not isinstance(entries, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in entries.items()
    ):
        raise CleanPromptError(f"vault {path!r} has a malformed 'entries' object")
    return Vault(entries, grammar_fingerprint=document.get("grammar_fingerprint"))


def _vault_tag_style(path: str) -> Any | None:
    """
    Return the placeholder grammar a vault was written under.

    Parameters
    ----------
    path : str
        The vault document.

    Returns
    -------
    TagStyle or None
        The grammar recorded in the document, or ``None`` when it records none
        or records one this build cannot construct.

    Notes
    -----
    **Developer notes.** ``decode`` is meant to need no arguments: that is the
    whole contract of the ``encode``/``decode`` pair. It nearly was not. A
    vault written with ``--style surrogate`` carries a different grammar
    fingerprint, so a bare ``decode`` — which assembled the default grammar
    from nothing — was refused with "this vault was issued under placeholder
    grammar 1a85… but restoration was asked for grammar 0e7c…".

    The refusal was right and the situation was wrong. The vault already
    records its own ``tag_style``; asking the user to restate on the way back
    something the file already knows is making them carry state the program is
    holding. So the grammar is read from the document, and an explicit
    ``--style`` still wins for anyone who means to override it.

    A malformed or unknown ``tag_style`` returns ``None`` rather than raising.
    Falling back to the default grammar produces the fingerprint mismatch
    above, which names the real problem; a constructor traceback here would
    not.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        recorded = document.get("tag_style")
        if not isinstance(recorded, dict):
            return None
        from ._policy import TagStyle  # ruff: ignore[import-outside-top-level]

        return TagStyle(**recorded)
    except Exception:  # noqa: BLE001 - any failure means "use the default"
        return None


def _seed_from_vault(path: str) -> tuple[Vault | None, tuple[Any, ...]]:
    """
    Read a vault for ``--vault-mode append``.

    Parameters
    ----------
    path : str
        The vault to continue from.

    Returns
    -------
    vault : Vault or None
        The existing vault, or ``None`` when the file does not exist yet, which
        is the ordinary first run and not an error.
    seed : tuple of Entry
        Entries to hand to :meth:`Redactor.redact` as ``seed``, so a value that
        was already redacted keeps the label it was given.

    Raises
    ------
    CleanPromptError
        If the vault exists but carries no index. See the notes.

    Notes
    -----
    **Developer notes — why a missing index is refused rather than guessed.**

    Appending without the previous run's labels is not a degraded append, it is
    a corruption. The new run would number from ``1`` again, so a fresh address
    would be issued ``[EMAIL-1]`` while the old vault already maps ``[EMAIL-1]``
    to a different address. Merging those two produces a vault in which one
    placeholder stands for two values, and restoration then silently puts the
    wrong one back — a redaction tool handing a user someone else's email
    address inside their own reply.

    Format 1 vaults, written before the index existed, are the only way to
    reach this. They restore normally; they just cannot be appended to, and the
    message says so along with the two ways forward.
    """
    if not os.path.exists(path):
        return None, ()

    with open(path, "r", encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except ValueError as exc:
            raise CleanPromptError(f"vault {path!r} is not valid JSON: {exc}") from exc

    vault = _read_vault(path)
    index = document.get("index") if isinstance(document, dict) else None
    values = vault.export()

    if not isinstance(index, list) or (values and not index):
        raise CleanPromptError(
            f"vault {_display_path(path)!r} was written before placeholder categories were "
            "recorded, so it cannot be appended to: a new run would reissue "
            "[KIND-1] for a different value and the two would collide. "
            "Either start a new vault (--vault-mode overwrite, or --vault "
            "pointing somewhere else) or restore from this one first."
        )

    from ._types import Entry  # ruff: ignore[import-outside-top-level]

    seed = []
    for item in index:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        original = values.get(label)
        if not isinstance(label, str) or original is None:
            continue  # an index row with no value cannot seed anything
        seed.append(
            Entry(
                label=label,
                kind=str(item.get("kind", "")),
                ordinal=int(item.get("ordinal", 0)),
                original=original,
                occurrences=(),
                detector="vault",
                confidence=1.0,
            )
        )
    return vault, tuple(seed)


def _color_flag(choice: str) -> bool | None:
    """Map the ``--color`` choice onto the tri-state used by the renderer."""
    return {"always": True, "never": False}.get(choice)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _cmd_redact(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """Run the ``redact`` subcommand."""
    redactor, policy, settings = _redactor_for(args)
    text = _resolve_text(args, stdin, stderr, _command_name(args))

    vault_path = resolve_vault_path(getattr(args, "vault", None))
    mode = getattr(args, "vault_mode", "overwrite")
    with _vault_lock(vault_path, stderr):
        prior_vault, seed = (
            _seed_from_vault(vault_path) if mode == "append" else (None, ())
        )

        result = redactor.redact(
            text,
            extra_terms=settings["hide"] or None,
            word_boundary=settings["word_boundary"],
            seed=seed,
        )

        merged, index = _merge_for_write(prior_vault, seed, result)
        _write_vault(
            vault_path,
            merged,
            policy,
            encrypt=args.encrypt,
            index=index,
            cipher=getattr(args, "cipher", "portable"),
        )

    color = _color_flag(args.color)
    to_stdout = args.output_path in (None, "-")
    rendered = (
        highlight_placeholders(result.text, policy, color=color, stream=stdout)
        if to_stdout
        else result.text
    )
    _write_text(args.output_path, rendered, stdout)

    if not args.quiet:
        outcome = describe_outcome(
            result,
            diagnose(policy, redactor.registry),
            suggest_terms(text, result) if result.stats.entries == 0 else (),
        )
        _write_outcome(outcome, stderr)
        if result.entries:
            stderr.write(
                summary_table(result, reveal=args.reveal, color=color, stream=stderr)
                + "\n"
            )
        stderr.write(_vault_note(vault_path, mode, merged) + "\n")
    return EXIT_OK


def _artifact_settings(args: argparse.Namespace) -> dict[str, Any]:
    """
    Read the artefact options off a namespace.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line.

    Returns
    -------
    dict
        ``fmt``, ``declared_roles``, ``extra_columns``, ``infer_roles``,
        ``hide_paths``, ``drop_binary`` and ``drop_outputs``.

    Raises
    ------
    PolicyError
        If ``--columns`` is malformed. A silently ignored role declaration
        would be worse than a refused one: the user would believe a column was
        classified when it was not.

    Notes
    -----
    **Developer notes.** ``--keep`` is read as a set of names rather than as
    three separate flags because the three are one decision — *how much of the
    artefact is data* — and a user who wants outputs usually wants figures too.
    Spelling it as one option also keeps the default visible in the help: the
    absence of ``--keep`` is what removes them.
    """
    keep = set(getattr(args, "keep", None) or ())
    declared: dict[str, str] = {}
    raw = getattr(args, "columns", None)
    if raw:
        for item in str(raw).split(","):
            item = item.strip()  # ruff: ignore[redefined-loop-name]
            if not item:
                continue
            if ":" not in item:
                msg = (
                    "--columns takes NAME:ROLE pairs; {!r} has no role. "
                    "Roles: {}".format(item, ", ".join(sorted(ROLES)))
                )
                raise PolicyError(msg)
            name, _, role = item.partition(":")
            name, role = name.strip(), role.strip()
            if role not in ROLES:
                msg_0 = "unknown role {!r} for column {!r}; choose from {}".format(
                    role, name, ", ".join(sorted(ROLES))
                )
                raise PolicyError(msg_0)
            declared[name] = role
    requested = getattr(args, "as_format", "auto") or "auto"
    return {
        "fmt": None if requested == "auto" else requested,
        "declared_roles": declared or None,
        "extra_columns": tuple(getattr(args, "add_columns", None) or ()),
        "infer_roles": bool(getattr(args, "infer_roles", False)),
        "hide_paths": "paths" not in keep,
        "drop_binary": "figures" not in keep,
        "drop_outputs": "outputs" not in keep,
    }


def _artifact_note(plan: ArtifactPlan) -> str:
    """
    Return the human-readable summary of what an artefact plan decided.

    Parameters
    ----------
    plan : ArtifactPlan
        The plan.

    Returns
    -------
    str
        A short report, or ``''`` when the input was ordinary prose.

    Notes
    -----
    **Developer notes.** The refusals and the unread cells are printed even
    when nothing else is, because they are the two places a column can exist
    and not be hidden. Leaving them out would turn this into the clean bill of
    health ``CP-046`` was filed against.
    """
    if plan.fmt == "text" and not plan.columns:
        return ""
    lines = []
    counts = {}
    for region in plan.regions:
        counts[region.role] = counts.get(region.role, 0) + 1
    shape = ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
    lines.append("read as {}{}".format(plan.fmt, ": " + shape if shape else ""))
    if plan.columns:
        established = sum(
            1 for _r, provenance, _s in plan.columns.values() if provenance != "none"
        )
        lines.append(
            f"  {len(plan.columns)} column name(s) hidden; {established} with an established role"
        )
        for name in sorted(plan.columns):
            role, provenance, standin = plan.columns[name]
            lines.append(f"    {standin:<14} <- role {role} ({provenance})")
    lines.extend(f"  NOT hidden: {plan.refused[name]}" for name in sorted(plan.refused))
    if plan.unparsed:
        lines.append(
            "  NOT read: {} — no column was discovered there".format(
                ", ".join(plan.unparsed)
            )
        )
    lines.extend(
        (
            f"  suggested column list {name}: re-run with --add-column to hide "
            "its entries"
        )
        for name in plan.suggestions
    )
    return "\n".join(lines)


def _cmd_clean(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``clean`` subcommand.

    Notes
    -----
    **User notes.** The shortest path from a paragraph to something you can
    paste into a chat::

        python -m scikitplot.cleanprompt clean --ner <<'END'
        ... paste anything ...
        END

    Standard output carries the redacted text and nothing else, so it can be
    piped, copied or redirected as it stands. The vault goes to the default
    location in append mode, which is what keeps ``[PERSON-1]`` meaning the
    same person across every prompt of a conversation — and what lets
    ``restore`` put the values back into the reply with no arguments at all.

    **Developer notes — why this is a command and not a flag.**

    ``redact`` already put the text on standard output and its report on
    standard error, so ``redact --vault v.json 2>/dev/null`` produced this
    output before ``clean`` existed. Nobody found it. The reported session
    shows a user reaching for ``inspect``, whose whole purpose is the report,
    and then asking how to get "just the clean prompt, no additional info".

    An affordance that exists only as a combination of three flags and a
    redirection is an affordance that is not there. ``clean`` is the same
    pipeline with the defaults a person pasting into a chat window actually
    wants: no vault path to invent, stable labels across turns, and one line
    of stderr instead of a table.

    That one line is deliberate. Writing a file that holds the removed values
    without saying so would be the kind of quiet side effect this submodule
    refuses everywhere else; ``-q`` silences it for scripts, which have no
    reader to inform.
    """
    redactor, policy, settings = _redactor_for(args)
    text = _resolve_text(args, stdin, stderr, _command_name(args))

    vault_path = resolve_vault_path(getattr(args, "vault", None))
    mode = getattr(args, "vault_mode", "append")
    with _vault_lock(vault_path, stderr):
        prior_vault, seed = (
            _seed_from_vault(vault_path) if mode == "append" else (None, ())
        )

        artifact = _artifact_settings(args)
        source_path = getattr(args, "input_path", None)
        resolved_format = artifact["fmt"] or detect_format(source_path, text)
        plan = None

        if resolved_format in ("python", "notebook"):
            # A structured artefact takes the artefact pipeline, which is the same
            # engine with a schema-aware detector set and a seeded vault. Prose
            # takes the path it always took, so nothing about the ordinary case
            # changes.
            result, plan = encode_artifact(
                text,
                fmt=resolved_format,
                path=source_path,
                policy=policy,
                declared_roles=artifact["declared_roles"],
                extra_columns=artifact["extra_columns"],
                infer_roles=artifact["infer_roles"],
                hide_paths=artifact["hide_paths"],
                drop_binary=artifact["drop_binary"],
                drop_outputs=artifact["drop_outputs"],
                extra_terms=settings["hide"] or None,
                word_boundary=settings["word_boundary"],
            )
        else:
            result = redactor.redact(
                text,
                extra_terms=settings["hide"] or None,
                word_boundary=settings["word_boundary"],
                seed=seed,
            )
        merged, index = _merge_for_write(prior_vault, seed, result)
        _write_vault(
            vault_path,
            merged,
            policy,
            encrypt=getattr(args, "encrypt", False),
            index=index,
            cipher=getattr(args, "cipher", "portable"),
        )

    to_stdout = args.output_path in (None, "-")
    rendered = (
        highlight_placeholders(
            result.text, policy, color=_color_flag(args.color), stream=stdout
        )
        if to_stdout
        else result.text
    )
    _write_text(args.output_path, rendered, stdout)

    if not args.quiet:
        stderr.write(_vault_note(vault_path, mode, merged) + "\n")
        if plan is not None:
            note = _artifact_note(plan)
            if note:
                stderr.write(note + "\n")
        # A gap in detection is the one thing worth interrupting for, because
        # the text is about to be sent somewhere. Everything else stays quiet.
        report = diagnose(policy, redactor.registry)
        for spot in report.blind_spots:
            if spot.severity == "high":
                stderr.write(f"warning: {spot.category} not detected\n")
                stderr.write(f"  {spot.remedy}\n")
                break
    return EXIT_OK


def _cmd_forget(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``forget`` subcommand.

    Notes
    -----
    **User notes.** When a conversation is over::

        python -m scikitplot.cleanprompt forget

    It deletes the vault, so the values it held are no longer on disk. After
    that, ``decode`` can no longer restore anything from that conversation —
    which is the point.

    **Developer notes — why this command had to exist.**

    ``encode`` defaults to a vault at a fixed path in append mode. Those two
    decisions are what make the conversational workflow work, and together
    they mean the removed values accumulate in one file, indefinitely, in
    clear text. A tool that starts collecting personal data by default and
    offers no way to stop is worse than one that never collected it.

    Deletion is unlinking, not shredding, and the message says so rather than
    implying more than the filesystem can promise. On a journalling or
    copy-on-write filesystem, on an SSD with wear levelling, or where a backup
    ran in between, the bytes may survive the unlink. Claiming otherwise would
    be the kind of security theatre this submodule avoids: ``--encrypt`` and a
    key the user holds is the answer for data that must not be recoverable,
    and the note points there.

    The report names how many values and of which categories, never a value.
    Someone deciding whether to delete should be able to see what they are
    losing without the act of looking putting the data on their screen.
    """
    del stdin, stdout
    path = resolve_vault_path(getattr(args, "vault", None))
    shown = _display_path(path)

    if not os.path.exists(path):
        stderr.write(f"no vault at {shown}; nothing to forget\n")
        return EXIT_OK

    try:
        kinds = _vault_summary(path)
    except (ValueError, OSError, CleanPromptError):
        # An unreadable vault is still a file of secrets, and refusing to
        # delete it because it cannot be parsed would strand exactly the user
        # who most wants it gone. Every way the read can fail is caught: a
        # truncated write leaves invalid JSON (ValueError), a permission or
        # device problem raises OSError, and a bad document raises our own
        # error. Only the summary is lost; the deletion still happens.
        kinds = None

    if not args.force:
        stderr.write(f"would delete {shown}\n")
        stderr.write(f"  {_forget_summary(kinds)}\n")
        stderr.write("  re-run with --force to delete it\n")
        return EXIT_OK

    with _vault_lock(path, stderr):
        # Every remover holds this lock, so the check cannot go stale: a
        # forget that waited behind another finds the vault already gone.
        if os.path.exists(path):
            os.remove(path)
    stderr.write(f"deleted {shown}\n")
    stderr.write(f"  {_forget_summary(kinds)}\n")
    stderr.write(
        "  note: this unlinks the file. On a journalling filesystem or an SSD "
        "the bytes may survive; use --encrypt with a key you hold for values "
        "that must not be recoverable.\n"
    )
    return EXIT_OK


def _vault_summary(path: str) -> dict[str, int]:
    """Return a count of vault entries by category, without their values."""
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    counts: dict[str, int] = {}
    index = document.get("index") if isinstance(document, dict) else None
    if isinstance(index, list):
        for item in index:
            if isinstance(item, dict):
                kind = str(item.get("kind") or "?")
                counts[kind] = counts.get(kind, 0) + 1
    elif isinstance(document, dict) and isinstance(document.get("entries"), dict):
        counts["?"] = len(document["entries"])
    return counts


def _forget_summary(kinds: dict[str, int] | None) -> str:
    """Describe what a vault holds, by category and count only."""
    if kinds is None:
        return "contents unreadable"
    total = sum(kinds.values())
    if not total:
        return "it held no values"
    detail = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
    return f"it held {total} value(s): {detail}"


def _merge_for_write(
    prior: Vault | None,
    seed: Sequence[Any],
    result: Any,
) -> tuple[Vault, tuple[Any, ...]]:
    """
    Combine a prior vault with this run's result for writing.

    Parameters
    ----------
    prior : Vault or None
        The vault read in append mode, or ``None`` for overwrite.
    seed : sequence of Entry
        The entries that came from ``prior``, carrying their categories.
    result : RedactionResult
        This run's result.

    Returns
    -------
    vault : Vault
        What to write.
    index : tuple of Entry
        The label index to record alongside it.

    Notes
    -----
    **Developer notes.** In overwrite mode this is the identity: the result's
    own vault and entries. In append mode the prior values come first and the
    new ones are added, and because the redactor was *seeded* with the prior
    entries, a value seen before arrives carrying the same label it had. So
    the union can never map one placeholder to two values — the engine's
    ``seed`` handling is what guarantees it, and this function only has to
    avoid undoing that.
    """
    if prior is None:
        return result.vault, tuple(result.entries)

    merged = Vault(dict(prior.export()), grammar_fingerprint=prior.grammar_fingerprint)
    for label, value in result.vault.items():
        merged.add(label, value)

    by_label = {entry.label: entry for entry in seed}
    for entry in result.entries:
        by_label[entry.label] = entry
    return merged, tuple(by_label.values())


def _vault_note(path: str, mode: str, vault: Vault) -> str:
    """Return the one-line stderr note naming where the vault went."""
    return f"vault: {_display_path(path)} ({mode}, {len(vault.labels())} value(s))"


def _write_outcome(outcome: dict[str, Any], stderr: IO[str]) -> None:
    """
    Write an outcome explanation to standard error.

    Notes
    -----
    **Developer notes.** The ``alert`` level is the one that matters: nothing
    was found *and* something was not looking. It must never render like a
    neutral all-clear, because the user reads an all-clear as "safe to send".
    """
    marker = {"ok": "", "warning": "warning: ", "alert": "ALERT: "}[outcome["level"]]
    stderr.write("{}{}\n".format(marker, outcome["headline"]))
    if outcome["level"] != "ok":
        stderr.write("  {}\n".format(outcome["detail"]))
    for action in outcome["actions"]:
        stderr.write(f"  -> {action}\n")


def _cmd_restore(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """Run the ``restore`` subcommand."""
    vault_path = resolve_vault_path(getattr(args, "vault", None))
    if not os.path.exists(vault_path):
        raise CleanPromptError(
            f"no vault at {_display_path(vault_path)}. Encode something first — 'encode' and "
            "'redact' write there by default — or point --vault at the vault "
            "you have."
        )
    vault = _read_vault(vault_path)
    text = _resolve_text(args, stdin, stderr, _command_name(args))

    # Read the grammar back from the vault rather than making the user restate
    # it. Without this a surrogate vault needed --style on the way back too,
    # and the pair stopped being the no-argument round trip it is meant to be.
    policy = None
    recorded = _vault_tag_style(vault_path)
    if recorded is not None:
        policy = DEFAULT_POLICY.evolve(tag_style=recorded)

    result = restore(
        text,
        vault,
        policy=policy,
        strict=args.strict,
        lenient=not getattr(args, "exact", False),
    )
    _write_text(args.output_path, result.text, stdout)
    if not args.quiet:
        stderr.write(restoration_note(result) + "\n")
    return EXIT_OK


def _cmd_inspect(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """Run the ``inspect`` subcommand."""
    redactor, policy, settings = _redactor_for(args)
    text = _resolve_text(args, stdin, stderr, _command_name(args))
    result = redactor.redact(
        text,
        extra_terms=settings["hide"] or None,
        word_boundary=settings["word_boundary"],
    )
    report = diagnose(policy, redactor.registry)
    suggestions = (
        ()
        if args.no_suggest
        else suggest_terms(text, result, min_tokens=args.suggest_min_tokens)
    )
    outcome = describe_outcome(result, report, suggestions)

    findings = []
    for entry in result.entries:
        item = {
            "placeholder": entry.label,
            "kind": entry.kind,
            "occurrences": entry.count,
            "detector": entry.detector,
            "confidence": entry.confidence,
        }
        if args.reveal:
            item["value"] = entry.original
        findings.append(item)

    data = {
        "outcome": outcome["level"],
        "headline": outcome["headline"],
        "detail": outcome["detail"],
        "would_redact": findings,
        "suggestions": outcome["suggestions"],
        "diagnosis": report.as_dict(),
        "actions": outcome["actions"],
        "status": "ok",
    }
    _emit(data, args.fmt, stdout)
    return EXIT_OK


def _cmd_scan(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``scan`` subcommand.

    Notes
    -----
    **Developer notes.** A distinct exit code for "found something" (``3``)
    rather than reusing ``1``: a pipeline needs to tell "this text contains
    personal data" apart from "the scanner crashed", and those demand different
    responses.
    """
    redactor, policy, settings = _redactor_for(args)
    text = _resolve_text(args, stdin, stderr, _command_name(args))
    result = redactor.redact(
        text,
        extra_terms=settings["hide"] or None,
        word_boundary=settings["word_boundary"],
    )
    report = diagnose(policy, redactor.registry)
    found = result.stats.entries
    over = found > max(0, args.max_findings)
    data = {
        "clean": not over,
        "findings": found,
        "max_findings": args.max_findings,
        "by_kind": dict(sorted(result.stats.by_kind.items())),
        "placeholders": list(result.labels),
        "blind_spots": [spot.as_dict() for spot in report.blind_spots],
        "status": "found" if over else "clean",
    }
    _emit(data, args.fmt, stdout)
    return EXIT_FOUND if over else EXIT_OK


def _simulated_reply(result: Any) -> str:
    """
    Return a stand-in model reply that uses the placeholders.

    Parameters
    ----------
    result : RedactionResult
        The redaction whose placeholders the reply should mention.

    Returns
    -------
    str
        A short sentence containing every placeholder that was issued.

    Notes
    -----
    **Developer notes.** The point of the demonstration is the *decode* step,
    and decoding an echo of the prompt would prove almost nothing: the
    placeholders would sit in the positions they already occupied. This puts
    them in a different sentence, in a different order, so that what the
    restoration actually does — match a placeholder anywhere in arbitrary text
    and put the value back — is what the reader sees.

    It is a fixed string, not a model call. The command says so in its own
    output, every time: presenting generated text as a model's answer would be
    a lie told by a tool whose entire purpose is trust.
    """
    labels = [entry.label for entry in result.entries]
    if not labels:
        return (
            "Nothing was redacted, so a model would see your text unchanged "
            "and there is nothing to restore."
        )
    if len(labels) == 1:
        listed = labels[0]
    else:
        listed = "{} and {}".format(", ".join(labels[:-1]), labels[-1])
    return f"Of course. I have made a note of {listed}, and I will follow up shortly."


def _cmd_roundtrip(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``roundtrip`` subcommand.

    Notes
    -----
    **Developer notes.** This exists because the two halves of the tool were
    only ever demonstrated apart. ``redact`` writes a vault, ``restore`` reads
    one, and nothing showed the loop closing — so the decode half, which is the
    part people doubt, was the part nobody saw. A reported session had a user
    trying to paste a paragraph at ``inspect`` and never reaching the question
    of how the values come back at all.

    Everything is held in memory. No vault file is written, because a file is
    ceremony for a demonstration and the first thing a newcomer has to be told
    to delete afterwards. For real work the vault is the point, and ``redact``
    still writes one.
    """
    redactor, policy, settings = _redactor_for(args)
    text = _resolve_text(args, stdin, stderr, _command_name(args))
    result = redactor.redact(
        text,
        extra_terms=settings["hide"] or None,
        word_boundary=settings["word_boundary"],
    )

    if args.reply_in:
        reply = _read_text(args.reply_in, stdin)
        reply_is_real = True
    elif args.reply:
        reply = args.reply
        reply_is_real = True
    else:
        reply = _simulated_reply(result)
        reply_is_real = False

    restored = restore(reply, result.vault, policy=policy)
    exact = restore(result.text, result.vault, policy=policy).text == text
    outcome = describe_outcome(result, diagnose(policy, redactor.registry))

    if args.fmt != "text":
        _emit(
            {
                "original": text,
                "safe": result.text,
                "entries": [
                    {
                        "placeholder": entry.label,
                        "kind": entry.kind,
                        "occurrences": entry.count,
                        "detector": entry.detector,
                    }
                    for entry in result.entries
                ],
                "reply": reply,
                "reply_is_simulated": not reply_is_real,
                "restored": restored.text,
                "unresolved_placeholders": list(restored.unknown),
                "round_trip_exact": exact,
                "outcome": outcome["level"],
                "headline": outcome["headline"],
            },
            args.fmt,
            stdout,
        )
        return EXIT_OK

    color = _color_flag(args.color)
    write = stdout.write

    write("\n1 · your text\n")
    write(_indent(text) + "\n")

    write("\n2 · what gets sent — copy this into the model\n")
    write(_indent(highlight_placeholders(result.text, color=color)) + "\n")

    write("\n3 · what was removed — kept on this machine, never sent\n")
    if result.entries:
        write(_indent(summary_table(result, reveal=args.reveal, color=color)) + "\n")
    else:
        write(_indent("nothing — no sensitive value was detected") + "\n")

    if reply_is_real:
        write("\n4 · the model's reply\n")
    else:
        write(
            "\n4 · a stand-in reply — NOT from a model; pass --reply to use a real one\n"
        )
    write(_indent(highlight_placeholders(reply, color=color)) + "\n")

    write("\n5 · restored — the values are back\n")
    write(_indent(restored.text) + "\n")

    write("\n")
    write("round trip exact: {}\n".format("yes" if exact else "NO"))
    if restored.unknown:
        write("unresolved placeholders: {}\n".format(", ".join(restored.unknown)))
    _write_outcome(outcome, stderr)
    return EXIT_OK if exact else EXIT_ERROR


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent every line of ``text``, so a block reads as one unit."""
    return "\n".join(prefix + line for line in text.splitlines() or [""])


def _cmd_doctor(
    args: argparse.Namespace,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """Run the ``doctor`` subcommand."""
    del stderr
    policy, settings = _policy_from(args)
    try:
        registry = _registry_for(policy, settings)
    except CapabilityError:
        registry = default_registry(kinds=policy.kinds)
    if getattr(args, "new_key", False):
        # A key generator that needs an optional package installed is no use to
        # the person who has not installed it, and that is exactly the person
        # being told to encrypt their vault. The portable passphrase comes from
        # the standard library; the Fernet key still needs the tier that reads
        # it, which is the honest constraint rather than an oversight.
        if getattr(args, "cipher", "portable") == "fernet":
            # deferred: the tier may be absent
            from ._crypto import new_key  # ruff: ignore[import-outside-top-level]

            stdout.write(new_key() + "\n")
        else:
            from ._vaultcrypt import (  # ruff: ignore[import-outside-top-level]
                new_passphrase,
            )

            stdout.write(new_passphrase() + "\n")
        return EXIT_OK

    report = diagnose(policy, registry)
    engines = describe_engines(settings["language"], settings["ner_engine"])
    languages = language_report(settings["language"], settings["model_size"])
    corpora: dict[str, Any] = {"checked": False}
    if engines["engines"].get("nltk", {}).get("status") == "AVAILABLE":
        from ._nltk import corpora_status  # ruff: ignore[import-outside-top-level]

        corpora = dict(corpora_status(), checked=True)

    data: dict[str, Any] = {
        "headline": report.headline(),
        "healthy": report.healthy,
        "tiers": report.tiers,
        "detection": {
            "active_kinds": list(report.active_kinds),
            "inactive_kinds": list(report.inactive_kinds),
            "ner_active": report.ner_active,
            "total_patterns": len(PATTERNS),
        },
        "blind_spots": [spot.as_dict() for spot in report.blind_spots],
        "entity_engines": engines,
        "nltk_corpora": corpora,
        "languages": languages,
        "configuration": {
            "profile": settings["profile"],
            "config_file": settings["config_path"],
            "ner_engine": settings["ner_engine"],
            "language": settings["language"],
            "model_size": settings["model_size"],
            "policy_fingerprint": policy.fingerprint,
            "grammar_fingerprint": policy.tag_style.fingerprint,
            # Where the values go when --vault is not given. Reported because
            # a file of secrets written to a path the user never typed must be
            # a path the user can find, and delete.
            "vault_path": default_vault_path(),
            "vault_exists": os.path.exists(default_vault_path()),
            "case_insensitive": policy.case_insensitive,
            "overlap": policy.overlap.value,
            "allow": list(policy.allow),
            "limits": policy.limits.as_dict(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cli_frontend": load_runner()[1],
            "vault_key_set": bool(os.environ.get("CLEANPROMPT_VAULT_KEY")),
            "web_secret_key_set": bool(os.environ.get("CLEANPROMPT_SECRET_KEY")),
        },
    }

    if args.input_path:
        text = _read_text(args.input_path, stdin)
        result = Redactor(policy=policy, registry=registry).redact(
            text,
            extra_terms=settings["hide"] or None,
            word_boundary=settings["word_boundary"],
        )
        suggestions = suggest_terms(text, result)
        outcome = describe_outcome(result, report, suggestions)
        data["text_report"] = {
            "characters": len(text),
            "would_redact": result.stats.entries,
            "by_kind": dict(sorted(result.stats.by_kind.items())),
            "outcome": outcome["level"],
            "headline": outcome["headline"],
            "suggestions": outcome["suggestions"],
        }

    data["status"] = "ok" if report.healthy else "degraded"
    _emit(data, args.fmt, stdout)
    return EXIT_OK


def _cmd_kinds(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """Run the ``kinds`` subcommand."""
    del stdin, stderr
    data = {
        "kinds": {
            spec.kind: {
                "intent": spec.intent,
                "priority": spec.priority,
                "enabled_by_default": spec.enabled_by_default,
                "validated": spec.validate is not None,
                "confidence": spec.confidence,
            }
            for spec in sorted(PATTERNS.values(), key=lambda s: s.kind)
        },
        "profiles": {name: sorted(PROFILES[name]) for name in sorted(PROFILES)},
        "status": "ok",
    }
    _emit(data, args.fmt, stdout)
    return EXIT_OK


def _plan_from(  # ruff: ignore[too-many-branches]
    args: argparse.Namespace,
) -> CleanPlan:
    """
    Build and validate the plan a ``batch`` run asked for.

    Returns
    -------
    CleanPlan
        Validated.

    Raises
    ------
    CleanPromptError
        Listing every problem with the selection at once.
    """
    plan_file = getattr(args, "plan", None)
    if plan_file:
        chosen = [
            flag
            for dest, flag in (
                ("packs", "--pack"),
                ("pack_files", "--pack-file"),
                ("file_formats", "--file-format"),
                ("profile", "--profile"),
                ("hide", "--hide"),
                ("allow", "--allow"),
                ("keep", "--keep"),
            )
            if getattr(args, dest, None)
        ]
        chosen += [
            flag
            for dest, flag in (
                ("no_core", "--no-core"),
                ("infer_roles", "--infer-roles"),
                ("ner", "--ner"),
                ("no_remember", "--no-remember"),
                ("replace_builtins", "--replace-builtins"),
            )
            if getattr(args, dest, False)
        ]
        if getattr(args, "style", "placeholder") != "placeholder":
            chosen.append("--style")
        if chosen:
            msg = "--plan fixes every choice; remove {} or save a new plan".format(
                ", ".join(chosen),
            )
            raise CleanPromptError(msg)
        from ._plan import load_plan  # ruff: ignore[import-outside-top-level]

        return load_plan(plan_file)
    builder = FluentCleanPrompt()
    if args.packs:
        builder = builder.packs(*args.packs)
    if getattr(args, "file_formats", None):
        builder = builder.formats(*args.file_formats)
    if args.pack_files:
        builder = builder.custom(
            *args.pack_files, replace_builtins=args.replace_builtins
        )
    if getattr(args, "profile", None):
        builder = builder.profile(args.profile)
    if getattr(args, "no_core", False):
        builder = builder.core(False)
    if getattr(args, "style", "placeholder") != "placeholder":
        builder = builder.style(args.style)
    if getattr(args, "keep", None):
        builder = builder.keep(*args.keep)
    if getattr(args, "infer_roles", False):
        builder = builder.infer_roles()
    if getattr(args, "hide", None):
        builder = builder.hide(*args.hide)
    if getattr(args, "allow", None):
        builder = builder.allow(*args.allow)
    if getattr(args, "ner", False):
        builder = builder.ner("auto")
    if getattr(args, "no_remember", False):
        builder = builder.remember(False)
    return builder.build()


def _cmd_packs(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``packs`` subcommand.

    Notes
    -----
    **Developer notes.** ``--check`` compares the compiled catalogue with the
    YAML only when PyYAML is installed; without it the compiled file is the
    only source there is, and the check says so rather than failing.
    """
    del stdin, stderr
    from ._catalog import (  # ruff: ignore[import-outside-top-level]
        check_compiled,
        write_compiled,
    )

    if args.compile:
        written = write_compiled()
        _emit(
            {"written": _display_path(str(written)), "status": "ok"}, args.fmt, stdout
        )
        return EXIT_OK
    problems: list[str] = []
    try:
        plan = FluentCleanPrompt()
        if args.pack_files:
            plan = plan.custom(*args.pack_files, replace_builtins=args.replace_builtins)
        if args.packs:
            plan = plan.packs(*args.packs)
        built = plan.build()
        catalog = built.catalog()
        selected = catalog.resolve_packs(
            built.packs_selection() if args.packs else "all"
        )
    except CleanPromptError as exc:
        if not args.check:
            raise
        problems.append(str(exc))
        catalog, selected = None, ()
    if args.check:
        try:
            drift = check_compiled()
        except CapabilityError:
            # Without PyYAML the compiled file is the only source there is, so
            # there is nothing to compare it with; the report says so.
            compiled = "not compared: PyYAML is not installed"
        else:
            compiled = "matches the YAML" if not drift else "differs from the YAML"
            problems.extend(drift)
        data = {
            "problems": problems,
            "compiled": compiled,
            "status": "ok" if not problems else "problems",
        }
        _emit(data, args.fmt, stdout)
        return EXIT_OK if not problems else EXIT_ERROR
    if args.show:
        document = catalog.documents.get(f"pack:{args.show}") or catalog.documents.get(
            f"format:{args.show}"
        )
        if document is None:
            known = sorted(list(catalog.packs) + list(catalog.formats))
            raise CleanPromptError(
                f"no pack or format named {args.show!r}; known: {', '.join(known)}"
            )
        _emit(
            json.loads(document) if isinstance(document, str) else document,
            args.fmt,
            stdout,
        )
        return EXIT_OK
    data = {
        "packs": {
            spec.name: {
                "summary": spec.summary,
                "requires": list(spec.requires),
                "fields": len(spec.fields),
                "patterns": len(spec.patterns),
                "source": spec.source,
            }
            for spec in selected
        },
        "formats": {
            spec.name: {
                "extensions": list(spec.extensions),
                "splitter": spec.splitter,
                "round_trip": spec.round_trip,
                "auto_packs": list(spec.packs),
            }
            for spec in (catalog.formats[name] for name in sorted(catalog.formats))
        },
        "status": "ok",
    }
    _emit(data, args.fmt, stdout)
    return EXIT_OK


def _cmd_batch(  # ruff: ignore[too-many-branches]
    args: argparse.Namespace,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """
    Run the ``batch`` subcommand.

    Notes
    -----
    **Developer notes.** Exit status 1 when any file was refused: the output
    is complete for what it holds, but something the user pointed at was not
    processed, and a pipeline must be able to tell. Skipped files (no format
    reads them) are expected and do not change the status. The vault is
    written before the report, and only after every file succeeded or was
    reported, so a crash never leaves encoded files whose values are nowhere.
    """
    del stdin
    from dataclasses import asdict  # ruff: ignore[import-outside-top-level]

    from ._runtime import (  # ruff: ignore[import-outside-top-level]
        Cleaner,
        kind_totals,
        restore_archive,
        restore_tree,
    )

    source = args.source
    is_zip = os.path.isfile(source) and source.lower().endswith(".zip")
    if not is_zip and not os.path.isdir(source):
        raise CleanPromptError(f"{source!r} is neither a folder nor a .zip archive")
    if args.dry_run:
        if args.decode or args.target:
            raise CleanPromptError("--dry-run writes nothing; drop --out and --decode")
        if is_zip:
            raise CleanPromptError(
                "--dry-run surveys a folder; for a zip, unpack it first or run without it"
            )
        items = Cleaner(_plan_from(args)).survey_tree(source)
        counts = {
            status: sum(1 for item in items if item.status == status)
            for status in ("encoded", "skipped", "refused")
        }
        data = {
            "items": [asdict(item) for item in items],
            **{
                ("would_" + status if status == "encoded" else status): count
                for status, count in counts.items()
                if count
            },
            "kinds": kind_totals(items),
            "status": "refused" if counts["refused"] else "ok",
        }
        _emit(data, args.fmt, stdout)
        return EXIT_ERROR if counts["refused"] else EXIT_OK
    if not args.target:
        raise CleanPromptError("--out is required (or use --dry-run)")
    vault_path = resolve_vault_path(args.vault)
    if args.decode:
        if not os.path.exists(vault_path):
            raise CleanPromptError(
                f"no vault at {_display_path(vault_path)}; nothing can be decoded"
            )
        vault = _read_vault(vault_path)
        recorded = _vault_tag_style(vault_path)
        policy = (
            DEFAULT_POLICY.evolve(tag_style=recorded) if recorded is not None else None
        )
        if is_zip:
            items = restore_archive(source, args.target, vault, policy)
        else:
            items = list(restore_tree(source, args.target, vault, policy))
    else:
        with _vault_lock(vault_path, stderr):
            prior: tuple[Any, ...] = ()
            if args.vault_mode == "append":
                _, prior = _seed_from_vault(vault_path)
            cleaner = Cleaner(_plan_from(args), prior=prior)
            try:
                if is_zip:
                    items = cleaner.encode_archive(source, args.target)
                else:
                    items = list(cleaner.encode_tree(source, args.target))
                handle = cleaner.handle()
                _write_vault(
                    vault_path,
                    handle.vault,
                    cleaner.policy,
                    encrypt=args.encrypt,
                    index=handle.entries,
                    cipher=args.cipher,
                )
            finally:
                cleaner.clear()
    counts = {
        status: sum(1 for item in items if item.status == status)
        for status in ("encoded", "decoded", "skipped", "refused")
    }
    data = {
        "items": [asdict(item) for item in items],
        **{status: count for status, count in counts.items() if count},
        "vault": _display_path(vault_path),
        "status": "refused" if counts["refused"] else "ok",
    }
    _emit(data, args.fmt, stdout)
    return EXIT_ERROR if counts["refused"] else EXIT_OK


def _cmd_ask(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``ask`` subcommand.

    Notes
    -----
    **Developer notes.** The exit status is the model command's, so a script
    can tell "the model failed" from "cleanprompt refused" (``1``, with the
    reason on standard error, and the command never started). The guard is
    cleared on every path out, so the values live exactly as long as the
    call.
    """
    from ._bridge import (  # ruff: ignore[import-outside-top-level]
        run_command,
        split_command,
    )
    from ._guard import Guard  # ruff: ignore[import-outside-top-level]
    from ._runtime import Cleaner  # ruff: ignore[import-outside-top-level]

    argv = split_command(args.via)
    text = _resolve_text(args, stdin, stderr, _command_name(args))
    with Guard(Cleaner(_plan_from(args)), format=args.read_as) as guard:
        if args.show_sent:
            stderr.write("sent:\n" + guard.outgoing(text) + "\n")
        return run_command(guard, argv, text, stdout, stderr, timeout=args.timeout)


def _cmd_mcp(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``mcp`` subcommand.

    Notes
    -----
    **Developer notes.** Standard output is the protocol channel, so nothing
    else may be written to it; the one human-readable line goes to standard
    error. The plan is validated before the first message is read, so a bad
    ``--pack`` fails at start-up rather than on an agent's first call.
    """
    from ._guard import Guard  # ruff: ignore[import-outside-top-level]
    from ._mcp import McpServer, serve  # ruff: ignore[import-outside-top-level]
    from ._runtime import Cleaner  # ruff: ignore[import-outside-top-level]

    plan = _plan_from(args)
    server = McpServer(
        lambda: Guard(Cleaner(plan)),
        args.roots or [os.getcwd()],
        version=_package_version(),
    )
    stderr.write(
        f"cleanprompt MCP server ready on stdio (roots: {len(args.roots or [1])})\n"
    )
    stderr.flush()
    return serve(server, stdin, stdout)


def _package_version() -> str:
    """Return this submodule's version without importing the package root."""
    from . import __version__  # ruff: ignore[import-outside-top-level]

    return __version__


def _cmd_plan(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``plan`` subcommand.

    Notes
    -----
    **Developer notes.** ``--check`` exits ``1`` on a stale or invalid plan
    so a CI job can pin a team's plan and fail the build when a dependency
    upgrade changes what it means.
    """
    del stdin
    from ._plan import load_plan, save_plan  # ruff: ignore[import-outside-top-level]

    if args.check:
        try:
            plan = load_plan(args.check)
        except CleanPromptError as exc:
            _emit(
                {
                    "plan": args.check,
                    "problems": str(exc).splitlines(),
                    "status": "stale",
                },
                args.fmt,
                stdout,
            )
            return EXIT_ERROR
        _emit(
            {"plan": args.check, "fingerprint": plan.fingerprint(), "status": "ok"},
            args.fmt,
            stdout,
        )
        return EXIT_OK
    plan = _plan_from(args)
    if args.write:
        fingerprint = save_plan(plan, args.write)
        stderr.write(
            f"saved {_display_path(args.write)} (fingerprint {fingerprint[:12]}…)\n"
        )
        return EXIT_OK
    _emit({**plan.as_dict(), "fingerprint": plan.fingerprint()}, args.fmt, stdout)
    return EXIT_OK


def _cmd_skill(
    args: argparse.Namespace, stdin: IO[str], stdout: IO[str], stderr: IO[str]
) -> int:
    """
    Run the ``skill`` subcommand.

    Notes
    -----
    **Developer notes.** The file is data shipped with the package
    (``_config/agent/SKILL.md``) so it is reviewed like code and versioned
    with the commands it names. An existing file is never replaced without
    ``--force``: it may hold the user's own edits.
    """
    del stdin
    from pathlib import Path  # ruff: ignore[import-outside-top-level]

    source = Path(__file__).resolve().parent / "_config" / "agent" / "SKILL.md"
    body = source.read_text(encoding="utf-8")
    if not args.write:
        stdout.write(body)
        return EXIT_OK
    target = Path(args.write) / "cleanprompt-guard" / "SKILL.md"
    if target.exists() and not args.force:
        raise CleanPromptError(
            f"{_display_path(str(target))} exists; pass --force to replace it"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    stderr.write(f"installed {_display_path(str(target))}\n")
    return EXIT_OK


def _cmd_session(
    args: argparse.Namespace,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """Run the interactive ``cli`` subcommand."""
    # deferred: keeps import cost off other paths
    from ._session import run_session  # ruff: ignore[import-outside-top-level]

    return run_session(args, stdin, stdout, stderr)


def _cmd_flask(
    args: argparse.Namespace,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """Run the ``flask`` subcommand."""
    del stdin, stdout
    # deferred: the web tier may be absent
    from ._serve import serve  # ruff: ignore[import-outside-top-level]

    return serve(args, stderr)


def _cmd_docker(
    args: argparse.Namespace,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """Run the ``docker`` subcommand."""
    del stdin
    from ._serve import container_files  # ruff: ignore[import-outside-top-level]

    files = container_files(port=args.port, with_ner=args.with_ner)
    if not args.write:
        for name, body in files.items():
            stdout.write(f"# ---- {name} ----\n{body}\n")
        return EXIT_OK
    os.makedirs(args.write, exist_ok=True)
    for name, body in files.items():
        target = os.path.join(args.write, name)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(body)
        stderr.write(f"wrote {target}\n")
    stderr.write(
        "build and run with:\n"
        f"  docker compose -f {args.write}/docker-compose.yml up --build\n"
    )
    return EXIT_OK


#: Handler lookup, resolved by the name each :class:`Command` declares.
_HANDLERS: dict[str, Callable[..., int]] = {}


def _handler_for(command_name: str) -> Callable[..., int]:
    """Return the handler for a command name or alias."""
    if not _HANDLERS:
        module = globals()
        for command in COMMANDS:
            _HANDLERS[command.name] = module[command.handler]
            for alias in command.aliases:
                _HANDLERS[alias] = module[command.handler]
    return _HANDLERS[command_name]


def main(  # ruff: ignore[too-many-branches, too-many-return-statements]
    argv: Sequence[str] | None = None,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    frontend: str | None = None,
) -> int:
    """
    Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name. Defaults to ``sys.argv[1:]``.
    stdin, stdout, stderr : file-like, optional
        Streams to use. Default to the interpreter's own. Injectable so the
        tests exercise the real entry point rather than a parallel one.
    frontend : {"argparse", "click"}, optional
        Force a frontend. Normally chosen automatically; see
        :func:`~scikitplot.cleanprompt._frontends.select_frontend`.

    Returns
    -------
    int
        Process exit status: ``0`` success, ``1`` handled error, ``2`` usage,
        ``3`` ``scan`` found something, ``69`` an optional tier is missing,
        ``130`` interrupted, ``141`` the reader closed the pipe (``EXIT_ERROR``
        on platforms without ``SIGPIPE``).

    Notes
    -----
    **User notes.** Two equivalent ways in::

        python -m scikitplot.cleanprompt doctor
        scikitplot cleanprompt doctor

    The second goes through the project-wide CLI, which forwards every argument
    here verbatim.

    **Developer notes.** This is the delegation target registered in the
    project-wide CLI, so its signature satisfies the delegate contract
    ``main(argv) -> int``. The extra keyword-only parameters have defaults and
    do not disturb it.

    Parsing is the frontend's job and everything after it is not, so both
    frontends converge on one :class:`argparse.Namespace` and one handler call.
    Only :class:`CleanPromptError` and :class:`OSError` become a message and a
    status; anything else propagates with its traceback, because an unexpected
    exception in a privacy tool is a bug report, not a user error to be
    summarised away.

    Examples
    --------
    >>> main(["kinds"]) == 0
    True
    >>> main(["kinds"], frontend="argparse") == 0
    True
    """
    in_ = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    chosen = frontend or _frontend_override(argv_list)

    try:
        runner, _frontend_name = load_runner(chosen)
        # Both frontends print help and usage errors through sys.stdout and
        # sys.stderr directly: argparse's help action and click's echo both
        # bypass any stream the caller passed. Redirecting around the parse is
        # the one place that fixes it for both, and it is what lets the tests
        # drive the real entry point instead of a parallel one.
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            name, args = runner(COMMANDS, argv_list, PROG, DESCRIPTION, EPILOG, out)
    except SystemExit as exc:  # --help and usage errors from either frontend
        code = exc.code
        if code is None:
            return EXIT_OK
        return code if isinstance(code, int) else EXIT_USAGE
    except ValueError as exc:  # an unknown --frontend value
        err.write(f"error: {exc}\n")
        return EXIT_USAGE

    if name == "--version":
        from . import __version__  # ruff: ignore[import-outside-top-level]

        out.write(f"scikitplot.cleanprompt {__version__}\n")
        return EXIT_OK
    if name is None:
        return EXIT_USAGE

    level = getattr(args, "log_level", None) or log_level_from_env()
    if level != "warning" or getattr(args, "log_level", None):
        # Logs go to stderr so that piping stdout stays clean.
        configure_logging(level, getattr(args, "log_format", "text") or "text", err)
        get_logger().debug("command=%s frontend=%s", name, _frontend_name)

    try:
        return _handler_for(name)(args, in_, out, err)
    except CapabilityError as exc:
        err.write(f"error: {exc}\n")
        if exc.install_hint:
            err.write(f"hint: {exc.install_hint}\n")
        return EXIT_UNAVAILABLE
    except CleanPromptError as exc:
        err.write(f"error: {exc}\n")
        hint = getattr(exc, "install_hint", None)
        if hint:
            err.write(f"hint: {hint}\n")
        return EXIT_ERROR
    except KeyboardInterrupt:
        err.write("\ninterrupted\n")
        return EXIT_INTERRUPTED
    except BrokenPipeError:
        # `cleanprompt kinds | head -1` is an ordinary thing to type, and the
        # reader closing the pipe is not an error this program can do anything
        # about. Reporting it as one printed `error: [Errno 32] Broken pipe`
        # into the terminal for a correct command line (CP-045).
        return _exit_broken_pipe(owns_stdout=out is sys.stdout)
    except OSError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_ERROR


def _exit_broken_pipe(owns_stdout: bool = True) -> int:
    """
    Detach this process from the closed pipe and report the conventional status.

    Parameters
    ----------
    owns_stdout : bool, default=True
        Whether the stream that failed really is the interpreter's
        :data:`sys.stdout`. ``False`` when the caller injected a stream, in
        which case descriptor ``1`` belongs to somebody else and must be left
        alone.

    Returns
    -------
    int
        :data:`EXIT_BROKEN_PIPE`.

    Notes
    -----
    **User notes.** Nothing to do. Piping a long report into ``head``, ``less``
    or ``grep -m1`` is normal, and this is what makes it quiet.

    **Developer notes — why the file descriptor is replaced.**

    Returning from :func:`main` is not the end of the story: the interpreter
    flushes ``sys.stdout`` during shutdown, and if the pipe is still closed that
    flush raises again, too late for any handler. Python prints
    ``Exception ignored in: <_io.TextIOWrapper ...>`` and the message the
    handler just suppressed reappears in a worse form.

    Duplicating ``os.devnull`` over descriptor ``1`` gives the final flush
    somewhere harmless to go. It is done on the descriptor rather than by
    reassigning :data:`sys.stdout` because the object that will be flushed is
    the one the interpreter holds, not the name this module can rebind.

    This is the recipe in CPython's own note on ``SIGPIPE``, with the exit
    status made portable: see :mod:`signal`.

    ``owns_stdout`` is the guard that keeps the repair proportionate, and it
    exists because the first version of this function did not have one.
    :func:`main` takes injectable streams so the tests can drive the real entry
    point, and replacing descriptor ``1`` when the failing stream was somebody
    else's object tore down a descriptor this process did not own — the test
    harness's, in the run that found it. The condition is not "a
    ``BrokenPipeError`` happened" but "the object the interpreter will flush at
    shutdown is the one that just refused a write".

    Failure here is not escalated. The process is already on its way out with a
    status that says why, and a second :class:`OSError` raised while trying to
    quieten the first would replace a clean exit with a traceback.
    """
    if not owns_stdout:
        return EXIT_BROKEN_PIPE
    try:
        descriptor = sys.stdout.fileno()
    except (OSError, ValueError, AttributeError):
        # A captured or non-file stdout has no descriptor to replace, and needs
        # none: nothing will flush it into a closed pipe.
        return EXIT_BROKEN_PIPE
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return EXIT_BROKEN_PIPE
    try:
        os.dup2(devnull, descriptor)
    except OSError:
        # There is no third thing to try and no stream left to report on: both
        # of this process's output descriptors are the ones in question. The
        # returned status is the whole of what can be communicated, and it
        # already says the pipe broke.
        return EXIT_BROKEN_PIPE
    finally:
        os.close(devnull)
    return EXIT_BROKEN_PIPE


def _frontend_override(argv: Sequence[str]) -> str | None:
    """
    Read ``--frontend NAME`` out of ``argv`` before parsing.

    Notes
    -----
    **Developer notes.** The option that chooses the parser cannot be read by
    the parser it chooses, so it is scanned for first. Only the exact long form
    is recognised; the frontends both declare it too, so it still appears in
    ``--help`` and is still validated there.
    """
    for index, _key in enumerate(argv):
        if _key == "--frontend" and index + 1 < len(argv):
            return argv[index + 1]
        if _key.startswith("--frontend="):
            return _key.split("=", 1)[1]
    return None
