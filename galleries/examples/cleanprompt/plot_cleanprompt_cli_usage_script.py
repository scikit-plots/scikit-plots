"""
Cleanprompt on the Command Line: Single Commands and Edge Cases
===============================================================

.. currentmodule:: scikitplot.cleanprompt

A reference for the command line, organised as *the ordinary form first, then
the things that bite*. The second half is the longer one, because the failures
a redaction tool can have at the shell are not evenly distributed: almost all
of them are about **how the text gets in**.

The command line is reachable three ways, all the same code:

.. code-block:: bash

    scikitplot cleanprompt doctor          # the project-wide CLI
    python -m scikitplot.cleanprompt doctor
    python -m scikitplot cleanprompt doctor

Two frontends parse it — ``click`` when installed, ``argparse`` otherwise — and
``--frontend`` forces either. They are required to read a command line the same
way; this example asserts that rather than claiming it.

Exit codes, which is what a script actually consumes:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Status
     - Meaning
   * - ``0``
     - the command did what it was asked
   * - ``1``
     - a handled error: no vault, unreadable file, wrong passphrase,
       ``--strict`` found a placeholder the vault does not hold
   * - ``2``
     - a usage error: unknown option, unknown command, an abbreviation
   * - ``3``
     - ``scan`` only: sensitive values were present
   * - ``69``
     - an optional tier is required and is not installed
   * - ``130``
     - interrupted
   * - ``141``
     - the reader closed the pipe, as ``| head`` does

Every value printed below is reserved and cannot reach anybody.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-cleanprompt-cli-")
_HOME = Path(_WORKSPACE.name)
_VAULT = _HOME / "vault.json"

os.environ["CLEANPROMPT_VAULT"] = str(_VAULT)


def _cli() -> list[str]:
    """Return the command prefix that reaches this submodule's CLI."""
    executable = shutil.which("scikitplot")
    if executable:
        return [executable, "cleanprompt"]
    return [sys.executable, "-m", "scikitplot.cleanprompt"]


CLI = _cli()


def _mask(text: str) -> str:
    return text.replace(_WORKSPACE.name, "<workspace>")


def _echo(argument: str) -> str:
    single_line = _mask(argument).replace("\n", "⏎")
    if len(single_line) > 52:
        single_line = single_line[:49] + "..."
    if single_line == "" or " " in argument:
        return '"{0}"'.format(single_line)
    return single_line


def run(
    *arguments: str,
    stdin_text: str | None = None,
    limit: int = 0,
    expect: int | None = None,
) -> subprocess.CompletedProcess:
    """Run one CLI command and show the command, both streams and the status."""
    completed = subprocess.run(
        [*CLI, *arguments],
        input=stdin_text if stdin_text is not None else "",
        capture_output=True,
        text=True,
    )
    print("$ cleanprompt", " ".join(_echo(one) for one in arguments))
    for label, stream in (("out", completed.stdout), ("err", completed.stderr)):
        if not stream:
            continue
        lines = _mask(stream).rstrip("\n").split("\n")
        if limit and len(lines) > limit:
            lines = lines[:limit] + ["... ({0} more)".format(len(lines) - limit)]
        print("  {0} │ {1}".format(label, "\n      │ ".join(lines)))
    print("  exit │", completed.returncode)
    if expect is not None:
        assert completed.returncode == expect, "expected exit {0}".format(expect)
    return completed


# %%
# Part one — the ordinary forms
# -----------------------------
#
# 1. The four ways to give it text
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# One command, four input routes.  They exist because a shell is not a good
# place to type a paragraph, and most of the edge cases later in this example
# are somebody discovering that.

PROMPT = "Mail ada@example.com about the outage on 192.0.2.10."

# %%
# **As an argument.** Fine for short text.  Quote it.

run("encode", "--quiet", "--vault-mode", "overwrite", PROMPT)

# %%
# **From a file**, with ``--in``.  No quoting rules apply at all.

source = _HOME / "prompt.txt"
source.write_text(PROMPT, encoding="utf-8")

run("encode", "--quiet", "--vault-mode", "overwrite", "--in", str(source))

# %%
# **From standard input**, with ``--in -`` or by omitting the text.  This is
# the route a pipeline uses.

run("encode", "--quiet", "--vault-mode", "overwrite", "--in", "-", stdin_text=PROMPT)

# %%
# **From a heredoc**, which is standard input with the shell's quoting turned
# off entirely.  This is the answer to "how do I paste a paragraph":
#
# .. code-block:: bash
#
#     cleanprompt encode --quiet <<'EOF'
#     Mail ada@example.com about the outage on 192.0.2.10.
#     It has (parentheses), $dollars, `backticks` and 'quotes'.
#     EOF
#
# The quoted delimiter ``<<'EOF'`` is the load-bearing part: it stops the shell
# expanding anything between the markers.  Without the quotes, ``$dollars``
# becomes empty and ``` `backticks` ``` are executed.

AWKWARD = (
    "Mail ada@example.com about the outage on 192.0.2.10.\n"
    "It has (parentheses), $dollars, `backticks` and 'quotes'."
)

run("encode", "--quiet", "--vault-mode", "overwrite", stdin_text=AWKWARD)

# %%
# 2. Where the output goes
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The prompt goes to **standard output**; every report, warning and summary
# goes to **standard error**.  That split is the whole reason the command is
# usable in a pipeline: ``-q`` silences the summary, and stdout carries nothing
# but the text to paste.

bare = run("encode", "--quiet", "--vault-mode", "overwrite", PROMPT)

print()
print("stdout is exactly the prompt:", repr(bare.stdout))
print("stderr is empty under --quiet:", bare.stderr == "")

# %%
# Without ``-q`` the same stdout is produced and the summary appears beside it.
# Redirecting one does not affect the other:
#
# .. code-block:: bash
#
#     cleanprompt encode --in draft.txt > clean.txt      # keep the summary on screen
#     cleanprompt encode --in draft.txt 2>/dev/null      # keep only the prompt
#     cleanprompt encode --in draft.txt | pbcopy         # straight to the clipboard
#     cleanprompt encode --in draft.txt --out clean.txt  # no shell redirection

run("encode", "--vault-mode", "overwrite", "--out", str(_HOME / "clean.txt"), PROMPT)
print("file written:", (_HOME / "clean.txt").read_text(encoding="utf-8").strip())

# %%
# 3. The commands, and when to reach for each
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``encode`` and ``decode`` are the pair.  The rest answer other questions.

run("kinds", limit=6)

# %%
# ``inspect`` is a dry run: it reports what *would* be removed and writes
# nothing at all — no vault, no output file.  Use it before trusting a policy.

run("inspect", "--format", "json", PROMPT, limit=12)

# %%
# ``scan`` is the same detection with a script's interface: status ``3`` when
# it finds something, ``0`` when it does not.  This is the CI gate.

run("scan", "--kinds", "EMAIL", "nothing sensitive here", limit=2, expect=0)
run("scan", "--kinds", "EMAIL", "mail ada@example.com", limit=2, expect=3)

# %%
# ``doctor`` reports what is active **and what is blind**.  A redaction tool
# that finds nothing looks exactly like a clean bill of health, so this command
# exists to make the difference visible.

run("doctor", "--format", "json", limit=6)

# %%
# Part two — the edge cases
# -------------------------
#
# 4. The shell eats the text before the program sees it
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The most common first failure, and it is not this program's error message:
#
# .. code-block:: console
#
#     $ cleanprompt encode Our customer (Acme Ltd) wrote from ada@example.com
#     bash: syntax error near unexpected token `('
#
# ``bash:`` is the clue — the shell failed, so the program never ran.  Three
# fixes, in increasing order of robustness:
#
# .. code-block:: bash
#
#     cleanprompt encode 'Our customer (Acme Ltd) wrote from ada@example.com'
#     cleanprompt encode --in draft.txt
#     cleanprompt encode <<'EOF'
#     Our customer (Acme Ltd) wrote from ada@example.com
#     EOF
#
# Single quotes are enough for parentheses.  For a paragraph with quotes of its
# own, use a file or a heredoc: there is no quoting to get wrong.

run(
    "encode",
    "--quiet",
    "--vault-mode",
    "overwrite",
    "Our customer (Acme Ltd) wrote from ada@example.com",
)

# %%
# 5. Text that starts with a dash
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A value beginning with ``-`` looks like an option.  ``--`` ends the options
# and everything after it is text — the POSIX convention, honoured identically
# by both frontends.

run("inspect", "--secret is ada@example.com", limit=4, expect=2)

# %%
# With the delimiter, the same text is read as text.

run("inspect", "--", "--secret is ada@example.com", limit=4, expect=0)

# %%
# Only the **first** ``--`` is consumed, so a literal ``--`` inside the text
# survives, and text that is exactly ``--`` is passed through.

run("encode", "--quiet", "--vault-mode", "overwrite", "--", "use -- to end options")
run("encode", "--quiet", "--vault-mode", "overwrite", "--", "--")

# %%
# The one place the two frontends still differ is a dash-leading **option
# value** given in separated form.  Use the attached spelling, which both
# accept:
#
# .. code-block:: bash
#
#     cleanprompt encode --hide=-secret "the -secret token"   # both frontends
#     cleanprompt encode --hide -secret "the -secret token"   # click only
#
# Closing the gap would mean overriding a private argparse method that has
# moved between releases, so it is documented rather than papered over.

run("encode", "--quiet", "--vault-mode", "overwrite", "--hide=-secret", "the -secret token")

for frontend in ("click", "argparse"):
    result = subprocess.run(
        [*CLI, "--frontend", frontend, "encode", "-q", "--vault-mode", "overwrite",
         "--hide", "-secret", "the -secret token"],
        capture_output=True,
        text=True,
        input="",
    )
    print("{0:<9} separated --hide: exit {1}".format(frontend, result.returncode))

# %%
# 6. A mistyped option must not become your prompt
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This is the failure mode worth being strict about.  If a mistyped flag were
# accepted as text, the tool would quietly redact the flag and send the *real*
# prompt — which is still sitting in the next argument — to the model.  Both
# frontends refuse with status ``2``.

run("inspect", "--nosuchoption", "ada@example.com", limit=4, expect=2)

# %%
# Abbreviations are refused for the same reason.  ``--form`` is an unambiguous
# prefix of ``--format`` *today*; it stops being one the day an option called
# ``--formal`` is added, and a script that worked for a year then breaks on an
# upgrade it never used.

run("inspect", "--form", "json", "ada@example.com", limit=4, expect=2)
run("inspect", "--format", "json", "ada@example.com", limit=2, expect=0)

# %%
# 7. Both frontends read the same command line
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A divergence here is a bug, not a preference.  The check is mechanical.

CASES = [
    ["inspect", "--format", "json", "ada@example.com"],
    ["inspect", "--form", "json", "ada@example.com"],
    ["inspect", "--", "--secret is ada@example.com"],
    ["inspect", "--nosuchoption", "x"],
    ["kinds"],
    ["-V"],
]

divergences = []
for case in CASES:
    statuses = {}
    for frontend in ("argparse", "click"):
        completed = subprocess.run(
            [*CLI, "--frontend", frontend, *case],
            capture_output=True,
            text=True,
            input="",
        )
        statuses[frontend] = completed.returncode
    agree = statuses["argparse"] == statuses["click"]
    if not agree:
        divergences.append((case, statuses))
    print(
        "{0:<6} argparse={1} click={2}  {3}".format(
            "agree" if agree else "DIFFER",
            statuses["argparse"],
            statuses["click"],
            " ".join(case),
        )
    )

assert divergences == [], divergences

# %%
# 8. Pipes, and a reader that leaves early
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``| head`` closes the pipe as soon as it has what it wants.  That is not an
# error this program can act on, so it exits quietly with the conventional
# ``141`` rather than printing ``error: [Errno 32] Broken pipe`` over a
# perfectly correct command line.

writer = subprocess.Popen(
    [*CLI, "kinds", "--format", "json"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
first_line = writer.stdout.readline() if writer.stdout else ""
if writer.stdout:
    writer.stdout.close()
pipe_errors = writer.stderr.read() if writer.stderr else ""
writer.wait(timeout=60)

print("first line read:", first_line.strip())
print("stderr from the writer:", repr(pipe_errors))
print("writer status:", writer.returncode)
assert "Broken pipe" not in pipe_errors
assert "Exception ignored" not in pipe_errors

# %%
# 9. Degenerate input
# ^^^^^^^^^^^^^^^^^^^
# Empty text, whitespace, and a text that is *already* redacted.  None of these
# is an error; all of them have to behave predictably.
#
# The third is invariant ``I6``, idempotence: running ``encode`` on its own
# output must not re-redact the placeholders into ``[CUSTOM-1]`` nonsense.

run("encode", "--quiet", "--vault-mode", "overwrite", "", expect=0)
run("encode", "--quiet", "--vault-mode", "overwrite", "   ", expect=0)

once = run("encode", "--quiet", "--vault-mode", "overwrite", PROMPT).stdout
twice = run("encode", "--quiet", "--vault-mode", "append", "--in", "-", stdin_text=once)

print()
print("idempotent:", once.strip() == twice.stdout.strip())
assert once.strip() == twice.stdout.strip()

# %%
# 10. A value at the end of a sentence
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Worth its own case because it was a real defect twice, in opposite
# directions: a sentence-final IP address was not detected at all, and a URL
# swallowed the sentence's full stop into the placeholder.  Both are checked by
# a class-level test now, so a pattern added next year is checked the same way.

run(
    "inspect",
    "--format",
    "json",
    "The host is 192.0.2.10. See https://example.com/docs.",
    limit=14,
)

# %%
# 11. ``decode`` when the model did not return the label you issued
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Between ``encode`` and ``decode`` the text passes through a language model,
# and a model rewrites tokens: it lower-cases the label, swaps the hyphen for
# an underscore or a space, escapes the brackets for Markdown, or wraps it
# across a line.
#
# ``decode`` accepts that bounded set and **reports every repair**.  It does
# not accept an invented label, and ordinary prose like ``[note 2]`` is
# returned untouched.

run("encode", "--quiet", "--vault-mode", "overwrite", "Mail ada@example.com or bob@example.com")
run("decode", "I wrote to [email_1] and [EMAIL 2]; see [note 2] for the rest.")

# %%
# ``--exact`` turns the leniency off for a caller who depends on the strict
# reading, and ``--strict`` turns an unknown placeholder into an error instead
# of leaving it in place.

run("decode", "--exact", "I wrote to [email_1].")
run("decode", "--strict", "what about [EMAIL-9]?", expect=1)

# %%
# 12. Cleanup
# ^^^^^^^^^^^

run("forget", "--force")

os.environ.pop("CLEANPROMPT_VAULT", None)
_WORKSPACE.cleanup()

print("Temporary workspace cleaned:", not Path(_WORKSPACE.name).exists())

# %%
#
# .. tags::
#
#    model-workflow: cleanprompt
#    plot-type: text
#    level: beginner
#    purpose: reference
