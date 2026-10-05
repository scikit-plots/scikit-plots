"""
Cleanprompt Basics: Hide a Value, Then Get It Back
==================================================

.. currentmodule:: scikitplot.cleanprompt

The whole submodule is one loop with one hop in the middle that it does not
control:

.. code-block:: text

    your text  ──encode──▶  a prompt you can paste anywhere
                                     │
                            the model answers
                                     │
    your answer ◀──decode──  the reply, still in placeholders

This example walks that loop twice — once on the command line and once from
Python — on a text small enough to read in full, and checks the round trip
rather than asserting it in prose.

Nothing here needs a third-party package. ``import scikitplot.cleanprompt``
imports no ``spacy``, no ``flask``, no ``cryptography`` and no ``numpy``, so
this example runs on a bare installation.

Every value below is a reserved one that cannot reach anybody: ``example.com``
is reserved by :rfc:`2606`, ``+1 555 0100``–``0199`` is the North American
fiction block, and ``192.0.2.0/24`` is the :rfc:`5737` documentation range.
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

# %%
# 1. Give the vault somewhere temporary to live
# ---------------------------------------------
# A vault holds the values that were removed, in clear text.  Left to itself
# the command line writes one into the platform state directory, which is the
# right default for a person at a terminal and the wrong one for a
# documentation build.  ``CLEANPROMPT_VAULT`` overrides it for every command in
# this process and every subprocess it starts.

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-cleanprompt-basics-")
_VAULT = Path(_WORKSPACE.name) / "vault.json"

os.environ["CLEANPROMPT_VAULT"] = str(_VAULT)

print("Vault for this example:", _VAULT.name, "(inside a temporary directory)")

# %%
# 2. The text we are not willing to send
# --------------------------------------
# One address, one telephone number, one host.  Short enough that you can check
# the output by eye, which is the point of a first example.

PROMPT = (
    "Draft a reply to ada@example.com. "
    "She called from +1 555 0142 about the outage on 192.0.2.10."
)

print(PROMPT)

# %%
# 3. The command line: ``encode``
# -------------------------------
# ``encode`` prints the prompt to **standard output** and its summary to
# **standard error**.  That split is what makes the command pipeable: the thing
# you paste into a chat window is the only thing on stdout.
#
# The gallery runs the module form, which always works.  An installed
# scikit-plots also provides ``scikitplot cleanprompt ...`` through the
# centralized CLI, and ``python -m scikitplot.cleanprompt ...`` directly.


def _cli() -> list[str]:
    """Return the command prefix that reaches this submodule's CLI."""
    executable = shutil.which("scikitplot")
    if executable:
        return [executable, "cleanprompt"]
    return [sys.executable, "-m", "scikitplot.cleanprompt"]


CLI = _cli()


def _mask(text: str) -> str:
    """Replace the temporary workspace path so the output is reproducible."""
    return text.replace(_WORKSPACE.name, "<workspace>")


def _echo(argument: str) -> str:
    """Render one argument the way a shell would show it, shortened."""
    single_line = argument.replace("\n", "⏎")
    if len(single_line) > 56:
        single_line = single_line[:53] + "..."
    return '"{0}"'.format(single_line) if " " in argument else single_line


def _stream(label: str, text: str, limit: int = 0) -> None:
    """Print one captured stream, indented under its label."""
    lines = _mask(text).rstrip("\n").split("\n")
    if limit and len(lines) > limit:
        lines = lines[:limit] + ["... ({0} more line(s))".format(len(lines) - limit)]
    print("  {0} │ {1}".format(label, "\n         │ ".join(lines)))


def run(
    *arguments: str,
    stdin_text: str | None = None,
    limit: int = 0,
) -> subprocess.CompletedProcess:
    """Run one CLI command, echo it, and show both streams separately."""
    completed = subprocess.run(
        [*CLI, *arguments],
        input=stdin_text,
        capture_output=True,
        text=True,
    )
    print("$ cleanprompt", " ".join(_echo(one) for one in arguments))
    if completed.stdout:
        _stream("stdout", completed.stdout, limit)
    if completed.stderr:
        _stream("stderr", completed.stderr, limit)
    print("  exit   │", completed.returncode)
    return completed


encoded = run("encode", PROMPT)

# %%
# The prompt on stdout is what goes to the model.  Check that, rather than
# trusting it: none of the three values may appear in it.

clean_prompt = encoded.stdout.strip()

for secret in ("ada@example.com", "+1 555 0142", "192.0.2.10"):
    assert secret not in clean_prompt, "{0!r} survived redaction".format(secret)

print("no original value survives in the prompt:", True)

# %%
# 4. The command line: ``decode``
# -------------------------------
# Paste the prompt into any chat.  Whatever the model answers still contains
# the placeholders, because that is all it ever saw.  ``decode`` reads the same
# vault ``encode`` wrote and puts the values back.
#
# The reply below stands in for a model.  Nothing in this example sends
# anything anywhere.

REPLY = (
    "Here is a draft:\n\n"
    "    Hi, thanks for calling from [PHONE-1]. I have looked at [IPV4-1] "
    "and will write back to [EMAIL-1] within the hour."
)

decoded = run("decode", REPLY)

# %%
# 5. Ending the conversation
# --------------------------
# The vault accumulates across ``encode`` calls so that a reply quoting an
# earlier turn still restores.  When the conversation is over, delete it.
# ``forget`` without ``--force`` is a dry run: it says what it would remove and
# removes nothing.

run("forget")
run("forget", "--force")

# %%
# 6. The same loop from Python
# ----------------------------
# :func:`encode` returns an :class:`EncodedPrompt`.  ``.text`` is what you send;
# ``.handle`` is what :func:`decode` needs to put the values back.  The handle
# never leaves your process.

from scikitplot.cleanprompt import decode, encode  # noqa: E402

encoded_prompt = encode(PROMPT)

print("send this:")
print("   ", encoded_prompt.text)
print()
print("kept here:")
for entry in encoded_prompt.result.entries:
    print("    {0:<12} {1:<6} {2}".format(entry.label, entry.kind, entry.original))

# %%
# The reply comes back in placeholders, and :func:`decode` closes the loop.

answer = decode(REPLY, encoded_prompt.handle)

print(answer)

# %%
# 7. Check the loop rather than describing it
# -------------------------------------------
# The property worth asserting is that a reply which is exactly the redacted
# prompt restores to exactly the original text.  That is invariant ``I1`` —
# the round trip — and it is what everything else in this submodule is built
# on.

restored = decode(encoded_prompt.text, encoded_prompt.handle)

print("round trip exact:", restored == PROMPT)
assert restored == PROMPT

# %%
# 8. One shot, if you only want to see it work
# --------------------------------------------
# ``roundtrip`` does all of the above in one command, with a stand-in reply, and
# prints the five stages side by side.  It is the fastest way to show somebody
# what this submodule does.
#
# It writes no vault, because the vault only has to survive between two
# commands and here there is one.

run("roundtrip", PROMPT)

# %%
# 9. What is *not* being detected
# -------------------------------
# Regular expressions find structured values.  They cannot find a person's
# name, and a redaction tool that quietly finds nothing looks exactly like a
# clean bill of health.  ``doctor`` is the command that refuses to let that
# happen: it reports what is active **and what is blind**.
#
# Run it before trusting a redaction, not after.

run("doctor", limit=14)

# %%
# 10. Cleanup
# -----------
# Remove the temporary vault directory.  ``forget --force`` above already
# unlinked the vault file itself; this removes the directory that held it.

os.environ.pop("CLEANPROMPT_VAULT", None)
_WORKSPACE.cleanup()

print("Temporary vault workspace cleaned:", not Path(_WORKSPACE.name).exists())

# %%
#
# .. tags::
#
#    model-workflow: cleanprompt
#    plot-type: text
#    level: beginner
#    purpose: showcase
