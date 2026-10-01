"""
Cleanprompt Recipes: CI Gates, Notebooks, Pipelines and Agents
==============================================================

.. currentmodule:: scikitplot.cleanprompt

Six complete, working patterns for the places this submodule actually gets
used. Each one is short, runs here, and is written so it can be lifted into a
project without editing anything but the paths.

.. list-table::
   :header-rows: 1
   :widths: 6 28 66

   * - #
     - Recipe
     - The question it answers
   * - 1
     - CI gate
     - "how do I fail a build when a file contains a value we must not ship?"
   * - 2
     - Pre-commit hook
     - "how do I catch it before it is committed, not after?"
   * - 3
     - Notebook helper
     - "how do I paste a prompt from a notebook without leaking the dataframe?"
   * - 4
     - Batch pipeline
     - "how do I redact a directory reproducibly?"
   * - 5
     - Agent loop
     - "how do I put this between an agent and a model API?"
   * - 6
     - Log hygiene
     - "how do I stop a value reaching the logs in the first place?"

Every value below is reserved and cannot reach anybody.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-cleanprompt-recipes-")
_HOME = Path(_WORKSPACE.name)

os.environ["CLEANPROMPT_VAULT"] = str(_HOME / "vault.json")


def _cli() -> list[str]:
    executable = shutil.which("scikitplot")
    if executable:
        return [executable, "cleanprompt"]
    return [sys.executable, "-m", "scikitplot.cleanprompt"]


CLI = _cli()


def run(*arguments: str, limit: int = 0):
    """Run one CLI command and show both streams, with paths masked."""
    completed = subprocess.run(
        [*CLI, *arguments], input="", capture_output=True, text=True
    )
    shown = " ".join(
        ('"{0}"'.format(one) if " " in one else one).replace(
            _WORKSPACE.name, "<workspace>"
        )
        for one in arguments
    )
    print("$ cleanprompt", shown)
    for label, stream in (("out", completed.stdout), ("err", completed.stderr)):
        if not stream:
            continue
        lines = stream.replace(_WORKSPACE.name, "<workspace>").rstrip("\n").split("\n")
        if limit and len(lines) > limit:
            lines = lines[:limit] + ["... ({0} more)".format(len(lines) - limit)]
        print("  {0} │ {1}".format(label, "\n      │ ".join(lines)))
    print("  exit │", completed.returncode)
    return completed


# %%
# Recipe 1 — a CI gate
# --------------------
# ``scan`` exists for exactly this.  It runs the same detection as everything
# else and reports through the only channel a build system reads: the exit
# status.  ``3`` means values were found; ``0`` means none were.
#
# .. code-block:: yaml
#
#     # .github/workflows/no-secrets.yml
#     - name: Reject prompts containing personal data
#       run: |
#         python -m scikitplot.cleanprompt scan \
#             --profile strict \
#             --format json \
#             --in prompts/support-macros.md
#
# Put ``--profile strict`` on a gate.  A gate is the one place where a false
# positive costs a minute and a false negative costs the thing you were
# protecting.

fixture = _HOME / "support-macros.md"
fixture.write_text(
    "# Macros\n\n"
    "Escalation contact: ada@example.com\n"
    "Bridge line: +1 555 0142\n",
    encoding="utf-8",
)

clean_fixture = _HOME / "clean-macros.md"
clean_fixture.write_text("# Macros\n\nEscalation contact: the on-call rota.\n", encoding="utf-8")

found = run("scan", "--profile", "strict", "--format", "json", "--in", str(fixture), limit=8)
passed = run("scan", "--profile", "strict", "--format", "json", "--in", str(clean_fixture), limit=4)

print()
print("gate fails on the dirty file:", found.returncode == 3)
print("gate passes on the clean file:", passed.returncode == 0)
assert found.returncode == 3 and passed.returncode == 0

# %%
# ``--max-findings`` exists for a repository that is adopting the gate on an
# existing codebase: set it to today's count, then ratchet it down.  It is a
# migration tool, not a setting — a gate that tolerates findings forever is not
# a gate.

run("scan", "--max-findings", "5", "--in", str(fixture), limit=3)

# %%
# Recipe 2 — a pre-commit hook
# ----------------------------
# The same gate, one step earlier.  Add this to ``.pre-commit-config.yaml``:
#
# .. code-block:: yaml
#
#     repos:
#       - repo: local
#         hooks:
#           - id: cleanprompt-scan
#             name: no personal data in prompts
#             entry: python -m scikitplot.cleanprompt scan --profile strict --in
#             language: system
#             files: '^prompts/.*\\.(md|txt)$'
#
# ``pre-commit`` appends the changed file names to ``entry``, which is why the
# command ends with ``--in``.  Because the hook runs on the *staged* content,
# it catches the value before it exists in history — which matters, because
# removing a value from a Git history is considerably harder than not
# committing it.

# %%
# Recipe 3 — a notebook helper
# ----------------------------
# In a notebook the prompt is usually built from data, so the risk is not a
# value you typed but one that arrived in a dataframe.  Wrap the send.

from scikitplot.cleanprompt import session  # noqa: E402


def ask(question: str, send) -> str:
    """Redact, send, restore — the whole loop as one call.

    Parameters
    ----------
    question : str
        The prompt, values and all.
    send : callable
        Takes the redacted prompt, returns the model's reply. It never sees a
        value.

    Returns
    -------
    str
        The reply, with the values put back.
    """
    with session() as chat:
        return chat.roundtrip(question, send)


ROWS = [
    {"name": "Marion Holt", "email": "ada@example.com", "host": "192.0.2.10"},
    {"name": "Devin Nakamura", "email": "bob@example.com", "host": "192.0.2.11"},
]

built = "Summarise these incidents:\n" + "\n".join(
    "- {name} ({email}) on {host}".format(**row) for row in ROWS
)


def fake_model(sent: str) -> str:
    """Stand in for a model, and assert what it was given."""
    for row in ROWS:
        assert row["email"] not in sent, "a value reached the model"
        assert row["host"] not in sent, "a value reached the model"
    return "Two incidents. I will follow up with the addresses in the text."


print(ask(built, fake_model))

# %%
# Note what the assertions inside ``fake_model`` are doing.  A wrapper like this
# is worth very little unless something checks that it worked, and the cheapest
# place to check is the boundary itself.  Keep those assertions in the real
# version too.

# %%
# Recipe 4 — a reproducible batch pass
# ------------------------------------
# For a directory of documents, use ``redact`` rather than ``encode``: it
# defaults to ``--vault-mode overwrite``, which is what a reproducible pipeline
# needs — each run starts from a known state instead of inheriting the last
# one's numbering.
#
# One vault per document, beside the output, so the two travel together.

corpus = _HOME / "corpus"
corpus.mkdir()
output = _HOME / "redacted"
output.mkdir()

for index, body in enumerate(
    [
        "Ticket 1: ada@example.com could not reach 192.0.2.10.",
        "Ticket 2: bob@example.com was billed on 4242 4242 4242 4242.",
        "Ticket 3: nothing sensitive in this one.",
    ],
    start=1,
):
    (corpus / "ticket-{0}.txt".format(index)).write_text(body, encoding="utf-8")

for source in sorted(corpus.glob("*.txt")):
    run(
        "redact",
        "--quiet",
        "--profile",
        "strict",
        "--in",
        str(source),
        "--out",
        str(output / source.name),
        "--vault",
        str(output / (source.stem + ".vault.json")),
    )

print()
for produced in sorted(output.glob("*.txt")):
    print("{0:<14} {1}".format(produced.name, produced.read_text(encoding="utf-8").strip()))

# %%
# The pass is reproducible in the sense that matters: running it twice gives
# byte-identical output, because nothing in the pipeline depends on process
# state or on :func:`hash`.

before = {p.name: p.read_bytes() for p in sorted(output.glob("*.txt"))}

for source in sorted(corpus.glob("*.txt")):
    run("redact", "--quiet", "--profile", "strict", "--in", str(source),
        "--out", str(output / source.name),
        "--vault", str(output / (source.stem + ".vault.json")))

after = {p.name: p.read_bytes() for p in sorted(output.glob("*.txt"))}

print()
print("byte-identical on a second run:", before == after)
assert before == after

# %%
# One caution for a batch pass, and it is the reason ``--vault`` is explicit
# above: a vault is clear text.  Writing it next to the output is convenient
# and is the wrong default for a repository — add ``*.vault.json`` to
# ``.gitignore``, or encrypt with ``--encrypt`` and keep the passphrase
# elsewhere.

# %%
# Recipe 5 — between an agent and a model API
# -------------------------------------------
# The shape that matters here is that the vault belongs to the **conversation**,
# not to the call.  A :class:`Session` held for the life of the exchange gives a
# value the same label in turn seven that it had in turn one, which is what lets
# the model refer back to it.

from scikitplot.cleanprompt import Session  # noqa: E402


class Conversation:
    """A model client with redaction on both sides of every call.

    Notes
    -----
    **User notes.** Construct one per conversation, use it as a context
    manager, and the values are cleared when the block ends.

    **Developer notes.** ``_transport`` is the only place a redacted prompt
    leaves this object, and the only place a reply comes back. Keeping it to
    one method is what makes the boundary auditable: there is exactly one line
    to check when somebody asks whether a value can escape.
    """

    def __init__(self, transport):
        self._transport = transport
        self._session = Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._session.clear()

    def ask(self, question: str) -> str:
        """Send one turn and return the reply with values restored."""
        sent = self._session.encode(question)
        received = self._transport(sent)
        return self._session.decode(received)

    @property
    def audit(self) -> dict:
        """Return what was withheld, for a log that must not hold values."""
        report = self._session.report()
        return {"turns": report["turns"], "labels": report["labels"]}


TRANSCRIPT = []


def recording_transport(sent: str) -> str:
    """Stand in for a model API, recording exactly what crossed the boundary."""
    TRANSCRIPT.append(sent)
    mentioned = re.findall(r"\[[A-Z_]+-\d+\]", sent)
    return "Understood. I will contact {0}.".format(
        " and ".join(mentioned) if mentioned else "the reporter"
    )


with Conversation(recording_transport) as chat:
    print(chat.ask("Open an incident for ada@example.com"))
    print(chat.ask("Add bob@example.com and ada@example.com to it"))
    print()
    print("audit:", json.dumps(chat.audit, sort_keys=True))

print()
print("what actually crossed the boundary:")
for line in TRANSCRIPT:
    print("   ", line)

for line in TRANSCRIPT:
    assert "ada@example.com" not in line
    assert "bob@example.com" not in line
print()
print("no value reached the transport:", True)

# %%
# Recipe 6 — keeping values out of the logs
# -----------------------------------------
# The loop above protects the model.  It does nothing about the log line your
# framework writes just before the call, which is a common way for a value to
# escape a system that redacts correctly everywhere it was designed to.
#
# :func:`redacting` installs a filter for literal strings you hold — tokens,
# passphrases, a customer identifier you were given.  It is deliberately
# literal rather than pattern-based, so treat it as a last line of defence for
# **named** values, never as a substitute for redacting the prompt.

from scikitplot.cleanprompt import redacting  # noqa: E402

buffer = io.StringIO()
app_log = logging.getLogger("cleanprompt.recipes")
app_log.handlers = [logging.StreamHandler(buffer)]
app_log.setLevel(logging.INFO)
app_log.propagate = False

API_TOKEN = "sk-demo-not-a-real-token"

with redacting(secrets=[API_TOKEN], logger=app_log):
    app_log.info("calling the model with token %s", API_TOKEN)
    app_log.info("prompt was: Open an incident for [EMAIL-1]")

written = buffer.getvalue()
print(written.strip())
print()
print("the named token is absent:", API_TOKEN not in written)
assert API_TOKEN not in written

# %%
# 7. Two more worth knowing about
# -------------------------------
# **The local web interface.** ``cleanprompt flask`` serves a paste-and-go page
# on loopback, and ``cleanprompt docker`` writes a Dockerfile and a compose
# file for it.  No server is started in a documentation build:
#
# .. code-block:: bash
#
#     cleanprompt flask --port 5000            # http://127.0.0.1:5000
#     cleanprompt flask --docker --port 5000   # binds 0.0.0.0 inside a container
#     cleanprompt docker --write ./deploy      # generate the files
#
# ``--docker`` binds ``0.0.0.0`` because a container's port mapping needs it.
# Outside a container that exposes an unauthenticated page to the network, so
# any other non-loopback bind additionally requires ``--allow-remote``.

run("docker", "--port", "5000", limit=10)

# %%
# **The interactive session.** ``cleanprompt cli`` is a paste-and-go terminal
# loop that holds one vault for the whole session: paste a prompt, copy the
# redacted version, paste the reply back, read the restored answer.  It needs a
# terminal, so it is shown rather than run.  The transcript below is real
# output, captured by driving the command with a pipe:
#
# .. code-block:: console
#
#     $ cleanprompt cli
#     CleanPrompt interactive session
#       Paste your text, then press Ctrl-D on a blank line to submit.
#       Commands: :hide TERM   :allow TERM   :suggest   :why   :again   :quit
#       12 structural detectors are active. Names, organisations and places are NOT being detected.
#       ! Names, organisations, places and other named entities
#         fix: spaCy is installed; enable it for this run (--ner ...)
#
#     Paste your text, then Ctrl-D:
#     Mail ada@example.com about the outage
#
#     --- send this ---
#     Mail [EMAIL-1] about the outage
#
#     placeholder  kind   count  detector
#     -----------  -----  -----  -----------
#     [EMAIL-1]    EMAIL  1      regex:EMAIL
#
#     cleanprompt>
#
# Note the banner: the session says what it is **blind to** before you paste
# anything, which is the one moment that warning is worth reading.

run("cli", "--help", limit=6)

# %%
# 8. Cleanup
# ----------

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
#    level: advanced
#    purpose: reference
