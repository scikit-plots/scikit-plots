"""
Clean a Prompt in a Notebook Cell with ``%%scikitplot``
=======================================================

.. currentmodule:: scikitplot._ipymagic

The cell form ``%%scikitplot COMMAND`` gives the cell's text to the command
on standard input. With ``cleanprompt`` that turns a notebook cell into a
guarded text box: what you paste is redacted before it goes anywhere, and
the vault that can restore it is written where you say.

The steps:

1. redact a cell, keep the vault in a folder of your choice;
2. restore a model's reply with the same vault;
3. read what ran (exit code, output, where it ran);
4. the edge cases: an empty cell, a command that needs no input.

``cleanprompt`` is a *delegated* command: it runs in a subprocess of the
kernel's own Python, streamed, and stopped by the interrupt button or by the
``timeout`` setting. Every value below is reserved for documentation.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# A vault for this example
# ------------------------
#
# ``cleanprompt`` writes the removed values to a vault. By default that is a
# file in your state directory; here a temporary folder keeps the example
# self-contained and leaves nothing behind.

import io
import shutil
import tempfile
from pathlib import Path

from scikitplot._ipymagic import MagicSettings, route, run_command

# In a notebook (this page's download), load the magics so the ``%%`` cells
# below run; a plain script has no IPython and skips this.
_shell = globals().get("get_ipython", lambda: None)()
if _shell is not None:
    _shell.run_line_magic("load_ext", "scikitplot")

work = Path(tempfile.mkdtemp(prefix="skplt-ipymagic-"))
vault = work / "vault.json"
print("cleanprompt runs in a subprocess:", route(["cleanprompt"]).delegated)

# %%
# Redact a cell
# -------------
#
# .. code-block:: ipython
#
#     %%scikitplot cleanprompt redact --vault {vault}
#     Mail me at jane.doe@example.com or call +1 202 555 0143.
#
# ``{vault}`` is a Python variable expanded by IPython. The redacted text is
# printed; the report and the vault path go to standard error.

cell = "Mail me at jane.doe@example.com or call +1 202 555 0143.\n"
redacted = run_command(["cleanprompt", "redact", "--vault", str(vault)], stdin=cell)
print(redacted.summary())

# %%
# The result keeps both streams, so a notebook can test the outcome instead
# of reading it:

assert redacted.ok and "jane.doe@example.com" not in redacted.stdout
assert vault.is_file()
print("safe to send:", redacted.stdout.strip())

# %%
# Restore a reply
# ---------------
#
# Send the redacted text to a model; its reply keeps the placeholders. The
# same vault puts the values back:
#
# .. code-block:: ipython
#
#     %%scikitplot cleanprompt restore --vault {vault}
#     I will write to [EMAIL-1] and call [PHONE-1] tomorrow.

reply = "I will write to [EMAIL-1] and call [PHONE-1] tomorrow.\n"
restored = run_command(["cleanprompt", "restore", "--vault", str(vault)], stdin=reply)
expected = "I will write to jane.doe@example.com and call +1 202 555 0143 tomorrow."
assert restored.ok and restored.stdout.strip() == expected

# %%
# A list of words, not a string, is the safe way to pass a path that may hold
# spaces or quotes; a string is split like a shell would (``"a b"`` is one
# word).
#
# Edge cases
# ----------
#
# An empty or whitespace-only cell gives the command no input at all, so
# ``%%scikitplot cleanprompt --help`` behaves like the line magic:

blank_cell = "   \n"
stdin = blank_cell if blank_cell.strip() else None  # what the magic does
help_result = run_command(["cleanprompt", "--help"], stdin=stdin, out=io.StringIO())
print(help_result.summary())
print(help_result.stdout.splitlines()[0])

# %%
# A command that fails shows its own message and returns its exit status;
# ``restore --strict`` refuses a placeholder the vault does not hold:

err = io.StringIO()
strict = run_command(
    ["cleanprompt", "restore", "--strict", "--vault", str(vault)],
    stdin="Who is [PERSON-9]?\n",
    err=err,
)
print(strict.summary())
print(err.getvalue().strip().splitlines()[-1])

# %%
# For a long-running command, ``timeout`` stops it (exit status 124, as GNU
# ``timeout``) together with every process it started:
#
# .. code-block:: ipython
#
#     %scikitplot_config timeout=120

guarded = MagicSettings(timeout=120)
print(guarded)

# %%
# Clean up the example's folder (the vault holds the removed values):

shutil.rmtree(work)

# %%
# .. tags::
#
#    model-workflow: ipymagic
#    model-workflow: cleanprompt
#    plot-type: text
#    level: beginner
