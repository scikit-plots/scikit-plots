"""
Run scikit-plots Commands from a Notebook
=========================================

.. currentmodule:: scikitplot._ipymagic

``%scikitplot`` runs any ``scikitplot`` command inside a notebook and returns
a :class:`CommandResult` you can keep, test and pass on. This example walks
through the everyday cases and the ones that go wrong, in the order a new
user meets them:

1. load the extension;
2. run a command and keep its result;
3. ask for JSON and use it as data;
4. what happens with a typo, a bad option and a failure;
5. session settings (``%scikitplot_config``);
6. tab completion.

Every magic is shown in an ``ipython`` block, then run with the function it
calls, so this page also runs as a plain script.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Load the extension
# ------------------
#
# In a notebook, once per session (``%reload_ext`` after an upgrade):
#
# .. code-block:: ipython
#
#     %load_ext scikitplot
#
# The functions behind the magics need no IPython at all:

import io
import json

from scikitplot._ipymagic import (
    CommandFailedError,
    MagicSettings,
    MagicUsageError,
    complete,
    route,
    run_command,
)

# In a notebook (this page's download), load the magics so the ``%%`` cells
# below run; a plain script has no IPython and skips this.
_shell = globals().get("get_ipython", lambda: None)()
if _shell is not None:
    _shell.run_line_magic("load_ext", "scikitplot")

# %%
# Run a command and keep the result
# ---------------------------------
#
# .. code-block:: ipython
#
#     %scikitplot info
#
# The command's output is shown as in a terminal; the magic returns a
# :class:`CommandResult`. ``info``, ``doctor``, ``show-versions`` run in the
# kernel itself (fast), and the kernel's logging is left as it was.

result = run_command("info")
print(result.summary())
print("exit code:", result.exit_code, "| ok:", result.ok, "| ran:", result.mode)

# %%
# Ask for JSON, get data
# ----------------------
#
# .. code-block:: ipython
#
#     r = %scikitplot info --format json
#     r.data["python"]["version"]
#
# With ``--format json`` the output is parsed into ``result.data`` and not
# printed again, so a notebook can compute with it.

quiet = io.StringIO()  # not shown here; the data is what we want
result = run_command("info --format json", out=quiet)
print(sorted(result.data))
print("Python", result.data["python"]["version"])

# %%
# ``to_dict()`` gives the whole run as plain, JSON-serialisable data, which is
# what an agent or a CI log keeps:

record = result.to_dict()
print(json.dumps({k: record[k] for k in ("argv", "command", "mode", "exit_code")}))

# %%
# A typo is refused, never guessed
# --------------------------------
#
# .. code-block:: ipython
#
#     %scikitplot evn plan
#
# A word that is not a command is refused before anything runs, with the
# closest names. Running "the nearest command" could run the wrong one.

try:
    run_command("evn plan")
except MagicUsageError as exc:
    print(f"{type(exc).__name__} [{exc.code}]: {exc}")
    print("  hint:", exc.hint)

# %%
# :func:`route` shows where a command would run, without running it. The
# decision comes from the CLI's own registry, so it cannot drift from it:
# delegated commands (``env``, ``mcp``, ``cleanprompt``) run in a subprocess.

for words in (["info"], ["-v", "env", "plan"], ["clean-prompt"], ["--version"]):
    print(f"{' '.join(words):<18} -> {route(words)}")

# %%
# A failing command returns, unless you ask it to raise
# -----------------------------------------------------
#
# An unknown option is the CLI's own usage error (exit status 2). By default
# the magic shows the message and returns the result, so one cell does not
# stop a notebook:

err = io.StringIO()
result = run_command("info --no-such-option", err=err)
print(result.summary())
print(err.getvalue().strip().splitlines()[-1])

# %%
# Session settings
# ----------------
#
# .. code-block:: ipython
#
#     %scikitplot_config raise_on_error=true timeout=600
#
# Settings live in ``%scikitplot_config``, never as magic options, so they
# cannot collide with a CLI option. All pairs are applied or none: a bad
# value changes nothing.

settings = MagicSettings().updated("raise_on_error=true timeout=600")
print(settings)
try:
    MagicSettings().updated("timeout=-1 raise_on_error=true")
except MagicUsageError as exc:
    print("refused:", exc)

# %%
# With ``raise_on_error=true`` a failure raises :class:`CommandFailedError`,
# which stops "Run all"; the result travels with the exception.

try:
    run_command("info --no-such-option", settings=settings, err=io.StringIO())
except CommandFailedError as exc:
    print("raised:", exc)
    print("exit code kept on the exception:", exc.result.exit_code)

# %%
# Tab completion
# --------------
#
# ``%scikitplot <TAB>`` completes command names, then a native command's
# options and fixed values. :func:`complete` is what the notebook calls:

print(complete("%scikitplot sh"))
print(complete("%scikitplot info --"))
print(complete("%scikitplot info --format "))

# %%
# .. tip::
#
#    ``{var}`` and ``$var`` in a magic line are expanded by IPython before the
#    command sees them: ``name = "analysis"`` then
#    ``%scikitplot env plan -n {name}``. Quote values with spaces:
#    ``-n "my env"`` stays one word.
#
# .. tags::
#
#    model-workflow: ipymagic
#    plot-type: text
#    level: beginner
