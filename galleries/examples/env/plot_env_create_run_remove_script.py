"""
Build, Use and Remove an Environment, Idempotently
==================================================

.. currentmodule:: scikitplot._env

``create`` runs exactly the plan you would have seen with ``plan``, then
writes a *receipt* into the environment. Run it again with the same request
and nothing happens. This example builds a real ``venv`` environment from
the standard library (no network), and shows the whole life cycle:

1. create, with an events log;
2. create again: up to date, nothing to do;
3. run a command inside it;
4. remove it: an approval, given explicitly.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# A temporary home
# ----------------

import json
import os
import shutil
import tempfile
from pathlib import Path

from scikitplot import _env

work = Path(tempfile.mkdtemp(prefix="skplt-env-build-"))
# Read when a function runs, not at import; restored at the end.
previous_home = os.environ.get("SKPLT_ENV_HOME")
os.environ["SKPLT_ENV_HOME"] = str(work / "home")


prefix = work / "demo"
events = work / "events.jsonl"


def show(text):
    """Print ``text`` with this example's temporary folder shortened."""
    print(str(text).replace(str(work), "<tmp>"))


# %%
# Create
# ------
#
# .. code-block:: console
#
#     $ scikitplot env create -p <tmp>/demo --backend venv --verify-import json \
#           --events <tmp>/events.jsonl
#
# ``verify_imports`` is the last step: the environment is checked by
# importing modules with its own interpreter, so "created" means "works".

result = _env.create(
    prefix=str(prefix), backend="venv", verify_imports=["json"], events=str(events)
)
print("ok:", result.ok, "| changed:", result.changed)
print("steps run:", [(step.id, step.ok, step.returncode) for step in result.steps])
show(f"receipt: {result.receipt}")

# %%
# The events file is JSON Lines, one record per stage, for a CI log or an
# agent that follows the run:

for line in events.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    print(record["event"], record.get("step", ""))

# %%
# Create again: nothing to do
# ---------------------------
#
# The receipt records what the environment contains (packages, sources and
# their hashes, Python, backend). The same request finds it up to date and
# runs no step:

again = _env.create(prefix=str(prefix), backend="venv", verify_imports=["json"])
print("changed the second time:", again.changed)
assert not again.changed

# %%
# ``plan`` says the same before anything runs:

print(
    "up to date:",
    _env.plan(prefix=str(prefix), backend="venv", verify_imports=["json"]).up_to_date,
)

# %%
# Run a command inside it
# -----------------------
#
# .. code-block:: console
#
#     $ scikitplot env run -p <tmp>/demo -- python -c "import sys; print(sys.prefix)"
#
# ``run`` uses the environment's interpreter and returns the exit status.
# The command's own output goes to the terminal; here it writes a file, so
# the example can show what it saw.

seen = work / "seen.txt"
status = _env.run(
    [
        "python",
        "-c",
        f"import sys, pathlib; pathlib.Path({str(seen)!r}).write_text(sys.prefix)",
    ],
    prefix=str(prefix),
)
print("exit status:", status)
show(f"sys.prefix inside: {seen.read_text(encoding='utf-8')}")

# %%
# Remove it: an explicit approval
# -------------------------------
#
# Deleting a directory is never implied. Without approval ``remove`` refuses
# before touching anything, and says how to approve:

try:
    _env.remove(prefix=str(prefix), interaction="never")
except _env.ApprovalRequiredError as exc:
    show(f"{type(exc).__name__} [{exc.code}]: {exc}")
    print("  hint:", exc.hint)
print("still there:", prefix.exists())

# %%
# .. code-block:: console
#
#     $ scikitplot env remove -p <tmp>/demo --approve remove

removed = _env.remove(prefix=str(prefix), approve="remove")
print("removed:", removed.changed, "| still there:", prefix.exists())

shutil.rmtree(work)
if previous_home is None:
    os.environ.pop("SKPLT_ENV_HOME", None)
else:
    os.environ["SKPLT_ENV_HOME"] = previous_home

# %%
# .. tags::
#
#    model-workflow: env
#    plot-type: text
#    level: beginner
