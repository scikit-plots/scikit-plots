"""
Plan an Environment Before Building It
======================================

.. currentmodule:: scikitplot._env

The first habit with ``scikitplot env`` is to *look before you build*.
``plan`` reads the request and the dependency files, picks a backend by
fixed rules, and returns every command it would run, which ones use the
network and which need an approval. It changes nothing.

1. plan a ``venv`` environment and read it;
2. let the builder choose the backend, and see why;
3. the plan as JSON, and its digest;
4. a plan that is not run here: the browser (Pyodide) target.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# A home for this example
# -----------------------
#
# Named environments live under ``SKPLT_ENV_HOME``. Pointing it at a
# temporary folder keeps the example from touching ``~/.scikitplot``.

import json
import os
import shutil
import tempfile
from pathlib import Path

from scikitplot import _env

work = Path(tempfile.mkdtemp(prefix="skplt-env-plan-"))
# Read when a function runs, not at import; restored at the end.
previous_home = os.environ.get("SKPLT_ENV_HOME")
os.environ["SKPLT_ENV_HOME"] = str(work / "home")


def show(text):
    """Print ``text`` with this example's temporary folder shortened."""
    print(str(text).replace(str(work), "<tmp>"))


# %%
# Plan a venv environment
# -----------------------
#
# .. code-block:: console
#
#     $ scikitplot env plan -p <tmp>/demo --backend venv --verify-import json
#
# A plan is numbered steps, each with its exact command line. ``[network]``
# and ``needs approval`` would be shown on a step that has them; this one has
# neither: ``venv`` and the standard library need nothing from outside.

plan = _env.plan(prefix=str(work / "demo"), backend="venv", verify_imports=["json"])
show(plan.render())

# %%
# The plan is also an object to check before anything runs, which is what a
# test or an agent does:

print("steps:", [step.id for step in plan.steps])
print("uses network:", plan.uses_network, "| approvals:", plan.approvals)
print("exists:", plan.exists, "| up to date:", plan.up_to_date)
assert not plan.uses_network and plan.approvals == ()

# %%
# Let the builder choose
# ----------------------
#
# With ``backend="auto"`` (the default) the choice follows fixed rules, in
# order (:ref:`env-index`, "How it works"): a cross-build target allows only
# micromamba and mamba; Conda packages take the first of
# micromamba, mamba, conda on the host; a locked ``Pipfile`` takes pipenv;
# otherwise uv when found, else ``venv`` for the running Python. The plan
# names the rule that decided (``backend_reason``), so the same machine and
# request always give the same answer. The output below is this build
# machine's.

auto = _env.plan(name="analysis", packages=["pandas>=2"])
print("backend:", auto.backend, "-", auto.backend_reason)
show(f"prefix:  {auto.prefix}")
print("uses network:", auto.uses_network)

# %%
# The plan as JSON, and its digest
# --------------------------------
#
# .. code-block:: console
#
#     $ scikitplot env plan -n analysis --package "pandas>=2" --format json
#
# ``to_dict()`` is the same document ``--format json`` prints (schema
# ``scikitplot.env.plan/v1``). The digest identifies the plan: the same
# request on the same machine gives the same digest, so a reviewed plan can
# be compared with the one that is about to run.

document = auto.to_dict()
print(document["schema"], "|", sorted(document)[:6], "...")
print(
    "first step:", json.dumps({k: document["steps"][0][k] for k in ("id", "network")})
)
again = _env.plan(name="analysis", packages=["pandas>=2"])
print("same request, same digest:", again.digest == auto.digest)

# %%
# A plan that is not run here
# ---------------------------
#
# .. code-block:: console
#
#     $ scikitplot env plan -n lite --target pyodide --package six
#
# Some targets are not built by running commands on this machine. For the
# browser (Pyodide: JupyterLite, PyScript) the plan carries an *artifact*
# instead, the code to run there:

lite = _env.plan(name="lite", target="pyodide", packages=["six"])
print(
    "target:",
    lite.target,
    "| backend:",
    lite.backend,
    "| executable here:",
    lite.executable,
)
artifacts = dict(lite.artifacts)  # (file name, content) pairs
print(artifacts["micropip.py"])

# %%
# A plan never writes: the example's folder holds no environment.

print("anything built?", any(work.rglob("pyvenv.cfg")))
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
