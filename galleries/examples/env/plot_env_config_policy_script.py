"""
One Configuration File for Laptops, CI and Agents
=================================================

.. currentmodule:: scikitplot._env

A committed ``scikitplot-env.toml`` describes the environment once; every
machine runs the same command. This example covers what an automation (a CI
job, a container build, an AI agent) relies on:

1. a configuration file with a dependency file and a preset;
2. where each value came from (``origins``);
3. offline: a plan that cannot reach the network;
4. what a file can *not* do: grant an approval;
5. Conda packages, a bootstrap and its approval;
6. a batch of requests, dry-run;
7. stable error codes.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# A project with a configuration file
# -----------------------------------

import json
import os
import shutil
import tempfile
from pathlib import Path

from scikitplot import _env

work = Path(tempfile.mkdtemp(prefix="skplt-env-config-"))
# Read when a function runs, not at import; restored at the end.
previous_home = os.environ.get("SKPLT_ENV_HOME")
os.environ["SKPLT_ENV_HOME"] = str(work / "home")


def show(text):
    """Print ``text`` with this example's temporary folder shortened."""
    print(str(text).replace(str(work), "<tmp>"))


(work / "requirements.txt").write_text("six>=1.16\n", encoding="utf-8")
config = work / "scikitplot-env.toml"
config.write_text(
    """\
schema = "scikitplot.env/v1"
prefix = ".venv"                 # relative to this file
backend = "venv"
sources = ["requirements.txt"]
profile = "ci"                   # preset: never ask a question

[policy]
network = "offline"              # pip gets --no-index

[verify]
imports = ["six"]
""",
    encoding="utf-8",
)

# %%
# Validate: the request, and where every value came from
# ------------------------------------------------------
#
# .. code-block:: console
#
#     $ scikitplot env validate --format json
#
# Unknown keys are refused, so a typo is an error, not an ignored setting.
# ``origins`` says, per field, whether it came from the file, a preset, an
# argument or a default: the answer to "why is this set?".

request = _env.validate(config)
print("policy:", request.to_dict()["policy"])
for field, origin in request.origins:
    show(f"  {field:<20} {origin}")

# %%
# Offline
# -------
#
# With ``network = "offline"`` every step is planned without the network:
# pip gets ``--no-index`` (give ``find_links`` for a wheel folder), uv and
# micromamba ``--offline``. A step that would need to download is refused at
# planning time, not half-way through a build.

plan = _env.plan(config=config)
show(plan.render())
print("uses network:", plan.uses_network)
assert not plan.uses_network

# %%
# A file cannot grant an approval
# -------------------------------
#
# A configuration expresses intent. Approvals (bootstrap a tool, edit a
# shell profile, delete an environment) come only from the person or program
# running the command: ``--yes``, ``--approve KIND``, ``approve=`` in Python.

sneaky = work / "sneaky.toml"
sneaky.write_text('name = "x"\n[policy]\napprove = ["bootstrap"]\n', encoding="utf-8")
try:
    _env.load_request(sneaky)
except _env.ConfigError as exc:
    print(f"{exc.code}: {exc}")
    print("  hint:", exc.hint)

# %%
# Conda packages, a bootstrap, an approval
# ----------------------------------------
#
# .. code-block:: console
#
#     $ scikitplot env plan -n geo --conda-package gdal --bootstrap if-missing
#
# Conda packages need micromamba, mamba or conda. When none is installed the
# builder can install micromamba itself (pinned, SHA-256 checked), only if
# the policy allows it *and* it is approved. The plan shows that step and its
# approval before anything runs. What the plan says depends on this machine.

geo = _env.plan(
    prefix=str(work / "geo"),
    conda_packages=["gdal"],
    policy={"bootstrap": "if-missing"},
)
print("backend:", geo.backend, "-", geo.backend_reason)
print("approvals needed:", geo.approvals)

# %%
# Without the approval, ``create`` stops before its first step: nothing is
# downloaded. (On a machine that already has micromamba there is nothing to
# approve, and this example does not build the environment.)

if geo.approvals:
    try:
        _env.create(
            prefix=str(work / "geo"),
            conda_packages=["gdal"],
            policy={"bootstrap": "if-missing"},
            interaction="never",
        )
    except _env.ApprovalRequiredError as exc:
        print(f"{exc.code}: refused before running anything")
        print("  hint:", exc.hint)
else:
    print("a Conda tool is installed here: no bootstrap, no approval needed")

# %%
# A batch of requests, dry-run
# ----------------------------
#
# .. code-block:: console
#
#     $ scikitplot env --format json batch requests.jsonl --dry-run --keep-going
#
# One JSON Lines file, one request per line, each with an ``id``. The whole
# file is validated first: one malformed line refuses the file, with its
# line number, before any request runs.

requests = work / "requests.jsonl"
schema = "scikitplot.env.request/v1"
requests.write_text(
    json.dumps({"schema": schema, "id": "typo", "request": {"name": "bad name!"}})
    + "\n",
    encoding="utf-8",
)
try:
    _env.batch(requests, dry_run=True)
except _env.ConfigError as exc:
    show(f"{exc.code}: {exc}")

# %%
# A valid file runs every request; a dry run returns the plans, and
# ``keep_going`` reports a request that cannot be planned (Conda packages for
# the PyPI-only ``venv``) instead of stopping there.

lines = [
    {
        "schema": schema,
        "id": "docs",
        "request": {"prefix": str(work / "docs"), "backend": "venv"},
    },
    {
        "schema": schema,
        "id": "lite",
        "request": {"name": "lite", "target": "pyodide", "packages": ["six"]},
    },
    {
        "schema": schema,
        "id": "mixed",
        "request": {
            "prefix": str(work / "mixed"),
            "backend": "venv",
            "conda_packages": ["gdal"],
        },
    },
]
requests.write_text(
    "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
)
for item in _env.batch(requests, dry_run=True, keep_going=True):
    if item["ok"]:
        print(f"{item['id']:<6} ok     backend={item['plan']['backend']}")
    else:
        print(f"{item['id']:<6} error  {item['error']['code']}")

# %%
# Stable error codes
# ------------------
#
# Every error has a ``code`` and a ``hint``; with ``--format json`` the
# command prints ``{"ok": false, "error": {...}}`` and exits with a fixed
# status, so a script branches on the code, never on the message.

try:
    _env.validate(name="bad name!")
except _env.EnvError as exc:
    print(json.dumps({"code": exc.code, "error": type(exc).__name__}))

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
#    level: intermediate
