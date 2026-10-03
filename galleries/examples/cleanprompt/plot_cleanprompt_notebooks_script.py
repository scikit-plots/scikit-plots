"""
Cleanprompt for Notebooks: Sending an Analysis Without Sending the Data
=======================================================================

.. currentmodule:: scikitplot.cleanprompt

A notebook is not prose, and what leaks out of one is not what leaks out of a
paragraph. Three classes run through every stage of a project — exploration,
analytics, modelling, model analysis — and none is reachable by matching
patterns in text:

**Identifiers, not values.** ``customer_ssn`` contains no social-security
number. It discloses that the dataset does, which is often the more sensitive
fact.

**Structure, not content.** ``/home/marion.holt/work/acme-churn/`` names a
person and a client project in nine tokens, none of which any pattern matches.

**Rendered data.** The code can be spotless while the *output* beneath it holds
two hundred real rows. This is the leak people are least aware of, because they
are reading the code.

The hard part is not hiding these. It is hiding them and still getting useful
help, because a model's advice depends on the schema's semantics:

.. code-block:: text

    df['acct_balance_usd']  →  numeric and skewed, wants a log
    df['region_code']       →  categorical, wants encoding
    df['signup_date']       →  temporal, a leakage risk
    df['customer_ssn']      →  an identifier, not a feature at all

Replace those four with ``[COLUMN-1..4]`` and the information the advice depends
on is exactly the information that was removed. So the stand-in keeps the
**role** and discards the name — ``id_1``, ``amount_1``, ``date_1``,
``category_1`` — and "drop ``id_1``, log-transform ``amount_1``, one-hot
``category_1``" decodes back into your own column names.

Every value in this example is reserved and cannot reach anybody.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-cleanprompt-nb-")
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
# 1. A notebook with the leaks that actually occur
# ------------------------------------------------
# Built here rather than shipped, so you can see exactly what goes in: a
# markdown cell naming the client, a load from a production path, a rendered
# ``head()`` with real rows, a ``value_counts()`` with real segment names, a
# traceback carrying a home directory, and a plot title.

NOTEBOOK = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Q1 churn model — Acme Financial\n",
                "\n",
                "Pull from `/mnt/prod/exports/2026_q1_acme_customers.parquet`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 1,
                    "metadata": {},
                    "data": {
                        "text/plain": [
                            "   customer_ssn  acct_balance_usd region_code\n",
                            "0  123-45-6789           14320.55    AC-NORTH\n",
                            "1  987-65-4321            2210.00    AC-SOUTH\n",
                        ]
                    },
                }
            ],
            "source": [
                "import pandas as pd\n",
                "df = pd.read_parquet("
                "'/mnt/prod/exports/2026_q1_acme_customers.parquet')\n",
                "df = df[['customer_ssn', 'acct_balance_usd', 'signup_date',\n",
                "         'region_code', 'internal_risk_score_v3', 'churned']]\n",
                "df.head(2)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["AC-NORTH    41822\nAC-SOUTH    18204\n"],
                }
            ],
            "source": ["print(df['region_code'].value_counts())\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "error",
                    "ename": "KeyError",
                    "evalue": "'acct_balance'",
                    "traceback": [
                        "File /home/marion.holt/work/acme-churn/features.py:18",
                        "KeyError: 'acct_balance'",
                    ],
                }
            ],
            "source": [
                "X = df.drop(columns=['churned', 'customer_ssn'])\n",
                "X['balance_log'] = np.log1p(X.acct_balance)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {"image/png": "iVBORw0KGgoAAAANSUhEUgAAA" * 12},
                }
            ],
            "source": [
                "plt.barh(X.columns, model.feature_importances_)\n",
                "plt.title('Acme drivers — internal_risk_score_v3 dominates')\n",
            ],
        },
    ],
    "metadata": {"kernelspec": {"name": "acme-churn", "display_name": "acme-churn"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

source = _HOME / "churn.ipynb"
source.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")

print("notebook written:", source.name, "-", source.stat().st_size, "bytes")

# %%
# 2. What the prose pipeline alone would find
# -------------------------------------------
# Worth seeing before the fix, because the failure is not that it finds
# nothing — it finds three things and reports success.

run("inspect", "--as", "text", "--format", "json", "--in", str(source), limit=3)

# %%
# The two real values in the rendered table are caught, and an address. The
# column names, the production path, the analyst's home directory, the segment
# names and the figure all go to the model, and nothing says so.

# %%
# 3. Reading it as a notebook instead
# -----------------------------------
# ``--as`` is inferred from the suffix, so in practice you type neither of
# these. ``--infer-roles`` is the one worth typing: without it a column becomes
# ``field_1``, which is safe and tells the model nothing.

clean = _HOME / "churn.clean.ipynb"

run(
    "encode",
    "--vault-mode",
    "overwrite",
    "--infer-roles",
    "--in",
    str(source),
    "--out",
    str(clean),
)

# %%
# Read that report rather than skimming it. Three of its lines are the ones
# that matter:
#
# * ``read as notebook: code=…, output=…, traceback=…`` — the document was
#   understood as a structure, not as a wall of text.
# * ``N column name(s) hidden; M with an established role`` — when ``M`` is
#   below ``N``, the rest became ``field_n`` and the model lost that context.
# * anything beginning ``NOT hidden`` or ``NOT read`` — the two places where a
#   column can exist and not be protected.

# %%
# 4. What the model receives
# --------------------------

redacted = json.loads(clean.read_text(encoding="utf-8"))

for index, cell in enumerate(redacted["cells"], start=1):
    body = "".join(cell["source"]).rstrip()
    print("--- cell {0} ({1}) ---".format(index, cell["cell_type"]))
    print(body)
    for output in cell.get("outputs", []):
        print("    [{0}] {1}".format(output["output_type"], json.dumps(output)[:96]))

# %%
# Four things in that output are worth naming.
#
# The **code still reads as code**. ``df[['id_1', 'amount_1', 'date_1',
# 'category_1', 'score_1', 'flag_1']]`` is something a model can work with.
#
# ``X.acct_balance`` became ``X.amount_…`` even though attribute access is
# deliberately *not* used to discover a column. Discovery is conservative and
# works only from unambiguous positions; rewriting, once the name is known,
# covers every occurrence. Collapsing the two would either miss columns or
# corrupt code.
#
# The **rendered outputs are gone**, not filtered. An output is not a
# description of the data, it *is* the data, and parsing inside a pandas repr
# means a silent leak every time the repr changes. The figure is gone for the
# same reason: a chart of real data shows real axis labels.
#
# The **structure survived**. ``output_type``, ``execution_count`` and the cell
# ids are byte-identical, so the file still opens.

# %%
# 5. The payoff: a reply, decoded
# -------------------------------
# Paste the clean notebook into any chat. The answer comes back in stand-ins,
# and ``decode`` turns it into your own schema — including code the model wrote
# that did not exist when you encoded.

REPLY = (
    "Three things. Drop id_1 — it is an identifier, not a feature, and "
    "leaving it in will leak. Log-transform amount_1, which is skewed. "
    "One-hot category_1, and check date_1 for target leakage:\n\n"
    "    X = df.drop(columns=['id_1'])\n"
    "    X['amount_1'] = np.log1p(X['amount_1'])\n"
    "    X = pd.get_dummies(X, columns=['category_1'])\n"
)

run("decode", REPLY)

# %%
# That is the property the whole design is for. The advice is specific and
# correct, the code is pasteable, and the model never saw a column name, a
# path, a row or a figure.

# %%
# 6. Everything is restorable, exactly
# ------------------------------------
# The notebook itself round-trips byte for byte, which is what makes it safe to
# work on the redacted copy and put the values back at the end.

restored = _HOME / "churn.restored.ipynb"
run("decode", "--quiet", "--in", str(clean), "--out", str(restored))

print()
print(
    "notebook round trip exact:",
    restored.read_text(encoding="utf-8") == source.read_text(encoding="utf-8"),
)
assert restored.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

# %%
# 7. What is still not hidden, and why
# ------------------------------------
# One thing survives on purpose, and the report says so every time.

text = clean.read_text(encoding="utf-8")
for probe in (
    "customer_ssn",
    "123-45-6789",
    "AC-NORTH",
    "/mnt/prod",
    "marion.holt",
    "internal_risk_score_v3",
    "Acme",
):
    print("  {0:<24} survives: {1}".format(probe, probe in text))

# %%
# ``Acme`` is a named entity, and no regular expression recognises one. That is
# the blind spot ``doctor`` reports and the entity tier exists for. Two
# remedies, and the second needs nothing installed:
#
# .. code-block:: bash
#
#     cleanprompt encode --ner --in analysis.ipynb --out analysis.clean.ipynb
#     cleanprompt encode --hide "Acme Financial" --hide Acme --in analysis.ipynb
#
# The point is not that the tool catches everything. It is that the tool says
# what it did not catch, every time, instead of reporting three values removed
# and calling the file clean.

run("encode", "--quiet", "--vault-mode", "overwrite", "--infer-roles",
    "--hide", "Acme", "--in", str(source), "--out", str(clean))

print("with --hide Acme, 'Acme' survives:", "Acme" in clean.read_text(encoding="utf-8"))

# %%
# 8. Declaring what inference cannot know
# ---------------------------------------
# ``--infer-roles`` reads names, and a name is not evidence of a type. Where it
# matters, say so: a declared role always wins and is reported as ``declared``
# rather than ``inferred``.
#
# ``--add-column`` answers the other half — a column the conservative discovery
# rule could not establish, which the report names for you.

run(
    "encode",
    "--quiet",
    "--vault-mode",
    "overwrite",
    "--columns",
    "churned:target,acct_balance_usd:amount,internal_risk_score_v3:score",
    "--in",
    str(source),
    "--out",
    str(clean),
)

print("target column reads as:", "target" in clean.read_text(encoding="utf-8"))

# %%
# 9. A module, and any other stage
# --------------------------------
# The same pipeline reads a ``.py`` file, so a feature builder, a training
# script or a scoring job goes through unchanged.

module = _HOME / "features.py"
module.write_text(
    '"""Feature builder. Reads /mnt/prod/exports/q1_acme.parquet."""\n'
    "import pandas as pd\n"
    "\n"
    'ID_COLUMNS = ["customer_ssn", "acct_external_ref"]\n'
    'TARGET = "churned"\n'
    "\n"
    "def build(path):\n"
    "    df = pd.read_parquet(path)\n"
    '    df["balance_log"] = np.log1p(df["acct_balance_usd"])\n'
    "    return df.drop(columns=ID_COLUMNS), df[TARGET]\n",
    encoding="utf-8",
)

run("encode", "--vault-mode", "overwrite", "--infer-roles", "--in", str(module))

# %%
# Note what happened to ``ID_COLUMNS`` and ``TARGET``. Neither is a column
# name, and neither was guessed at: they are names bound to string literals,
# and the discovery walk resolves a name used as a selector against the literal
# it was bound to — possibly several cells earlier. Without that, a module
# written the way careful people write them would leak every column it
# declared at the top.

# %%
# 10. Cleanup
# -----------

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
#    level: intermediate
#    purpose: showcase
