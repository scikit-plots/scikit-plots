"""
Gather the Evidence on a Lean Proof with LeanIndex
==================================================

.. currentmodule:: scikitplot.lean

A reviewer of a formal proof asks three questions first: what does the
theorem claim, what does its proof rest on, and does any of that rest on
``sorry``, an axiom, ``native_decide`` or code that adds declarations when
Lean builds the project? :class:`LeanIndex` answers them from the source
text, every item with its file and lines. This example goes from a project
of three files to a generated one of a few hundred:

1. a project summary;
2. a dossier: the statement, what it unfolds to, the path to each flag;
3. what a proof can see: imports decide, not names alone;
4. what a proof could use without naming it;
5. code that adds declarations no source line spells;
6. an incomplete index hides nothing;
7. discovery: rules, patterns, claims and anomalies;
8. helper documents for people and models, exported twice;
9. a pipeline check with ``--fail-on``;
10. a generated project, timed.

Everything runs on files the example writes: no Lean toolchain, no download.

.. note::

   The index is lexical: it reads the text, it does not check proofs. A
   dossier without red flags is not a proof; that is for Lean's kernel
   (``lake build``, ``#print axioms``, an independent replay), which every
   report names.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import textwrap
import time
from pathlib import Path

from scikitplot.lean import AUDIT_FLAGS, LeanIndex
from scikitplot.lean._index import main

work = Path(tempfile.mkdtemp(prefix="lean_index_"))


def project(name, files):
    """Write *files* (path to Lean source) under a new folder and return it."""
    root = work / name
    for path, source in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


def where(item):
    return f"{item['file']}:{item['line_start']}"


# %%
# 1. A project summary
# --------------------
# Three files: definitions that state a claim, lemmas (one of them left
# ``sorry``), and the main results. :meth:`LeanIndex.summary` covers the
# whole project; every flag comes with where it is.

demo = project(
    "demo",
    {
        "Demo/Defs.lean": """
            /-- What the project claims. -/
            def Claim : Prop := ∀ n : Nat, Good n
            def Good (n : Nat) : Prop := n = n
        """,
        "Demo/Lemmas.lean": """
            import Demo.Defs
            namespace Demo
            theorem good_all (n : Nat) : Good n := rfl
            theorem shaky : 1 = 2 := sorry
            end Demo
        """,
        "Demo/Main.lean": """
            import Demo.Lemmas
            open Demo
            theorem main : Claim := fun n => good_all n
            theorem main_shaky (k : Nat) : Claim ∧ k = k :=
              ⟨fun n => by have := shaky; exact good_all n, rfl⟩
        """,
    },
)
index = LeanIndex.from_folder(demo)
summary = index.summary()
print(summary["files"], "files,", summary["declarations"], "declarations")
print("flags:", {flag: [where(i) for i in listing["items"]] for flag, listing in summary["flags"].items()})
print("missing project modules:", summary["imports"]["missing_modules"]["count"])

# %%
# 2. A dossier
# ------------
# :meth:`LeanIndex.dossier` quotes the statement and the project
# definitions it unfolds to (what the claim *means*), follows the proof
# through every declaration it names, and gives the path to each flag. The
# verdict never claims more than the text shows.

clean = index.dossier("main")
print(clean["verdict"])
print("statement:", clean["target"]["statement"])
print("unfolds to:", [d["declaration"] for d in clean["statement"]["local_definitions"]["items"]])

flagged = index.dossier("main_shaky")
print(flagged["verdict"], flagged["flags_found"])
for item in flagged["dependencies"]["red_flags"]["items"]:
    print(" ", item["declaration"], "at", where(item), "via",
          " -> ".join(step["from"] for step in item["path"]) or "a direct use")
print("\n".join(f"  {c['check']}: {c['status']}" for c in flagged["checks"]))

# %%
# :meth:`LeanIndex.render` turns any report into Markdown, every item
# keeping its ``file:line``.

print(LeanIndex.render(clean)[:600])

# %%
# 3. What a proof can see
# -----------------------
# A name links only into the proof's own file and the files it imports, as
# Lean sees them. Here ``helper`` is declared twice: once in an imported
# file (clean), once in a scratch file nothing imports (``sorry``). Only
# the first is a dependency; ``visibility="all"`` links by text alone.

scoped = project(
    "scoped",
    {
        "S/Core.lean": "theorem helper : True := trivial\n",
        "S/Main.lean": "import S.Core\ntheorem main : True := helper\n",
        "Scratch/Try.lean": "theorem helper' : True := sorry\ntheorem helper : True := sorry\n",
    },
)
for visibility in ("imports", "all"):
    report = LeanIndex.from_folder(scoped, visibility=visibility).dossier("main")
    print(f"{visibility:>7}: flags {report['flags_found']}")

# %%
# 4. What a proof could use without naming it
# -------------------------------------------
# Lean finds instances, ``@[simp]`` lemmas and notation without their names
# in the proof. A dossier lists the flagged ones in the proof's environment
# (files it imports; its own file before it) as risks to review.

implicit = project(
    "implicit",
    {
        "I/Inst.lean": "instance seen : Inhabited Nat := sorry\n",
        "I/Far.lean": "instance unseen : Inhabited Nat := sorry\n",
        "I/Main.lean": "import I.Inst\ntheorem main : True := trivial\n",
    },
)
report = LeanIndex.from_folder(implicit).dossier("main")
print(report["verdict"])
print([(r["declaration"], where(r)) for r in report["implicit_risks"]["items"]])

# %%
# 5. Code that adds declarations
# ------------------------------
# A command the project defines (here ``copy_all``) can run code that adds
# declarations when Lean builds the file: their source is nowhere. Commands
# are learned from the project, and each invocation that reaches
# ``addDecl`` is reported with the path to it.

generated = project(
    "generated",
    {
        "G/Meta.lean": """
            open Lean Elab Command
            def mkCopies : CommandElabM Unit := do
              liftCoreM (addDecl default)
            elab "copy_all" : command => mkCopies
        """,
        "G/Use.lean": "import G.Meta\ncopy_all\ntheorem plain : 1 = 1 := rfl\n",
    },
)
for item in LeanIndex.from_folder(generated).summary()["generators"]["items"]:
    adder = item["adds_declarations_at"]
    print(item["invocation"]["kind"], "at", where(item["invocation"]),
          "runs", adder["declaration"], "at", where(adder))

# %%
# 6. An incomplete index hides nothing
# ------------------------------------
# ``main`` names ``stale``, a ``sorry`` in a file it does not import: from
# the project root, that is no dependency. Indexed from the subfolder
# ``S/``, ``import S.Core`` matches no file, so what ``main`` sees is
# unknown: it sees everything rather than nothing, and the check on
# imported modules fails and says why. Index a project from its root.

partial = project(
    "partial",
    {
        "S/Core.lean": "theorem helper : True := trivial\n",
        "S/Old.lean": "theorem stale : True := sorry\n",
        "S/Main.lean": "import S.Core\ntheorem main : True := stale\n",
    },
)
for root in (partial, partial / "S"):
    report = LeanIndex.from_folder(root).dossier("main")
    status = {c["check"]: c["status"] for c in report["checks"]}["imported_modules_present"]
    print(f"from {root.name}/: flags {report['flags_found']}, imported modules: {status}")

# %%
# 7. Discovery
# ------------
# :meth:`LeanIndex.discover` lists candidates, each by a stated rule:
# results named from many files (rules), theorems of one statement shape
# (patterns), what is stated without proof (claims), and questions the
# structure raises (anomalies). A candidate is not a conclusion.

found = index.discover()
print("rules:", [i["declaration"] for i in found["rules"]["load_bearing_theorems"]["items"]])
print("concepts:", [i["declaration"] for i in found["rules"]["central_concepts"]["items"]])
print("claims:", [i["declaration"] for i in found["claims"]["claim_definitions"]["items"]])
print("unproven:", [i["declaration"] for i in found["claims"]["unproven"]["items"]])
print("headline results:", [i["declaration"] for i in found["headline_results"]["items"]])

# %%
# 8. Helper documents, exported twice
# -----------------------------------
# :meth:`LeanIndex.export` writes the summary, discovery, a dossier per
# headline result (and per target) as Markdown and JSON, and one JSON
# record per item for a corpus or an MCP server. Exporting again with
# fewer targets removes what it no longer writes, and nothing else.

out = work / "helpers"
first = index.export(out, targets=["Demo.good_all"])  # not a headline result
second = index.export(out)
print(len(first), "then", len(second), "files;", sorted(set(first) - set(second)))
record = json.loads((out / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
print({k: record[k] for k in ("section", "title", "file", "line_start")})

# %%
# 9. A pipeline check
# -------------------
# ``python -m scikitplot.lean FOLDER --target NAME --fail-on sorry`` (or
# ``scikitplot lean ...``) exits 1 when the flag is found. A misspelled flag
# is refused, so a typo cannot pass a pipeline.

print(sorted(AUDIT_FLAGS))
for target in ("main", "main_shaky"):
    with contextlib.redirect_stdout(io.StringIO()):
        status = main([str(demo), "--target", target, "--fail-on", "sorry"])
    print(f"{target}: exit {status}")

# %%
# 10. A generated project
# -----------------------
# A chain of 300 modules of 20 theorems each, one ``sorry`` near the start.
# Indexing is linear in the text; a dossier at the far end follows the
# proof back through every module to the flag.

files = {}
for m in range(300):
    head = f"import Big.M{m - 1}\n" if m else ""
    body = [f"theorem t{m}_{i} : {i} = {i} := rfl" for i in range(20)]
    if m:
        body.append(f"theorem link{m} : True := link{m - 1}")
    else:
        body.append("theorem link0 : True := sorry")
    files[f"Big/M{m}.lean"] = head + "\n".join(body) + "\n"
big = project("big", files)

start = time.perf_counter()
big_index = LeanIndex.from_folder(big)
indexed = time.perf_counter() - start
start = time.perf_counter()
far = big_index.dossier("link299")
asked = time.perf_counter() - start
print(f"{len(files)} files indexed in {indexed:.2f} s; dossier in {asked:.3f} s")
print(far["verdict"], "-", far["dependencies"]["declarations"], "declarations reached")

# %%
# .. tags::
#
#    model-workflow: lean
#    plot-type: text
#    level: intermediate
#    purpose: showcase
