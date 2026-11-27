"""
Index a Lean Project over its Saved Dependencies with LeanIndex
===============================================================

.. currentmodule:: scikitplot.lean

A Lean project sits on dependencies far larger than itself: a few hundred
files over mathlib's ten thousand. Reading the dependency once, saving its
index, and adding each project over it as a *layer* keeps the cost of a
review to the project's own files, with the same reports as one index of
everything. This example:

1. indexes a dependency once and saves it to one file;
2. loads it and adds a project over it;
3. checks the layered index against one index of both folders;
4. reports on the project's layer only, while every use still counts;
5. lists a module two layers both provide;
6. puts two projects over one saved dependency;
7. runs the same from the command line, failing on the project's flags only;
8. refreshes after an edit, reading again only the files that changed;
9. times reading everything against loading and adding, on a generated
   dependency.

Everything runs on files the example writes: no Lean toolchain, no download.

.. note::

   The index is lexical: it reads the text, it does not check proofs. Its
   reports name the kernel checks (``lake build``, ``#print axioms``) that
   decide.
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

from scikitplot.lean import LeanIndex
from scikitplot.lean._index import main

work = Path(tempfile.mkdtemp(prefix="lean_layers_"))


def project(name, files):
    """Write *files* (path to Lean source) under a new folder and return it."""
    root = work / name
    for path, source in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


# %%
# 1. A dependency, indexed once and saved
# ---------------------------------------
# :meth:`LeanIndex.save` writes the index to one file: numbers and text only,
# a checksum, the same bytes for the same files. Nothing in it runs when it
# is loaded.

library = project(
    "library",
    {
        "Lib/Defs.lean": """
            def Good (n : Nat) : Prop := n = n
            theorem shaky : 1 = 1 := sorry
        """,
        "Lib/Util.lean": """
            import Lib.Defs
            theorem good_all (n : Nat) : Good n := rfl
            theorem uses_shaky : 1 = 1 := shaky
            theorem unused : 2 = 2 := rfl
        """,
    },
)
saved = LeanIndex.from_folder(library).save(work / "library.leanidx")
again = LeanIndex.from_folder(library).save(work / "again.leanidx")
print(f"{saved.name}: {saved.stat().st_size:,} bytes")
print("same bytes from a second build:", saved.read_bytes() == again.read_bytes())

# %%
# 2. A project over the dependency
# --------------------------------
# :meth:`LeanIndex.load` reads the saved dependency (``root`` is where its
# files are, to quote statements); :meth:`LeanIndex.add_folder` reads only
# the project's files, labelled ``name:path``. The project's names resolve
# into the layer below.

app = project(
    "app",
    {
        "App/Main.lean": """
            import Lib.Util
            theorem main : ∀ n, Good n := fun n => good_all n
            theorem main_shaky : 1 = 1 := uses_shaky
        """,
    },
)
index = LeanIndex.load(saved, root=library)
index.add_folder(app, name="app")
dossier = index.dossier("main_shaky")
print(dossier["target"]["file"], "-", dossier["verdict"])
for item in dossier["dependencies"]["red_flags"]["items"]:
    steps = " -> ".join(f"{s['declaration']} ({s['file']})" for s in item["path"])
    print("  ", item["flags"], "via main_shaky ->", steps)

# %%
# 3. The same as one index of both folders
# ----------------------------------------
# Copy both into one folder and read it whole: once the ``app:`` prefix is
# set aside, the dossiers are the same.

both = project(
    "both",
    {
        str(p.relative_to(folder)): p.read_text(encoding="utf-8")
        for folder in (library, app)
        for p in folder.rglob("*.lean")
    },
)
whole = LeanIndex.from_folder(both)


def essence(report):
    """A dossier without the layer prefix of its labels, nor quoted text."""
    found = json.loads(json.dumps(report, sort_keys=True).replace('"app:', '"'))
    found["target"].pop("statement", None)
    for item in found["statement"]["local_definitions"]["items"]:
        item.pop("text", None)
        item.pop("text_cut", None)
    return found


for name in ("main", "main_shaky"):
    print(name, "same:", essence(index.dossier(name)) == essence(whole.dossier(name)))

# %%
# 4. Reports on one layer
# -----------------------
# ``layer=`` in :meth:`LeanIndex.summary`, :meth:`LeanIndex.discover` and
# :meth:`LeanIndex.export` describes one layer's files (``""`` is the
# first). Uses are still counted across layers: the dependency's
# ``good_all`` is load-bearing because the project names it, and listed
# only in the dependency's layer.

for layer in ("", "app"):
    summary = index.summary(layer=layer)
    found = index.discover(layer=layer)
    print(
        f"layer {layer!r}: {summary['files']} file(s),"
        f" {summary['declarations']} declarations, flags {sorted(summary['flags'])};"
        f" headline {[i['declaration'] for i in found['headline_results']['items']]}"
    )
rules = index.discover(layer="")["rules"]["load_bearing_theorems"]["items"]
print({i["declaration"]: i["named_from_files"] for i in rules})

# %%
# 5. A module two layers provide
# ------------------------------
# Lean gives each module one name across all packages, and refuses a build
# where two provide it. The lower layer's file is used; the other is listed.

fork = project("fork", {"Lib/Defs.lean": "def Good (n : Nat) : Prop := True\n"})
index.add_folder(fork, name="fork")
print(index.summary()["imports"]["module_clashes"]["items"])

# %%
# 6. Two projects over one dependency
# -----------------------------------
# Each review loads the same saved file: one index per project, one read of
# the dependency for all of them. Loading again costs the load, not a
# second reading of the library.

other = project(
    "other",
    {"Other/A.lean": "import Lib.Defs\ntheorem other : Good 0 := rfl\n"},
)
for folder, name in ((app, "app"), (other, "other")):
    each = LeanIndex.load(saved)
    each.add_folder(folder, name=name)
    print(name, "-", each.summary(layer=name)["declarations"], "declaration(s) of its own")

# %%
# 7. The command line
# -------------------
# ``--base FILE`` loads the dependency and adds ``FOLDER`` over it, named
# ``--name`` or the folder's name; ``--layer`` limits the summary and
# ``--fail-on`` to one layer, so a pipeline fails on the project's own
# ``sorry`` and not on a dependency's.

for layer in ("app", ""):
    with contextlib.redirect_stdout(io.StringIO()):
        status = main(
            [str(app), "--base", str(saved), "--layer", layer, "--fail-on", "sorry"]
        )
    print(f"--layer {layer!r}: exit {status}")

# %%
# 8. Refresh after an edit
# ------------------------
# :meth:`LeanIndex.refresh` reads every file of one layer once, for its
# digest and the commands it defines, and parses only the new and changed
# ones. The refreshed index saves to the same bytes as a new read.

index = LeanIndex.load(saved, root=library)
index.add_folder(app, name="app")
(app / "App" / "Main.lean").write_text(
    "import Lib.Util\ntheorem main : ∀ n, Good n := sorry\n", encoding="utf-8"
)
(app / "App" / "Extra.lean").write_text(
    "import App.Main\ntheorem extra : Good 3 := main 3\n", encoding="utf-8"
)
report = index.refresh("app")
print(
    "changed", report["changed"]["items"], "added", report["added"]["items"],
    "removed", report["removed"]["items"], "- files read:", report["files_read"],
)  # fmt: skip
print("extra:", index.dossier("extra")["flags_found"])
again = LeanIndex.load(saved, root=library)
again.add_folder(app, name="app")
same = index.save(work / "r1.leanidx").read_bytes() == again.save(
    work / "r2.leanidx"
).read_bytes()
print("same bytes as a new read:", same)

# %%
# 9. Read everything, or load and add
# -----------------------------------
# A generated dependency of 400 modules and a project of 5. Reading both
# grows with the dependency; loading it and adding the project grows with
# the project.

big = {}
for m in range(400):
    head = f"import Dep.M{m - 1}\n" if m else ""
    body = [f"theorem t{m}_{i} : {i} = {i} := rfl" for i in range(20)]
    body.append(f"theorem link{m} : True := {f'link{m - 1}' if m else 'trivial'}")
    big[f"Dep/M{m}.lean"] = head + "\n".join(body) + "\n"
dep = project("dep", big)
mine = {
    f"Mine/P{k}.lean": f"import Dep.M399\ntheorem mine{k} : True := link399\n"
    for k in range(5)
}
proj = project("proj", mine)
combined = project("combined", {**big, **mine})

start = time.perf_counter()
LeanIndex.from_folder(combined).dossier("mine4")
read_all = time.perf_counter() - start
dep_saved = LeanIndex.from_folder(dep).save(work / "dep.leanidx")
start = time.perf_counter()
layered = LeanIndex.load(dep_saved)
layered.add_folder(proj, name="proj")
reached = layered.dossier("mine4")["dependencies"]["declarations"]
load_add = time.perf_counter() - start
print(f"read both: {read_all:.2f} s; load and add: {load_add:.2f} s")
print("mine4 reaches", reached, "declarations through the dependency")

# %%
# .. tags::
#
#    model-workflow: lean
#    plot-type: text
#    level: intermediate
#    purpose: showcase
