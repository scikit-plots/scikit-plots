"""
Read Lean Proofs with LeanReader: Tracing Every Line of a Formal Development
=============================================================================

.. currentmodule:: scikitplot.corpus

`Lean <https://lean-lang.org/>`_ proofs can run to millions of lines, far past
what a person can keep track of. :class:`LeanReader` splits a ``.lean`` file
into one chunk per Lean command: a ``theorem``, a ``def``, a ``namespace``.
Each chunk records where it is, what it declares and the audit flags it
raises. This example goes from the simplest case to the hard ones:

1. one file: what a chunk carries;
2. statements without proofs (``proofs="drop"``), for reading claims;
3. audit flags: ``sorry``, ``native_decide``, ``axiom``, code after ``#exit``;
4. choosing what to read: ``kinds``, ``watch``, ``commands``;
5. what a declaration refers to (``references=True``), and chunks an index
   reads (``statements=True``);
6. very long proofs read in parts (``lines_per_chunk``);
7. Lean 3 sources (``dialect="lean3"``);
8. hostile input: unclosed comments, too-deep scopes, and the source
   problems Lean would refuse;
9. a generated project of thousands of declarations, with a check that
   every line lands in exactly one chunk.

Everything runs on files the example writes: no Lean toolchain, no download.

.. note::

   The reader is lexical: it reads the text, it does not check proofs.
   A file without flags is not proved correct; that is for Lean's kernel
   (``lake build``, ``#print axioms``).
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from scikitplot.corpus import LEAN_FLAGS, LeanReader

work = Path(tempfile.mkdtemp(prefix="lean_reader_"))


def read(text, name="Demo.lean", **options):
    """Write *text* to a file and return the chunks LeanReader makes of it."""
    path = work / name
    path.write_text(text, encoding="utf-8")
    return list(LeanReader(input_path=path, **options).get_raw_chunks())


def show(chunks, *keys):
    for c in chunks:
        extra = {k: c[k] for k in keys if k in c}
        print(f"{c['line_start']:>3}-{c['line_end']:<3} {c['lean_kind']:<10}",
              c.get("declaration", ""), extra)


# %%
# 1. One file
# -----------
# Imports, namespaces, doc comments and declarations each become a chunk.
# A doc comment stays with the declaration it documents; ``declaration`` is
# the full name, namespace included.

source = """\
import Mathlib.Data.Nat.Basic

namespace Demo

/-- Adding zero changes nothing. -/
theorem add_zero' (n : Nat) : n + 0 = n := by
  simp

def double (n : Nat) : Nat := 2 * n

end Demo
"""
show(read(source), "namespace", "imports")

# %%
# 2. Statements without proofs
# ----------------------------
# ``proofs="drop"`` keeps what a declaration claims and leaves out how it is
# proved. ``proof_line_start``/``proof_line_end`` say where the proof was.

for c in read(source, proofs="drop"):
    if c["lean_kind"] == "theorem":
        print(repr(c["text"]), c.get("proof_line_start"), c.get("proof_line_end"))

# %%
# 3. Audit flags
# --------------
# Some constructs let a "proof" through without the kernel checking it.
# The reader flags them in code only, never in comments or strings. A
# comment line belongs to the declaration after it (``fast`` starts at 2).
# Everything after ``#exit`` is flagged too, since Lean never reads it.

print(sorted(set(LEAN_FLAGS.values())))
risky = """\
theorem gap : 1 = 2 := by sorry
-- a comment saying sorry is not flagged
theorem fast : 2 ^ 10 = 1024 := by native_decide
axiom trust_me : False
#exit
theorem never_checked : True := trivial
"""
show(read(risky, "Risky.lean"), "flags")

# %%
# 4. Choosing what to read
# ------------------------
# ``kinds`` keeps some kinds of command; ``watch`` marks chunks naming the
# identifiers you care about; ``commands`` adds a project's own command
# keywords, so their uses become chunks of their own.

print([c["declaration"] for c in read(source, kinds=["theorem"])])
print([c.get("watched") for c in read(source, watch=["double"]) if "watched" in c])
custom = "theorem a : True := trivial\nmy_check a\ntheorem b : True := trivial\n"
print([c["lean_kind"] for c in read(custom, "Custom.lean")])
print([c["lean_kind"] for c in read(custom, "Custom.lean", commands=["my_check"])])

# %%
# 5. What a declaration refers to
# -------------------------------
# With ``references=True`` each chunk lists the names it uses, and the
# ``open`` namespaces in force, so a dependency graph can be built.

refs = """\
def base : Nat := 1
def next : Nat := base + 1
open Nat in
theorem uses : next = 2 := by simp [next, base]
"""
show(read(refs, "Refs.lean", references=True), "references", "opens")

# %%
# With ``statements=True`` (whole commands: ``lines_per_chunk=None``) each
# theorem also carries its statement. Chunks read with both are what
# :meth:`scikitplot.lean.LeanIndex.add_chunks` indexes, for a file inside an
# archive, say: the same index as reading the file. A project that defines
# commands gives them first (``learn_commands``, ``defined_commands``) and
# reads each file with ``commands=index.commands()``.

from scikitplot.lean import LeanIndex  # noqa: E402

chunks = read(refs, "Refs.lean", references=True, statements=True, lines_per_chunk=None)
print([c["statement"] for c in chunks if "statement" in c])
index = LeanIndex()
index.add_chunks(chunks, label="Refs.lean")
print(index.dossier("uses")["dependencies"]["declarations"], "declarations reached by uses")

# %%
# 6. Very long proofs
# -------------------
# A proof of thousands of lines is read in parts of ``lines_per_chunk``
# lines; ``part`` numbers them, and every part keeps the declaration name.

long_proof = "theorem big : True := by\n" + "  skip\n" * 1_000 + "  trivial\n"
parts = read(long_proof, "Long.lean", lines_per_chunk=300)
print([(c["part"], c["line_start"], c["line_end"]) for c in parts])

# %%
# 7. Lean 3
# ---------
# Lean 3 proofs use ``begin ... end`` blocks; ``dialect="lean3"`` reads them
# as one command each.

lean3 = """\
import data.nat.basic

theorem add_comm' (a b : ℕ) : a + b = b + a :=
begin
  exact nat.add_comm a b,
end
"""
show(read(lean3, "Old.lean", dialect="lean3"))

# %%
# 8. Hostile input
# ----------------
# A comment that never closes swallows the rest of the file as comment, as
# Lean would. Scopes nested deeper than ``max_scope_depth`` are refused with
# a clear error instead of using unbounded memory.

print([c["section_type"] for c in read("/- never closed\ntheorem x : True := trivial\n", "Open.lean")])

# %%
# What Lean refuses to read is on the chunk holding its line, in
# ``source_problems``: an ``end`` that closes nothing, a file ending inside a
# comment, bytes that are not UTF-8.

problems = read("namespace A\nend B\n/- never closed\n", "Bad.lean")
print([p for c in problems for p in c.get("source_problems", ())])
try:
    read("namespace N\n" * 1_000, "Deep.lean", max_scope_depth=256)
except ValueError as exc:
    print("refused:", str(exc)[:70])

# %%
# 9. A generated project
# ----------------------
# Thousands of small declarations, a few of them flagged. Counting the
# flags points straight at the lines to review, and the partition check
# proves no line was lost or read twice.

lines = ["namespace Big"]
for i in range(5_000):
    proof = "by sorry" if i % 1_000 == 999 else "rfl"
    lines.append(f"theorem t{i} : {i} = {i} := {proof}")
lines.append("end Big")
big = read("\n".join(lines) + "\n", "Big.lean")

flags = Counter(flag for c in big for flag in c.get("flags", ()))
print(f"{len(big)} chunks; flags: {dict(flags)}")
print([f"{c['declaration']} at line {c['line_start']}" for c in big if c.get("flags")])

covered = [n for c in big for n in range(c["line_start"], c["line_end"] + 1)]
assert covered == list(range(1, len(lines) + 1)), "lines lost or repeated"
print(f"all {len(lines):,} lines are in exactly one chunk")

# %%
# .. tags::
#
#    model-workflow: corpus
#    plot-type: text
#    level: intermediate
#    purpose: showcase
