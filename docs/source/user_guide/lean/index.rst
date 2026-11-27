.. currentmodule:: scikitplot.lean

.. _lean-index:
.. _lean-user-guide:

Evidence on Lean projects
=========================

:mod:`scikitplot.lean` reads a Lean 4 (or Lean 3) project from its source
text and answers the questions a reviewer of a formal proof asks first: what
does this theorem claim, what does its proof rest on, and is any of that
``sorry``, an axiom, ``native_decide`` or code that adds declarations at
build time? Every answer names its file and lines. Nothing here is a proof:
Lean's kernel decides proofs (``lake build``, ``#print axioms``, an
independent replay), and every report says so.

It uses the Python standard library only. To read ``.lean`` files as
documents of a retrieval corpus, use :class:`scikitplot.corpus.LeanReader`,
which is built on the same language core.

Quick start
-----------

.. code-block:: python

    from scikitplot.lean import LeanIndex

    index = LeanIndex.from_folder("path/to/project")  # the project's root
    report = index.dossier("MyProject.main_theorem")
    print(report["verdict"])           # never claims more than the text shows
    print(LeanIndex.render(report))    # Markdown, every item with file:line

    summary = index.summary()          # the whole project
    found = index.discover()           # rules, patterns, claims, anomalies
    index.export("helpers/")           # all of it as documents and JSONL

From the shell, and in a pipeline (exit status 1 when a flag is found, 2
when a target is not found or ambiguous, so a renamed theorem never passes
a gate)::

    python -m scikitplot.lean path/to/project
    python -m scikitplot.lean path/to/project --target main_theorem --fail-on sorry

The reports
-----------

``summary()``
    Files, lines and declarations; every audit flag with where it is;
    names declared more than once; imported project modules no file
    provides; syntax extensions by kind, with the flagged ones apart
    (Lean applies them without their names); commands the project defines; code that
    adds declarations no source line spells; files that were skipped, with
    the reason; text Lean refuses to read (an ``end`` that names another
    scope or closes none, a file ending inside a comment or string), by
    file and line.

``dossier(name)``
    The theorem's statement and the project definitions it unfolds to
    (quoted, each vacuous one marked: a ``Prop`` whose body is ``True``,
    ``False`` and connectives only, or uses none of its parameters); every
    declaration its proof names, directly or through
    others, with the path to each flagged one; what it could use without
    naming it (flagged instances, tagged lemmas, syntax extensions, and
    generated declarations, in its environment only); text Lean refuses to
    read in the files it depends on; checks with a status each, a verdict,
    and the kernel steps that settle it.

``discover()`` and ``export(folder)``
    Candidates, each chosen by a stated rule: theorems and definitions
    named from the most files; theorems of one statement shape and
    numbered series of names; ``Prop`` definitions, vacuous predicates,
    axioms, ``proof_wanted`` and ``sorry``; names declared in parallel; results
    nothing names. ``export`` writes them as Markdown, JSON and one JSON
    record per item (``records.jsonl``) for a corpus or an MCP server, with
    a dossier per headline result.

Reports are JSON-ready and deterministic: the same files give the same
report, in any order of adding files and queries.

Policies
--------

``visibility="imports"`` (default)
    A name links only into its own file and the files it imports, as Lean
    sees them. A file with a project import no file provides (a missing
    module, or an index built from a subfolder) sees every declaration, so
    an incomplete index links too much, never too little.
    ``visibility="all"`` links by text alone.

``source_root``
    Module names start at the project's source root, found from the
    imports (``Mathlib/Data/Nat.lean`` is ``Mathlib.Data.Nat`` from the
    repository root; ``src/A/B.lean`` is ``A.B``). Set it (``"src"``, or
    ``""`` for the index root) when a small project's imports could name
    either a dependency or a project file.

``suffix="unique"`` (default) or ``"none"``
    Whether ``h.myLemma`` (dot notation on a hypothesis) links to the one
    declaration whose last component is ``myLemma``.

Limits
    ``max_file_bytes`` and ``max_scope_depth`` skip a file (listed with the
    reason) instead of stopping. Hidden folders such as ``.lake/packages``
    are left out unless ``include_hidden=True``. Links to folders are never
    followed, and a link to a file is read only when it stays inside the
    project.

``--fail-on FLAG``
    One of :data:`AUDIT_FLAGS`; an unknown flag is refused, so a typo
    cannot pass a pipeline.

What a text index cannot see
----------------------------

Lean finds instances, ``@[simp]`` lemmas, coercions and notation without
their names appearing in a proof, and metaprograms can create declarations
under names no source line spells. A dossier lists those in the theorem's
environment as risks to review rather than assume them unused. Names are
resolved lexically, as Lean looks them up (namespaces, ``open``,
``_root_``, derived names), keeping every candidate: a dossier may hold
more than the proof uses, never less of what its text names.

Scale
-----

Indexing is linear in the text and keeps a few integers and the used
names per declaration. On mathlib4 (9,351 files, 288,006 declarations)
with two workers: indexing about 21 s, a summary about 6.8 s (12.5 s with
one), discovery about 1.5 s, a dossier in milliseconds, peak memory about
400 MB.

Two options change how the work is done, never what is reported:

- ``workers=`` (``--workers``) reads files in that many processes;
  ``None`` (``--workers 0``) uses every CPU. On Linux the same workers
  also resolve names when a step needs every edge (a summary, discovery,
  an export), for projects of 20,000 declarations or more: forked, they
  start from the index in memory. Results are taken in order, so every
  report is the same for any number of workers. Memory grows with them:
  each worker copies the parts of the index it reads and keeps its own
  lookup cache (see the table below), so choose ``workers`` by the
  machine's memory as well as its CPUs.
- ``graph=`` (``--graph``) chooses the graph backend: ``"python"`` (the
  default, standard library only), ``"numpy"`` (edges packed into
  arrays, use counts and searches vectorised) or ``"scipy"`` (packed as
  for ``"numpy"``, searches in :mod:`scipy.sparse.csgraph`). A missing
  optional dependency raises :class:`ImportError` when the index is made.

Each file keeps the set of project files it sees, one bit per file, so
these sets grow with the square of the file count. Past 20,000 files they
are stored compressed and expanded on use: on a 175,187-file project the
sets took 18 times less memory, and peak memory fell from 5.5 GB to
2.1 GB.

On that project (175,187 files, 1,068,089 declarations; two CPUs), the
same report either way; memory is the peak proportional set size (PSS)
summed over every process, so a page the workers share is counted once:

==========================  ===========  ===========
\                           one worker   two workers
==========================  ===========  ===========
indexing                    434 s        210 s
summary                     318 s        173 s
peak memory while reading   1.2 GB       1.5 GB
peak memory, summary        1.7 GB       3.0 GB
==========================  ===========  ===========

Reading costs each worker a small, fixed amount: it holds at most 64
files' records before sending them back. Name resolution costs more,
since each worker copies what it reads of the index (Python's reference
counts write to it); the names each declaration uses are kept in flat
arrays, which reading does not copy. Freezing the garbage collector before
the workers start (``gc.freeze``) saves under 100 MB: the copies come from
reference counts, not from collection.

To compare the backends, or numbers of workers, on a project of your own,
run the benchmark helper from a source checkout; it runs each backend in
its own process, times every stage, gives each stage's peak memory (PSS,
on Linux), and exits 1 if two backends' reports differ::

    python maintenances/corpus/_maintenance/tools/lean_index_benchmark.py \
        FOLDER --compare --workers 2
