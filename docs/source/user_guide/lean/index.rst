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

Saving an index
---------------

Reading a large project takes minutes; loading its saved index takes
seconds and gives the same reports::

    index = LeanIndex.from_folder("path/to/project", workers=2)
    index.save("project.leanidx")
    index = LeanIndex.load("project.leanidx", root="path/to/project")

From the shell, ``--save FILE`` after indexing, and ``--load FILE`` instead
of indexing (``FOLDER`` is then where the files are)::

    python -m scikitplot.lean path/to/project --save project.leanidx
    python -m scikitplot.lean path/to/project --load project.leanidx --target main_theorem

A snapshot (format ``lean.index.snapshot/1``) holds numbers and text only:
nothing in it runs when it is loaded. It carries a checksum, and a
damaged or altered file is refused with :class:`ValueError`, also when its
checksum was made to match: every number is checked against what it
refers to. The same files give the same bytes, whatever order they were
added in. Files are named by label, never by path: without ``root`` no
file is read and reports give each item's file and lines; with ``root``,
statements are quoted only from files inside it.

==============  =============  =========  =========  ==========
\                build          save       load       file
==============  =============  =========  =========  ==========
mathlib4        30 s           1.4 s      1.4 s      54 MB
175,187 files   245 s (two     10 s       13 s       404 MB
                workers)
==============  =============  =========  =========  ==========

A project over its dependencies
-------------------------------

A project that depends on mathlib need not read mathlib each time: load
the saved dependency and add the project over it as a *layer*::

    index = LeanIndex.load("mathlib.leanidx", root="path/to/mathlib")
    index.add_folder("path/to/project", name="project")
    index.dossier("main_theorem")

From the shell, ``--base FILE`` loads the dependency (``--base-root`` is
where its files are, to quote them) and ``FOLDER`` is added over it,
named ``--name`` or the folder's own name::

    python -m scikitplot.lean path/to/project --base mathlib.leanidx \
        --base-root path/to/mathlib --target main_theorem

The reports are those of one index of both folders: names in the project
resolve into the layer below, and dossiers follow proofs through both. A
layer's files are labelled ``name:path`` (``project:Project/Main.lean``),
so the same path in two folders stays two files. Each layer finds its own
source root, from the imports of that layer and the ones above it that no
lower layer provides. A module two layers provide is listed under
``imports`` / ``module_clashes``: the lower layer's file is used, as Lean
refuses both in one build. A layer learns the commands its own files
define (``learn_commands``). Saving a layered index keeps its layers;
``load(root={"": "path/to/mathlib", "project": "path/to/project"})``
quotes each from its own folder.

On mathlib4 with the PFR project over it, loading mathlib4 and adding PFR
takes 1.4 s against 24 s to read both folders, and the dossiers of PFR's
theorems are the same, labels aside.

Reports on one layer
~~~~~~~~~~~~~~~~~~~~

``layer=`` in :meth:`~LeanIndex.summary`, :meth:`~LeanIndex.discover` and
:meth:`~LeanIndex.export` (``--layer NAME`` on the command line) describes
one layer's files: the project without its dependencies, or ``""`` for the
first layer::

    index.summary(layer="project")    # its files, lines, kinds, flags
    index.discover(layer="project")   # its rules, patterns, claims, headlines

Names still resolve through every layer and uses are counted from every
layer: a dependency's theorem the project names is load-bearing in the
dependency's layer, and a project theorem only a later layer names is not
isolated. A duplicate name, a module clash or a generator is listed when
one of its files is in the layer. With ``--layer``, ``--fail-on`` looks at
that layer's flags only, so a pipeline fails on the project's own
``sorry`` and not on a dependency's::

    python -m scikitplot.lean path/to/project --base mathlib.leanidx \
        --layer project --fail-on sorry

See :ref:`sphx_glr_auto_examples_lean_plot_lean_layers_script.py` for both,
run end to end.

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
indexing                    434 s        227 s
summary                     318 s        187 s
peak memory while reading   1.2 GB       1.5 GB
peak memory, summary        1.7 GB       2.6 GB
==========================  ===========  ===========

Reading costs each worker a small, fixed amount: it holds at most 64
files' records before sending them back. Name resolution costs more,
since each worker copies what it reads of the index (Python's reference
counts write to it). What the workers read most is kept flat so they do
not copy it: the names each declaration uses, as numbers in arrays, and
in the workers each name and context as text in one shared buffer,
decoded privately. Freezing the garbage collector before
the workers start (``gc.freeze``) saves under 100 MB: the copies come from
reference counts, not from collection.

To compare the backends, or numbers of workers, on a project of your own,
run the benchmark helper from a source checkout; it runs each backend in
its own process, times every stage, gives each stage's peak memory (PSS,
on Linux), and exits 1 if two backends' reports differ::

    python maintenances/corpus/_maintenance/tools/lean_index_benchmark.py \
        FOLDER --compare --workers 2
