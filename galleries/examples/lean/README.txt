.. _lean_examples:

Lean
====

.. currentmodule:: scikitplot.lean

Evidence on Lean projects, read from source text with :class:`LeanIndex`:
what a theorem claims, what its proof rests on, and whether any of that is
``sorry``, an axiom or code that adds declarations at build time. Every
item names its file and lines. The examples write their own projects: no
Lean toolchain, no download.

.. code-block:: text

   .lean files
        ↓
   LeanIndex.from_folder      (declarations, names used, audit flags)
        ↓                     save / load: one file, read in seconds;
        ↓                     refresh: read again only what changed
   add_folder                 (a project over its saved dependencies)
        ↓
   summary   dossier   discover / export     (layer=: one project's files)
        ↓
   lake build, #print axioms  (the kernel decides)

To read ``.lean`` files as documents of a retrieval corpus, see
:class:`scikitplot.corpus.LeanReader` in the corpus examples.
