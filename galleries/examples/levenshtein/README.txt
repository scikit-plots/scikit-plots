.. _levenshtein_examples:

Levenshtein
===========

.. currentmodule:: scikitplot.levenshtein

These examples form a small learning path for exact edit distance and fuzzy
ranking. They are deterministic, local and network-free.

.. code-block:: text

   two sequences
        ↓
   unit-cost edit distance
        ↓
   normalized similarity
        ↓
   optional ranking / cutoff
        ↓
   actual backend provenance

Start here
----------

1. **Basics** — the classic ``kitten`` → ``sitting`` distance and normalized
   metrics.
2. **Backends** — inspect internal, RapidFuzz, explicit GPL and pure-Python
   capabilities without requiring any optional backend to be installed.
3. **Ranking** — stable ties, ``limit``, ``score_cutoff`` and object ``key``.
4. **Sequences** — bytes, token sequences and explicit Unicode normalization.
5. **Corpus** — use Levenshtein as a deterministic local Corpus retrieval
   scorer.

Reliability rules
-----------------

Every script:

- uses only local literal data;
- performs no network access or downloads;
- has a pure-Python path;
- asserts the result it demonstrates;
- does not turn a wrong distance into a skip.

An unavailable optional accelerator can be reported as unavailable. Metric
correctness cannot be skipped.

Backend note
------------

``auto`` never chooses the external GPL ``Levenshtein`` backend. The examples
do not require it.

The gallery demonstrates the facade contract rather than benchmarking one
machine's accelerator installation.
