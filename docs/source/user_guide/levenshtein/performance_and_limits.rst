.. _levenshtein-performance:

Performance and limits
======================

.. currentmodule:: scikitplot.levenshtein

Pure-Python complexity
----------------------

The fallback uses a memory-bounded Wagner--Fischer dynamic program.

For lengths ``N`` and ``M``:

- time is ``O(N*M)``;
- working memory is ``O(min(N, M))``.

Accelerators can use substantially faster implementations for unit-cost edit
distance, but the public result must remain identical.

Ranking cost
------------

Ranking performs one distance computation per consumed choice.

``limit`` reduces retained match memory, not the number of choices evaluated.
A top-k ranking still examines every input candidate.

``score_cutoff`` currently filters after exact computation and therefore is not
a performance bound.

Large collections
-----------------

For large corpora, do not use edit distance as an accidental all-pairs search
engine.

Prefer a staged design:

.. code-block:: text

   lexical/vector candidate retrieval
              ↓
        small candidate set
              ↓
        Levenshtein reranking

This gives edit distance the role it is best at: local spelling/sequence
similarity rather than semantic retrieval.

Unicode
-------

Distance operates on the sequence Python receives. Canonically equivalent
Unicode spellings can therefore have non-zero edit distance.

Normalize explicitly when your application requires that:

.. code-block:: python

   import unicodedata

   normalize = lambda s: unicodedata.normalize("NFC", s)

   left = normalize("cafe\u0301")
   right = normalize("café")

No normalization is implicit because that would change metric semantics.

Backend probing
---------------

Capability discovery should not be placed inside extremely hot application
loops. `rank()` currently resolves backend state per candidate; maintenance
finding ``LV-002`` tracks a future per-operation execution plan.
