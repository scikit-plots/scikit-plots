.. _levenshtein-user-guide:

Levenshtein distance and fuzzy matching
=======================================

.. currentmodule:: scikitplot.levenshtein

:mod:`scikitplot.levenshtein` provides one small, stable facade for exact
unit-cost Levenshtein edit distance, normalized similarity, deterministic
ranking and optional Corpus retrieval.

The base behavior never depends on a third-party package. Optional
accelerators are discovered lazily.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Start here
      :link: levenshtein-getting-started
      :link-type: ref

      Distance, normalized similarity and the first ranked match.

   .. grid-item-card:: How it works
      :link: levenshtein-how-it-works
      :link-type: ref

      Metric semantics, fallback boundaries and provenance.

   .. grid-item-card:: Backends
      :link: levenshtein-backends
      :link-type: ref

      Internal, RapidFuzz, explicit GPL backend and pure Python.

   .. grid-item-card:: Ranking
      :link: levenshtein-ranking
      :link-type: ref

      Stable ties, ``limit``, ``key`` and ``score_cutoff``.

   .. grid-item-card:: Corpus integration
      :link: levenshtein-corpus
      :link-type: ref

      Use normalized edit similarity as a retrieval scorer.

   .. grid-item-card:: Performance and limits
      :link: levenshtein-performance
      :link-type: ref

      Complexity, streaming choices and what cutoff does today.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   how_it_works
   backends_and_fallbacks
   ranking_and_matching
   corpus_integration
   python_api
   performance_and_limits
   troubleshooting

Scope
-----

The facade intentionally does **not** try to expose every feature from
RapidFuzz or the external ``Levenshtein`` project. It keeps one portable metric
contract and lets optional implementations accelerate it.

For transpositions, weighted operations, edit scripts, median strings or a
broader fuzzy-matching toolbox, use a specialized library directly.

Examples
--------

See :ref:`levenshtein_examples` for executable, network-free examples.
