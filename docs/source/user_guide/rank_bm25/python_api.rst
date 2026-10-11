.. _rank-bm25-python-api:

Python API
==========

.. autosummary::
   :nosignatures:

   BM25
   BM25Okapi
   BM25L
   BM25Plus

The base :class:`BM25` owns corpus statistics, validation, sparse candidate
state, persistence and identity. Instantiate a concrete scorer for normal use.

The implementation contains analyzer/recipe identity machinery, but analyzer
declarations are not currently exported as the stable top-level public API.
Do not build application code against private import paths merely to persist a
custom tokenizer.
