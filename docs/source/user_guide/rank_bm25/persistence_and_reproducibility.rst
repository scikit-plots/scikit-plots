.. _rank-bm25-persistence-and-reproducibility:

Persistence and reproducibility
===============================

``save(directory)`` publishes a candidate generation and then updates
``current.json`` only after the candidate can be reloaded. ``load`` accepts the
publication root or a generation directory.

The saved artifact contains derived lexical state—terms, term frequencies, IDF,
document lengths, parameters and optional ``doc_ids``. Derived does not mean
non-sensitive; vocabulary and document identities can reveal source content.

``index.identity`` hashes the scoring recipe, declared analyzer and parameters.
It describes **how** an index was built, not **which corpus** it contains. Equal
identities do not imply equal documents.

A bare callable tokenizer has no durable analyzer identity and therefore cannot
be safely persisted as a reproducible declaration.

Generation format
-----------------

A generation written by this version (schema ``bm25-index/2``) holds
``index.json`` and three arrays: ``indptr.npy``, ``rows.npy`` and ``tfs.npy``,
the postings term by term (rows and counts as 32-bit integers when they fit).
``index.json`` records each array's SHA-256, dtype and shape, and the
generation's name is the digest of ``index.json``, so it covers the arrays too.
``load`` reads the arrays with ``allow_pickle=False``, refuses any whose digest,
dtype or shape differs, and recomputes the precomputed contributions from them
with the recipe's own formula, bit for bit; ``load(..., mmap_mode="r")``
memory-maps the arrays. The per-document term counts are rebuilt from the
arrays only if a path needs them (``backend="reference"``,
``get_batch_scores``).

Measured on a synthetic 100 000-document index: the previous format
(``bm25-index/1``, every document's counts in the JSON) took 173 MB and 4.2 s to
load and answer a first query; ``bm25-index/2`` took 82 MB and 0.6 s.
Generations of ``bm25-index/1`` still load.
