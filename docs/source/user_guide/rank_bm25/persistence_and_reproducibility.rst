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
