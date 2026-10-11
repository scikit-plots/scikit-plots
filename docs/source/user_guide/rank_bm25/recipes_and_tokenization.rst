.. _rank-bm25-recipes-and-tokenization:

Recipes and tokenization
========================

The concrete scorers are :class:`BM25Okapi`, :class:`BM25L` and
:class:`BM25Plus`. Each exposes a named/versioned ``recipe`` so persisted or
reported results can identify the implemented formula rather than relying only
on a class name.

Tokenization is part of correctness
-----------------------------------

Without ``tokenizer=`` each document must already be a sequence of string
tokens. A raw string is rejected rather than indexed character-by-character.
If a tokenizer is provided, use the same case folding, stemming, punctuation,
stopword and normalization policy for queries.

``workers`` affects corpus tokenization during construction. Multiprocessing
requires a picklable tokenizer; closures and lambdas are poor choices for that
mode.
