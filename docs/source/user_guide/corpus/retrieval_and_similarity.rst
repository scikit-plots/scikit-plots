.. currentmodule:: scikitplot.corpus

.. _corpus-retrieval-similarity:

Retrieval and similarity
========================

Corpus supports its existing lexical/vector/hybrid retrieval stack and can also
consume user scorers through :class:`CustomRetrievalIndex`.

Hybrid fusion
-------------

Every hybrid path fuses ranks by one rule, :class:`FusionPolicy` (weighted
reciprocal rank fusion): :meth:`RetrievalIndex.search` with
``match_mode="hybrid"``, :func:`fuse_by_rank`, and the MCP hybrid retriever,
which carries a byte-identical copy of the contract. A document scores the sum
of ``weight / (k + rank)`` over the legs that list it, ranks counted from 1.

* A document counts once per leg, at its best rank.
* A weight-0 leg contributes nothing and adds no document, so
  ``hybrid_alpha=0`` returns the lexical leg alone and ``1`` the dense leg
  alone, exactly; the other leg is not run and is reported ``SKIPPED``.
  ``hybrid_alpha`` is shorthand for the weights
  ``{lexical: 1 - alpha, dense: alpha}``.
* Scores are compared exactly; a tie falls to the lower ``doc_id``.
* Each corpus hybrid hit carries its per-leg ``contributions``; MCP results
  carry only the fused score.

.. code-block:: python

    from scikitplot.corpus import FusionPolicy

    fused = FusionPolicy(k=60).fuse(
        [("lexical", ["A", "B", "C"]), ("dense", ["C", "A", "D"])]
    )
    [item.key for item in fused]                       # ['A', 'C', 'B', 'D']
    [(c.leg, c.rank) for c in fused[0].contributions]  # [('lexical', 1), ('dense', 2)]
    dict(FusionPolicy.from_alpha(1.0).weights)         # {'lexical': 0.0, 'dense': 1.0}

Filtering a search
------------------

:meth:`RetrievalIndex.search` takes ``filter=``: a :class:`Filter` expression
of the nine operators (``Eq``, ``NotEq``, ``In``, ``NotIn``, ``Range``,
``Exists``, ``And``, ``Or``, ``Not``) or its JSON form. Every leg applies it
before ranking, so a filtered query returns the best ``top_k`` matching
documents, not the best ``top_k`` documents minus the excluded ones. Each leg
that ran reports ``filter_capability``: here every operator is ``EMULATED``,
evaluated in Python over the indexed documents. A filter is never dropped.

A filter names scalar :class:`CorpusDocument` fields (``language``,
``source_type``, ``collection_id``, ``chunk_index`` ...) or one metadata key as
``"metadata.<key>"``; any other name is refused, so a misspelt field cannot
silently read as missing. A value that cannot be compared (an array, a list in
``In``) does not match, and ``True`` does not equal ``1``.

A :class:`SecurityFilter` is bound to the index when it is built
(``RetrievalIndex(security_filter=...)``, ``CustomRetrievalIndex``, or
``BuilderConfig(security_filter=...)`` for every index a builder builds) and
AND-ed with every query's filter, on ``search`` and on the vector seam
``query``: a caller can narrow what it sees, never widen it. Keyword (BM25)
statistics are computed over the scope only, so documents outside it cannot
change a score, and a response reports the scope by its ``label``, not its
expression. ``n_documents`` still counts every indexed document.
``RuntimeCorpus`` and the MCP corpus factory do not take a scope yet. :func:`filter_to_dict` and :func:`filter_from_dict` carry a filter
through JSON (an MCP tool call, an agent's tool arguments); reading refuses
unknown operators, keys and fields, non-scalar values, strings over 4096
characters, integers beyond 2**63, and trees deeper than 32 levels or larger
than 256 nodes (filter cost grows with nodes times documents).

.. code-block:: python

    from scikitplot.corpus import (
        And, Eq, RetrievalIndex, SecurityFilter, filter_from_dict, filter_to_dict,
    )

    index = RetrievalIndex(security_filter=SecurityFilter(Eq("collection_id", "public")))
    query_filter = filter_from_dict({"op": "eq", "field": "language", "value": "en"})
    filter_to_dict(And(Eq("collection_id", "public"), query_filter))["op"]  # 'and'
    # index.build(documents); index.search("...", filter=query_filter)

Levenshtein facade
------------------

:mod:`scikitplot.levenshtein` is intentionally independent from Corpus and is
always importable. The automatic backend preference is:

.. code-block:: text

   bundled scikitplot.cexternals._editdistance
       -> RapidFuzz when installed
       -> dependency-free Python fallback

The separately distributed ``Levenshtein``/``python-Levenshtein`` backend is
supported only when explicitly requested; it is not silently selected by the
automatic chain.

.. code-block:: python

   from scikitplot.levenshtein import distance, rank

   distance("kitten", "sitting")
   # 3

   rank(
       "colour",
       ["color", "collar", "cloud"],
       score_cutoff=0.60,
   )

Corpus compatibility is lazy:

.. code-block:: python

   from scikitplot.corpus import CustomRetrievalIndex, make_levenshtein_ranker

   index = CustomRetrievalIndex(
       custom_scorer_fn=make_levenshtein_ranker(
           backend="auto",
           score_cutoff=0.70,
       )
   )

Corpus owns the adapter: importing :mod:`scikitplot.levenshtein` never imports
Corpus. :func:`make_pairwise_ranker` does the same for any pairwise function,
``make_pairwise_ranker(edit_similarity)`` included.

``score_cutoff`` uses the same normalized similarity scale ``[0, 1]`` for every
backend, so switching between bundled, RapidFuzz and pure-Python implementations
does not change the filtering contract. ``top_k`` is therefore an upper bound:
a cutoff can intentionally return fewer hits when the remaining candidates are
too dissimilar.

Choosing a metric
-----------------

Levenshtein distance is a character/sequence edit metric, not a semantic
embedding model. It is useful for spelling variants, OCR noise, identifiers,
near-duplicate labels and fuzzy lexical matching. Do not present edit-distance
similarity as semantic equivalence.

Embedding cache
---------------

:class:`EmbeddingEngine` caches vectors on disk under a key that names the
texts, the model and its revision, the backend, the provider and the output
settings. A sentence-transformers model that is not pinned to a commit
(``model_revision=""``, or a branch or tag such as ``"main"``, which can move)
names whatever the hub serves today, so by default its embeddings are computed
and not cached; the call says so (:meth:`EmbeddingEngine.embed_cached` returns
a :class:`CacheStatus`). Pin ``model_revision`` to a commit hash (40 or 64
lowercase hex characters), or choose a :class:`CachePolicy`:

* ``CachePolicy(unpinned="resolve", revision_resolver=resolve_hub_revision)``
  resolves the commit once per engine (a full Hub id ``"owner/name"``, not a
  local directory), then loads and keys exactly that commit; when resolution
  fails the model loads unpinned and the cache is bypassed (a branch or tag
  given in ``model_revision`` is not resolved);
* ``CachePolicy(unpinned="allow")`` caches under the name given, as before:
  an entry can outlive an upstream change of the model.

Other backends do not load a revision: for them a non-empty
``model_revision`` is your statement of the model version, and it enables the
cache. A custom provider is cached when it has a ``custom_fn_id``. ``max_entries``
and ``max_bytes`` bound the cache directory: a hit marks its entry as used,
and after a store the least recently used entries are evicted (only
``<64 hex>.npy`` files are counted or removed); one result larger than
``max_bytes`` is not kept, and evicts nothing. There is no time-to-live:
whether an entry is stale is a question of which revision produced it, which
the key answers.

.. code-block:: python

   from scikitplot.corpus import CachePolicy, EmbeddingEngine, resolve_hub_revision

   name = "sentence-transformers/all-MiniLM-L6-v2"
   commit = resolve_hub_revision(name)  # once; record it with your corpus
   engine = EmbeddingEngine(
       model_name=name,
       model_revision=commit,
       cache_policy=CachePolicy(max_entries=10_000),
   )
   # engine.embed_cached(texts).status -> CacheStatus.STORED, then HIT

:class:`MultimodalEmbeddingEngine` keys its cache on the content (every
backend and model, the fusion, projection and normalisation, and each
document's modality, text and ``raw_tensor``), not on the source path and its
modification time. Its image and audio models have no revision, so it caches
only with ``CachePolicy(unpinned="allow")``; a custom backend or projection is
never cached.

Retrieval execution plan
------------------------

:class:`RetrievalExecutionPlan` runs a query through typed stages, always in
this order: ``transform`` (the query to one or more queries), ``retrieve``
(``index.search`` for each, unchanged), ``fuse`` (the queries' rankings, by
the same :class:`FusionPolicy`), ``rerank`` (new scores for the first
``rerank_depth`` hits) and ``compress`` (character spans of each hit's text,
so the context stays a citation of its source). Only ``retrieve`` is
required; an empty plan returns exactly what ``index.search`` returns.

Each stage reports a :class:`StageOutcome`: its :class:`StageStatus`
(``success``, ``empty``, ``degraded``, ``failed``, ``skipped`` or
``cancelled``), the component that ran, items in and out, calls, seconds and
budget, and an error record whenever it did not finish. Limits are checked
before each call, so a spent limit stops the next call and says which limit
stopped it; a running call is not interrupted. That bounds only a stage of
several calls: a :class:`StageBudget` limits the calls and seconds of
``retrieve`` (one call per query) and is refused for any other stage, and
``deadline_seconds`` stops any stage or query from starting after it.
``fuse``, an in-process computation, always runs; a single call is bounded by
the component's own timeout. With
``on_error="degrade"`` (the default) a component that raises, or returns
something malformed, fails its stage and the previous stage's output is kept;
``on_error="raise"`` propagates the error. The result's status is derived from
the stages and the responses, as for a search.

.. code-block:: python

   from scikitplot.corpus import (
       CharBudgetCompressor,
       RetrievalConfig,
       RetrievalExecutionPlan,
       StageBudget,
       evaluate_retrieval,
   )

   plan = RetrievalExecutionPlan(
       retrieval=RetrievalConfig(match_mode="hybrid", top_k=20),
       transform=lambda q: [q + " definition"],  # run with the original query
       rerank=my_cross_encoder,  # (query, hits) -> one score per hit
       rerank_depth=20,
       top_k=10,
       compress=CharBudgetCompressor(4000),
       budgets={"retrieve": StageBudget(max_calls=4, max_seconds=5.0)},
       deadline_seconds=10.0,  # no stage starts after 10 s
   )
   result = plan.run(index, "what is a span?", embed_query=embedder.embed_query)
   result.status, result.stage("rerank").seconds, result.passages
   report = evaluate_retrieval(plan.bind(index, embed_query=embedder.embed_query), queries, qrels)

A plan is measured by the same harness as an index. On the pinned corpus of
the chunking table above (``recursive (500)``, keyword search, 82 queries made
of the first half of a guide sentence), with a deterministic reranker that
scores the longest run of query words a chunk contains. The queries are
verbatim prefixes of the guide, so that reranker is close to an oracle: the
table shows the stages compose and are measured, not what a learned reranker
gains. The multi-query plan runs the half sentence and its two halves:

==========================  ========  ========  ========
Plan                        Recall@1  Recall@5  MRR@10
==========================  ========  ========  ========
search                      0.91      0.94      0.93
empty plan                  0.91      0.94      0.93
rerank                      0.94      0.94      0.94
multi-query (and halves)    0.90      0.94      0.92
==========================  ========  ========  ========

With whole sentences as queries no stage changes these numbers: every miss
there is a sentence that no chunk contains, which no ranking stage can
recover. ``CharBudgetCompressor(1500)`` kept on average 1492 characters; the
context held the sentence for 0.94 of the queries, against 0.95 whose ten hits
held it: a budget drops the evidence that ranks too low to fit. No stage is a
default.

Corrective retrieval
--------------------

:class:`CorrectiveRetrieval` grades what a search returned and decides what to
do with it: ``correct`` evidence is ``accepted`` (``corrected`` when a later
round's is); ``incorrect`` or ``ambiguous``
evidence is retried with a rewritten query while rounds remain
(``max_rounds``, the call budget, and ``deadline_seconds``, checked before
each rewrite and again before each further round); when no round is accepted, the ambiguous rounds are fused by the
same :class:`FusionPolicy` and returned as ``partial`` evidence, or, when
there are none, the retrieval ``abstained`` and returns no hit. Every round
is graded against the original query, so a rewrite that drifts from it is
not accepted for it. Each round is one ``index.search`` (or one
:class:`RetrievalExecutionPlan` run, with ``plan=``), and the result reads
like a search response, so ``evaluate_retrieval(corrective.bind(index), ...)``
measures it.

.. code-block:: python

   from scikitplot.corpus import CorrectiveRetrieval, TermCoverageGrader

   corrective = CorrectiveRetrieval(
       grader=TermCoverageGrader(top_k=1, accept=0.9, reject=0.6),
       rewrite=my_rewriter,  # (query, rounds) -> next query, or None
       max_rounds=3,
   )
   result = corrective.run(index, "what keeps a chunk's span?")
   result.verdict, [r.grade.grade for r in result.rounds], list(result)

:class:`TermCoverageGrader` is a deterministic rule: the share of the query's
words that the top hits contain, against two thresholds. It measures overlap,
not truth: a hit can hold every word and not answer the question. A model
grader is any callable ``grader(query, hits) -> GradeResult``, and its name
is recorded with each grade. On the corpus of the tables above, with the
82 half-sentence queries (answerable), the same queries with three words of
an unrelated page appended (noisy), and 14 half sentences of the Levenshtein
guide, which is not indexed (off-corpus: the right answer is none):

==========================  ==========  =====  ===========  ==========
Retrieval                   Answerable  Noisy  Noisy        Off-corpus
                            Recall@1    R@1    answered     abstained
==========================  ==========  =====  ===========  ==========
search                      0.91        0.91   1.00         0.00
graded, top 3, 0.8 / 0.3    0.91        0.91   1.00         0.00
graded, top 1, 0.9 / 0.6    0.91        0.82   0.89         0.71
graded, top 1, 1.0 / 0.8    0.91        0.29   0.33         0.79
==========================  ==========  =====  ===========  ==========

The default grader reads three hits, where some hit holds most words of any
query, so it never abstains. Reading the first hit only, a strict grader
abstains on most off-corpus queries and keeps every answerable one, at the
price of noisy queries it no longer answers. Retrying with the halves of the
query (up to three rounds) did not help: noisy recall went from 0.91 to 0.88
with the default grader, from 0.82 to 0.80 with the middle one (and
off-corpus abstention from 0.71 to 0.57), and barely changed with the
strictest (0.29 to 0.30), because a half query finds partial evidence that is
then returned. The rewrite, not the loop, decides whether correction helps;
the benchmark is ``maintenances/corpus/_maintenance/evidence/bench_r5b.py``.
The grader by default is ``TermCoverageGrader()``, the table's second row
(``graded, top 3, 0.8 / 0.3``), which never abstains here; no rewrite is a
default.
