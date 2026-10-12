..
  docs/source/user_guide/corpus/evaluation.rst

.. currentmodule:: scikitplot.corpus

.. _corpus-evaluation:

Evaluating retrieval and answers
================================

Measure a retriever before tuning it. The evaluation layer answers three
questions with one set of formulas:

* *How good is this ranking?* Metric functions such as :func:`recall_at_k`
  and :func:`ndcg_at_k`, over precomputed rankings.
* *How good is this system on my query set?* :func:`evaluate_retrieval` runs a
  retriever, times it, scores it and returns an exportable report.
* *Which configuration is best?* :func:`make_corpus_scorer` turns any metric
  into a scikit-learn scorer for ``cross_validate`` and ``GridSearchCV``.

Inputs
------

Relevance judgments (*qrels*) say which documents answer each query, either as
a collection of relevant identifiers or as graded relevance ``{doc_id:
grade}`` with grades >= 0. A ranking is a sequence of identifiers, best first.

* Identifiers are ``str`` or ``int``, one type per evaluation; mixing them is
  refused, because ``"1"`` and ``1`` would silently never match.
* A ``str`` is never read as a collection of identifiers: wrap a single
  identifier in a list.
* An identifier that is not judged has grade 0 (non-relevant).
* A ranking that repeats an identifier is refused unless
  ``duplicates="first"``, which keeps its best position. Use it when several
  chunks map to one source document.
* A pandas ``Series`` is refused as one query's judgments or ranking: its index
  or its values could be the identifiers, and reading the wrong one gives
  plausible, wrong numbers. Pass ``series.to_dict()`` for ``{doc_id: grade}``
  or ``series.tolist()`` for identifiers. Whole ``y_true``/``y_pred`` series are
  aligned by position, as in scikit-learn, and two series with different
  indexes are refused.

Ranking metrics
---------------

With ``R`` the relevant identifiers (grade >= ``relevance_threshold``,
default 1) and ``top`` the first ``k`` ranked identifiers:

================================  ==============================================
Function (name)                   Per-query value
================================  ==============================================
:func:`precision_at_k`            ``|R & top| / k``
:func:`recall_at_k`               ``|R & top| / |R|``
:func:`hit_rate_at_k`             1 if ``R & top`` is not empty, else 0
:func:`mean_reciprocal_rank`      ``1 / rank`` of the first relevant, else 0
:func:`mean_average_precision`    mean of precision at each relevant position,
                                  divided by ``|R|``
:func:`r_precision`               precision at ``|R|``
:func:`ndcg_at_k`                 ``DCG@k / IDCG@k`` over the grades
================================  ==============================================

.. code-block:: python

   from scikitplot.corpus import ndcg_at_k, ranking_metrics, recall_at_k

   qrels = {"q1": {"d1": 2, "d4": 1}, "q2": ["d3"]}
   runs = {"q1": ["d4", "d2", "d1"], "q2": ["d5", "d3"]}

   recall_at_k(qrels, runs, k=2)          # 0.75
   round(ndcg_at_k(qrels, runs, k=3), 4)  # 0.6956
   ranking_metrics(qrels, runs, metrics=["mrr", "hit_rate@1"])
   # {'mrr': 0.75, 'hit_rate@1': 0.5}

:func:`ranking_metrics` computes several metrics in one validated pass. Pass
``average=None`` for per-query values. Mappings keyed by query identifier and
sequences aligned by position are both accepted.

Undefined values
----------------

A query without a relevant document has no recall, MAP or R-precision
(division by ``|R|``), and no nDCG when no grade is positive. Precision, hit
rate and MRR stay defined (they score 0). ``zero_division`` decides, as in
scikit-learn: ``"warn"`` (default) scores 0.0 and emits
:class:`UndefinedMetricWarning`; ``0.0`` or ``1.0`` score that value;
``np.nan`` leaves the value undefined and excludes it from the mean.

Evaluate a retriever
--------------------

:func:`evaluate_retrieval` accepts anything with ``search(query)``, such as
:class:`RetrievalIndex` or :class:`CustomRetrievalIndex`, or a plain callable.

.. code-block:: python

   from scikitplot.corpus import (
       CorpusDocument,
       RetrievalConfig,
       RetrievalIndex,
       evaluate_retrieval,
   )

   texts = {
       "d1": "hemoglobin carries oxygen in red blood cells",
       "d2": "platelets help blood clot",
       "d3": "white blood cells fight infection",
   }
   documents = [
       CorpusDocument(doc_id=key, input_path=f"memory://{key}", chunk_index=0, text=text)
       for key, text in texts.items()
   ]
   index = RetrievalIndex(RetrievalConfig(match_mode="keyword", top_k=3))
   index.build(documents)

   queries = {"q1": "what carries oxygen", "q2": "blood clotting cells"}
   qrels = {"q1": ["d1"], "q2": ["d2"]}

   report = evaluate_retrieval(index, queries, qrels, metrics=["recall@1", "mrr", "ndcg@3"])
   report.metrics          # {'recall@1': 0.5, 'mrr': 0.666..., 'ndcg@3': 0.75}
   report.status_counts    # {'success': 2}
   report.latency          # mean, p50, p95, max, total (seconds)

A response whose status says a requested leg did not run (a hybrid search
without a query embedding, for example) is a partial ranking. The harness
raises on it by default; ``on_degraded="warn"`` evaluates it anyway and keeps
the statuses in the report.

Reports
-------

:meth:`RetrievalEvaluation.write` picks the format from the suffix:

* ``.json``: the full report, strict JSON (an undefined value is ``null``);
* ``.jsonl``: one row per query, for dataframes and log pipelines;
* ``.yaml`` / ``.yml``: needs PyYAML;
* ``.toml``: needs ``tomli-w`` to write (and ``tomli`` to read before
  Python 3.11). TOML has no null, so ``None`` fields are omitted.

.. code-block:: python

   report.write("baseline.json")
   report.write("baseline.jsonl")
   same = type(report).read("baseline.json")

Compare two systems
-------------------

:meth:`RetrievalEvaluation.compare` pairs two evaluations of the same queries
and reports, per metric, the mean difference, wins, ties, losses and a
two-sided paired sign-flip randomisation p-value (exact up to 16 pairs, seeded
sampling beyond):

.. code-block:: python

   strict = RetrievalIndex(RetrievalConfig(match_mode="strict", top_k=3))
   strict.build(documents)
   candidate = evaluate_retrieval(strict, queries, qrels, metrics=["recall@1", "mrr", "ndcg@3"])
   comparison = candidate.compare(report)
   [(m.metric, m.delta, m.p_value) for m in comparison.metrics]
   # [('recall@1', -0.5, 1.0), ('mrr', -0.666..., 0.5), ('ndcg@3', -0.75, 0.5)]

Two queries cannot show a significant difference: with ``n`` pairs the
smallest exact two-sided p-value is ``2 / 2**n``, here 0.5. Evaluate on a query
set large enough for the difference you need to detect.

Answer metrics
--------------

For generated answers against reference answers (a reference may be a list of
acceptable strings; the best one counts):

* :func:`exact_match_score` and :func:`token_f1_score` use the SQuAD
  normalisation by default (``normalize="squad"``: lower case, ASCII
  punctuation and the English articles removed).
* :func:`anls_score` is the average normalised Levenshtein similarity with a
  0.5 threshold, after ``normalize="basic"`` by default;
  :func:`edit_similarity_score` averages the similarity without a threshold and,
  by default, without normalisation (``normalize=None``). Both use
  :mod:`scikitplot.levenshtein` when it is installed and an exact pure-Python
  fallback otherwise.

.. code-block:: python

   from scikitplot.corpus import anls_score, exact_match_score, token_f1_score

   references = ["The Eiffel Tower", ["Paris", "City of Paris"]]
   answers = ["eiffel tower.", "Paris, France"]
   exact_match_score(references, answers)          # 0.5
   round(token_f1_score(references, answers), 3)   # 0.833
   round(anls_score(references, answers), 3)       # 0.344

scikit-learn scorers
--------------------

:func:`make_corpus_scorer` mirrors :func:`sklearn.metrics.make_scorer`
(``score_func``, ``greater_is_better``, ``**kwargs``), and
:func:`get_corpus_scorer` mirrors :func:`sklearn.metrics.get_scorer` with names
such as ``"ndcg@10"``. Both add two ways to get predictions:

* ``response_method="predict"``: ``estimator.predict(X)`` returns one ranking
  (or answer) per query. From scikit-learn 1.4, a scikit-learn estimator
  shares one ``predict`` call across several scorers; 1.3 calls it once per
  scorer.
* ``response_method="search"``: ``estimator.search(query)`` for each query, so
  a corpus index can be scored directly. :func:`sklearn.metrics.make_scorer`
  cannot do this: its scorer calls ``is_regressor(estimator)``, which needs
  scikit-learn's estimator tags.

.. code-block:: python

   from scikitplot.corpus import get_corpus_scorer, make_corpus_scorer, recall_at_k

   recall = make_corpus_scorer(recall_at_k, response_method="search", k=1)
   recall(index, list(queries.values()), list(qrels.values()))   # 0.5

   get_corpus_scorer("ndcg@3", response_method="search")

Tune a retriever with scikit-learn model selection by giving it ``fit`` and
``predict``:

.. code-block:: python

   from sklearn.base import BaseEstimator
   from sklearn.model_selection import GridSearchCV, KFold

   class KeywordRetriever(BaseEstimator):
       def __init__(self, documents=None, top_k=3):
           self.documents = documents
           self.top_k = top_k

       def fit(self, X, y=None):
           self.index_ = RetrievalIndex(RetrievalConfig(match_mode="keyword", top_k=self.top_k))
           self.index_.build(self.documents)
           return self

       def predict(self, X):
           return [[hit.doc.doc_id for hit in self.index_.search(query)] for query in X]

   search = GridSearchCV(
       KeywordRetriever(documents=documents),
       {"top_k": [1, 2]},
       scoring={"recall@2": get_corpus_scorer("recall@2"), "mrr": get_corpus_scorer("mrr")},
       refit="recall@2",
       cv=KFold(2),
   ).fit(list(queries.values()), list(qrels.values()))
   search.best_params_   # {'top_k': 1}

:func:`scorer_backend_info` reports the base class in use: scikit-learn's own
``_BaseScorer`` when it is installed (checked on 1.3.2 through 1.9.1), and a
vendored copy of the same contract otherwise, so the scorers work in a
``scikit-plots-corpus`` installation without scikit-learn. Metadata routing
(``set_score_request``) needs scikit-learn.

.. seealso::

   * :ref:`corpus-llm-evaluation` to judge generated answers (faithfulness,
     answer relevance, context precision and recall, preference) with one or
     several language models.
   * :ref:`corpus-retrieval-similarity` for the retrievers being measured.
   * :func:`make_pairwise_ranker` and :func:`make_levenshtein_ranker` to build
     a ranker from a pairwise function.
