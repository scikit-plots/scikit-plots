.. currentmodule:: scikitplot.rank_bm25

.. _rank-bm25-calibration:

Score calibration
=================

A BM25 score orders documents for one query. It is not a probability: its scale
changes with the corpus, the query length and the recipe, so "0.8" or "12.4"
cannot be compared across queries, thresholded, or combined with another
retriever's score. :class:`ScoreCalibrator` learns a monotone map from scores
to ``P(relevant | score)`` from labelled (query, document) pairs.

.. code-block:: python

   from scikitplot.corpus import calibration_report
   from scikitplot.rank_bm25 import ScoreCalibrator

   cal = ScoreCalibrator("sigmoid").fit(
       train_scores, train_labels,
       query_ids=train_query_ids,
       population="bm25-top50",          # what the labelled pairs are
   )
   cal.require_held_out(test_query_ids)  # raises if a test query was trained on
   probs = cal.predict_proba(test_scores)
   calibration_report(test_labels, probs)["expected_calibration_error"]

Methods
-------

``"sigmoid"`` (default)
    Platt scaling, ``sigmoid(a * score + b)``: two parameters, works with few
    labels, strictly increasing, so the ranking is unchanged. Fitted to Platt's
    smoothed targets by default (``platt_targets=True``), which keeps the fit
    finite when the labels are perfectly separated by the scores.
``"isotonic"``
    The best non-decreasing step fit: more flexible, needs more labels, and can
    map different scores to the same probability, which merges their ranks.

Both agree with scikit-learn's sigmoid calibration and
``IsotonicRegression(out_of_bounds="clip")``; scikit-learn is not needed.

What the calibrator refuses
---------------------------

* **Evaluating on training queries.** ``fit`` requires ``query_ids`` and keeps
  their SHA-256 digests (never the text); :meth:`ScoreCalibrator.require_held_out`
  raises when an evaluation query was used to fit. Pairs of one query are
  correlated, so a random split of pairs leaks.
* **An undeclared population.** ``population`` names what the labelled pairs
  were (the top 50 BM25 candidates, the rerank pool, ...). The share of relevant
  documents differs between populations, and so do the probabilities.
* **A reversing fit.** A sigmoid slope below zero (or zero on varying scores)
  means higher scores are not more often relevant on these labels; the fit
  raises instead of returning a map that would reverse the ranking.
* **One class only**, non-binary labels and non-finite scores.

There is no unfitted transform: a score passed through a fixed sigmoid is not a
probability, and offering one would invite treating it as one.

Moving to another population
----------------------------

:meth:`ScoreCalibrator.shift_prior` returns a copy for a population whose share
of relevant pairs is known to differ, by adding
``logit(new rate) - logit(training prevalence)`` to every log-odds. This is
exact when only the mix of relevant and non-relevant pairs changes, as in a
subsample; it is not a substitute for refitting when the candidate generator,
the recipe or the analyzer changes.

Measured example
----------------

A synthetic corpus of 8 000 documents in 40 topics, 400 three-term queries,
relevance = same topic, population = BM25 top 50 per query, fitted on 200
queries and evaluated on the other 200 (10 000 pairs, prevalence 0.88):

=====================  =====  =====  ========
probability            ECE    Brier  log loss
=====================  =====  =====  ========
score / query maximum  0.237  0.163  0.503
sigmoid                0.014  0.101  0.346
isotonic               0.009  0.101  0.345
=====================  =====  =====  ========

Dividing by the best score of each query is a common shortcut; it orders
documents the same way but its values are far from relevance frequencies.

Persistence
-----------

``to_dict()`` returns a JSON-serialisable state and ``ScoreCalibrator.from_dict``
restores it; predictions after a round trip are identical. Store the calibrator
with the index identity it was fitted for, and refit when that identity changes.
