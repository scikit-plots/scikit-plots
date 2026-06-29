:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-re-scoring:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Re-scoring</b></div>`

============
Re-scoring
============

*Recomputing model scores on data, e.g. after retraining or recalibration.*

What it is
----------

**Re-scoring** is **adjusting or recomputing a model's scores** — predictions, probabilities
or rankings — *after* the initial output, by layering on extra information, calibration or
rules. It improves **relevance, fairness or calibration without retraining**, and shows up in
ranking, recommendations, search, fraud detection and NLP pipelines.

Where it's used
---------------

Five common settings. In **search and recommendation**, a base ranking is re-scored with
business rules, diversity constraints or personalisation (boost new items, demote
duplicates). In **classification**, raw probabilities are re-scored by **calibration** (Platt
scaling, isotonic regression, Bayesian correction). In **ensembles**, several models' outputs
are combined — a weighted average of fraud scores, say. For **fairness**, re-scoring enforces
guardrails so no group is systematically disadvantaged. And in **NLP reranking**, an N-best
list from an acoustic or translation model is re-scored by a second model (a language model)
to pick the most fluent candidate.

Examples
--------

A fraud model's **0.7** probability is re-scored against the customer's risk profile to
**0.85**; an e-commerce relevance score gets a **+0.2** promotion boost; a speech system's
N-best hypotheses are re-scored by a language model to choose the most fluent sentence.

Benefits and challenges
-----------------------

It buys **accuracy without full retraining**, **flexibility** (rules, fairness) and
**real-time, context-aware** adjustment. The risks: a complex re-scoring stage adds
**latency**, weighting must be tuned to **avoid introducing bias**, and leaning on it too hard
can **paper over** model weaknesses instead of fixing their root cause.

----

**Mind map — connected ideas**

   :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Recalibration <159-recalibration>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>`

----

**More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Re-scoring <https://insightful-data-lab.com/2025/08/24/re-scoring/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
