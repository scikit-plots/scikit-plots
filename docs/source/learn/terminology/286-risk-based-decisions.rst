:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-risk-based-decisions:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Risk-Based Decisions</b></div>`

======================
Risk-Based Decisions
======================

*Choosing actions by weighing outcome probabilities against their costs.*

What it is
----------

A **risk-based decision** chooses the action that best balances **predicted probabilities against costs** —
it acts on **expected risk**, not on the raw label. The basic definition of risk is the **expected cost**: a
loss weighted by its probability,

.. math::

   R = \mathbb{E}[L] = \sum_i L_i \, p_i.

How it works
------------

Under **Bayesian decision theory**, you pick the action that **minimizes total expected risk** given the
**cost** of each error. Because false positives and false negatives usually cost **differently**, the
optimal **decision threshold** on a probability is generally **not 0.5** — a cheap-to-check,
expensive-to-miss event (fraud, disease) warrants a **lower** threshold.

Why calibration matters
-----------------------

The whole scheme assumes the probabilities are **honest** — an **overconfident** model makes the
expected-cost arithmetic wrong and triggers bad actions. So risk-based decisions rest on **calibrated**
probability forecasts.

----

**Mind map — connected ideas**

   :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Overconfident <284-overconfident>` · :doc:`Likelihood <304-likelihood>` · :doc:`Binary Classification <293-binary-classification>`

----

**More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>`

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Risk-Based Decisions <https://insightful-data-lab.com/2025/08/21/risk-based-decisions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
