:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probability-forecasts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Probability Forecasts</b></div>`

=======================
Probability Forecasts
=======================

*Forecasts stated as probabilities of events.*

What it is
----------

A **probability forecast** states the **probability of a specific outcome or event** rather than a single
deterministic value — "a 70% chance of rain," "a 4% probability of default." It is the event-focused face of
**probabilistic forecasting**: instead of committing to one number, it quantifies **how likely** each
outcome is.

How it's expressed
------------------

As a probability per event, or as **certainty levels** on a distribution — a **P80** forecast is 80% certain
(a 20% chance of being exceeded), a **P50** is the median. A full probability forecast carries the **whole
distribution** of outcomes.

Why it matters
--------------

Probabilities are what **risk-based decisions** consume — a threshold on the forecast (escalate if
:math:`P(\text{default}) > 5\%`) turns uncertainty into action. They must be **calibrated** to be trusted,
and are scored with **strictly proper scoring rules** (e.g. the log score), which reward honest
probabilities.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probability Forecasts <https://insightful-data-lab.com/2025/08/22/probability-forecasts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
