:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-deterministic-forecasts:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Deterministic forecasts</b></div>`

=========================
Deterministic forecasts
=========================

*Single-valued forecasts that carry no explicit uncertainty.*

What it is
----------

A **deterministic forecast** outputs a **single value** for each future point — a **point estimate** with
**no uncertainty** attached. It is the counterpart of a **probabilistic** forecast, which predicts a whole
distribution; here all the probability sits on **one number**.

What it hides
-------------

Two deterministic forecasts can agree on the number yet face very different **risk**, and the value is
usually a **summary** of an implicit distribution — the **mean** (if fit by minimizing RMSE) or the
**median** (if fit by MAE).

When it's enough
----------------

It is simple to produce, communicate and act on, and fine when uncertainty is small or irrelevant — but
where the **cost of being wrong is asymmetric**, a **probabilistic** or **quantile** forecast conveys far
more.

----

**Mind map — connected ideas**

   :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Full Distribution <229-full-distribution>`

----

**More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Deterministic forecasts <https://insightful-data-lab.com/2025/08/22/deterministic-forecasts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
