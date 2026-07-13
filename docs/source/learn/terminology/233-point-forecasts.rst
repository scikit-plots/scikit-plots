:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-point-forecasts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Point Forecasts</b></div>`

=================
Point Forecasts
=================

*A single best-estimate prediction, without an uncertainty range.*

What it is
----------

A **point forecast** gives a **single predicted value** for each future period — the **deterministic**
forecast, with all the outcome probability mass placed on one number. Historically the dominant approach
because it is **easy to interpret and act on**: a demand at 5 p.m., a price in the day-ahead market.

What it hides
-------------

A point forecast says nothing about **uncertainty** — two forecasts can share the *same* point estimate
yet imply very different **risk**. The value reported is usually a **summary** of an underlying
predictive distribution (typically its **mean** or **median**). As the saying goes, *it is better to be
vaguely right than exactly wrong*.

Scoring and context
-------------------

Point forecasts are scored against the realized value with **error metrics** — MAE, MSE / RMSE — which
reward closeness but ignore calibration. Probabilistic forecasting does not *eliminate* point forecasts;
it **places them in context** as one functional (mean, median, a quantile) of the full distribution.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Time Series Forecasting <256-time-series-forecasting>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Point Forecasts <https://insightful-data-lab.com/2025/08/23/point-forecasts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
