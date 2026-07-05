:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-continuous-ranked-probability-score-crps:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Continuous Ranked Probability Score (CRPS)</b></div>`

============================================
Continuous Ranked Probability Score (CRPS)
============================================

*A proper score comparing a full predicted distribution to the outcome.*

What it is
----------

The **Continuous Ranked Probability Score** grades a **full probabilistic** forecast by comparing its
predicted **CDF** to the observed outcome — the integrated squared gap between the forecast distribution and a
step at the truth:

.. math::

   \text{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(z) - \mathbb{1}\{y \le z\}\big)^2\, dz.

**Lower** is better, and it rewards mass placed **near** the observation.

Its key properties
------------------

CRPS is a **strictly proper scoring rule** — it is minimized only by **honest**, well-calibrated
distributions, penalizing **overconfidence** — and it reports in the target's **units** (like MAE). It equals
the integral of **pinball loss** over **all** quantiles, tying the whole family together.

How it relates to MAE
---------------------

For a **point** (degenerate) forecast, the predicted CDF becomes a step function and CRPS **collapses to the
MAE**. So CRPS is literally MAE **generalized** to distributions — the natural score for **weather**,
**energy**, and **demand** probabilistic forecasting, though it is **unbounded**.

----

**Mind map — connected ideas**

   :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Brier Score <418-brier-score>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>`

----

**More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Continuous Ranked Probability Score (CRPS) <https://insightful-data-lab.com/2025/08/19/continuous-ranked-probability-score-crps/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
