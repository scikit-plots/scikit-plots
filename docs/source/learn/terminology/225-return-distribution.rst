:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-return-distribution:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Return Distribution</b></div>`

=====================
Return Distribution
=====================

*The distribution of asset returns, central to financial risk modelling.*

What it is
----------

A **return distribution** is the probability distribution of an asset's (or portfolio's) **returns** over
a period — the range of possible returns and how likely each is. From a price series
:math:`(S_0, \dots, S_n)`, the simple return is

.. math::

   R_t = \frac{S_t - S_{t-1}}{S_{t-1}}

(or, equivalently for many purposes, the **log return**).

Why it matters
--------------

Every downside-risk measure is read **off this distribution** — for instance, **Value-at-Risk** is a
tail **quantile** of it. Model the return distribution and you can **price** risk.

The reality: fat tails
----------------------

Empirical returns are **not** normal — they have **heavy tails** and **skewness**, so extreme moves
happen far more often than a Gaussian predicts. Assuming normality **understates** tail risk;
heavy-tailed (**Student-t**) or **location-scale** models fit better.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Value-at-Risk (VaR) <226-value-at-risk-var>` · :doc:`Risk Forecast <227-risk-forecast>` · :doc:`Probability Distribution <240-probability-distribution>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Return Distribution <https://insightful-data-lab.com/2025/08/23/return-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
