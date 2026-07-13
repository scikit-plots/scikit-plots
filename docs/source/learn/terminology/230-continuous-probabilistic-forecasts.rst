:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-continuous-probabilistic-forecasts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Continuous Probabilistic Forecasts</b></div>`

====================================
Continuous Probabilistic Forecasts
====================================

*Forecasts expressed as continuous probability distributions over outcomes.*

What it is
----------

A **continuous probabilistic forecast** is a probabilistic forecast for a **continuous (real-valued)**
outcome — a full **predictive distribution** over the variable (a density or CDF), rather than a single
value or a class probability. It answers *what is the whole distribution of tomorrow's demand, price, or
temperature?*

How it's represented
--------------------

It can be given as a **parametric** distribution (e.g. a normal :math:`\mathcal{N}(\mu, \sigma^2)` with a
forecast mean and variance), a set of **quantiles**, or an **ensemble** of sampled trajectories — each a
way to describe the continuous outcome's uncertainty.

Why it's useful
---------------

From one object it exposes **every** downstream quantity — the mean, any **quantile**, a **prediction
interval**, or a **tail probability**. Because it lives on a continuum (unlike a discrete / categorical
probabilistic forecast), it is scored with the **CRPS**.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Continuous Probabilistic Forecasts <https://insightful-data-lab.com/2025/08/23/continuous-probabilistic-forecasts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
