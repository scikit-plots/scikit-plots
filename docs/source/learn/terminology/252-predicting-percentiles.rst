:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-predicting-percentiles:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Predicting Percentiles</b></div>`

========================
Predicting Percentiles
========================

*Forecasting specific percentiles to convey the outcome distribution.*

What it is
----------

**Predicting percentiles** means forecasting specific **percentiles** (quantiles) of the outcome
distribution — the value below which a given **percentage** of outcomes fall — instead of only a single
mean. A percentile is a quantile stated as a percent: the 0.9 quantile is the **90th percentile**.

Why percentiles
---------------

A handful of percentiles (say the **10th, 50th and 90th**) sketch the **range** of outcomes and their
**best- and worst-case** scenarios, exposing **uncertainty** and enabling **asymmetric** decisions —
without committing to a parametric distribution.

How it's done
-------------

Percentiles are produced by **quantile regression** (and its tree / boosting variants), each trained on
the **pinball loss** for its level; stacking many percentiles approximates the **full distribution**.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Point Forecasts <233-point-forecasts>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Predicting Percentiles <https://insightful-data-lab.com/2025/08/22/predicting-percentiles/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
