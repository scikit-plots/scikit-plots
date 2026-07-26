:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-full-distribution:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Full Distribution</b></div>`

===================
Full Distribution
===================

*Predicting the entire outcome distribution rather than a single value.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

Forecasting the **full distribution** means predicting the **entire** predictive distribution — the
complete CDF / PDF over all possible outcomes — rather than a **summary** of it. A point forecast
collapses it to one number; a quantile forecast reports a few points; the full distribution keeps
**everything**.

The richest target
------------------

From the full distribution you can **derive any summary** after the fact — the mean, the median, any
**quantile**, a **prediction interval**, the probability of exceeding a threshold, or a risk measure such
as **VaR**. Nothing about the uncertainty is discarded.

How it's judged
---------------

Because it is a whole distribution, it is scored by a rule that reads the **entire shape** against the
outcome — the **CRPS**, which compares the forecast CDF to the observation's step CDF — not a point-error
metric like MAE.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Probability Distribution <240-probability-distribution>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Full Distribution <https://insightful-data-lab.com/2025/08/23/full-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
