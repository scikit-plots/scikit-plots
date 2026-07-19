:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probabilistic-scoring:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Probabilistic Scoring</b></div>`

=======================
Probabilistic Scoring
=======================

*Evaluating forecasts by how well their predicted probabilities match outcomes.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Probabilistic scoring** evaluates a **probabilistic forecast** — a whole predictive distribution —
against the outcome that occurs, using a **scoring rule** (a loss function for distributions). To be
trustworthy the rule should be **strictly proper**, so a forecaster **minimizes** the expected score
*only* by reporting their true distribution.

The workhorse: CRPS
-------------------

The **Continuous Ranked Probability Score** is the most-used score for real-valued forecasts. It
integrates the **squared gap** between the forecast CDF :math:`F` and the step CDF of the observation
:math:`y`, and is **negatively oriented** (lower is better):

.. math::

   \mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(x) - \mathbb{1}\{x \ge y\}\big)^2 \, dx.

It **generalizes the MAE** (for point forecasts) and the **Brier score** (for binary ones), reducing to
them in those cases.

What it captures
----------------

A proper score rewards both **calibration** (probabilities match reality) and **sharpness** (tight
distributions) — the CRPS in fact **decomposes** into calibration, discrimination and uncertainty parts.
The **log score** is a **local** alternative that looks only at the density assigned to the outcome.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>` · :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Quantile Forecasts <232-quantile-forecasts>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probabilistic Scoring <https://insightful-data-lab.com/2025/08/23/probabilistic-scoring/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
