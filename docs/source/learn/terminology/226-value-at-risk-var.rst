:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-value-at-risk-var:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Value-at-Risk (VaR)</b></div>`

=====================
Value-at-Risk (VaR)
=====================

*A threshold loss unlikely to be exceeded at a given confidence over a horizon.*

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

**Value-at-Risk** summarizes downside risk in a single number: the **maximum loss** over a holding period
:math:`h` that will **not be exceeded** with confidence :math:`\alpha` (typically 95% or 99%) — anything
worse occurs only with probability :math:`1 - \alpha`. Formally it is the :math:`\alpha`-**quantile** of
the loss distribution (the negative :math:`\alpha`-quantile of returns):

.. math::

   \mathrm{VaR}_\alpha = -F_r^{-1}(\alpha), \qquad \Pr\!\left(L > \mathrm{VaR}_\alpha\right) = 1 - \alpha,

with :math:`F_r` the return CDF and :math:`L` the loss.

Where it comes from
-------------------

VaR was introduced by J. P. Morgan's **RiskMetrics** (1994) and enshrined by the **Basel** framework for
bank regulatory capital. It is estimated by **historical simulation** (the empirical quantile over a
rolling window), **parametric** methods (assume a normal / t distribution and scale by volatility, often
via **GARCH**), or **Monte Carlo**.

Its blind spot
--------------

VaR says **nothing about how bad** losses beyond the threshold are, and it is **not coherent** — it can
violate **subadditivity**, so a diversified portfolio's VaR may exceed the sum of its parts. **Expected
Shortfall** (CVaR) — the *average* loss **given** VaR is breached — repairs both and is coherent.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Return Distribution <225-return-distribution>` · :doc:`Risk Forecast <227-risk-forecast>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Value-at-Risk (VaR) <https://insightful-data-lab.com/2025/08/23/value-at-risk-var/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
