:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-risk-forecast:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Risk Forecast</b></div>`

===============
Risk Forecast
===============

*A forward-looking estimate of potential loss or adverse outcomes.*

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

A **risk forecast** predicts a risk measure — most often **VaR** or **Expected Shortfall** — for a
**future** period. Because VaR is a quantile, forecasting it means forecasting the :math:`\tau`-**quantile
of future returns** given today's information; the quantity is **unobserved** and estimated ahead of time.

How it's done
-------------

Methods forecast the future **return distribution** (or just its **scale**): **GARCH**-family volatility
models (forecast the variance, then scale a distributional quantile), **historical simulation**, **Extreme
Value Theory** for the far tail, **quantile regression**, and hybrids of these.

How it's judged
---------------

By **backtesting**: over a long out-of-sample run, the fraction of days the loss **breaches** the forecast
VaR should match the stated level (about 1% of days for 99% VaR). Too many breaches means risk was
**under-forecast**. This discipline is vital for banks, risk managers and regulators.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Value-at-Risk (VaR) <226-value-at-risk-var>` · :doc:`Return Distribution <225-return-distribution>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Forecast Error <250-forecast-error>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Risk Forecast <https://insightful-data-lab.com/2025/08/23/risk-forecast/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
