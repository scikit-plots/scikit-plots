:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probabilistic-forecasts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Probabilistic Forecasts</b></div>`

=========================
Probabilistic Forecasts
=========================

*Forecasts that quantify uncertainty as a full probability distribution.*

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

A **probabilistic forecast** produces a **full predictive distribution** — a range of possible outcomes
together with their probabilities — rather than a single value. By **quantifying uncertainty**, it lets
decision-makers weigh risk instead of betting on one number.

Forms and quality
-----------------

It can be expressed as **prediction intervals**, **quantiles**, a **density**, or **samples**. Quality is
judged on two axes: **calibration** — do the stated probabilities match observed frequencies? — and
**sharpness** — are the intervals as **tight** as possible *subject to* being calibrated? Sharp but
miscalibrated is misleading; calibrated but diffuse is uninformative.

Why it matters
--------------

Probabilistic forecasts support **decisions under uncertainty** — hedging, safety stock, capacity
planning — especially where the **cost of outcomes is asymmetric**. Narrow bands signal agreement; wide
spreads flag where more flexibility or hedging is needed. A point estimate alone cannot convey this.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probabilistic Forecasts <https://insightful-data-lab.com/2025/08/22/probabilistic-forecasts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
