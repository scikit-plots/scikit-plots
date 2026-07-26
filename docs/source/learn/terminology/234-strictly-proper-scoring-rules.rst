:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-strictly-proper-scoring-rules:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Strictly Proper Scoring Rules</b></div>`

===============================
Strictly Proper Scoring Rules
===============================

*Scoring rules minimised only by reporting the true probability distribution.*

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

A **scoring rule** assigns a numerical score to a **probabilistic forecast** given the outcome that
materializes. It is **proper** if the forecaster's *expected* score is maximized by reporting the **true**
distribution, and **strictly proper** if that maximum is **unique** — attained *only* at the truth:

.. math::

   S(p, q) \;\le\; S(q, q) \quad \text{for all } p, q, \qquad \text{with equality} \iff p = q.

Why it matters
--------------

Strict propriety makes **honesty optimal**: a forecaster cannot improve the expected score by hedging or
shading probabilities away from their true beliefs. That single property is why these rules serve **both**
as **training objectives** (to calibrate probabilistic models) and as **evaluation metrics** (to rank
forecasts fairly). Foundational reference: Gneiting & Raftery (2007).

Common examples
---------------

The **logarithmic score**, which is also **local** (it depends only on the density assigned to what
actually happened):

.. math::

   S(q, x) = \log q(x) \quad\text{(its negative, } -\log q(x)\text{, is the log loss / NLL).}

Others include the **Brier score** for probabilities and the **CRPS** and **pinball (quantile) loss** for
full distributions and quantiles.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Forecast Error <250-forecast-error>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Strictly Proper Scoring Rules <https://insightful-data-lab.com/2025/08/22/strictly-proper-scoring-rules/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
