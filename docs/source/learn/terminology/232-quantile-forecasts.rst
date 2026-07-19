:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-quantile-forecasts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Quantile Forecasts</b></div>`

====================
Quantile Forecasts
====================

*Forecasts of specific quantiles of the outcome distribution.*

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

A **quantile forecast** expresses the prediction not as one number but as one or more **conditional
quantiles** of the future value's distribution — for example the 10th, 50th and 90th percentiles. The
0.5 level is the **median**; a pair like 0.1 / 0.9 brackets the outcome, saying it is unlikely (:math:`\le`
10%) to fall below the lower or above the upper. Formally the :math:`\tau`-quantile forecast satisfies

.. math::

   \Pr\!\left(Y \le \hat{y}_\tau \mid X\right) = \tau.

Why use them
------------

They convey **uncertainty and asymmetry** directly: you can act differently when under- and over-shooting
carry different costs, **without** assuming a parametric (e.g. Gaussian) shape for the distribution.

How they're scored
------------------

Quantile forecasts are evaluated with the **pinball (quantile) loss**, matched to each level; the
**CRPS** (Continuous Ranked Probability Score) generalizes it by integrating the pinball loss across
**all** quantiles, giving a single distributional score.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Point Forecasts <233-point-forecasts>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Quantile Forecasts <https://insightful-data-lab.com/2025/08/23/quantile-forecasts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
