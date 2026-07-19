:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-pinball-loss-a-k-a-quantile-loss:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Pinball Loss (a.k.a. Quantile Loss)</b></div>`

=====================================
Pinball Loss (a.k.a. Quantile Loss)
=====================================

*The loss minimised by an accurate quantile forecast.*

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

**Pinball loss** (a.k.a. **quantile loss**) scores a **quantile** forecast by penalizing errors
**asymmetrically** — under- and over-prediction get different weights set by the target quantile
:math:`\tau`:

.. math::

   L_\tau(y, \hat{y}) = \max\big(\tau(y - \hat{y}),\ (\tau - 1)(y - \hat{y})\big).

Minimizing it makes :math:`\hat{y}` approach the true :math:`\tau`-quantile.

Why asymmetry
-------------

For a high quantile (say :math:`\tau = 0.9`), **under**-predicting is penalized far more than over-predicting,
pushing the forecast **up** to cover the upper tail — exactly what you want for a 90% **prediction interval**.
At :math:`\tau = 0.5` the two weights match and pinball loss reduces to (half) the **MAE**.

Where it's used
---------------

It trains and evaluates **quantile regressors** and probabilistic models that output **intervals** rather than
points, without assuming any distribution. A caveat: fitting several quantiles independently can cause
**quantile crossing**, where a lower quantile's forecast exceeds a higher one's.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Pinball Loss (a.k.a. Quantile Loss) <https://insightful-data-lab.com/2025/08/19/pinball-loss-a-k-a-quantile-loss/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
