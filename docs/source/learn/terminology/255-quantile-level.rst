:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-quantile-level:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Quantile Level</b></div>`

================
Quantile Level
================

*The probability level (e.g. 0.9) targeted by a quantile forecast.*

What it is
----------

The **quantile level** :math:`\tau \in (0,1)` (sometimes written :math:`\alpha`) is the probability that
**names which quantile** a forecast targets — the value :math:`q_\tau` below which the outcome is expected
to fall a fraction :math:`\tau` of the time:

.. math::

   \Pr\!\left(Y \le q_\tau\right) = \tau, \qquad \tau \in (0, 1).

Reading levels
--------------

:math:`\tau = 0.5` is the **median**; :math:`\tau = 0.1` is the 10th percentile (a pessimistic lower value
the outcome undercuts 10% of the time); :math:`\tau = 0.9` the 90th (an optimistic upper value). A **set**
of levels :math:`\{0.1, 0.5, 0.9, \dots\}` traces out the whole predictive distribution.

Monotonicity
------------

In **quantile regression** the level is the **tilting parameter** of the pinball loss. Estimated quantiles
should be **monotonically increasing** in :math:`\tau` — when a lower level's forecast exceeds a higher
one's, that is **quantile crossing**, an error to constrain away.

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Quantile Level <https://insightful-data-lab.com/2025/08/22/quantile-level/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
