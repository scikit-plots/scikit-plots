:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-quantile-regression:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Quantile Regression</b></div>`

=====================
Quantile Regression
=====================

*Regression that estimates conditional quantiles rather than the mean.*

What it is
----------

**Quantile regression** estimates a **conditional quantile** of the target instead of its mean: for a
chosen level :math:`\tau \in (0, 1)` it predicts the :math:`\tau`-th quantile given the features. Fit
every level and you recover the **inverse CDF** — the whole conditional distribution.

The pinball loss and τ
----------------------

It minimizes the **pinball loss**, which weights over- and under-prediction **asymmetrically** by
:math:`\tau`:

.. math::

   \ell_\tau(y, \hat{y}) =
   \begin{cases}
     \tau\,(y - \hat{y}) & y \ge \hat{y}, \\[2pt]
     (1 - \tau)(\hat{y} - y) & y < \hat{y}.
   \end{cases}

At :math:`\tau = 0.5` this is symmetric and recovers the **median** (equivalent to minimizing MAE);
:math:`\tau < 0.5` pushes the model to **under-predict**, :math:`\tau > 0.5` to **over-predict**, and the
further :math:`\tau` is from 0.5 the stronger the asymmetry.

In practice
-----------

It is **distribution-free** and robust (built on absolute differences), but fits each quantile
**separately**, which can cause **quantile crossing** (a lower quantile predicted above a higher one)
unless constrained. Common estimators: the linear ``QuantileRegressor``, gradient-boosted quantile
models, and quantile random forests.

.. code-block:: python

   from sklearn.linear_model import QuantileRegressor

   lower = QuantileRegressor(quantile=0.05, alpha=0.0).fit(X_train, y_train)
   upper = QuantileRegressor(quantile=0.95, alpha=0.0).fit(X_train, y_train)
   # [lower.predict(X), upper.predict(X)] is a 90% prediction interval

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`R² (R-squared) <259-r2-r-squared>`

----

.. hint::
   **More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Return Distribution <225-return-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Quantile Regression <https://insightful-data-lab.com/2025/08/22/quantile-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
