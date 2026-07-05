:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-prediction-intervals-pi:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📉&nbsp;&nbsp;<b>Prediction Intervals (PI)</b></div>`

===========================
Prediction Intervals (PI)
===========================

*A range expected to contain the outcome with a stated probability.*

What it is
----------

A **prediction interval** is a range :math:`[\,\text{lower},\ \text{upper}\,]` expected to contain the
**future observation** with a stated probability — its **nominal coverage**, e.g. 90%. It is typically
built from a **pair of quantiles**, the :math:`(1-\alpha)/2` and :math:`(1+\alpha)/2` levels (the 5th and
95th percentiles for 90% coverage):

.. math::

   \big[\, \hat{Q}_{(1-\alpha)/2},\ \ \hat{Q}_{(1+\alpha)/2} \,\big]
   \quad\Rightarrow\quad \text{nominal coverage } 1-\alpha.

Coverage vs width
-----------------

Quality trades **coverage** — does the empirical fraction of actuals landing inside match the nominal
level? — against **width / sharpness** — narrower is more useful, but only if coverage holds. (A PI is
about a *future value*, distinct from a **confidence interval**, which is about a *parameter*.)

Calibrating them
----------------

**Conformal prediction** is a model-agnostic wrapper that adjusts interval width on a held-out
**calibration** set to guarantee the target coverage in **finite samples**, under mild assumptions.

----

**Mind map — connected ideas**

   :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>`

----

**More in Risk & Probabilistic Forecasting**

   :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Deterministic forecasts <242-deterministic-forecasts>` · :doc:`Full Distribution <229-full-distribution>` · :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Predicting Percentiles <252-predicting-percentiles>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Probabilistic Scoring <228-probabilistic-scoring>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Quantile Forecasts <232-quantile-forecasts>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Return Distribution <225-return-distribution>`

----

*Theme:* :ref:`Risk & Probabilistic Forecasting <term-theme-risk>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Prediction Intervals (PI) <https://insightful-data-lab.com/2025/08/22/prediction-intervals-pi/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
