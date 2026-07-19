:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-r2-r-squared:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>R² (R-squared)</b></div>`

================
R² (R-squared)
================

*The share of variance in the target explained by a regression model.*

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

**R²**, the **coefficient of determination**, is the **proportion of variance** in the target that the
model explains. It compares the model's **residual** error to the variance around the **mean**:

.. math::

   R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}, \qquad
   SS_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2, \quad SS_{\text{tot}} = \sum_i (y_i - \bar{y})^2.

How to read it
--------------

It usually runs **0 to 1** — **1** is a perfect fit, **0** means the model does no better than predicting
the **mean** (and it can go **negative** for a model worse than that). Being **dimensionless**, it
complements MAE / RMSE, which report error in the target's units; in simple regression it equals
:math:`r^2`, the squared **Pearson correlation**.

Caveats
-------

R² **never decreases** when predictors are added (even noise), so use **adjusted R²** to compare models
of different size; it is sensitive to **outliers**, assumes the modeled relationship, and a high value
implies **neither causation nor good out-of-sample** performance.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Forecast Error <250-forecast-error>` · :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Relative accuracy <258-relative-accuracy>` · :doc:`Regression Coefficient <090-regression-coefficient>` · :doc:`Target Variable <236-target-variable>` · :doc:`Point Forecasts <233-point-forecasts>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `R² (R-squared) <https://insightful-data-lab.com/2025/08/22/r%c2%b2-r-squared/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
