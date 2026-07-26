:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-mean-absolute-percentage-error-mape:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Mean Absolute Percentage Error (MAPE)</b></div>`

=======================================
Mean Absolute Percentage Error (MAPE)
=======================================

*The average absolute error expressed as a percentage of actuals.*

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

**Mean absolute percentage error** expresses each error as a **percentage of the actual** value, averaged:

.. math::

   \text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|.

This makes it **scale-free** — comparable across series of wildly different magnitudes.

Why it's popular
----------------

"8% off" is instantly meaningful to non-specialists and lets you compare accuracy across products or regions
on different scales — hence its ubiquity in **demand forecasting** and business reporting.

The pitfalls
------------

MAPE **explodes when actuals are zero or near-zero** (the denominator → 0), and it is **asymmetric** —
over-forecasts can incur unbounded percentage error while under-forecasts are capped at 100%, biasing it
toward models that **under-predict**. For intermittent or zero-heavy data, **scaled** errors like **MASE**
are safer.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Relative accuracy <258-relative-accuracy>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`R² (R-squared) <259-r2-r-squared>` · :doc:`Regression Models <309-regression-models>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Mean Absolute Percentage Error (MAPE) <https://insightful-data-lab.com/2025/08/17/mean-absolute-percentage-error-mape/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
