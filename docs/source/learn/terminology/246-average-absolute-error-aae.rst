:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-average-absolute-error-aae:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Average Absolute Error (AAE)</b></div>`

==============================
Average Absolute Error (AAE)
==============================

*The mean of absolute differences between forecasts and outcomes.*

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

The **average absolute error** is the mean of the **absolute** errors — the average of
:math:`|\text{forecast} - \text{actual}|` across all points. It answers *how big is the typical error?*
in the **same units** as the data, and is identical to the **Mean Absolute Error (MAE)** (often also
called the Mean Absolute Deviation):

.. math::

   \mathrm{AAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|.

Properties
----------

Because it uses **absolute** (not squared) errors, AAE is **robust to outliers** — a few large misses do
not dominate — and it treats over- and under-prediction **symmetrically**, ignoring the *direction* of
error. The forecast that minimizes it is the **median**.

Limitation
----------

AAE is **scale-dependent**: you cannot compare it across series on different scales (an AAE of 10 is tiny
for house prices, huge for temperatures). For that, switch to a **percentage** or **relative** metric.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Forecast Error <250-forecast-error>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Relative accuracy <258-relative-accuracy>` · :doc:`R² (R-squared) <259-r2-r-squared>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Time Series Forecasting <256-time-series-forecasting>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Average Absolute Error (AAE) <https://insightful-data-lab.com/2025/08/22/average-absolute-error-aae/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
