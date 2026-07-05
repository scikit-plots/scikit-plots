:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-smape-symmetric-mean-absolute-percentage-error:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>sMAPE (Symmetric Mean Absolute Percentage Error)</b></div>`

==================================================
sMAPE (Symmetric Mean Absolute Percentage Error)
==================================================

*A symmetric percentage error bounded between 0 and 200%.*

What it is
----------

**Symmetric Mean Absolute Percentage Error** fixes MAPE's asymmetry by putting the **average** of actual and
forecast in the denominator:

.. math::

   \text{sMAPE} = \frac{1}{n}\sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}.

In its common form it is **bounded** between 0% and 200%.

What it fixes (and doesn't)
---------------------------

Plain MAPE penalizes **over-forecasts** more than under-forecasts and explodes as actuals approach zero;
sMAPE is more **balanced** and **bounded**, which is why it served as the official metric of the
**M-competitions**. But it is **not** perfectly symmetric, and it still misbehaves when both actual and
forecast are near **zero** (the error jumps toward 100–200%).

When to use it
--------------

Reach for sMAPE when you want a **bounded**, roughly symmetric percentage error for comparing across series —
but avoid it on **intermittent** or zero-heavy demand, where **MASE** is the safer scale-free choice.

----

**Mind map — connected ideas**

   :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`WMAPE (Weighted Mean Absolute Percentage Error) <405-wmape-weighted-mean-absolute-percentage-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `sMAPE (Symmetric Mean Absolute Percentage Error) <https://insightful-data-lab.com/2025/08/19/smape-symmetric-mean-absolute-percentage-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
