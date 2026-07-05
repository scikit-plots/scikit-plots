:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-mean-squared-error-mse:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Mean Squared Error (MSE)</b></div>`

==========================
Mean Squared Error (MSE)
==========================

*The average of squared differences between predictions and actuals.*

What it is
----------

**Mean squared error** is the average of the **squared** differences between predictions and truth — the
workhorse **loss** and metric for **regression** (also called **L2** or quadratic loss):

.. math::

   \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2.

Squaring makes every error positive and **weights large errors far more** than small ones.

How it behaves
--------------

Because errors are squared, MSE is dominated by **big misses** and is **sensitive to outliers** — one large
error can swamp many small ones. Its units are the **square** of the target's, so it doesn't read directly;
minimizing MSE yields the **mean** (conditional expectation) as the optimal prediction.

Why it's used
-------------

It is **smooth and differentiable**, ideal for **gradient descent** (it is the loss regression networks
minimize), and it is the **maximum-likelihood** loss under **Gaussian** noise. When outliers should count
less, **MAE** is preferred.

----

**Mind map — connected ideas**

   :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`R² (R-squared) <259-r2-r-squared>` · :doc:`Regression Models <309-regression-models>` · :doc:`Forecast Error <250-forecast-error>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Mean Squared Error (MSE) <https://insightful-data-lab.com/2025/08/21/mean-squared-error-mse/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
