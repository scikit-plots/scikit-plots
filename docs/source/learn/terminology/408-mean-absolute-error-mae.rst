:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-mean-absolute-error-mae:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Mean Absolute Error (MAE)</b></div>`

===========================
Mean Absolute Error (MAE)
===========================

*The average absolute difference between predictions and actuals.*

What it is
----------

**Mean Absolute Error** is the average **absolute** gap between prediction and truth — the **L1** error:

.. math::

   \text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|.

It reports the typical error in the **same units** as the target, with no squaring.

How it behaves
--------------

Because it takes **absolute** values rather than squares, MAE weights **all** errors **linearly** and is far
more **robust to outliers** than MSE / RMSE — one huge miss doesn't dominate. The forecast that minimizes MAE
is the **median** of the target (for RMSE it is the mean).

When to use it
--------------

MAE is the right choice when you want an **interpretable**, outlier-**resistant** measure of typical error and
don't need to punish large mistakes extra hard. Its main limits: it is **scale-dependent** (not comparable
across series — use **MASE** for that) and, being **point-only**, it can't score probabilistic forecasts.

----

**Mind map — connected ideas**

   :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Mean Absolute Error (MAE) <https://insightful-data-lab.com/2025/08/19/mean-absolute-error-mae/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
