:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-mase-mean-absolute-scaled-error:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>MASE (Mean Absolute Scaled Error)</b></div>`

===================================
MASE (Mean Absolute Scaled Error)
===================================

*Forecast error scaled by a naive baseline's error, comparable across series.*

What it is
----------

**Mean Absolute Scaled Error** divides a forecast's **mean absolute error** by the MAE of an **in-sample
naive** benchmark — the seasonal-naive or last-value forecast — giving a pure ratio:

.. math::

   \text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}.

It answers a single question: *did the model beat the trivial baseline?*

Reading it
----------

**MASE < 1** means the forecast **outperforms** naive; **= 1** ties it; **> 1** means the naive forecast
**wins** and the model should be reconsidered. Because numerator and denominator share **units**, MASE is
**scale-free** and comparable across series of wildly different magnitudes.

Why it's the gold standard
--------------------------

Unlike percentage errors, MASE is **symmetric** (over- and under-forecasts penalized equally), **robust** to
**zeros** and outliers (the naive step is bounded away from zero unless the series is constant), and
**interpretable**. Proposed by Hyndman & Koehler (2006), it is a default for forecasting **competitions** and
multi-SKU demand.

----

**Mind map — connected ideas**

   :doc:`WMAPE (Weighted Mean Absolute Percentage Error) <405-wmape-weighted-mean-absolute-percentage-error>` · :doc:`sMAPE (Symmetric Mean Absolute Percentage Error) <406-smape-symmetric-mean-absolute-percentage-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Forecasting Competitions <251-forecasting-competitions>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `MASE (Mean Absolute Scaled Error) <https://insightful-data-lab.com/2025/08/19/mase-mean-absolute-scaled-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
