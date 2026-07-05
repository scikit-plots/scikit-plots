:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-wape-weighted-absolute-percentage-error:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>WAPE (Weighted Absolute Percentage Error)</b></div>`

===========================================
WAPE (Weighted Absolute Percentage Error)
===========================================

*Total absolute error divided by total actuals.*

What it is
----------

**Weighted Absolute Percentage Error** divides the **total** absolute forecast error by the **total** actual
demand:

.. math::

   \text{WAPE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}.

It expresses aggregate error as a single percentage, weighting each item by its **volume**.

Why retailers use it
--------------------

Unlike **MAPE**, WAPE doesn't **blow up** when individual actuals are near **zero** (common for slow-moving
SKUs), and it lets **high-volume** items dominate — reflecting real demand. It stays defined as long as total
demand isn't zero, which makes it the **default** accuracy metric in demand planning.

How it connects
---------------

WAPE is effectively the same quantity as **WMAPE**, and it is the number that **drives inventory** — high WAPE
means forecasts are far off, feeding both **overstock** and **stockouts**. Cutting WAPE is how planners
**shrink** those costly failure modes.

----

**Mind map — connected ideas**

   :doc:`WMAPE (Weighted Mean Absolute Percentage Error) <405-wmape-weighted-mean-absolute-percentage-error>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Overstock % <400-overstock>` · :doc:`Stockouts <401-stockouts>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `WAPE (Weighted Absolute Percentage Error) <https://insightful-data-lab.com/2025/08/17/wape-weighted-absolute-percentage-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
