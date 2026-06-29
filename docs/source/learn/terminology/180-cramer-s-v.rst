:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-cramer-s-v:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Cramér's V</b></div>`

============
Cramér's V
============

*A measure of association between two categorical variables, derived from chi-squared.*

What it is
----------

**Cramér's V** measures the **strength of association between two categorical variables**. Built on
the **chi-square statistic**, it **normalises** the result to lie between **0 and 1**: 0 means the
variables are completely independent, 1 means one fully determines the other.

The formula
-----------

.. math::

   V = \sqrt{\frac{\chi^2}{n \,(k - 1)}},

where :math:`\chi^2` is the chi-square statistic, :math:`n` the total sample size, and :math:`k` the
smaller of the number of rows and columns in the contingency table. Dividing by :math:`k - 1` is
what keeps :math:`V` bounded regardless of table size.

Reading the number
------------------

A rough guide: **0.0-0.1** very weak, **0.1-0.3** weak, **0.3-0.5** moderate, and **above 0.5**
strong — though exact thresholds vary by field. Surveying 1,000 people on gender (male/female) and
drink preference (coffee/tea), a computed **V = 0.25** signals a weak-to-moderate link between the
two.

Where it's used
---------------

In data science it does three jobs: detecting **categorical drift** by comparing distributions over
time, flagging **redundant features** (two categoricals so strongly associated that one can be
dropped), and testing **feature-target association** in classification.

----

**Mind map — connected ideas**

   :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Data Drift <331-data-drift>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Categorical Explosions <182-categorical-explosions>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Cramér's V <https://insightful-data-lab.com/2025/08/23/cramers-v/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
