:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-ks-statistic-kolmogorovsmirnov-statistic:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>KS Statistic (Kolmogorov–Smirnov Statistic)</b></div>`

=============================================
KS Statistic (Kolmogorov–Smirnov Statistic)
=============================================

*The maximum gap between two cumulative distributions; measures separation or shift.*

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

The **KS statistic** measures the **maximum difference between two cumulative distribution functions
(CDFs)**. In statistics it underpins the **KS test** — do two samples come from the same
distribution? In ML, especially credit scoring and binary classification, it measures how well a
model **separates positives from negatives**: far-apart distributions give a large KS, heavily
overlapping ones a small KS.

The formula
-----------

For a binary problem, build the CDF of positives and the CDF of negatives across the score, then take
the largest vertical gap between them:

.. math::

   KS = \max_x \left| F_{\text{pos}}(x) - F_{\text{neg}}(x) \right|,

where :math:`F` is a cumulative distribution. Because it scans every threshold for the single point
of greatest separation, KS is **threshold-independent**.

A worked example
----------------

Score 1,000 customers — 500 good, 500 bad — and sort by predicted score. At each threshold track the
share of bads captured against the share of goods captured; KS is the largest gap between those
curves. If at score 0.65 the CDF of bads is 0.70 and the CDF of goods is 0.30, the gap is 0.40, so
**KS = 40%**.

Reading it, and its cousin AUC
--------------------------------

KS runs from **0 to 1**: 0 means no separating power (identical distributions), **0.4-0.6** is strong
(typical in credit risk), below 0.2 is weak. It is closely related to **AUC** — both measure class
separability and both are threshold-independent — but where AUC integrates the whole ROC curve, KS
reports only the **single maximum point of separation**.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `KS Statistic (Kolmogorov–Smirnov Statistic) <https://insightful-data-lab.com/2025/08/23/ks-statistic-kolmogorov-smirnov-statistic/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
