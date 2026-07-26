:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-wmape-weighted-mean-absolute-percentage-error:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>WMAPE (Weighted Mean Absolute Percentage Error)</b></div>`

=================================================
WMAPE (Weighted Mean Absolute Percentage Error)
=================================================

*MAPE weighted by volume so large items count more.*

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

**Weighted Mean Absolute Percentage Error** divides the **total** absolute error by the **total** actual
demand — the sum of errors over the sum of actuals:

.. math::

   \text{WMAPE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}.

Rather than averaging per-item percentages, it weights each error by its **volume**.

Why weighting matters
---------------------

Plain MAPE treats a 50% miss on a **tiny** item the same as on a **huge** one and blows up when actuals are
near **zero**. WMAPE lets **high-volume** items dominate — reflecting real **business impact** — and stays
defined as long as total demand isn't zero, making it a **retail** and demand-planning staple.

Its trade-off
-------------

Because big items dominate, WMAPE can **hide** poor accuracy on the **long tail** of small items — a model can
score well while badly missing many low-volume SKUs. It is closely related to **WAPE**, and best read
**alongside** a per-item metric to catch tail errors.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`sMAPE (Symmetric Mean Absolute Percentage Error) <406-smape-symmetric-mean-absolute-percentage-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`WAPE (Weighted Absolute Percentage Error) <422-wape-weighted-absolute-percentage-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `WMAPE (Weighted Mean Absolute Percentage Error) <https://insightful-data-lab.com/2025/08/19/wmape-weighted-mean-absolute-percentage-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
