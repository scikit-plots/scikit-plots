:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-rmsle-root-mean-squared-logarithmic-error:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>RMSLE (Root Mean Squared Logarithmic Error)</b></div>`

=============================================
RMSLE (Root Mean Squared Logarithmic Error)
=============================================

*RMSE on log-scaled values, penalising under-prediction and easing large magnitudes.*

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

**Root Mean Squared Logarithmic Error** is RMSE computed on the **logarithms** of the predictions and
actuals — the root-mean-square of the differences in **log space**:

.. math::

   \text{RMSLE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\big(\log(\hat{y}_i + 1) - \log(y_i + 1)\big)^2}.

The **+1** lets it handle zeros.

What the log changes
--------------------

Taking logs turns absolute errors into **relative** ones, so a fixed **percentage** miss costs the same
whether the value is small or huge — making RMSLE **robust to scale** and to large **outliers**. It also
becomes **asymmetric**: it penalizes **under**-prediction more than over-prediction.

When to use it
--------------

RMSLE suits **positive**, **right-skewed** targets that span orders of magnitude — prices, counts, demand —
and situations where **under-forecasting** is the costlier mistake. Its limits: it **can't** take negative
values, and its log scaling makes the raw number **less intuitive** than RMSE.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`WAPE (Weighted Absolute Percentage Error) <422-wape-weighted-absolute-percentage-error>` · :doc:`Outlier <307-outlier>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `RMSLE (Root Mean Squared Logarithmic Error) <https://insightful-data-lab.com/2025/08/19/rmsle-root-mean-squared-logarithmic-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
