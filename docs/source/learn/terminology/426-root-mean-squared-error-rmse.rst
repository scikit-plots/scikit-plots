:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-root-mean-squared-error-rmse:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Root Mean Squared Error (RMSE)</b></div>`

================================
Root Mean Squared Error (RMSE)
================================

*The square root of mean squared error, in the target's units.*

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

**Root mean squared error** is the **square root** of the MSE:

.. math::

   \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2}.

The root returns the error to the target's **original units**, so it reads as a **typical error magnitude**.

How it behaves
--------------

RMSE keeps MSE's heavy **penalty on large errors** and its **outlier sensitivity**, but is far more
**interpretable** — an RMSE of 5 means predictions are off by about **5 units** on average. It ranks models
**identically** to MSE, is always **≥ the MAE**, and the RMSE–MAE gap widens as the **error variance** grows.

When to use it
--------------

Report RMSE for regression when **large errors are costly** and you want a number in the data's units; pair
it with **R²** for a scale-free complement. Like R², it also **falls** as you add variables, so watch
**overfitting**.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`R² (R-squared) <259-r2-r-squared>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Regression Models <309-regression-models>` · :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Root Mean Squared Error (RMSE) <https://insightful-data-lab.com/2025/08/17/root-mean-squared-error-rmse/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
