:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-baseline-heuristics:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Baseline Heuristics</b></div>`

=====================
Baseline Heuristics
=====================

*Simple rules used as reference points to judge whether a model adds value.*

What it is
----------

**Baseline heuristics** are the **simple, naive** reference models a real system must **beat** to earn its
complexity — the "**dumb**" benchmark. Predicting the **mean** or **median**, the **majority class**, the
**last value**, or a basic **if-then rule** are all baselines.

Why they're essential
---------------------

A metric is **meaningless** in isolation — 90% accuracy is trivial if the majority class is already 90%. A
baseline sets the **floor**: if a complex model **can't** beat it, the model adds **no value** and may even
hide a **bug**. Baselines are **cheap**, fast, and interpretable, so they cost almost nothing to run.

Where they show up
------------------

Baselines frame every honest **evaluation** and are **built into** metrics — **MASE**, for instance, divides a
forecast's error by a **naive** baseline's, so a score below 1 literally means "**better than the
heuristic**." Always establish the baseline **first**.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>` · :doc:`Accuracy <323-accuracy>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Model Score <364-model-score>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Baseline Heuristics <https://insightful-data-lab.com/2025/08/17/baseline-heuristics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
