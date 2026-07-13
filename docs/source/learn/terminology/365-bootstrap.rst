:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bootstrap:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Bootstrap</b></div>`

===========
Bootstrap
===========

*Resampling with replacement to estimate the variability of a statistic.*

What it is
----------

The **bootstrap** is a resampling method that draws new samples **with replacement** from the observed data
— each resample the same size as the original, with some points repeated and others omitted. From many such
resamples it estimates a statistic's **sampling distribution**.

What it's for
-------------

By recomputing a statistic (a mean, an AUC) across hundreds or thousands of bootstrap resamples, you get its
**standard error** and **confidence intervals** **without** assuming a formula or a distribution. That makes
it a flexible, **non-parametric** way to quantify **uncertainty**.

Where it appears
----------------

The same idea powers **bagging** (bootstrap aggregating) and **random forests**, which train each model on a
different bootstrap sample to reduce variance. Its main cost is **compute** — many refits — and it can
struggle with very small samples or extreme statistics.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Upsampling <367-upsampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bootstrap <https://insightful-data-lab.com/2025/08/20/bootstrap/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
