:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bootstrap-confidence-intervals-cis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Bootstrap Confidence Intervals (CIs)</b></div>`

======================================
Bootstrap Confidence Intervals (CIs)
======================================

*Interval estimates built by resampling the data with replacement and recomputing the statistic many times.*

What it is
----------

A **bootstrap confidence interval** estimates the uncertainty of a statistic — a
mean, median, regression coefficient, even an AUROC — by **resampling the data**
rather than relying on a parametric formula. The idea: when the population
distribution is unknown, approximate it by drawing from the sample you already have.
It shines when sample sizes are small, the data are non-normal, or no clean
standard-error formula exists.

The procedure
-------------

1. Start with the original sample of size :math:`n`.
2. **Resample with replacement** to build :math:`B` bootstrap samples (often
   :math:`B = 1000` or more), each of size :math:`n`.
3. Compute the statistic on each bootstrap sample.
4. The spread of those :math:`B` values is the **bootstrap distribution** of the
   statistic.
5. Read a confidence interval off that distribution.

Three ways to build the interval
--------------------------------

- **Percentile** — take the :math:`\alpha/2` and :math:`1-\alpha/2` quantiles
  directly (a 95% CI is the 2.5th–97.5th percentiles).
- **Basic (reverse percentile)** — reflect the percentile interval around the
  observed statistic to correct simple bias.
- **BCa (bias-corrected and accelerated)** — adjusts for both bias and skew in the
  bootstrap distribution; usually the most accurate and the default recommendation.

Worked example
--------------

For :math:`X = [5, 7, 9, 10, 12, 8, 6, 7, 9, 11]` the mean is 8.4. Draw 1000
resamples, take each mean, and read the 2.5th and 97.5th percentiles — about 7.2 and
9.6 — giving a **95% CI of [7.2, 9.6]**.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(42)
   boot = [rng.choice(X, size=len(X), replace=True).mean() for _ in range(1000)]
   lo, hi = np.percentile(boot, [2.5, 97.5])

Pitfalls and edge cases
-----------------------

- **Too few resamples** — small :math:`B` makes the interval itself noisy; prefer
  thousands.
- **A bad sample stays bad** — the bootstrap can only resample what you have; a tiny
  or unrepresentative sample yields a confident-looking but wrong interval.
- **Dependent data break it** — for time series or grouped data the plain bootstrap
  destroys the dependence structure; use a **block bootstrap** instead.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Subsampling <001-subsampling>` · :doc:`Probability <025-probability>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bootstrap Confidence Intervals (CIs) <https://insightful-data-lab.com/2025/08/30/bootstrap-confidence-intervals-cis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
