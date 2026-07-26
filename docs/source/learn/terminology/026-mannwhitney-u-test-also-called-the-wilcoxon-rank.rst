:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-mannwhitney-u-test-also-called-the-wilcoxon-rank-sum-test:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Mann–Whitney U Test (also called the Wilcoxon rank-sum test)</b></div>`

==============================================================
Mann–Whitney U Test (also called the Wilcoxon rank-sum test)
==============================================================

*A nonparametric, rank-based test of whether one group's values tend to exceed another's.*

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

The **Mann–Whitney U test** (equivalently the **Wilcoxon rank-sum test**) is a
**non-parametric** test for whether two *independent* groups come from the same
distribution. Unlike the t-test it makes **no normality assumption** — it works on
the **ranks** of the pooled data, so it is the natural choice for skewed or ordinal
data.

Assumptions
-----------

- The two samples are **independent**.
- The outcome is **ordinal or continuous**.
- Strictly, it tests *stochastic dominance* (whether one group tends to be larger);
  only when the two distributions have the **same shape** does it become a test of
  **medians**.

The U statistic
---------------

Pool both groups, rank everything, and sum the ranks of each group
(:math:`R_1, R_2`). Then

.. math::

   U_1 = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1, \qquad
   U_2 = n_1 n_2 + \frac{n_2(n_2 + 1)}{2} - R_2, \qquad
   U = \min(U_1, U_2).

For large samples :math:`U` is approximately normal, giving a z-based p-value.

Hypotheses
----------

- **H₀** — the two groups come from the same distribution.
- **H₁** — one group tends to produce larger (or smaller) values than the other.

Worked example
--------------

Group A = {88, 92, 100, 75, 85}, Group B = {60, 70, 65, 80, 72}. Rank all ten
values, sum the ranks, compute :math:`U`, and look up the p-value; :math:`p < 0.05`
means the groups differ significantly.

.. code-block:: python

   from scipy.stats import mannwhitneyu

   U, p = mannwhitneyu(group_a, group_b, alternative="two-sided")

The link to ROC-AUC
-------------------

The U statistic is **mathematically equivalent to AUROC**:

.. math::

   \text{AUC} = \frac{U}{n_1 n_2},

i.e. the probability that a randomly chosen value from one group outranks a randomly
chosen value from the other — exactly the definition of AUROC. Testing whether two
score distributions differ is the same computation as measuring a classifier's
ranking power.

Pitfalls and edge cases
-----------------------

- **Ties** — many tied values need a tie correction (most libraries apply one).
- **The "median" shortcut** — only valid under equal-shape distributions; otherwise
  report it as a test of stochastic dominance.
- **Paired data** — for *matched* samples use the Wilcoxon *signed-rank* test
  instead; this test is for *independent* groups.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Probability <025-probability>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <https://insightful-data-lab.com/2025/08/30/mann-whitney-u-test-also-called-the-wilcoxon-rank-sum-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
