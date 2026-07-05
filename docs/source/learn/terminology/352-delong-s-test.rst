:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-delong-s-test:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>DeLong's Test</b></div>`

===============
DeLong's Test
===============

*A statistical test comparing the AUCs of two models.*

What it is
----------

**DeLong's test** is a **non-parametric statistical test** for comparing the **ROC-AUCs of two
correlated models** — answering *is model A's AUC significantly better than model B's?* Proposed by
DeLong, DeLong & Clarke-Pearson (1988), it matters because two models are usually evaluated on the
**same test set**, so their AUCs are **correlated** and their difference must be judged with a p-value,
not by eye.

Why you need it
---------------

AUCs are **random variables** — they depend on the sample — so a raw ``AUC_A - AUC_B`` gap could be
noise. DeLong's test supplies the **standard error** of each AUC, the SE of their **difference**, and a
**z-statistic and p-value**.

How it works
------------

It treats AUC as the probability that a random positive outranks a random negative, and uses
**U-statistics** to estimate the variances and covariances of those pairwise comparisons, giving

.. math::

   z = \frac{\text{AUC}_1 - \text{AUC}_2}{\text{SE}(\text{AUC}_1 - \text{AUC}_2)},

which is referred to the standard normal for a p-value.

A worked example
----------------

Model A scores **AUC = 0.88** and Model B **0.84**. The difference of **0.04** with an SE of **0.015**
gives :math:`z \approx 2.67` and :math:`p \approx 0.0076` — so A is **significantly better**
(p < 0.01).

.. code-block:: python

   # delong_roc_test is a small helper (from a gist/package)
   from delong import delong_roc_test

   # y_true = true labels; y_pred1, y_pred2 = the two models' scores
   p_value = delong_roc_test(y_true, y_pred1, y_pred2)
   print("p-value:", p_value)

Alternatives
------------

Other options include **bootstrap confidence intervals** (resample and recompute the AUC-difference
distribution), the older, less accurate **Hanley-McNeil** approximation, and **permutation tests**
(shuffle labels to build the null).

----

**Mind map — connected ideas**

   :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `DeLong's Test <https://insightful-data-lab.com/2025/08/20/delongs-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
