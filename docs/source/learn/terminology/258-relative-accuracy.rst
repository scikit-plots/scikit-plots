:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-relative-accuracy:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Relative accuracy</b></div>`

===================
Relative accuracy
===================

*Forecast accuracy measured against a baseline rather than in absolute terms.*

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

**Relative accuracy** measures forecast accuracy **relative to a benchmark** rather than in absolute
units. You **normalize** the error by a reference method's error — usually the **naïve** (or
seasonal-naïve) forecast — so results land on a **common, scale-free** scale comparable across series:

.. math::

   \text{relative error} = \frac{\mathrm{MAE}_{\text{model}}}{\mathrm{MAE}_{\text{benchmark}}}.

How to read it
--------------

A value **< 1** means the model **beats** the benchmark, **= 1** means it **matches** it, and **> 1**
means it is **worse** — a relative error of 0.6 is roughly **40% better** than the benchmark. This family
includes **MASE**, **Theil's U** (:math:`<1` beats a naïve guess), and relative / bounded relative
absolute errors.

Why it matters
--------------

Absolute errors like MAE and RMSE are **meaningless without a reference** — *is an MAE of 10 good?*
depends entirely on the problem — whereas relative accuracy is **interpretable** and puts easy and
hard-to-forecast series on equal footing.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Forecast Error <250-forecast-error>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`R² (R-squared) <259-r2-r-squared>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Relative accuracy <https://insightful-data-lab.com/2025/08/22/relative-accuracy/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
