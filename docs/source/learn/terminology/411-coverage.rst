:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-coverage:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Coverage</b></div>`

==========
Coverage
==========

*The share of outcomes that fall within predicted intervals.*

What it is
----------

**Coverage** measures how much of the **catalog** a recommender actually uses — the share of available items
it is able to, or chooses to, recommend. A system can be accurate yet only ever surface a **handful** of
popular items, ignoring the rest.

Two flavors
-----------

**Prediction coverage** is the fraction of items for which the model **can** make a prediction at all;
**catalog coverage** is the fraction of items that actually **appear** in the recommendation lists users see.
The latter is the usual beyond-accuracy target.

Why it matters
--------------

High coverage means the **long tail** gets exposure and the catalog isn't wasted — countering **popularity
bias**. A limitation of plain coverage: it counts an item shown **once** the same as one shown **thousands**
of times, which is why **Gini** and **entropy** refine it to capture how **evenly** exposure is spread.

----

**Mind map — connected ideas**

   :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>`

----

**More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Coverage <https://insightful-data-lab.com/2025/08/19/coverage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
