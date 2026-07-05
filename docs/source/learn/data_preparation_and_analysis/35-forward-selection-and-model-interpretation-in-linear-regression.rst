.. _dpa-forward-selection-and-model-interpretation-in-linear-regression:

========================================================================
Forward Selection and Model Interpretation in Linear Regression
========================================================================

**Stage 5 · 📈 Regression**  ·  Lesson 35 of 56  ·  *intermediate*

:doc:`◀ Previous · Forward Selection: Definition and Core Idea <34-forward-selection-definition-and-core-idea>`   ·   :doc:`Next · Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression> ▶`


Building the model
--------------------

Put forward selection to work on a real regression and it produces a **compact, fitted model** — a
handful of predictors, each earning its place. Starting from the intercept alone, the procedure admits
features one at a time until none of the leftovers improve the fit, leaving a **parsimonious** equation
that is far easier to reason about than one stuffed with every available column.

Reading the order of entry
----------------------------

The **order in which features enter** is itself informative. The first variable admitted is the
**single strongest** predictor of the outcome; the second adds the most **on top of** the first, and
so on. This sequence gives a rough **importance ranking** — though a subtle one, because each entry is
judged given those already in, so a feature's rank reflects its **added** value, not its value in
isolation. On the taxi data, distance might enter first, with duration adding power beyond it.

Comparing nested models
-------------------------

Each step yields a slightly larger model **nested** inside the next, which invites comparison. Because
adding any feature can only **increase** ordinary :math:`R^2`, that raw measure always favours the
bigger model and cannot judge whether an addition is worthwhile. Penalised criteria fix this:
**adjusted** :math:`R^2`, **AIC** and **BIC** all reward fit but **charge** for each extra parameter,
so they rise only when a feature earns its complexity. These are the yardsticks forward selection
actually optimises.

Interpret with care
---------------------

Interpret the result **cautiously**. Because selection is **greedy** and driven by the data, the
chosen model can be **unstable** — a slightly different sample might select different features — and
repeatedly testing many features inflates apparent significance, so p-values from the final model read
**optimistically**. The coefficients are still interpreted the usual way (effect per unit, holding
others fixed), but the honest test of the model is its performance on the **held-out** data, not the
selection statistics.

.. seealso::

   **Related lessons:** :doc:`Forward Selection: Definition and Core Idea <34-forward-selection-definition-and-core-idea>`  ·  :doc:`Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`  ·  :doc:`Multiple Linear Regression <32-multiple-linear-regression>`  ·  :doc:`How Shapley Values Work <37-how-shapley-values-work>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/16/forward-selection-and-model-interpretation-in-linear-regression/ <https://insightful-data-lab.com/2026/01/16/forward-selection-and-model-interpretation-in-linear-regression/>`__

.. tags:: purpose: reference, topic: data preparation, level: intermediate
