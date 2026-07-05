.. _dpa-forward-selection-definition-and-core-idea:

========================================================================
Forward Selection: Definition and Core Idea
========================================================================

**Stage 5 · 📈 Regression**  ·  Lesson 34 of 56  ·  *intermediate*

:doc:`◀ Previous · Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`   ·   :doc:`Next · Forward Selection and Model Interpretation in Linear Regression <35-forward-selection-and-model-interpretation-in-linear-regression> ▶`


Too many features
-------------------

A dataset often offers **more candidate predictors than you should use**. Throwing every feature into
a regression risks **overfitting**, obscures which variables truly matter, and can break down entirely
when features outnumber observations. **Feature selection** chooses a smaller, better subset — and
**forward selection** is the most intuitive way to do it.

Start empty, add the best
---------------------------

Forward selection **builds the model up from nothing**. Begin with the **null model** — no predictors,
just the intercept. Then, at each step, try adding **each** remaining feature in turn and keep the
**single one** that improves the model most, by a chosen **criterion** (the largest drop in a score
like **AIC**, or the most significant feature by **p-value**). Add it, then repeat: search the
features not yet in the model, and again admit the best.

When to stop
--------------

The process halts when **no remaining feature is worth adding** — when the best candidate fails to
clear the entry criterion (say, its p-value exceeds a threshold like 0.05, or it no longer lowers
AIC). Because it starts empty and grows, forward selection can handle **very wide** data — even more
candidate features than data points — since it never has to fit the full model at once.

Greedy, not exhaustive
------------------------

One honest limitation: forward selection is **greedy**. It makes the **locally** best choice at each
step and never reconsiders, so it does **not** guarantee finding the **globally** best subset — a
feature that shines only in combination with another might never be picked. Checking every possible
subset would be exact but explodes combinatorially (echoing the Apriori scale problem). Forward
selection trades that guarantee for speed and simplicity; the next lessons apply and extend it.

.. seealso::

   **Related lessons:** :doc:`Forward Selection and Model Interpretation in Linear Regression <35-forward-selection-and-model-interpretation-in-linear-regression>`  ·  :doc:`Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`  ·  :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`  ·  :doc:`Forward Selection with Nested Models and Deviance Tests <42-forward-selection-with-nested-models-and-deviance-tests>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/16/forward-selection-definition-and-core-idea/ <https://insightful-data-lab.com/2026/01/16/forward-selection-definition-and-core-idea/>`__

.. tags:: purpose: reference, topic: data preparation, level: intermediate
