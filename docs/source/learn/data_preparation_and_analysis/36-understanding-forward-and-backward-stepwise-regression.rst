.. _dpa-understanding-forward-and-backward-stepwise-regression:

========================================================================
Understanding Forward and Backward Stepwise Regression
========================================================================

**Stage 5 · 📈 Regression**  ·  Lesson 36 of 56  ·  *intermediate*

:doc:`◀ Previous · Forward Selection and Model Interpretation in Linear Regression <35-forward-selection-and-model-interpretation-in-linear-regression>`   ·   :doc:`Next · How Shapley Values Work ▶ <37-how-shapley-values-work>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Three directions
------------------

Forward selection is one of **three** stepwise strategies, distinguished by the **direction** they
move. **Forward** starts empty and **adds**; **backward elimination** starts full and **removes**;
**bidirectional** does **both** at every step. All three share the same goal — a parsimonious model —
and the same criteria (p-values, AIC, BIC, adjusted :math:`R^2`), differing only in how they search.

Backward elimination
----------------------

**Backward elimination** works in reverse. Begin with the **full model** containing **all** candidate
predictors, then repeatedly drop the **least useful** one — the feature with the **highest p-value**
(least significant), or whose removal most improves the criterion — until every remaining feature
earns its place. Its advantage is that it weighs all variables **together** from the start, which can
handle **correlated** predictors more gracefully than forward selection. Its cost: it must fit the full
model, so it needs **more observations than features**.

Bidirectional stepwise
------------------------

**Bidirectional** (or plain "stepwise") selection **combines** the two. At each step it can **add** a
promising feature the way forward does, but also **re-examine** features already included and **drop**
any that have become redundant now that others are present. This flexibility corrects a weakness of
pure forward selection, where a feature admitted early can never be removed even if later additions
make it unnecessary.

Use with caution
------------------

All three are **greedy** — they explore only a sliver of the possible models and offer **no
guarantee** of the best subset. And all carry real hazards: on small samples they **overfit**, they
produce **biased** coefficient estimates, and the selected model can be **non-reproducible** — a
different sample yields a different set. Use them as **exploratory** tools when candidates are many and
theory is thin, always confirming the final model on held-out data. When you can, methods that assess
a feature's contribution more fairly — like the **Shapley values** of the next lesson — sidestep some
of these pitfalls.

.. hint::

   **Related lessons:** :doc:`Forward Selection: Definition and Core Idea <34-forward-selection-definition-and-core-idea>`  ·  :doc:`Forward Selection and Model Interpretation in Linear Regression <35-forward-selection-and-model-interpretation-in-linear-regression>`  ·  :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`  ·  :doc:`Forward Selection with Nested Models and Deviance Tests <42-forward-selection-with-nested-models-and-deviance-tests>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/understanding-forward-and-backward-stepwise-regression/ <https://insightful-data-lab.com/2026/01/16/understanding-forward-and-backward-stepwise-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
