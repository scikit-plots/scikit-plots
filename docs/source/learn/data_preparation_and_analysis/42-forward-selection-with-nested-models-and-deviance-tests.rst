.. _dpa-forward-selection-with-nested-models-and-deviance-tests:

========================================================================
Forward Selection with Nested Models and Deviance Tests
========================================================================

**Stage 6 · 🎯 Classification & Logistic Regression**  ·  Lesson 42 of 56  ·  *advanced*

:doc:`◀ Previous · Complete and Quasi-Complete Separation in Logistic Regression <41-complete-and-quasi-complete-separation-in-logistic-regression>`   ·   :doc:`Next · Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention ▶ <43-interpreting-and-assessing-a-forward-selection-logistic-regression-model-for-college-student-retention>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Selecting features by deviance
--------------------------------

The **deviance** — that :math:`-2` log-likelihood measure of misfit — does more than grade a single
model. Its real power is in **comparing** models, and it is the natural tool for **feature selection**
in logistic regression, playing the role that the F-test and residual sum of squares play in linear
regression.

Nested models
---------------

The comparison requires the models to be **nested**: one model's predictors must be a **subset** of the
other's, so the smaller (reduced) model is a special case of the larger (full) one. Adding predictors
can only **lower** the deviance (improve the fit on the training data), so the full model **always** has
deviance less than or equal to the reduced model. The question is whether that improvement is **real**
or just the inevitable reward of extra parameters.

The deviance test
-------------------

The answer is the **likelihood-ratio (deviance) test**. The **difference** in deviance between the
reduced and full models is itself a statistic that, under the hypothesis that the extra features add
nothing, follows a **chi-square** distribution — with degrees of freedom equal to the **number of
added parameters**:

.. math::

   \Delta D = D_{\text{reduced}} - D_{\text{full}} \sim \chi^2_{\,k}.

A **large** deviance drop is unlikely by chance, so a significant test means the added features
**genuinely improve** the model; a small, non-significant drop means they can be dropped. This
procedure is often called an **analysis of deviance**, the logistic cousin of analysis of variance.

Forward selection, revisited
------------------------------

This gives forward selection a principled **entry rule** for logistic models. Start from the null
model and, at each step, consider adding each remaining feature; admit the one whose deviance drop is
**largest and significant** by the chi-square test. Stop when no candidate produces a significant
improvement. It is the same greedy search as before, now driven by **deviance** rather than an
F-statistic — and with the same caution that data-driven selection inflates significance, so the final
model earns its keep only on **held-out** data. The next lesson works a full example on real data.

.. hint::

   **Related lessons:** :doc:`Assessing Model Fit in Logistic Regression <40-assessing-model-fit-in-logistic-regression>`  ·  :doc:`Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`  ·  :doc:`Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`  ·  :doc:`Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention <43-interpreting-and-assessing-a-forward-selection-logistic-regression-model-for-college-student-retention>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/forward-selection-with-nested-models-and-deviance-tests/ <https://insightful-data-lab.com/2026/01/16/forward-selection-with-nested-models-and-deviance-tests/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced
