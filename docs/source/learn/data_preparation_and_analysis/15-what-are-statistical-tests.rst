.. _dpa-what-are-statistical-tests:

========================================================================
What Are Statistical Tests?
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 15 of 56  ·  *beginner*

:doc:`◀ Previous · Harald Cramér <14-harald-cramer>`   ·   :doc:`Next · Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova> ▶`


Is it real or chance?
-----------------------

When you measure an association in a sample, one question always lurks: **is it real, or could it be
chance?** A correlation of 0.2 in a handful of taxi trips might vanish in the next batch. **Statistical
tests** answer this — they quantify how likely your finding is to have arisen by luck alone.

Null and alternative
----------------------

A test pits two hypotheses against each other. The **null hypothesis** (:math:`H_0`) is the sceptic's
position: **no effect, no association** — any pattern seen is chance. The **alternative**
(:math:`H_1`) is the claim you are investigating: a real effect exists. The test computes a **test
statistic** from the data, measuring how far the sample departs from what :math:`H_0` would predict.

The p-value
-------------

That departure is summarised in a **p-value**: the probability of seeing data **at least as extreme**
as yours **if the null hypothesis were true**. A **small** p-value means your result would be
surprising under "pure chance", so chance is an unconvincing explanation; a **large** one means the
data is unremarkable and the null stands. You compare it to a chosen threshold, the **significance
level** :math:`\alpha` (commonly 0.05), and **reject** :math:`H_0` when :math:`p < \alpha`.

Reading the result
--------------------

Two cautions make tests trustworthy. Rejecting a **true** null is a **false positive** (a Type I
error, at rate :math:`\alpha`); failing to detect a **real** effect is a **false negative** (Type II).
And **significance is not importance** — with enough data a trivially small effect becomes
"significant", which is why the next lesson pairs tests with a measure of **effect size**. The
chi-square, t-test and ANOVA F-test are all instances of this one logic.

.. seealso::

   **Related lessons:** :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova>`  ·  :doc:`Karl Pearson <13-karl-pearson>`  ·  :doc:`Forward Selection with Nested Models and Deviance Tests <42-forward-selection-with-nested-models-and-deviance-tests>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/14/what-are-statistical-tests/ <https://insightful-data-lab.com/2026/01/14/what-are-statistical-tests/>`__

.. tags:: purpose: reference, topic: data preparation, level: beginner
