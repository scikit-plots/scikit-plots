.. _dpa-feature-importance-in-linear-regression:

========================================================================
Feature Importance in Linear Regression
========================================================================

**Stage 5 · 📈 Regression**  ·  Lesson 33 of 56  ·  *intermediate*

:doc:`◀ Previous · Multiple Linear Regression <32-multiple-linear-regression>`   ·   :doc:`Next · Forward Selection: Definition and Core Idea ▶ <34-forward-selection-definition-and-core-idea>`   ·   :doc:`↑ Section <index>`


Which features matter?
------------------------

Once a multiple regression is fitted, a natural question follows: **which predictors matter most?**
Which features do the heavy lifting in explaining the outcome, and which barely register? The
coefficients seem like the obvious answer — bigger coefficient, bigger effect — but read naively, they
**mislead**.

The units trap
----------------

The problem is **units**. A coefficient's size depends on the **scale** of its feature. Suppose fare
rises by **one dollar** per mile and by **one cent** per second travelled; the mile coefficient (1.0)
dwarfs the second coefficient (0.01), yet that reflects the **units**, not the importance — a mile is
simply a bigger step than a second. Comparing raw coefficients across features measured in different
units is comparing apples to oranges.

Standardized coefficients
---------------------------

The fix is to put every feature on the **same scale** before comparing. **Standardising** each
predictor — subtracting its mean and dividing by its standard deviation, so all are measured in
**standard-deviation units** — yields **standardised coefficients** (often called **beta
coefficients**). Now each answers a comparable question: how much does :math:`y` move for a
**one-standard-deviation** change in this feature? Their **magnitudes** can be ranked, giving a genuine
measure of importance. Statistical significance (the **t-statistic** and **p-value** from
``statsmodels``) tells a complementary story: whether an effect is distinguishable from zero at all.

Cautions
----------

Two caveats. **Multicollinearity** — predictors that are themselves strongly correlated — makes
individual coefficients **unstable** and their importances hard to trust, since the model cannot
cleanly separate overlapping features (recall the correlation lessons of Stage 2). And linear-model
importance only captures **linear** effects; a feature with a strong curved relationship may look
unimportant. For a model-agnostic alternative, **permutation importance** measures how much
performance drops when a feature is shuffled — an idea that generalises to the trees and other models
ahead. The next lessons turn to choosing *which* features to include at all.

.. hint::

   **Related lessons:** :doc:`Multiple Linear Regression <32-multiple-linear-regression>`  ·  :doc:`Least Squares Regression <31-least-squares-regression>`  ·  :doc:`How Shapley Values Work <37-how-shapley-values-work>`  ·  :doc:`Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/feature-importance-in-linear-regression/ <https://insightful-data-lab.com/2026/01/16/feature-importance-in-linear-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
