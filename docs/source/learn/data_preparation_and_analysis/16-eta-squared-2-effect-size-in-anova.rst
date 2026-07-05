.. _dpa-eta-squared-2-effect-size-in-anova:

========================================================================
Eta Squared (η²): Effect Size in ANOVA
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 16 of 56  ·  *beginner*

:doc:`◀ Previous · What Are Statistical Tests? <15-what-are-statistical-tests>`   ·   :doc:`Next · Understanding Market Baskets and Ideal Customers <17-understanding-market-baskets-and-ideal-customers> ▶`


How big, not just whether
---------------------------

A statistical test tells you **whether** a categorical variable relates to a numeric one — do taxi
fares **differ** by payment type? But "significant" does not mean "large". **Eta squared**
(:math:`\eta^2`) answers the other half: **how big** is the effect? It is the **effect size** that
partners the ANOVA test, filling the "continuous ↔ categorical" slot of the association taxonomy.

A number to a category
------------------------

The setting is **ANOVA** (analysis of variance): one **categorical** variable splits the data into
groups, and you ask how much of the spread in a **continuous** variable is explained by which group a
point falls in. ANOVA works by splitting the **total** variability into a **between-group** part
(differences among the group means) and a **within-group** part (scatter inside each group).

The formula
-------------

Eta squared is simply the **between-group** share of the total:

.. math::

   \eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}, \qquad
   SS_{\text{total}} = SS_{\text{between}} + SS_{\text{within}}.

It ranges from :math:`0` (the grouping explains nothing) to :math:`1` (the grouping explains
everything). An :math:`\eta^2` of 0.30 means **30%** of the variance in the numeric variable is
accounted for by the category.

The cousin of r²
------------------

Eta squared is the categorical twin of a familiar quantity: it is analogous to :math:`r^2`, the
**coefficient of determination**. Both report the **proportion of variance explained** in a continuous
variable — :math:`r^2` when the predictor is also continuous (as in regression, Stage 5),
:math:`\eta^2` when the predictor is a category. Seen this way, correlation and ANOVA are two faces of
the same question: **how much does one variable tell you about another?** (In multi-factor models, a
variant called *partial* eta squared isolates one factor's contribution.)

.. seealso::

   **Related lessons:** :doc:`What Are Statistical Tests? <15-what-are-statistical-tests>`  ·  :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`Harald Cramér <14-harald-cramer>`  ·  :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/14/eta-squared-%ce%b7%c2%b2-effect-size-in-anova/ <https://insightful-data-lab.com/2026/01/14/eta-squared-%ce%b7%c2%b2-effect-size-in-anova/>`__

.. tags:: purpose: reference, topic: data preparation, level: beginner
