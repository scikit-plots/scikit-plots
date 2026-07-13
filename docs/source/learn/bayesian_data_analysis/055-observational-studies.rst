.. _bda-observational-studies:

========================================================================
Observational studies
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 055 of 144  ·  *intermediate*

:doc:`◀ Previous · Sensitivity and the role of randomization <054-sensitivity-and-the-role-of-randomization>`   ·   :doc:`Next · Censoring and truncation ▶ <056-censoring-and-truncation>`   ·   :doc:`↑ Section <index>`


When you cannot randomise
---------------------------

Most questions cannot be settled by an experiment: you cannot randomise people to smoke, to be poor, or
to live near a highway. In an **observational study** the treatment was chosen — by the units, by
doctors, by circumstance — and the assignment mechanism is **unknown**. Everything then depends on
whether ignorability can be recovered by conditioning.

Ignorability conditional on covariates
----------------------------------------

The working assumption is that treatment is as good as random **given the observed covariates**:

.. math::

   \Pr\bigl(W \mid y(0), y(1), X\bigr) = \Pr(W \mid X)
   \qquad \text{(``no unmeasured confounding'')}.

This is the MAR condition transposed to causal inference. Its opposite — assignment depending on the
unobserved potential outcomes — is the MNAR case, and it is **not testable**: two datasets identical in
:math:`(y_{\text{obs}}, W, X)` are consistent with both. No amount of data settles it; only knowledge of
*why* treatment was assigned can.

Two routes to conditioning
----------------------------

Given the assumption, the estimand is recovered by conditioning on :math:`X` — either by **modelling the
outcome** :math:`p(y \mid W, X)` and averaging its predictions over the covariate distribution, or by
modelling the **propensity** :math:`e(x_i) = \Pr(W_i = 1 \mid x_i)` and comparing units of like
propensity.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       tau  = pm.Normal("tau", 0, 1)
       pm.Normal("y", X @ beta + tau * W, pm.HalfNormal("s", 1), observed=y)
       idata = pm.sample()
   # ATE = posterior mean of predictions with W=1 minus W=0, averaged over the sample

Bayesian practice tends to favour the outcome model — it uses the posterior directly, propagates
uncertainty, and permits hierarchical structure — while treating the propensity as a **diagnostic**:
where the treated and untreated propensity distributions barely overlap, the comparison is an
extrapolation, and no model can rescue it.

Which covariates?
-------------------

Include the variables that **affect assignment**, and the pre-treatment variables that affect the
outcome. But conditioning is not universally benign: adjusting for a variable on the causal pathway
*between* treatment and outcome removes part of the effect you want, and adjusting for a **collider**
can create bias where none existed. Covariate choice requires **causal reasoning about the mechanism**,
not a search over predictors — and the final report should say, plainly, how strong an unmeasured
confounder would need to be to change the conclusion.

.. hint::

   **Related lessons:** :doc:`Sensitivity and the role of randomization <054-sensitivity-and-the-role-of-randomization>`  ·  :doc:`Designed experiments <053-designed-experiments>`  ·  :doc:`Regression for causal inference: incumbency and voting <093-regression-for-causal-inference-incumbency-and-voting>`  ·  :doc:`Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/observational-studies/ <https://insightful-data-lab.com/2025/11/11/observational-studies/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
