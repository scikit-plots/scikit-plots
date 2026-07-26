.. _bda-regularization-and-dimension-reduction:

========================================================================
Regularization and dimension reduction
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 096 of 144  ·  *advanced*

:doc:`◀ Previous · Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>`   ·   :doc:`Next · Unequal variances and correlations ▶ <097-unequal-variances-and-correlations>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When there are too many predictors
------------------------------------

With :math:`k` comparable to or larger than :math:`n`, least squares breaks: :math:`X^{\top}X` is
singular or nearly so, coefficients are wild, and the noninformative posterior is diffuse or improper.
The Bayesian answer is not to delete predictors but to **shrink** them — with a prior that expresses what
you believe about their sizes.

The ridge and lasso, and their limitation
-------------------------------------------

From Stage 4: a :math:`\mathrm{N}(0, \tau^2)` prior on each coefficient gives **ridge** at the mode; a
Laplace prior gives the **lasso**. Both apply the **same** amount of shrinkage to every coefficient,
governed by one global scale. That is exactly wrong when the truth is **sparse**: a single global scale
must either shrink the null coefficients enough (and thereby crush the real signals) or preserve the
signals (and leave noise unshrunk).

Global–local shrinkage: the horseshoe
---------------------------------------

The fix is a scale **per coefficient**, drawn from a heavy-tailed distribution. The **horseshoe** prior
of Carvalho, Polson and Scott:

.. math::

   \beta_j \mid \lambda_j, \tau \sim \mathrm{N}(0, \; \tau^2 \lambda_j^2), \qquad
   \lambda_j \sim \mathrm{C}^{+}(0, 1), \qquad \tau \sim \mathrm{C}^{+}(0, 1).

The **global** scale :math:`\tau` pulls everything toward zero; the **local** scales :math:`\lambda_j`,
with their Cauchy tails, let a genuinely large coefficient **escape** the shrinkage entirely. The result
is aggressive shrinkage of noise and near-zero shrinkage of signal — behaviour the lasso cannot achieve,
and achieved **continuously**, without the discrete mixture of spike-and-slab.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfCauchy("tau", 1)                       # global
       lam = pm.HalfCauchy("lam", 1, shape=k)              # local, heavy-tailed
       z = pm.Normal("z", 0, 1, shape=k)
       beta = pm.Deterministic("beta", z * lam * tau)      # non-centred: the funnel again
       pm.Normal("y", X @ beta, pm.HalfNormal("s", 1), observed=y)

The horseshoe's own pathology
-------------------------------

Those Cauchy tails create an **extreme funnel** in :math:`(\tau, \lambda, \beta)` — the geometry of Stage
9, worse — and NUTS reports divergences even on simple problems. Piironen and Vehtari's **regularized
horseshoe** repairs it: a slab caps how large the escaped coefficients may be, and the prior on
:math:`\tau` is set from a **guess at the number of nonzero coefficients**, turning a vague default into
a statement of sparsity you can defend.

Alternatives and the caveat
-----------------------------

Principal components and other **dimension reduction** replace predictors with fewer combinations,
buying stability with interpretability. Whatever the route, one caution: **regularisation is a prior**,
and a prior distorts causal estimands. Shrink the nuisance coefficients; leave the effect you came to
measure weakly informative, so the data — not the penalty — determine it.

.. hint::

   **Related lessons:** :doc:`Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>`  ·  :doc:`Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>`  ·  :doc:`Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`  ·  :doc:`Basis selection and shrinkage of coeﬃcients <127-basis-selection-and-shrinkage-of-coefficients>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/regularization-and-dimension-reduction/ <https://insightful-data-lab.com/2025/11/23/regularization-and-dimension-reduction/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
