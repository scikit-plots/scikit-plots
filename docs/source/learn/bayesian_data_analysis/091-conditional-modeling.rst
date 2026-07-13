.. _bda-conditional-modeling:

========================================================================
Conditional modeling
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 091 of 144  ·  *advanced*

:doc:`◀ Previous · Unknown normalizing factors <090-unknown-normalizing-factors>`   ·   :doc:`Next · Bayesian analysis of classical regression ▶ <092-bayesian-analysis-of-classical-regression>`   ·   :doc:`↑ Section <index>`


Model y given x, not (x, y)
-----------------------------

Almost every applied model is a **regression**: a distribution for an outcome :math:`y` **conditional
on** predictors :math:`x`. The move is so routine that its justification goes unexamined. Why is it
legitimate to model :math:`p(y \mid x, \theta)` and simply **ignore** the distribution of :math:`x`?

The factorisation
-------------------

Write the joint model for everything observed, with parameters :math:`\theta` for the conditional and
:math:`\psi` for the predictors:

.. math::

   p(x, y \mid \theta, \psi) = \underbrace{p(y \mid x, \theta)}_{\text{what you care about}}
                               \; \underbrace{p(x \mid \psi)}_{\text{nuisance}} .

If :math:`\theta` and :math:`\psi` are **distinct** and **a priori independent**, then the second factor
contributes nothing to :math:`p(\theta \mid x, y)`: it cancels in the normalisation. Conditional
modelling is therefore not an approximation but an exact consequence — under exactly the conditions of
the **ignorability** lesson in Stage 7, transposed from data collection to predictors.

When it fails
---------------

The conditions are not automatic, and each failure is a known pathology:

* **Measurement error in** :math:`x` — the observed predictor is not the one the outcome responds to, so
  the parameters are linked and :math:`p(x \mid \psi)` re-enters the likelihood.
* **Missing predictors** — imputing them requires a model for :math:`x`.
* **Selection on** :math:`x` **or** :math:`y` — if inclusion depends on the outcome, the conditional
  likelihood is wrong (Stage 7 again).
* **Shared parameters** — a design in which :math:`\psi` informs :math:`\theta` breaks distinctness.

What conditioning buys
------------------------

You are relieved of modelling the predictors' joint distribution, which is usually complicated,
high-dimensional, and irrelevant to the question. You may then choose :math:`p(y \mid x, \theta)` to
suit the outcome — normal, binomial, Poisson (Stage 13) — and the same posterior machinery applies.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])    # predictors standardised
       sigma = pm.HalfNormal("sigma", 1)
       pm.Normal("y", X @ beta, sigma, observed=y)           # x is conditioned on, not modelled
       idata = pm.sample()

One caution to carry into Part IV: conditioning on :math:`x` makes the model **agnostic about causation**.
:math:`p(y \mid x)` is a statement about association; whether the coefficient is a causal effect depends
on the data-collection lessons, not on the regression.

.. hint::

   **Related lessons:** :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`  ·  :doc:`Goals of regression analysis <094-goals-of-regression-analysis>`  ·  :doc:`Data-collection models and ignorability <051-data-collection-models-and-ignorability>`  ·  :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/conditional-modeling/ <https://insightful-data-lab.com/2025/11/23/conditional-modeling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
