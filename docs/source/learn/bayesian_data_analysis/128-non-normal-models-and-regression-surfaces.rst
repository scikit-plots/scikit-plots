.. _bda-non-normal-models-and-regression-surfaces:

========================================================================
Non-normal models and regression surfaces
========================================================================

**Part 5 · Stage 15 · 🌊 Basis Functions & Gaussian Processes**  ·  Lesson 128 of 144  ·  *advanced*

:doc:`◀ Previous · Basis selection and shrinkage of coeﬃcients <127-basis-selection-and-shrinkage-of-coefficients>`   ·   :doc:`Next · Gaussian process regression ▶ <129-gaussian-process-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Flexibility in two directions at once
---------------------------------------

Basis expansions handle a nonlinear function of **one** predictor. Two generalisations complete the
flexible-regression picture: non-normal **outcomes** with a flexible predictor, and flexible functions of
**several** predictors — a regression **surface** rather than a curve.

Flexible GLMs
--------------

The basis idea composes with the generalized linear model. Keep the link and the non-normal likelihood,
but let the linear predictor be a **spline** instead of a straight line — a logistic regression whose
log-odds is a smooth function of a continuous covariate, a Poisson whose log-rate bends:

.. math::

   g\bigl(\mathrm{E}[y_i]\bigr) = \sum_k \beta_k B_k(x_i),

with the smoothing prior of the previous lesson on the :math:`\beta_k`. Binary and count outcomes gain
flexible dose-response shapes without abandoning the GLM machinery.

Regression surfaces
---------------------

For a function of several predictors, one-dimensional bases combine. **Tensor-product** splines build a
multivariate basis from the products of univariate ones; **additive models** keep things tractable by
summing separate smooth functions, :math:`f(x_1, x_2) = f_1(x_1) + f_2(x_2)`, sacrificing interactions
for interpretability and far fewer parameters.

.. math::

   \mathrm{E}[y] = f_1(x_1) + f_2(x_2) + \cdots, \qquad f_j \text{ each a smooth spline.}

.. code-block:: python

   import numpy as np, pymc as pm
   # additive model: a smooth term per predictor, each with its own smoothing scale
   with pm.Model():
       contributions = []
       for j, Bj in enumerate(basis_matrices):             # one B-spline basis per predictor
           tau_j = pm.HalfNormal(f"tau_{j}", 1)
           z = pm.Normal(f"z_{j}", 0, 1, shape=Bj.shape[1])
           contributions.append(Bj @ (pm.math.cumsum(z) * tau_j))
       eta = sum(contributions)
       pm.Bernoulli("y", logit_p=eta, observed=y)           # flexible additive logistic

The curse, and the bridge
---------------------------

Flexibility in many dimensions meets the **curse of dimensionality**: a full tensor-product basis grows
exponentially with the number of predictors, so a surface over ten inputs is hopeless by basis expansion
alone. Additive models dodge it by forbidding interactions; the truly general tool handles arbitrary
smooth surfaces **without** an explicit growing basis — the **Gaussian process**, which specifies
smoothness through a covariance function over inputs and is the subject of the next lessons. Basis methods
take flexible regression a long way; Gaussian processes take the same idea to its infinite-dimensional
conclusion.

.. hint::

   **Related lessons:** :doc:`Splines and weighted sums of basis functions <126-splines-and-weighted-sums-of-basis-functions>`  ·  :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Gaussian process regression <129-gaussian-process-regression>`  ·  :doc:`Basis selection and shrinkage of coeﬃcients <127-basis-selection-and-shrinkage-of-coefficients>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/non-normal-models-and-regression-surfaces/ <https://insightful-data-lab.com/2025/12/09/non-normal-models-and-regression-surfaces/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
