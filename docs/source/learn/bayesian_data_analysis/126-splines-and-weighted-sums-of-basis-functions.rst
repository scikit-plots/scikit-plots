.. _bda-splines-and-weighted-sums-of-basis-functions:

========================================================================
Splines and weighted sums of basis functions
========================================================================

**Part 5 · Stage 15 · 🌊 Basis Functions & Gaussian Processes**  ·  Lesson 126 of 144  ·  *advanced*

:doc:`◀ Previous · Example: population toxicokinetics <125-example-population-toxicokinetics>`   ·   :doc:`Next · Basis selection and shrinkage of coeﬃcients ▶ <127-basis-selection-and-shrinkage-of-coefficients>`   ·   :doc:`↑ Section <index>`


Flexibility from fixed pieces
-------------------------------

When a relationship is clearly nonlinear but no mechanism dictates its form, model it as a **weighted sum
of basis functions** — simple fixed building blocks whose linear combination can approximate any smooth
curve. The model stays **linear in its coefficients**, so all the regression machinery still applies;
only the design matrix changes.

The construction
------------------

Replace the predictor :math:`x` with a set of :math:`K` **basis functions** :math:`B_k(x)` and regress on
those:

.. math::

   f(x) = \sum_{k=1}^{K} \beta_k \, B_k(x), \qquad
   y_i = \sum_k \beta_k B_k(x_i) + \epsilon_i .

The bases are chosen, not learned. **B-splines** — smooth, local, bump-shaped functions each supported on
a short interval — are the standard choice: local support means each coefficient controls the curve only
near its knot, giving stable, interpretable flexibility. **Splines** stitch low-degree polynomials
together at **knots** with smoothness enforced at the joins.

.. code-block:: python

   import numpy as np, pymc as pm
   from patsy import dmatrix
   B = np.asarray(dmatrix("bs(x, df=12, degree=3)", {"x": x}))   # B-spline basis matrix
   with pm.Model():
       beta = pm.Normal("beta", 0, 1, shape=B.shape[1])          # basis coefficients
       pm.Normal("y", B @ beta, pm.HalfNormal("s", 1), observed=y)

The bias-variance knob
------------------------

The number of basis functions sets flexibility. **Too few** knots and the curve is too stiff to follow
the data — bias. **Too many** and it chases noise — variance, a wiggly overfit. The classical response is
to tune :math:`K` by cross-validation; the Bayesian response, in the next lesson, is more elegant — use
**many** basis functions and let a **prior shrink** their coefficients, so the data decide the effective
smoothness rather than a discrete knot count.

Why the basis view unifies
----------------------------

Splines are just **regression with a cleverly chosen design matrix**, which is why everything transfers:
priors on the coefficients, hierarchical structure, the same sampler. A basis expansion also connects
directly to the batching stage — the spline coefficients are a **batch** of related parameters, ripe for
a shared prior — and forward to Gaussian processes, which are the infinite-basis limit of this same idea.
Flexible curves, built from fixed pieces and fit by ordinary means.

.. hint::

   **Related lessons:** :doc:`Basis selection and shrinkage of coeﬃcients <127-basis-selection-and-shrinkage-of-coefficients>`  ·  :doc:`Non-normal models and regression surfaces <128-non-normal-models-and-regression-surfaces>`  ·  :doc:`Assembling the matrix of explanatory variables <095-assembling-the-matrix-of-explanatory-variables>`  ·  :doc:`Gaussian process regression <129-gaussian-process-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/splines-and-weighted-sums-of-basis-functions/ <https://insightful-data-lab.com/2025/12/09/splines-and-weighted-sums-of-basis-functions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
