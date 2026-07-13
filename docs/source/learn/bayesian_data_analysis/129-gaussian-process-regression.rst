.. _bda-gaussian-process-regression:

========================================================================
Gaussian process regression
========================================================================

**Part 5 · Stage 15 · 🌊 Basis Functions & Gaussian Processes**  ·  Lesson 129 of 144  ·  *advanced*

:doc:`◀ Previous · Non-normal models and regression surfaces <128-non-normal-models-and-regression-surfaces>`   ·   :doc:`Next · Example: birthdays and birthdates ▶ <130-example-birthdays-and-birthdates>`   ·   :doc:`↑ Section <index>`


A prior over functions
------------------------

Basis expansions build a flexible curve from a **finite** set of functions. A **Gaussian process** takes
the idea to its limit: a prior directly over **functions themselves**, with no fixed basis. Instead of
parameterising a curve and putting priors on coefficients, a GP places a distribution on the space of
smooth functions and conditions it on the data.

The definition
----------------

A Gaussian process is a distribution over functions such that the values at **any finite set of inputs**
are jointly multivariate normal. It is specified completely by two objects: a **mean function**
:math:`m(x)` (often taken as zero) and a **covariance function** — the **kernel** — :math:`k(x, x')`:

.. math::

   f \sim \mathcal{GP}\bigl(m(x), \, k(x, x')\bigr), \qquad
   \bigl(f(x_1), \dots, f(x_n)\bigr) \sim \mathrm{N}(m, K), \;\; K_{ij} = k(x_i, x_j).

The kernel is the whole model. It sets how strongly the function values at two inputs covary, and
therefore what kinds of functions are probable — smooth, wiggly, periodic, or trending.

The kernel encodes assumptions
--------------------------------

The most common choice, the **squared-exponential** kernel, encodes smooth, infinitely differentiable
functions:

.. math::

   k(x, x') = \sigma^2 \exp\!\left(-\frac{\lVert x - x' \rVert^2}{2 \ell^2}\right),

with two interpretable hyperparameters: the **length-scale** :math:`\ell` — how far apart inputs must be
before their function values decorrelate, i.e. how wiggly the curve — and the **signal variance**
:math:`\sigma^2` — the amplitude. Other kernels encode other beliefs: **Matérn** for rougher functions,
**periodic** for cycles, **linear** for trends. And kernels **compose** — sums and products of kernels
build structured functions from simple parts, the key to the next lesson.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       ell = pm.Gamma("ell", 2, 1)                          # length-scale
       eta = pm.HalfNormal("eta", 1)                        # amplitude
       cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)          # squared-exponential kernel
       gp = pm.gp.Marginal(cov_func=cov)
       gp.marginal_likelihood("y", X=x[:, None], y=y, sigma=pm.HalfNormal("s", 1))

Prediction and the cost
-------------------------

Conditioning the joint normal on observed points gives a **closed-form** posterior over the function at
new inputs — a predictive mean and, crucially, a predictive **variance** that widens away from the data,
so a GP says "I don't know" where it has seen nothing. The catch is computational: prediction requires
**inverting the** :math:`n \times n` **covariance matrix**, an :math:`O(n^3)` operation that becomes
prohibitive beyond a few thousand points. Sparse approximations, inducing points, and basis-function
approximations (the previous lesson, in reverse) are the standard escapes. What you buy for that cost is
nonparametric flexibility with calibrated uncertainty — a curve that adapts to the data and honestly
reports where it cannot.

.. hint::

   **Related lessons:** :doc:`Non-normal models and regression surfaces <128-non-normal-models-and-regression-surfaces>`  ·  :doc:`Example: birthdays and birthdates <130-example-birthdays-and-birthdates>`  ·  :doc:`Latent Gaussian process models <131-latent-gaussian-process-models>`  ·  :doc:`Splines and weighted sums of basis functions <126-splines-and-weighted-sums-of-basis-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/gaussian-process-regression/ <https://insightful-data-lab.com/2025/12/09/gaussian-process-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
