.. _bda-posterior-inference-and-computation:

========================================================================
Posterior inference and computation
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 115 of 144  ·  *advanced*

:doc:`◀ Previous · Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>`   ·   :doc:`Next · Robust inference for the eight schools ▶ <116-robust-inference-for-the-eight-schools>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Fitting robust models
-----------------------

Heavy-tailed and overdispersed likelihoods are more expressive, and modestly harder to compute.
Two techniques — one classical, one modern — make them routine, and the classical one reveals *why* the
Student-:math:`t` works.

The t as a scale mixture of normals
-------------------------------------

The key representation: a Student-:math:`t` is a **normal whose variance is itself random**, drawn from
an inverse-gamma (equivalently, a normal whose precision is Gamma). Introduce a per-observation scale
:math:`\lambda_i`:

.. math::

   y_i \mid \lambda_i \sim \mathrm{N}(\mu, \sigma^2 / \lambda_i), \qquad
   \lambda_i \sim \mathrm{Gamma}(\nu/2, \nu/2)
   \quad \Longrightarrow \quad y_i \sim t_\nu(\mu, \sigma).

This is not just a computational trick; it is the **mechanism of robustness made visible**. Each
:math:`\lambda_i` is a weight, and an outlier is simply an observation whose posterior :math:`\lambda_i`
is **small** — its variance inflates, so it pulls on :math:`\mu` weakly. The model down-weights
aberrant points *automatically*, and the weights are readable: a small :math:`\lambda_i` flags exactly
which observations the model treated as outliers.

.. code-block:: python

   import pymc as pm
   # explicit scale-mixture form: the lambda_i ARE the outlier weights
   with pm.Model():
       nu = pm.Gamma("nu", 2, 0.1)
       lam = pm.Gamma("lam", nu / 2, nu / 2, shape=n)       # per-obs weights
       sigma = pm.HalfNormal("sigma", 1)
       pm.Normal("y", mu=X @ beta, sigma=sigma / pm.math.sqrt(lam), observed=y)
       # posterior mean of lam[i] << 1  <=>  observation i is an outlier

With augmentation, every conditional is standard, so **Gibbs** samples the model directly — the classical
route, and the reason robust models were tractable before HMC. In modern practice you simply write the
:math:`t` and let **NUTS** handle it; the mixture form is then a diagnostic and an explanation rather
than a necessity.

Computational cautions
------------------------

Two recur. The degrees of freedom :math:`\nu` is often **weakly identified** — data rarely pin down tail
weight precisely — so it needs a sensible prior (a Gamma keeping :math:`\nu` away from zero) and should
be reported with its full, often wide, posterior. And heavy-tailed models can induce **funnel** geometry
between :math:`\nu`, the scales, and the coefficients, so the non-centred parameterisation and the
divergence diagnostics of Stage 9 apply here too.

The payoff
------------

Robust computation costs little beyond a standard fit and returns two things: estimates that **do not
lurch** when one point is extreme, and an explicit account — through the weights :math:`\lambda_i` or the
posterior of :math:`\nu` — of *how much* robustness the data actually demanded. The next lesson puts the
whole apparatus to work on a familiar dataset.

.. hint::

   **Related lessons:** :doc:`Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>`  ·  :doc:`Robust regression using t-distributed errors <117-robust-regression-using-t-distributed-errors>`  ·  :doc:`Gibbs sampler <069-gibbs-sampler>`  ·  :doc:`Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/posterior-inference-and-computation/ <https://insightful-data-lab.com/2025/12/07/posterior-inference-and-computation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
