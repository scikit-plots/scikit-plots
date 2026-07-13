.. _bda-multivariate-normal-with-unknown-mean-and-variance:

========================================================================
Multivariate Normal with Unknown Mean and Variance
========================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 025 of 144  ·  *beginner*

:doc:`◀ Previous · Multivariate Normal Model with Known Variance <024-multivariate-normal-model-with-known-variance>`   ·   :doc:`Next · Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate) ▶ <026-example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate>`   ·   :doc:`↑ Section <index>`


The full multivariate model
-----------------------------

Drop the fiction that :math:`\Sigma` is known. With :math:`y_i \sim \mathrm{N}(\theta, \Sigma)` and
both unknown, the conjugate prior is the matrix version of the normal--inverse-:math:`\chi^2` from
earlier in this stage: a **normal--inverse-Wishart**,

.. math::

   \Sigma \sim \text{Inv-Wishart}(\nu_0, \Lambda_0^{-1}), \qquad
   \theta \mid \Sigma \sim \mathrm{N}\!\left(\mu_0,\, \frac{\Sigma}{\kappa_0}\right).

Again the mean's prior is scaled by the very covariance being estimated, and again :math:`\kappa_0`
and :math:`\nu_0` are **equivalent sample sizes** — for the mean and for the covariance respectively.

The update
------------

Conjugacy holds, and the hyperparameters update by addition exactly as in the scalar case:

.. math::

   \kappa_n = \kappa_0 + n, \qquad \nu_n = \nu_0 + n, \qquad
   \mu_n = \frac{\kappa_0 \mu_0 + n \bar{y}}{\kappa_0 + n},

.. math::

   \Lambda_n = \Lambda_0 + S + \frac{\kappa_0 n}{\kappa_0 + n}
               (\bar{y} - \mu_0)(\bar{y} - \mu_0)^{\top},

where :math:`S = \sum_i (y_i - \bar{y})(y_i - \bar{y})^{\top}`. The last term is the multivariate
**conflict detector**: prior–data disagreement about the mean inflates the posterior covariance.
Marginalising :math:`\Sigma` gives a **multivariate** :math:`t` for :math:`\theta`, the vector analogue
of the scalar result.

Why the inverse-Wishart disappoints
-------------------------------------

The inverse-Wishart is conjugate, guarantees positive-definiteness, and makes Gibbs sampling trivial —
which is why it dominated Bayesian software for decades. It is also, in Gelman's own later assessment,
a **poor default**. A *single* degrees-of-freedom parameter :math:`\nu_0` controls the certainty of
**every** variance at once; the implied marginal for each variance has **little density near zero**,
biasing small variances upward; and it forces an **a priori dependence between correlations and
variances** that nobody believes.

The modern alternative
------------------------

Modern practice decomposes :math:`\Sigma` into **scales** and a **correlation matrix**, giving each its
own prior — an **LKJ** prior on the correlations, half-normal or half-Cauchy on the standard
deviations:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       sd = pm.HalfNormal("sd", sigma=1, shape=d)             # scales, separately
       chol, corr, _ = pm.LKJCholeskyCov("Sigma", n=d, eta=2,  # correlations, separately
                                          sd_dist=pm.HalfNormal.dist(1))
       pm.MvNormal("y", mu=theta, chol=chol, observed=Y)

Conjugacy bought tractability when computation was scarce. With HMC available, the honest prior wins —
a theme that recurs whenever Part I's closed forms meet Part III's samplers.

.. hint::

   **Related lessons:** :doc:`Multivariate Normal Model with Known Variance <024-multivariate-normal-model-with-known-variance>`  ·  :doc:`Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`  ·  :doc:`Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/multivariate-normal-with-unknown-mean-and-variance/ <https://insightful-data-lab.com/2025/11/09/multivariate-normal-with-unknown-mean-and-variance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
