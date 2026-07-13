.. _bda-setting-up-and-interpreting-mixture-models:

========================================================================
Setting up and interpreting mixture models
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 134 of 144  ·  *advanced*

:doc:`◀ Previous · Density estimation and regression <133-density-estimation-and-regression>`   ·   :doc:`Next · Example: reaction times and schizophrenia ▶ <135-example-reaction-times-and-schizophrenia>`   ·   :doc:`↑ Section <index>`


The latent-class formulation
------------------------------

A mixture becomes easy to fit once written with an explicit **membership indicator**. Introduce, for each
observation, an unobserved label :math:`z_i` saying which component generated it:

.. math::

   z_i \sim \mathrm{Categorical}(\pi), \qquad
   y_i \mid z_i = k \sim \mathrm{N}(\mu_k, \sigma_k^2).

Marginalising :math:`z_i` recovers the weighted-sum density of the previous lesson, but *keeping*
:math:`z_i` as a latent variable turns an awkward sum inside the likelihood into a two-level model with
simple conditionals — the same data-augmentation trick used for the :math:`t` distribution and for probit
regression.

Fitting: EM and Gibbs
-----------------------

The augmented form yields two classic algorithms, both alternating between the labels and the parameters.
**EM** (from the modal-approximation stage) computes, in its E-step, the posterior **responsibilities**
:math:`\Pr(z_i = k \mid y_i)` — soft assignments — then in the M-step updates :math:`(\pi, \mu, \sigma)`
given them. **Gibbs** does the Bayesian version: sample each :math:`z_i` from its conditional, then sample
the parameters given the assignments, and repeat.

.. code-block:: python

   import numpy as np
   # EM for a Gaussian mixture: responsibilities, then weighted updates
   for _ in range(n_iter):
       # E-step: responsibility of component k for point i
       r = pi * norm_pdf(y[:, None], mu, sigma)             # (n, K)
       r /= r.sum(axis=1, keepdims=True)
       # M-step: update weights, means, sds from soft assignments
       Nk = r.sum(axis=0)
       pi = Nk / len(y)
       mu = (r * y[:, None]).sum(0) / Nk
       sigma = np.sqrt((r * (y[:, None] - mu)**2).sum(0) / Nk)

Interpreting the fit
----------------------

The responsibilities are the useful output when the goal is **clustering**: each :math:`\Pr(z_i = k)` is
a *soft* membership, honestly expressing that a point between two components belongs partly to each —
better than a hard assignment that hides the ambiguity. When the goal is **density estimation**, the
labels are a computational device and only the resulting smooth density matters.

Cautions
----------

Two recur through the stage. Mixture likelihoods are **multimodal** — genuinely, beyond label switching —
so optimisation and sampling can stick in local solutions; multiple starts and careful diagnostics are
needed. And a component's variance can **collapse** toward zero as it latches onto a single point, sending
the likelihood to infinity — the same boundary pathology met in hierarchical models, cured the same way,
with a prior that keeps variances off zero. The next lesson puts the setup to work; the one after
confronts label switching head-on.

.. hint::

   **Related lessons:** :doc:`Density estimation and regression <133-density-estimation-and-regression>`  ·  :doc:`Example: reaction times and schizophrenia <135-example-reaction-times-and-schizophrenia>`  ·  :doc:`Label switching and posterior computation <136-label-switching-and-posterior-computation>`  ·  :doc:`Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/setting-up-and-interpreting-mixture-models/ <https://insightful-data-lab.com/2025/12/09/setting-up-and-interpreting-mixture-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
