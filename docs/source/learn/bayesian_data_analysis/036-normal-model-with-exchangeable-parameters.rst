.. _bda-normal-model-with-exchangeable-parameters:

========================================================================
Normal model with exchangeable parameters
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 036 of 144  ·  *beginner*

:doc:`◀ Previous · Bayesian analysis of conjugate hierarchical models <035-bayesian-analysis-of-conjugate-hierarchical-models>`   ·   :doc:`Next · Example: parallel experiments in eight schools ▶ <037-example-parallel-experiments-in-eight-schools>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The hierarchical normal
-------------------------

Replace binomial groups with normal ones and the hierarchy becomes fully transparent — every quantity
has a closed form, and the mechanics of shrinkage can be read off directly. Each of :math:`J` groups
supplies an estimate :math:`\bar{y}_j` of its own mean :math:`\theta_j`, with **known** standard error
:math:`\sigma_j`:

.. math::

   \bar{y}_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2), \qquad
   (\mu, \tau) \sim p(\mu, \tau).

Here :math:`\mu` is the **population mean** and :math:`\tau` the **between-group standard deviation**:
how different the groups really are.

Shrinkage, in closed form
---------------------------

Conditional on :math:`(\mu, \tau)`, each group's posterior is the Stage 2 normal update — precisions
add, and the mean is precision-weighted:

.. math::

   \theta_j \mid \mu, \tau, y \;\sim\; \mathrm{N}(\hat{\theta}_j,\; V_j), \qquad
   \hat{\theta}_j = \frac{\frac{1}{\sigma_j^2}\bar{y}_j + \frac{1}{\tau^2}\mu}
                         {\frac{1}{\sigma_j^2} + \frac{1}{\tau^2}}, \qquad
   \frac{1}{V_j} = \frac{1}{\sigma_j^2} + \frac{1}{\tau^2}.

Define the **shrinkage factor** :math:`B_j = \sigma_j^2 / (\sigma_j^2 + \tau^2)`; then
:math:`\hat{\theta}_j = (1 - B_j)\, \bar{y}_j + B_j\, \mu`. A noisy group (large :math:`\sigma_j`) has
:math:`B_j` near 1 and is pulled almost entirely to the population mean; a precise group keeps its own
estimate. And crucially, :math:`\tau` is **not chosen** — it is inferred, so the data decide how much
pooling is warranted.

The whole model in code
-------------------------

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 10)
       tau = pm.HalfNormal("tau", 10)               # weakly informative; never inverse-gamma(eps,eps)
       theta = pm.Normal("theta", mu, tau, shape=J)
       pm.Normal("y", theta, sigma=sigma_j, observed=ybar)   # sigma_j known
       idata = pm.sample(target_accept=0.95)

Two warnings
--------------

When :math:`\tau` is near **zero**, the posterior sits at a boundary — the counterexample from Stage 4 —
and the geometry becomes a funnel that samplers negotiate badly; the standard repair is the
**non-centred** parameterisation, :math:`\theta_j = \mu + \tau \eta_j` with
:math:`\eta_j \sim \mathrm{N}(0,1)`. And the prior on :math:`\tau` **matters**, especially with few
groups: with :math:`J = 8`, a careless inverse-gamma can dominate. Both issues are met head-on in the
eight-schools example that follows.

.. hint::

   **Related lessons:** :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/normal-model-with-exchangeable-parameters/ <https://insightful-data-lab.com/2025/11/09/normal-model-with-exchangeable-parameters/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
