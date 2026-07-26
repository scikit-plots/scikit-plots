.. _bda-example-hierarchical-normal-model-continued:

========================================================================
Example: hierarchical normal model (continued)
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 086 of 144  ·  *intermediate*

:doc:`◀ Previous · Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>`   ·   :doc:`Next · Variational inference ▶ <087-variational-inference>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The same model, approximated
------------------------------

Return to the hierarchical normal model that Stage 9 sampled with Gibbs, and apply the machinery of this
stage instead. It is a fair test: the answers are known, so every approximation can be scored.

.. math::

   y_{ij} \sim \mathrm{N}(\theta_j, \sigma^2), \qquad
   \theta_j \sim \mathrm{N}(\mu, \tau^2), \qquad \phi = (\mu, \sigma, \tau).

Why the joint mode fails
--------------------------

Maximise :math:`p(\theta, \phi \mid y)` jointly and the optimiser walks straight to the degenerate point:
every :math:`\theta_j` equal to :math:`\mu`, and :math:`\tau = 0`. The joint density there is
**unbounded** — the likelihood of the group parameters concentrates without limit as the population
variance shrinks. The joint mode is not merely a poor summary; it does not exist as a finite maximum.

Marginal mode, via EM
-----------------------

Integrate the :math:`\theta_j` out and maximise the marginal :math:`p(\phi \mid y)` instead. EM does this
without doing the integral: the E-step supplies conditional means **and variances** of the
:math:`\theta_j`, and the M-step's update for :math:`\tau^2` includes those variances, so it cannot
collapse to zero. The degeneracy disappears the moment the nuisance parameters are averaged rather than
maximised.

.. code-block:: python

   import numpy as np
   for _ in range(200):
       V = 1 / (n_j / sigma**2 + 1 / tau**2)                  # E-step
       Etheta = V * (n_j * ybar_j / sigma**2 + mu / tau**2)
       mu = Etheta.mean()                                     # M-step
       tau = np.sqrt(np.mean((Etheta - mu) ** 2 + V))         # + V prevents collapse
   # then: Laplace around (mu, log sigma, log tau) for approximate uncertainty

Then the conditional
----------------------

Having a marginal for :math:`\phi` — from EM plus curvature, or from a grid — the previous lesson's
recipe completes the picture: draw :math:`\phi`, then draw each :math:`\theta_j` **exactly** from its
normal conditional. The group-level answers inherit the hyperparameter uncertainty, which is what
distinguishes this from empirical Bayes.

What the comparison teaches
-----------------------------

Against long-run MCMC, the modal approximation is accurate for :math:`\mu` and for the :math:`\theta_j`
in data-rich groups, and least accurate for :math:`\tau` — the parameter whose posterior is skewed,
bounded below, and often heaped near zero. That is the general pattern: **approximations fail on the
variance parameters of hierarchies**, which are precisely the parameters that decide how much pooling
occurs. Use modal methods for speed and starting values; check the conclusions that hinge on
:math:`\tau` against a sampler.

.. hint::

   **Related lessons:** :doc:`Example: hierarchical normal model <074-example-hierarchical-normal-model>`  ·  :doc:`Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>`  ·  :doc:`Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>`  ·  :doc:`Boundary-avoiding priors for modal summaries <082-boundary-avoiding-priors-for-modal-summaries>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/example-hierarchical-normal-model-2/ <https://insightful-data-lab.com/2025/11/22/example-hierarchical-normal-model-2/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
