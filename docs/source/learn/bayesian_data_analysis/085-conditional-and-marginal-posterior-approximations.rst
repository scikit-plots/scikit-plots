.. _bda-conditional-and-marginal-posterior-approximations:

========================================================================
Conditional and marginal posterior approximations
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 085 of 144  ·  *intermediate*

:doc:`◀ Previous · Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>`   ·   :doc:`Next · Example: hierarchical normal model (continued) ▶ <086-example-hierarchical-normal-model-continued>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Approximate in stages
-----------------------

A hierarchical posterior :math:`p(\theta, \phi \mid y)` is hard as a whole and easy in pieces. Factor it,

.. math::

   p(\theta, \phi \mid y) = \underbrace{p(\phi \mid y)}_{\text{marginal, hard}}
                            \; \underbrace{p(\theta \mid \phi, y)}_{\text{conditional, often easy}} ,

and treat the two factors differently. The **conditional** is frequently a standard distribution — the
Gibbs conditionals of Stage 9 are exactly these. The **marginal** for the hyperparameters is
low-dimensional, so it can be approximated well, or evaluated on a grid.

The recipe
------------

Approximate :math:`p(\phi \mid y)` — by a normal at its mode, or on a grid — then draw:

1. draw :math:`\phi^{(s)}` from the approximate marginal;
2. draw :math:`\theta^{(s)} \sim p(\theta \mid \phi^{(s)}, y)`, **exactly**, from the conditional.

The result is approximate joint draws, and crucially they **propagate uncertainty in** :math:`\phi`, in
contrast to empirical Bayes, which fixes :math:`\hat{\phi}` and understates every interval. This is
precisely what the conjugate hierarchical model of Stage 5 did with its two-dimensional grid.

.. code-block:: python

   import numpy as np
   from scipy import stats
   # 1. approximate the low-dimensional marginal p(phi | y) on a grid
   logm = np.array([log_marginal(p, y) for p in grid])       # theta integrated out
   w = np.exp(logm - logm.max()); w /= w.sum()
   phi = np.random.choice(grid, size=4000, p=w)              # draws from the marginal
   # 2. exact conditional draws given each phi
   theta = stats.norm(cond_mean(phi, y), cond_sd(phi, y)).rvs()

The marginal is where the work is
-----------------------------------

Computing :math:`p(\phi \mid y) = \int p(\theta, \phi \mid y) \, d\theta` requires integrating the group
parameters out. Conjugacy does it in closed form; otherwise a **Laplace approximation of the inner
integral**, evaluated at each :math:`\phi`, is the standard device. That nested-Laplace idea, applied to
latent Gaussian models, is the engine of **INLA** — accurate, and far faster than MCMC for the model
class it covers.

Why it still matters
----------------------

Two reasons, both practical. The factorisation tells you **where the difficulty lives**: almost always in
the hyperparameters, whose posterior is the funnel-shaped, weakly identified part. And it explains the
family relationship among methods — EM maximises the marginal, empirical Bayes plugs in its maximiser,
this approach **integrates** over it, and full MCMC samples the joint. They differ only in how honestly
they treat :math:`p(\phi \mid y)`.

.. hint::

   **Related lessons:** :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`  ·  :doc:`Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>`  ·  :doc:`Example: hierarchical normal model (continued) <086-example-hierarchical-normal-model-continued>`  ·  :doc:`Other approximations <089-other-approximations>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/conditional-and-marginal-posterior-approximations/ <https://insightful-data-lab.com/2025/11/22/conditional-and-marginal-posterior-approximations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
