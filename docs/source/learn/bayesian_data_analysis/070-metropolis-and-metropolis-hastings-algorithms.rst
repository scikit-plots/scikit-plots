.. _bda-metropolis-and-metropolis-hastings-algorithms:

========================================================================
Metropolis and Metropolis-Hastings algorithms
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 070 of 144  ·  *intermediate*

:doc:`◀ Previous · Gibbs sampler <069-gibbs-sampler>`   ·   :doc:`Next · Using Gibbs and Metropolis as building blocks ▶ <071-using-gibbs-and-metropolis-as-building-blocks>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Propose, then decide
----------------------

Gibbs needs full conditionals you can sample. **Metropolis** needs only the ability to **evaluate** the
unnormalised posterior :math:`q(\theta) \propto p(\theta \mid y)`. Propose a move; accept it with a
probability chosen so that the chain's stationary distribution is exactly the posterior.

The Metropolis algorithm
--------------------------

From the current :math:`\theta^{(t-1)}`, draw a proposal :math:`\theta^{*}` from a **symmetric** jumping
distribution, :math:`J(\theta^{*} \mid \theta^{(t-1)}) = J(\theta^{(t-1)} \mid \theta^{*})` — a random
walk, :math:`\mathrm{N}(\theta^{(t-1)}, c^2 \Sigma)`, is standard. Accept with probability

.. math::

   r = \min\left(1, \; \frac{q(\theta^{*})}{q(\theta^{(t-1)})}\right).

Moves **uphill** are always accepted; moves **downhill** are accepted with probability equal to the
density ratio. That occasional downhill step is what makes the chain explore rather than optimise. The
normalising constant cancels — the reason MCMC never needs the evidence.

Metropolis–Hastings
---------------------

Drop the symmetry requirement and correct for the asymmetry with a **proposal ratio**:

.. math::

   r = \min\left(1, \; \frac{q(\theta^{*}) \; J(\theta^{(t-1)} \mid \theta^{*})}
                            {q(\theta^{(t-1)}) \; J(\theta^{*} \mid \theta^{(t-1)})}\right).

This generality is what allows asymmetric proposals near boundaries, independence samplers, and — with
the proposal set to the full conditional — recovers **Gibbs**, whose ratio is identically 1. Both satisfy
**detailed balance**, which together with ergodicity guarantees the posterior is the stationary
distribution.

.. code-block:: python

   import numpy as np
   theta, out = init, []
   for t in range(n_iter):
       prop = theta + step * np.random.standard_normal(d)      # symmetric random walk
       if np.log(np.random.rand()) < log_q(prop) - log_q(theta):   # always work in logs
           theta = prop                                        # accept
       out.append(theta.copy())                                # note: reject means REPEAT the draw

Tuning is everything
----------------------

The step size :math:`c` decides the outcome. **Too small**: nearly every proposal is accepted, but the
chain inches along and successive draws are highly correlated. **Too large**: proposals land in the
tails, almost all are rejected, and the chain **stands still** — repeating the same value, which is a
legitimate draw but adds no information. The classic guidance targets an acceptance rate near
:math:`0.44` in one dimension and about :math:`0.23` in high dimensions.

Even tuned, a random walk explores by **diffusion**: to travel a distance :math:`L` it needs roughly
:math:`(L/c)^2` steps. That quadratic cost is what Hamiltonian Monte Carlo was invented to escape.

.. hint::

   **Related lessons:** :doc:`Gibbs sampler <069-gibbs-sampler>`  ·  :doc:`Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>`  ·  :doc:`Eﬃcient Metropolis jumping rules <076-efficient-metropolis-jumping-rules>`  ·  :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/12/metropolis-and-metropolis-hastings-algorithms/ <https://insightful-data-lab.com/2025/11/12/metropolis-and-metropolis-hastings-algorithms/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
