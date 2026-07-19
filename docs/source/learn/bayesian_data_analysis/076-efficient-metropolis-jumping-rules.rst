.. _bda-efficient-metropolis-jumping-rules:

========================================================================
Eﬃcient Metropolis jumping rules
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 076 of 144  ·  *intermediate*

:doc:`◀ Previous · Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>`   ·   :doc:`Next · Further extensions to Gibbs and Metropolis ▶ <077-further-extensions-to-gibbs-and-metropolis>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The proposal is the algorithm
-------------------------------

Metropolis is correct for *any* symmetric proposal, and useless for most of them. The **jumping rule**
determines how far the chain travels per unit of computation, and tuning it is not decoration but the
whole game.

The trade-off, quantified
---------------------------

Too small a step: proposals almost always accepted, but each move is negligible and the draws are
strongly autocorrelated. Too large: proposals land in the tails, are rejected, and the chain **repeats
its current value** — draws that are valid but carry no new information. The optimum sits between, and
for a random-walk proposal :math:`\mathrm{N}(\theta^{(t-1)}, c^2 \Sigma)` targeting a roughly normal
posterior with covariance :math:`\Sigma`, the asymptotically optimal scale is

.. math::

   c \;\approx\; \frac{2.4}{\sqrt{d}},

giving an acceptance rate near :math:`0.44` in one dimension and settling to about :math:`0.23` as
:math:`d` grows. Efficiency falls like :math:`1/d`: each added dimension makes the random walk worse.

Adapt, then stop
------------------

Practical samplers tune :math:`c` during **warm-up**, raising it when acceptance is too high and lowering
it when too low, and estimate :math:`\Sigma` from the draws so far so that proposals follow the
posterior's shape rather than a sphere.

.. code-block:: python

   import numpy as np
   accepted, step = 0, 0.1
   for t in range(n_warmup):
       ...                                            # one Metropolis update
       if (t + 1) % 50 == 0:                          # adapt in batches
           rate = accepted / 50
           step *= np.exp(0.5 * (rate - 0.234))       # nudge toward the target rate
           accepted = 0
   # then FREEZE step and Sigma; sampling draws must come from a time-homogeneous chain

The final clause matters: adaptation that continues during sampling destroys the Markov property, and the
chain no longer targets the posterior. Adapt in warm-up, freeze, then collect.

Better proposals, and their ceiling
-------------------------------------

Beyond scaling, the proposal can be improved by **preconditioning** with an estimate of :math:`\Sigma`
(equivalently, sampling in a whitened space), by **blocking** correlated parameters, or by adaptive
schemes that learn the covariance online. Each buys a constant factor.

None of them changes the fundamental cost. A random walk needs :math:`\mathcal{O}((L/c)^2)` steps to
cross a distance :math:`L`, and :math:`c` must shrink like :math:`1/\sqrt{d}`. **Gradients** break this
scaling by proposing *directed* moves, which is what the next lesson introduces.

.. hint::

   **Related lessons:** :doc:`Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`  ·  :doc:`Eﬀective number of simulation draws <073-effective-number-of-simulation-draws>`  ·  :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`  ·  :doc:`Further extensions to Gibbs and Metropolis <077-further-extensions-to-gibbs-and-metropolis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/21/e%ef%ac%83cient-metropolis-jumping-rules/ <https://insightful-data-lab.com/2025/11/21/e%ef%ac%83cient-metropolis-jumping-rules/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
