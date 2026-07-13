.. _bda-hamiltonian-monte-carlo:

========================================================================
Hamiltonian Monte Carlo
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 078 of 144  ·  *intermediate*

:doc:`◀ Previous · Further extensions to Gibbs and Metropolis <077-further-extensions-to-gibbs-and-metropolis>`   ·   :doc:`Next · Hamiltonian Monte Carlo for a hierarchical model ▶ <079-hamiltonian-monte-carlo-for-a-hierarchical-model>`   ·   :doc:`↑ Section <index>`


Stop wandering; use the gradient
----------------------------------

Random-walk Metropolis explores by diffusion, needing :math:`\mathcal{O}((L/c)^2)` steps to travel a
distance :math:`L`. **Hamiltonian Monte Carlo** replaces the aimless jump with a **directed trajectory**,
computed from the **gradient of the log posterior**. It is the reason modern Bayesian models with
thousands of parameters are fittable at all.

Physics on the posterior
--------------------------

Treat :math:`\theta` as the position of a particle on a landscape whose **potential energy** is the
negative log posterior. Add an auxiliary **momentum** :math:`\rho`, drawn afresh each iteration from
:math:`\mathrm{N}(0, M)`. The total energy is the **Hamiltonian**

.. math::

   H(\theta, \rho) = \underbrace{-\log p(\theta \mid y)}_{\text{potential}}
                   + \underbrace{\tfrac{1}{2} \rho^{\top} M^{-1} \rho}_{\text{kinetic}},

and the joint density :math:`p(\theta, \rho) \propto e^{-H(\theta, \rho)}` has the posterior as its
:math:`\theta`-marginal — so simulating the particle's motion and **discarding the momentum** samples the
posterior. Hamiltonian flow conserves :math:`H`, is time-reversible, and preserves phase-space volume:
exactly the properties a valid MCMC proposal needs.

Leapfrog, and the correction
------------------------------

The dynamics cannot be solved exactly, so they are integrated with the **leapfrog** scheme — a half-step
of momentum along the gradient, a full step of position, another half-step of momentum. Leapfrog is
time-reversible and volume-preserving **exactly**, but conserves energy only **approximately**. The
residual error is removed by a Metropolis accept/reject on :math:`e^{-H}`, whose acceptance is near one
when the step size is small.

.. code-block:: python

   def leapfrog(theta, rho, eps, L, grad_logp):
       rho = rho + 0.5 * eps * grad_logp(theta)          # half step
       for _ in range(L):
           theta = theta + eps * rho                     # full position step
           rho = rho + eps * grad_logp(theta)            # full momentum step
       rho = rho - 0.5 * eps * grad_logp(theta)          # correct the last half step
       return theta, -rho                                # negate for reversibility

Divergences: not a warning, a wound
-------------------------------------

The leapfrog integrator is only **conditionally stable**. Where the posterior's curvature is sharp
relative to the step size, the simulated trajectory departs from the true one and the energy error
explodes — a **divergent transition**. Stan and PyMC flag these, and the flag matters: positions after a
divergence are **never selected** as draws, so the sampler cannot explore that region. HMC silently
degenerates toward a random walk, and estimates are **biased** by the systematic omission.

Non-negotiable: divergences must be **fixed**, not suppressed. Raise ``target_accept`` (smaller steps),
rescale parameters, or — usually the real answer — **reparameterise** the geometry that caused them. The
next lesson is that story.

.. hint::

   **Related lessons:** :doc:`Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`  ·  :doc:`Hamiltonian Monte Carlo for a hierarchical model <079-hamiltonian-monte-carlo-for-a-hierarchical-model>`  ·  :doc:`Stan: developing a computing environment <080-stan-developing-a-computing-environment>`  ·  :doc:`Inference and assessing convergence <072-inference-and-assessing-convergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/hamiltonian-monte-carlo/ <https://insightful-data-lab.com/2025/11/22/hamiltonian-monte-carlo/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
