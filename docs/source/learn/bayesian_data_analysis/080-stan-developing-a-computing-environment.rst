.. _bda-stan-developing-a-computing-environment:

========================================================================
Stan: developing a computing environment
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 080 of 144  ·  *intermediate*

:doc:`◀ Previous · Hamiltonian Monte Carlo for a hierarchical model <079-hamiltonian-monte-carlo-for-a-hierarchical-model>`   ·   :doc:`Next · Finding posterior modes ▶ <081-finding-posterior-modes>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A language for models
-----------------------

Stan is a probabilistic programming language: you declare data, parameters and a model, and Stan compiles
that declaration into C++ that computes the log posterior **and its gradient** by automatic
differentiation, then samples it with an adaptive HMC variant. The point is separation of concerns — the
statistician states the model; the machine derives the calculus and tunes the sampler.

The blocks
------------

A Stan program is organised so that its structure mirrors the probability model, which is itself a
discipline: what is data, what is unknown, what is derived.

.. code-block:: stan

   data {
     int<lower=0> J;                 // number of schools
     vector[J] y;                    // estimated effects
     vector<lower=0>[J] sigma;       // known standard errors
   }
   parameters {
     real mu;
     real<lower=0> tau;
     vector[J] eta;                  // non-centred
   }
   transformed parameters {
     vector[J] theta = mu + tau * eta;
   }
   model {
     mu ~ normal(0, 5);
     tau ~ cauchy(0, 5);
     eta ~ std_normal();
     y ~ normal(theta, sigma);       // likelihood
   }

Constraints (``<lower=0>``) are not assertions but **transformations**: Stan samples an unconstrained
:math:`\log \tau` internally and applies the Jacobian, which is why HMC never proposes a negative
variance.

NUTS
------

Plain HMC has two free parameters: the step size and the **number of leapfrog steps**. Too few and the
sampler random-walks; too many and the trajectory curls back on itself, wasting computation. The
**No-U-Turn Sampler** (Hoffman and Gelman) removes the second: it doubles the trajectory forward and
backward until the path begins to double back on itself, then samples a point from it. Doubling in both
directions is what preserves reversibility. The step size is tuned during warm-up to hit a target
acceptance rate.

.. code-block:: python

   from cmdstanpy import CmdStanModel
   import arviz as az
   fit = CmdStanModel(stan_file="eight_schools.stan").sample(data=data, adapt_delta=0.9)
   az.summary(az.from_cmdstanpy(fit))         # same diagnostics, whatever the backend

Why it changed practice
-------------------------

Three things. **Gradients for free**, so HMC became usable by people who do not want to differentiate
hierarchical likelihoods by hand. **Diagnostics by default** — R-hat, ESS, divergences and energy
reported without asking, which made unchecked sampling embarrassing rather than normal. And a
**declarative model** that can be read, reviewed and reparameterised as a statement about the world
rather than as an algorithm. PyMC, NumPyro and BlackJAX offer the same bargain in Python. The remaining
craft is the modelling — which is where Parts IV and V go.

.. hint::

   **Related lessons:** :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`  ·  :doc:`Computing environments <067-computing-environments>`  ·  :doc:`Computation and Software <009-computation-and-software>`  ·  :doc:`Debugging Bayesian computing <068-debugging-bayesian-computing>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/stan-developing-a-computing-environment/ <https://insightful-data-lab.com/2025/11/22/stan-developing-a-computing-environment/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
