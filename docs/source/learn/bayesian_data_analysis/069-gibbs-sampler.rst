.. _bda-gibbs-sampler:

========================================================================
Gibbs sampler
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 069 of 144  ·  *intermediate*

:doc:`◀ Previous · Debugging Bayesian computing <068-debugging-bayesian-computing>`   ·   :doc:`Next · Metropolis and Metropolis-Hastings algorithms ▶ <070-metropolis-and-metropolis-hastings-algorithms>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Sampling one coordinate at a time
-----------------------------------

Rejection and importance sampling collapse in high dimensions because they try to hit the posterior in
**all** coordinates at once. **Markov chain Monte Carlo** gives up independent draws and instead builds
a chain whose stationary distribution *is* the posterior. The **Gibbs sampler** is the simplest such
chain: update each parameter in turn, drawing it from its **full conditional** distribution given the
current values of all the others.

The algorithm
---------------

For :math:`\theta = (\theta_1, \dots, \theta_d)`, one sweep at iteration :math:`t` is

.. math::

   \theta_1^{(t)} \sim p\bigl(\theta_1 \mid \theta_2^{(t-1)}, \dots, \theta_d^{(t-1)}, y\bigr), \quad
   \theta_2^{(t)} \sim p\bigl(\theta_2 \mid \theta_1^{(t)}, \theta_3^{(t-1)}, \dots, y\bigr), \;\dots

Each draw conditions on the **most recent** value of every other coordinate. Note there is no accept/reject
step — Gibbs is the special case of Metropolis–Hastings whose proposal *is* the full conditional, for
which the acceptance probability is identically **one**. A :math:`d`-dimensional problem becomes
:math:`d` one-dimensional ones.

Where the conditionals come from
----------------------------------

Conjugacy, which is why Part I's algebra returns here as **infrastructure**. In the normal hierarchical
model each conditional is a standard distribution:

.. code-block:: python

   import numpy as np
   from scipy import stats
   # y_j ~ N(theta_j, sigma_j^2);  theta_j ~ N(mu, tau^2)
   for t in range(n_iter):
       V = 1 / (1 / sigma**2 + 1 / tau**2)                    # theta_j | mu, tau, y
       m = V * (ybar / sigma**2 + mu / tau**2)
       theta = stats.norm(m, np.sqrt(V)).rvs()
       mu = stats.norm(theta.mean(), tau / np.sqrt(J)).rvs()  # mu | theta, tau
       ss = ((theta - mu) ** 2).sum()                         # tau^2 | theta, mu
       tau = np.sqrt(ss / stats.chi2(J - 1).rvs())

Strengths and the failure mode
--------------------------------

Gibbs is **tuning-free** and every draw is accepted, which made it the engine of BUGS and JAGS. But it
moves only **along the coordinate axes**, so when parameters are strongly **correlated** in the
posterior, each step is tiny relative to the diagonal ridge the chain must traverse. Successive draws
become nearly identical, mixing crawls, and the effective sample size collapses.

Two remedies define the rest of this stage: **reparameterise** so the coordinates are less correlated,
or **block** highly dependent parameters and update them jointly. And where a conditional has no
closed form, a Metropolis step can be substituted for that coordinate — the hybrid that the next two
lessons build.

.. hint::

   **Related lessons:** :doc:`Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`  ·  :doc:`Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>`  ·  :doc:`Example: hierarchical normal model <074-example-hierarchical-normal-model>`  ·  :doc:`Direct simulation and rejection sampling <064-direct-simulation-and-rejection-sampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/12/gibbs-sampler/ <https://insightful-data-lab.com/2025/11/12/gibbs-sampler/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
