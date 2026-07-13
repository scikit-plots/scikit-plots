.. _bda-example-hierarchical-normal-model:

========================================================================
Example: hierarchical normal model
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 074 of 144  ·  *intermediate*

:doc:`◀ Previous · Eﬀective number of simulation draws <073-effective-number-of-simulation-draws>`   ·   :doc:`Next · Eﬃcient Gibbs samplers ▶ <075-efficient-gibbs-samplers>`   ·   :doc:`↑ Section <index>`


Gibbs, worked through
-----------------------

The hierarchical normal model of Stage 5 has closed-form conditionals at every level, which makes it the
canonical Gibbs example — and a useful place to watch a sampler misbehave. Take :math:`J` groups,
:math:`n_j` observations each:

.. math::

   y_{ij} \sim \mathrm{N}(\theta_j, \sigma^2), \qquad
   \theta_j \sim \mathrm{N}(\mu, \tau^2), \qquad
   p(\mu, \log \sigma, \log \tau) \propto 1 .

The unknowns are the group means :math:`\theta`, the population mean :math:`\mu`, and the two variances.

The four conditionals
-----------------------

Each is a Stage 2 or Stage 3 result, reused:

.. math::

   \theta_j \mid \cdot \sim \mathrm{N}\!\left(
     \frac{\frac{n_j}{\sigma^2}\bar{y}_{\cdot j} + \frac{1}{\tau^2}\mu}
          {\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}}, \;
     \left(\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}\right)^{-1}\right),
   \qquad
   \mu \mid \cdot \sim \mathrm{N}\!\left(\bar{\theta}, \; \frac{\tau^2}{J}\right),

with :math:`\sigma^2` and :math:`\tau^2` drawn from scaled inverse-:math:`\chi^2` distributions built
from the within- and between-group sums of squares. Cycle through them and the chain converges to the
joint posterior — no tuning, no rejections.

.. code-block:: python

   import numpy as np
   from scipy import stats
   for t in range(n_iter):
       V = 1.0 / (n_j / sigma**2 + 1.0 / tau**2)                  # theta | mu, sigma, tau, y
       m = V * (n_j * ybar_j / sigma**2 + mu / tau**2)
       theta = stats.norm(m, np.sqrt(V)).rvs()
       mu = stats.norm(theta.mean(), tau / np.sqrt(J)).rvs()      # mu | theta, tau
       ss_w = ((y - theta[group]) ** 2).sum()                     # sigma^2 | theta, y
       sigma = np.sqrt(ss_w / stats.chi2(n).rvs())
       ss_b = ((theta - mu) ** 2).sum()                           # tau^2 | theta, mu
       tau = np.sqrt(ss_b / stats.chi2(J - 1).rvs())

Watch it stick
----------------

Now look at what the chain does when :math:`\tau` wanders near **zero**. The conditional for
:math:`\theta_j` collapses onto :math:`\mu`; the conditional for :math:`\tau^2` is then built from
:math:`\theta_j` that are all nearly equal, so it stays small. The two conditionals **trap each other**,
and the sampler crawls through the neck of the funnel from eight schools. Gibbs does not diverge or warn
— it simply mixes so slowly that the low-:math:`\tau` region is under-visited, and :math:`\hat{R}` may
look fine on a short run.

The lesson
------------

Closed-form conditionals guarantee **correctness in the limit**, not **efficiency in practice**. The
diagnosis is posterior **correlation between levels**, and the two cures are the subject of the next
lesson: reparameterise so the levels decouple, or update them jointly.

.. hint::

   **Related lessons:** :doc:`Gibbs sampler <069-gibbs-sampler>`  ·  :doc:`Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>`  ·  :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`  ·  :doc:`Example: hierarchical normal model (continued) <086-example-hierarchical-normal-model-continued>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/12/example-hierarchical-normal-model/ <https://insightful-data-lab.com/2025/11/12/example-hierarchical-normal-model/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
