.. _bda-importance-sampling:

========================================================================
Importance sampling
========================================================================

**Part 3 · Stage 8 · 🧰 Simulation Basics**  ·  Lesson 065 of 144  ·  *intermediate*

:doc:`◀ Previous · Direct simulation and rejection sampling <064-direct-simulation-and-rejection-sampling>`   ·   :doc:`Next · How many simulation draws are needed? ▶ <066-how-many-simulation-draws-are-needed>`   ·   :doc:`↑ Section <index>`


Draw from the wrong distribution, then correct
------------------------------------------------

Rejection sampling throws draws away. **Importance sampling** keeps them all and **reweights** instead.
Sample from a convenient proposal :math:`g`, and correct for having used the wrong distribution by
weighting each draw by how much the target wants it relative to the proposal.

The identity
--------------

For any :math:`g` whose support covers the target,

.. math::

   \mathrm{E}_{p}[h(\theta)] = \int h(\theta) \, \frac{p(\theta \mid y)}{g(\theta)} \, g(\theta) \,
   d\theta \;\approx\; \frac{\sum_{s=1}^{S} w_s \, h(\theta^{(s)})}{\sum_{s=1}^{S} w_s},
   \qquad w_s = \frac{q(\theta^{(s)})}{g(\theta^{(s)})},

with :math:`\theta^{(s)} \sim g` and :math:`q \propto p(\theta \mid y)` the unnormalised posterior.
Dividing by :math:`\sum_s w_s` gives the **self-normalised** estimator — which is why the normalising
constant of the posterior is never needed.

The weights are the whole story
---------------------------------

Everything hinges on the **variance of the weights**. If a few draws carry nearly all the weight, the
estimate is effectively based on a handful of points. The standard summary is the **effective sample
size**:

.. math::

   S_{\text{eff}} = \frac{\bigl(\sum_{s} w_s\bigr)^2}{\sum_{s} w_s^2} .

Equal weights give :math:`S_{\text{eff}} = S`; one dominant weight gives :math:`S_{\text{eff}} \approx 1`.
Worse, when :math:`g` has **lighter tails** than the target, the weights can have **infinite variance**
and the estimator has no central limit theorem at all — it will look stable, then jump.

Pareto smoothing
------------------

The modern repair, from Stage 6's LOO lesson, is **PSIS**: fit a generalised Pareto distribution to the
largest weights and replace them with the expected order statistics of that fit. This stabilises the
estimate and — the real prize — the fitted shape :math:`\hat{k}` **diagnoses** the problem. Weights have
finite variance when :math:`\hat{k} < 1/2`; values above :math:`0.7` warn that the proposal is a poor
match.

.. code-block:: python

   import numpy as np
   logw = log_q(draws) - g.logpdf(draws)                 # always work in logs
   logw -= logw.max()                                    # stabilise before exponentiating
   w = np.exp(logw)
   ess = w.sum() ** 2 / (w ** 2).sum()                   # effective sample size
   est = (w * h(draws)).sum() / w.sum()                  # self-normalised estimate

Where it is used
------------------

Importance sampling rarely fits a posterior from scratch — like rejection, it degrades exponentially
with dimension. Its value is as a **correction**: reweighting a variational or Laplace approximation
toward the true posterior; computing **leave-one-out** predictions from a single MCMC fit (PSIS-LOO);
diagnosing prior sensitivity by reweighting to a perturbed prior; and inside particle filters. Use it to
adjust a nearly-right answer, not to find one.

.. hint::

   **Related lessons:** :doc:`Direct simulation and rejection sampling <064-direct-simulation-and-rejection-sampling>`  ·  :doc:`Distributional approximations <063-distributional-approximations>`  ·  :doc:`Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`  ·  :doc:`How many simulation draws are needed? <066-how-many-simulation-draws-are-needed>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/importance-sampling/ <https://insightful-data-lab.com/2025/11/11/importance-sampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
