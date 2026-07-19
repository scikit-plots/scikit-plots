.. _bda-expectation-propagation:

========================================================================
Expectation propagation
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 088 of 144  ·  *intermediate*

:doc:`◀ Previous · Variational inference <087-variational-inference>`   ·   :doc:`Next · Other approximations ▶ <089-other-approximations>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Approximate one factor at a time
----------------------------------

The posterior is a **product** of factors — a prior and one likelihood term per observation (or per
group):

.. math::

   p(\theta \mid y) \;\propto\; p(\theta) \prod_{i=1}^{n} p(y_i \mid \theta).

**Expectation propagation** replaces each awkward factor :math:`p(y_i \mid \theta)` by a tractable
**site approximation** :math:`\tilde{t}_i(\theta)` — typically an unnormalised Gaussian — so that the
product is Gaussian and every quantity is available in closed form.

The iteration
---------------

Sites are refined one at a time. To update site :math:`i`:

1. form the **cavity** distribution by removing that site from the current approximation,
   :math:`q_{-i} \propto q / \tilde{t}_i`;
2. form the **tilted** distribution :math:`q_{-i}(\theta) \, p(y_i \mid \theta)` — the cavity times the
   *true* factor;
3. **match moments**: choose a Gaussian with the same mean and variance as the tilted distribution;
4. divide the cavity back out to recover the new :math:`\tilde{t}_i`.

Each moment-matching step is the local minimiser of :math:`\mathrm{KL}(p \| q)` — the **opposite**
direction from variational inference. That direction is **mass-covering**: it penalises :math:`q` for
missing regions where :math:`p` has mass, so EP tends to produce approximations that are **wider**, with
better-calibrated variances than mean-field VI.

.. code-block:: python

   # sketch: Gaussian sites, natural parameters, one sweep
   for i in range(n):
       cav_prec = q_prec - site_prec[i]                    # 1. cavity
       cav_mean = (q_prec * q_mean - site_prec[i] * site_mean[i]) / cav_prec
       m, v = tilted_moments(cav_mean, cav_prec, y[i])     # 2-3. moment match
       new_prec = 1 / v - cav_prec                          # 4. new site
       site_prec[i], site_mean[i] = new_prec, (m / v - cav_mean * cav_prec) / new_prec
       q_prec, q_mean = 1 / v, m                            # refresh global approximation

Strengths and cautions
------------------------

EP is often strikingly accurate for **latent Gaussian** models — Gaussian-process classification, probit
regression — and it is naturally parallel across sites. It also yields an estimate of the **marginal
likelihood** as a by-product, which VI's ELBO only bounds.

But EP carries no guarantees. It **need not converge** — there is no objective function being decreased,
so cycles are possible and damping is often required. Cavity variances can go **negative**, breaking the
Gaussian assumption. And, like all approximations in this stage, it is silent about its own quality. The
place of EP is alongside VI and Laplace: fast, sometimes excellent, always to be **checked against a
sampler** on a case you can afford to sample.

.. hint::

   **Related lessons:** :doc:`Variational inference <087-variational-inference>`  ·  :doc:`Other approximations <089-other-approximations>`  ·  :doc:`Normal and related mixture approximations <083-normal-and-related-mixture-approximations>`  ·  :doc:`Gaussian process regression <129-gaussian-process-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/expectation-propagation/ <https://insightful-data-lab.com/2025/11/23/expectation-propagation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
