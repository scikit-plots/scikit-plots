.. _bda-other-standard-single-parameter-models:

========================================================================
Other Standard Single-Parameter Models
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 016 of 144  ·  *beginner*

:doc:`◀ Previous · Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`   ·   :doc:`Next · Informative Prior Distribution for Cancer Rates ▶ <017-informative-prior-distribution-for-cancer-rates>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The same pattern, different data
----------------------------------

Binomial and normal are two instances of one recipe. Whenever the likelihood belongs to the
**exponential family**, a conjugate prior exists, the posterior stays in the prior's family, and the
update amounts to **adding sufficient statistics** to the hyperparameters. Three more cases cover most
of applied practice.

Poisson–Gamma (counts)
------------------------

For counts, :math:`y_i \sim \mathrm{Poisson}(\theta)` with a :math:`\mathrm{Gamma}(\alpha, \beta)`
prior on the rate:

.. math::

   \theta \mid y \;\sim\; \mathrm{Gamma}\Bigl(\alpha + \textstyle\sum_i y_i,\; \beta + n\Bigr),
   \qquad
   \mathrm{E}[\theta \mid y] = \frac{\alpha + \sum_i y_i}{\beta + n}.

The hyperparameters read as **pseudo-events** :math:`\alpha` observed in **pseudo-exposure**
:math:`\beta`, and the posterior mean is again a weighted average of the prior mean
:math:`\alpha/\beta` and the sample rate :math:`\bar{y}`. With exposures :math:`x_i` (person-years,
say), the update uses :math:`\beta + \sum_i x_i`.

Exponential and normal-variance
---------------------------------

For waiting times, :math:`y_i \sim \mathrm{Exponential}(\theta)` also takes a **Gamma** conjugate
prior, updating to :math:`\mathrm{Gamma}(\alpha + n,\ \beta + \sum_i y_i)`. And for a normal with
**known mean and unknown variance**, the conjugate prior on :math:`\sigma^2` is the
**inverse-gamma** (equivalently, scaled inverse-:math:`\chi^2`) — the piece needed to complete the
normal model in the next stage.

.. code-block:: python

   from scipy import stats
   y = [3, 5, 2, 4]                       # counts over n = 4 periods
   a, b = 2, 1                            # Gamma prior: 2 pseudo-events / 1 pseudo-period
   post = stats.gamma(a + sum(y), scale=1 / (b + len(y)))
   post.mean(), post.interval(0.95)       # posterior rate and 95% CI

Convenient, not sacred
------------------------

Conjugacy is prized for being **exact, instant and interpretable** — ideal for teaching, for sequential
updating, and as a building block inside larger samplers (Gibbs, in Part III). But conjugate families
exist for **algebraic convenience**, not because they encode anyone's real prior beliefs, and most
useful models have no conjugate form at all. Modern practice keeps the intuition — priors as pseudo-data,
posteriors as compromises — and reaches for **MCMC** whenever the model demands it.

.. hint::

   **Related lessons:** :doc:`Normal Distribution with Known Variance <015-normal-distribution-with-known-variance>`  ·  :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`  ·  :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/other-standard-single-parameter-models/ <https://insightful-data-lab.com/2025/11/09/other-standard-single-parameter-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
