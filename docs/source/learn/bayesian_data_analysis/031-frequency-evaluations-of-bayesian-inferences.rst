.. _bda-frequency-evaluations-of-bayesian-inferences:

========================================================================
Frequency Evaluations of Bayesian Inferences
========================================================================

**Part 1 · Stage 4 · 📏 Asymptotics & Frequentist Ties**  ·  Lesson 031 of 144  ·  *beginner*

:doc:`◀ Previous · Counterexamples to large-sample (asymptotic) Bayesian theorems <030-counterexamples-to-large-sample-asymptotic-bayesian-theorems>`   ·   :doc:`Next · Bayesian interpretations of other statistical methods ▶ <032-bayesian-interpretations-of-other-statistical-methods>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A different question
----------------------

Bayesian inference conditions on the data you have. **Frequency evaluation** asks a different, entirely
legitimate question: if this procedure were applied repeatedly, across many datasets, how would it
behave? The two views are not rivals — the frequency properties of a Bayesian method are a way of
**checking** it, and Gelman treats them as diagnostics rather than definitions.

Coverage
----------

The headline criterion is **calibration** of intervals: do 95% credible intervals contain the true
parameter 95% of the time? Bernstein–von Mises promises this **asymptotically**. In small samples the
answer depends on the prior — and often, informatively, in the Bayesian method's favour: a
weakly informative prior that shrinks noisy estimates can achieve **better** coverage and much smaller
average error than an unbiased estimator, particularly in the many-parameter, small-:math:`n` settings
(the cancer-rate map) where classical methods flounder.

Simulate to check
-------------------

The evaluation is mechanical: draw parameters from the prior, simulate data from the model, fit, and
count. This **simulation-based calibration** checks the prior, the likelihood **and** the sampler
together — if a 50% interval covers 70% of the time, something is wrong somewhere.

.. code-block:: python

   import numpy as np
   from scipy import stats
   rng = np.random.default_rng(0)
   n, a, b, cover = 20, 2, 8, 0
   for _ in range(2000):
       theta = stats.beta(a, b).rvs(random_state=rng)      # draw from the prior
       y = stats.binom(n, theta).rvs(random_state=rng)     # simulate data
       lo, hi = stats.beta(a + y, b + n - y).interval(0.95)
       cover += lo <= theta <= hi
   cover / 2000        # ≈ 0.95 when prior, model and computation all agree

Bias, variance, and honesty
-----------------------------

Bayesian estimators are typically **biased** — that is what shrinkage means — and typically achieve
lower **mean squared error** for it, trading a little bias for a lot of variance. Frequency evaluation
makes the bargain explicit rather than hiding it. Two honest limits: the calibration above averages
over the **prior**, so it certifies the procedure only if that prior is the one you believe; and no
amount of frequency checking rescues a **misspecified likelihood**. Coverage is a necessary condition
for trust, not a sufficient one.

.. hint::

   **Related lessons:** :doc:`Large-Sample Theory <029-large-sample-theory>`  ·  :doc:`Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>`  ·  :doc:`Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>`  ·  :doc:`Posterior predictive checking <042-posterior-predictive-checking>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/frequency-evaluations-of-bayesian-inferences/ <https://insightful-data-lab.com/2025/11/09/frequency-evaluations-of-bayesian-inferences/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
