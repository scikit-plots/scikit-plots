.. _bda-large-sample-theory:

========================================================================
Large-Sample Theory
========================================================================

**Part 1 · Stage 4 · 📏 Asymptotics & Frequentist Ties**  ·  Lesson 029 of 144  ·  *beginner*

:doc:`◀ Previous · Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`   ·   :doc:`Next · Counterexamples to large-sample (asymptotic) Bayesian theorems ▶ <030-counterexamples-to-large-sample-asymptotic-bayesian-theorems>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What happens as data pile up
------------------------------

The normal approximation of the last lesson is not merely convenient — under regularity conditions it
becomes **exact in the limit**. Large-sample theory describes the two things a posterior does as
:math:`n \to \infty`: it **concentrates**, and its shape becomes **normal**.

Consistency: the posterior concentrates
-----------------------------------------

If the model is correctly specified and the prior gives positive probability to a neighbourhood of the
true value :math:`\theta_0`, the posterior piles up on :math:`\theta_0`: for any :math:`\epsilon > 0`,
:math:`\Pr(|\theta - \theta_0| > \epsilon \mid y) \to 0`. Two consequences follow, and both were
foreshadowed in Stage 2. The **prior washes out** — its fixed weight :math:`\alpha + \beta` is swamped
by growing :math:`n` — so analysts with different reasonable priors are driven into agreement. And a
prior that assigns **zero** probability to the truth can never recover: *Cromwell's rule*, the reason
weakly informative beats dogmatically informative.

Bernstein–von Mises: the shape becomes normal
-----------------------------------------------

More sharply, the **Bernstein–von Mises theorem** says the posterior converges (in total variation) to
a normal centred at the maximum-likelihood estimator, with covariance the inverse Fisher information:

.. math::

   p(\theta \mid y) \;\longrightarrow\;
   \mathrm{N}\!\left(\hat{\theta}_{\mathrm{MLE}},\; \frac{1}{n}\, I(\theta_0)^{-1}\right).

The posterior standard deviation therefore shrinks like :math:`1/\sqrt{n}` — halving it costs **four
times** the data. The prior enters the limit **not at all**; only the likelihood survives.

Why it matters
----------------

Bernstein–von Mises is the bridge between the two schools: asymptotically, a Bayesian **credible**
interval *is* a frequentist **confidence** interval, with correct long-run coverage. It licenses the
normal approximation as a fast substitute for MCMC in large, regular problems, and it explains why
Bayesian and classical answers so often agree when data are plentiful.

.. code-block:: python

   import numpy as np
   from scipy import stats
   theta_true, a, b = 0.3, 2, 8               # informative prior, mean 0.2
   for n in (10, 100, 10_000):
       y = stats.binom(n, theta_true).rvs(random_state=0)
       post = stats.beta(a + y, b + n - y)
       print(n, round(post.mean(), 3), round(post.std(), 4))
   # posterior mean -> 0.3 ; posterior sd shrinks ~ 1/sqrt(n)

The fine print
----------------

Every word above rests on **regularity conditions**: the true value lies in the **interior** of the
parameter space, the parameter is **identified**, the dimension is **fixed** (not growing with
:math:`n`), and the model is **correctly specified**. Under misspecification the posterior still
becomes normal around the best-fitting parameter, but its variance is **no longer** the inverse Fisher
information — so credible intervals lose their coverage guarantee. The next lesson collects the cases
where these conditions break.

.. hint::

   **Related lessons:** :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`  ·  :doc:`Counterexamples to large-sample (asymptotic) Bayesian theorems <030-counterexamples-to-large-sample-asymptotic-bayesian-theorems>`  ·  :doc:`Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`  ·  :doc:`Frequency Evaluations of Bayesian Inferences <031-frequency-evaluations-of-bayesian-inferences>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/large-sample-theory/ <https://insightful-data-lab.com/2025/11/09/large-sample-theory/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
