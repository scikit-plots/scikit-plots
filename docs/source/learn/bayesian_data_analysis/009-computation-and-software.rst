.. _bda-computation-and-software:

========================================================================
Computation and Software
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 009 of 144  ·  *beginner*

:doc:`◀ Previous · Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>`   ·   :doc:`Next · Bayesian Inference in Applied Statistics ▶ <010-bayesian-inference-in-applied-statistics>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Why computation matters
-------------------------

Bayes' rule is one line, but the posterior it defines is usually **intractable**: the evidence
:math:`p(y) = \int p(y \mid \theta) p(\theta) d\theta` has no closed form once the model has more than
a couple of parameters. For most of the twentieth century that was a fatal obstacle. What made
Bayesian analysis practical is **computation** — and specifically, the decision to stop trying to
*evaluate* posteriors and instead **sample** from them.

Samples are enough
--------------------

The pivotal insight is that a large collection of draws :math:`\theta^{(1)}, \dots, \theta^{(S)}` from
:math:`p(\theta \mid y)` answers every question you would have asked of the density. Means, intervals,
tail probabilities, and the posterior of **any function** of the parameters are all obtained by
summarising draws:

.. math::

   \mathrm{E}[h(\theta) \mid y] \;\approx\; \frac{1}{S} \sum_{s=1}^{S} h\bigl(\theta^{(s)}\bigr),

with Monte Carlo error shrinking like :math:`1/\sqrt{S}`. Simulation also propagates uncertainty for
free — no delta method, no Jacobians (as the previous lesson noted).

The modern stack
------------------

In Python the workflow is standardised. **``scipy.stats``** covers the conjugate and closed-form cases
directly; **PyMC** (or **Stan**) expresses a model declaratively and samples it with **HMC/NUTS**; and
**ArviZ** handles the diagnostics and plots that step 3 demands:

.. code-block:: python

   import pymc as pm, arviz as az

   with pm.Model() as m:
       theta = pm.Beta("theta", 1, 1)
       pm.Binomial("y", n=10, p=theta, observed=8)
       idata = pm.sample(2000, tune=1000, chains=4)

   az.summary(idata)      # mean, sd, 94% HDI, r_hat, ess
   az.plot_trace(idata)   # visual convergence check

Never trust unchecked draws
-----------------------------

Sampling is approximate, so the output must be **audited** before it is believed: :math:`\hat{R}`
close to 1.0 (chains agree), an adequate **effective sample size**, and **no divergences**. A sampler
that has not converged produces confident nonsense. This is why the computation stages of this course
(Part III) devote as much attention to **diagnosing** samplers as to running them — and why "the folk
theorem" holds that computational trouble usually signals a problem with the **model**, not just the
algorithm.

.. hint::

   **Related lessons:** :doc:`The three steps of Bayesian data analysis <001-the-three-steps-of-bayesian-data-analysis>`  ·  :doc:`Bayesian Inference in Applied Statistics <010-bayesian-inference-in-applied-statistics>`  ·  :doc:`Inference and assessing convergence <072-inference-and-assessing-convergence>`  ·  :doc:`Stan: developing a computing environment <080-stan-developing-a-computing-environment>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/computation-and-software/ <https://insightful-data-lab.com/2025/11/09/computation-and-software/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
