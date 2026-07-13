.. _bda-weakly-informative-priors-for-variance-parameters:

========================================================================
Weakly Informative Priors for Variance Parameters
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 039 of 144  ·  *beginner*

:doc:`◀ Previous · Hierarchical modeling applied to a meta-analysis <038-hierarchical-modeling-applied-to-a-meta-analysis>`   ·   :doc:`Next · The Place of Model Checking in Applied Bayesian Statistics ▶ <040-the-place-of-model-checking-in-applied-bayesian-statistics>`   ·   :doc:`↑ Section <index>`


The parameter that decides everything
---------------------------------------

In a hierarchical model, :math:`\tau` — the group-level standard deviation — governs how much the
groups pool. It is also the parameter the data constrain **least**, especially when the number of
groups :math:`J` is small (eight schools: :math:`J = 8`). So the prior on :math:`\tau` is not a
formality; it can decide the answer.

Why the old default failed
----------------------------

For years the reflex was an **inverse-gamma**(:math:`\epsilon, \epsilon`) prior on :math:`\tau^2`, with
:math:`\epsilon = 0.001`, chosen because it is conjugate and "nearly noninformative". Gelman's 2006
analysis showed it is neither. As :math:`\epsilon \to 0` the prior approaches an improper limit whose
posterior may not exist; and for any small :math:`\epsilon` the prior has **almost no mass near zero**
while placing weight far out in the tail. When the data genuinely suggest :math:`\tau \approx 0` — the
eight-schools case — this prior **fights them**, inflating :math:`\tau` and under-pooling. Worse, the
answer is sensitive to :math:`\epsilon`, a number chosen for its irrelevance.

What to use instead
---------------------

Put the prior on the **standard deviation** :math:`\tau`, not the variance, and choose a density that is
**positive at zero** and has a finite scale:

* **half-normal** — :math:`\tau \sim \mathrm{N}^{+}(0, s)`, light tails, a sensible default when a rough
  scale is known;
* **half-Cauchy** — :math:`\tau \sim \mathrm{C}^{+}(0, s)`, heavy-tailed, allowing large
  :math:`\tau` if the data insist while still regularising;
* **uniform on** :math:`(0, A)` — acceptable when :math:`J` is large and :math:`A` is generous.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfNormal("tau", sigma=5)     # or pm.HalfCauchy("tau", beta=5)
       # NOT: pm.InverseGamma("tau2", alpha=0.001, beta=0.001)
       ...

Scale, and honesty
--------------------

The scale :math:`s` is a real statement: it should be large relative to the plausible group-level
variation and small relative to absurdity. On SAT points (sd ≈ 100) a half-Cauchy(0, 5) says
between-school effects of a few points are ordinary and of fifty points are not. **Standardise, or
scale the prior to the data.**

Two closing notes. Allowing :math:`\tau` to be near zero is a **feature** — the model can discover that
complete pooling is right — but it creates the funnel geometry that demands the non-centred
parameterisation. And with very small :math:`J` (say :math:`J \le 5`), no prior is truly weak: report
the sensitivity, because the prior is doing visible work.

.. hint::

   **Related lessons:** :doc:`Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`  ·  :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`  ·  :doc:`Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`  ·  :doc:`Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/weakly-informative-priors-for-variance-parameters/ <https://insightful-data-lab.com/2025/11/10/weakly-informative-priors-for-variance-parameters/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
