.. _bda-weakly-informative-prior-distributions:

========================================================================
Weakly Informative Prior Distributions
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 019 of 144  ·  *beginner*

:doc:`◀ Previous · Noninformative Prior Distributions <018-noninformative-prior-distributions>`   ·   :doc:`Next · Averaging Over Nuisance Parameters ▶ <020-averaging-over-nuisance-parameters>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The sensible middle
---------------------

Between an informative prior that asserts a specific belief and a noninformative one that tries to
assert nothing lies the **weakly informative** prior — the modern default. It deliberately contains
**less** information than is actually available, but enough to **regularise**: to keep the inference
inside the range of physically or scientifically plausible values, and to stabilise estimation when the
data are weak.

What it does
--------------

A weakly informative prior is chosen by asking what values are **conceivable**, not what values are
*likely*. On a standardised logistic-regression coefficient, a :math:`\mathrm{N}(0, 2.5^2)` prior says:
an odds ratio of 2 or 5 is unremarkable, an odds ratio of :math:`10^6` is not. That single, mild
statement:

* **prevents separation** — the infinite coefficients of a perfectly separated logistic fit become
  finite (a problem met again in Part IV);
* **tames weak identification** — parameters the data barely constrain get finite posteriors instead
  of wandering chains;
* **shrinks noise** — as in the cancer-rate example, small-:math:`n` estimates stop being extreme;
* leaves conclusions **essentially unchanged** where the data are strong.

Common defaults
-----------------

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # coefficients (predictors standardised): mild, symmetric, finite-tailed
       beta = pm.Normal("beta", mu=0, sigma=2.5, shape=k)
       # scale parameters: positive, heavy-tailed near zero, no hard upper bound
       sigma = pm.HalfNormal("sigma", sigma=1)        # or pm.HalfCauchy("sigma", 1)

For **variance** parameters in hierarchies, a **half-normal** or **half-Cauchy** on the standard
deviation is standard practice — it allows the group-level scale to be near zero (complete pooling)
without the pathologies of the once-popular inverse-gamma-with-tiny-parameters.

Scale matters
---------------

A weakly informative prior is a statement about **units**. :math:`\mathrm{N}(0, 2.5^2)` is mild for a
coefficient on a standardised predictor and wildly informative for one measured in dollars, so
**standardise predictors** (or scale the prior to the data). And the standard defence applies: run the
**prior predictive check** — simulate data from the prior alone and confirm the simulated datasets are
merely varied, not absurd. A prior generating impossible data is too weak, not too strong.

.. hint::

   **Related lessons:** :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`  ·  :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`  ·  :doc:`Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/weakly-informative-prior-distributions/ <https://insightful-data-lab.com/2025/11/09/weakly-informative-prior-distributions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
