.. _bda-sensitivity-and-the-role-of-randomization:

========================================================================
Sensitivity and the role of randomization
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 054 of 144  ·  *intermediate*

:doc:`◀ Previous · Designed experiments <053-designed-experiments>`   ·   :doc:`Next · Observational studies ▶ <055-observational-studies>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What randomisation actually does
----------------------------------

Randomisation is often described as "balancing the covariates". That is a consequence, not the
mechanism. What randomisation does, precisely, is make the assignment mechanism **known and independent
of the potential outcomes**, so that :math:`p(W \mid y(0), y(1), \phi) = p(W \mid \phi)`. Ignorability
then follows **by construction**, and the analyst need not enumerate the confounders — including the
ones nobody thought of.

Not a guarantee of balance
----------------------------

In any single randomised experiment, covariates may be **imbalanced** by chance: the treatment group
happens to be older. Randomisation does not prevent this; it makes the *probability* of each imbalance
known. The Bayesian response is neither to re-randomise nor to ignore it, but to **condition on the
observed covariates** — include age in the model. Adjustment is legitimate because it is *conditioning*,
not because it repairs a broken randomisation.

Sensitivity analysis, where ignorability is assumed
-----------------------------------------------------

Where the design does **not** guarantee ignorability, the conclusions rest on an assumption the data
cannot verify. The honest reply is **sensitivity analysis**: posit an unmeasured confounder of a given
strength, refit, and report how strong it would have to be to overturn the finding.

.. code-block:: python

   import numpy as np, pymc as pm
   # How strong must an unmeasured confounder U be to erase the effect?
   for gamma in [0.0, 0.25, 0.5, 1.0]:                # U's effect on the outcome
       with pm.Model():
           U   = pm.Normal("U", 0, 1, shape=n)        # unobserved, correlated with W
           tau = pm.Normal("tau", 0, 1)
           mu  = alpha + tau * W + gamma * U
           pm.Normal("y", mu, sigma, observed=y)
           # report the posterior for tau at each assumed gamma

If the effect survives confounders far stronger than any measured covariate, the conclusion is robust.
If a modest one erases it, say so.

The limits
------------

Randomisation protects against **confounding**, and against nothing else. It does not fix
**non-compliance** (assignment is not receipt), **attrition** (drop-out may depend on outcomes,
reintroducing MNAR), **interference** (one unit's treatment affecting another's outcome, violating the
stable-unit assumption), or **generalisation** to a population the units were not sampled from. Each is
a separate modelling problem. Randomisation is the cheapest way to buy ignorability of *assignment* — a
genuine and rare gift — but the phrase "randomised, therefore unbiased" quietly assumes that nothing
else went wrong.

.. hint::

   **Related lessons:** :doc:`Designed experiments <053-designed-experiments>`  ·  :doc:`Observational studies <055-observational-studies>`  ·  :doc:`Data-collection models and ignorability <051-data-collection-models-and-ignorability>`  ·  :doc:`Aspects of robustness <113-aspects-of-robustness>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/sensitivity-and-the-role-of-randomization/ <https://insightful-data-lab.com/2025/11/11/sensitivity-and-the-role-of-randomization/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
