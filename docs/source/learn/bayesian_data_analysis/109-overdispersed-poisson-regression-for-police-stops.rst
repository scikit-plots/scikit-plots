.. _bda-overdispersed-poisson-regression-for-police-stops:

========================================================================
Overdispersed Poisson regression for police stops
========================================================================

**Part 4 · Stage 13 · 🔗 Generalized Linear Models**  ·  Lesson 109 of 144  ·  *advanced*

:doc:`◀ Previous · Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`   ·   :doc:`Next · State-level opinons from national polls ▶ <110-state-level-opinons-from-national-polls>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Counts with too much variance
-------------------------------

The Poisson model forces a rigid tie: variance equals mean. Real count data almost never obey it —
stop counts, disease cases, accident tallies vary far more than Poisson allows, because unmodelled
heterogeneity inflates the spread. This lesson works through the canonical applied example and the fix
for **overdispersion**.

The stop-and-frisk study
--------------------------

Gelman, Fagan and Kiss analysed roughly 125,000 pedestrian stops by the New York police over fifteen
months, asking whether minority pedestrians were stopped more often than a race-neutral policy would
predict. The count of stops of ethnic group :math:`e` in precinct :math:`p` is modelled as Poisson,
with an **offset** for the group's baseline rate of prior arrests — the appropriate denominator, since
stops should be compared against involvement in crime, not raw population:

.. math::

   y_{ep} \sim \mathrm{Poisson}\bigl(n_{ep} \, e^{\mu + \alpha_e + \beta_p + \epsilon_{ep}}\bigr),
   \qquad \epsilon_{ep} \sim \mathrm{N}(0, \sigma^2),

where :math:`\alpha_e` is the ethnic-group effect, :math:`\beta_p` a precinct effect, and
:math:`n_{ep}` the arrest-based offset. The per-observation :math:`\epsilon_{ep}` is the crucial term.

Overdispersion, modelled honestly
-----------------------------------

That :math:`\epsilon_{ep}` — an individual-level normal error inside the log rate — is what turns a
Poisson into an **overdispersed** Poisson. It lets each cell's rate deviate from what the fixed effects
predict, absorbing the excess variance that a plain Poisson would deny. Without it, the model forces
variance to equal mean, the residuals blow past that ceiling, and every standard error comes out far
too small — a confident, wrong answer about a charged question.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5)
       a = pm.Normal("a", 0, 1, shape=n_eth)                # ethnic-group effects
       s_p = pm.HalfNormal("s_p", 1)
       b = pm.Normal("b", 0, s_p, shape=n_prec)             # precinct effects (pooled)
       s_e = pm.HalfNormal("s_e", 1)
       eps = pm.Normal("eps", 0, s_e, shape=n_obs)          # <- overdispersion term
       rate = pm.math.exp(mu + a[eth] + b[prec] + eps) * n_offset
       pm.Poisson("y", rate, observed=stops)

The finding, and the method's point
-------------------------------------

Controlling for precinct and for race-specific arrest rates, the analysis found that Black and Hispanic
pedestrians were stopped **more** than whites — the disparity survived the controls that might have
explained it away. Two methodological lessons carry beyond the application. The **offset** encodes the
right comparison: rates per unit of a relevant baseline, not raw counts. And the **overdispersion term**
is what makes the uncertainty honest — an ordinary Poisson would have reported spuriously precise
estimates on a question where precision claims have real consequences. Diagnosing variance in excess of
the mean, and modelling it, is not optional refinement; it is the difference between a defensible finding
and an artefact.

.. hint::

   **Related lessons:** :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Working with generalized linear models <107-working-with-generalized-linear-models>`  ·  :doc:`Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>`  ·  :doc:`Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/overdispersed-poisson-regression-for-police-stops/ <https://insightful-data-lab.com/2025/12/06/overdispersed-poisson-regression-for-police-stops/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
