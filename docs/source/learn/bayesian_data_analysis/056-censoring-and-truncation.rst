.. _bda-censoring-and-truncation:

========================================================================
Censoring and truncation
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 056 of 144  ·  *intermediate*

:doc:`◀ Previous · Observational studies <055-observational-studies>`   ·   :doc:`Next · Bayesian decision theory in diﬀerent contexts ▶ <057-bayesian-decision-theory-in-different-contexts>`   ·   :doc:`↑ Section <index>`


Two ways data go missing
--------------------------

Both words describe incomplete observation, and confusing them yields a wrong likelihood. The
distinction is what you know about the units you did not fully see.

* **Censoring**: the unit **is** in your dataset, but its value is known only to lie in a range. A
  patient still alive at the end of a trial has survival time :math:`> c`. A scale reading "over 100 kg".
* **Truncation**: the unit is **absent entirely**. Only patients who survived long enough to enrol
  appear; light sources fainter than the telescope's limit are never recorded. You do not know how many
  you missed.

The likelihoods differ
------------------------

For **censoring**, the censored observation contributes the probability of the event it represents —
the survival function, not the density:

.. math::

   p(y \mid \theta) = \prod_{i \, \text{obs}} f(y_i \mid \theta)
                      \prod_{i \, \text{cens}} \bigl[1 - F(c_i \mid \theta)\bigr] .

For **truncation**, every observed value must be **renormalised** by the probability of being observed
at all, because the sample space itself is restricted:

.. math::

   p(y \mid \theta) = \prod_{i} \frac{f(y_i \mid \theta)}{\Pr(\text{observed} \mid \theta)}
                    = \prod_{i} \frac{f(y_i \mid \theta)}{1 - F(c \mid \theta)} .

Ignore the denominator and you fit a model to a biased sample — the truncated-normal demonstration from
the data-collection lesson, where the sample mean converged confidently to the wrong number.

In code
---------

Both are one line in a modern PPL, which is precisely why the distinction must be made **before**
coding:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu, sigma = pm.Normal("mu", 0, 10), pm.HalfNormal("sigma", 5)
       # censoring: value known only to exceed c (unit IS in the data)
       pm.Censored("y_cens", pm.Normal.dist(mu, sigma), lower=None, upper=c, observed=y)
       # truncation: values beyond c never enter the sample at all
       pm.Truncated("y_trunc", pm.Normal.dist(mu, sigma), lower=c, observed=y)

Censoring is ignorable, truncation less so
--------------------------------------------

Where the censoring **time** is fixed in advance, or depends only on observed quantities, the mechanism
is ignorable in the technical sense — the censored contributions must appear in the likelihood, but no
separate model for :math:`\phi` is needed. When censoring depends on the **unobserved** value itself —
patients withdrawing *because* they are deteriorating — the mechanism is **MNAR**, and the analysis
must model why they left. That is not a computational difficulty but an **identification** one: the
data are silent, and the answer will move with the assumption. Report the sensitivity.

.. hint::

   **Related lessons:** :doc:`Bayesian inference requires a model for data collection <050-bayesian-inference-requires-a-model-for-data-collection>`  ·  :doc:`Data-collection models and ignorability <051-data-collection-models-and-ignorability>`  ·  :doc:`Missing values with counted data <122-missing-values-with-counted-data>`  ·  :doc:`Multiple imputation <119-multiple-imputation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/censoring-and-truncation/ <https://insightful-data-lab.com/2025/11/11/censoring-and-truncation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
