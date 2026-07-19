.. _bda-data-collection-models-and-ignorability:

========================================================================
Data-collection models and ignorability
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 051 of 144  ·  *intermediate*

:doc:`◀ Previous · Bayesian inference requires a model for data collection <050-bayesian-inference-requires-a-model-for-data-collection>`   ·   :doc:`Next · Sample surveys ▶ <052-sample-surveys>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When can the design be ignored?
---------------------------------

The previous lesson wrote the joint model :math:`p(y \mid \theta) \, p(I \mid y, \phi)`. Ignorability
is the precise statement of when the second factor can be **dropped** — when the posterior obtained from
modelling :math:`y` alone equals the posterior obtained from modelling :math:`y` **and** :math:`I`.

Two conditions
----------------

Ignorability requires **both**:

* **Missing at random (MAR).** The inclusion mechanism does not depend on the values you failed to
  observe, given the ones you did:

  .. math::

     p(I \mid y_{\text{obs}}, y_{\text{mis}}, \phi) = p(I \mid y_{\text{obs}}, \phi).

* **Parameter distinctness.** The parameters :math:`\theta` of the data model and :math:`\phi` of the
  mechanism are **a priori independent**, :math:`p(\theta, \phi) = p(\theta) p(\phi)`.

Grant both, and the mechanism contributes a factor free of :math:`\theta` that cancels in the
normalisation. The likelihood you wanted to use is the one you may use.

The hierarchy of mechanisms
-----------------------------

* **MCAR** (missing completely at random) — :math:`I` is independent of the data altogether; a simple
  random sample. Strong, and rarely true.
* **MAR** — :math:`I` may depend on **observed** quantities: sampling more heavily in some strata, or
  non-response predicted by recorded covariates. Ignorable, *provided those quantities are in the
  model*.
* **MNAR** (missing not at random) — :math:`I` depends on the unobserved values themselves: people with
  high incomes decline to state their income. **Not ignorable**; the mechanism must be modelled, and the
  answer will depend on assumptions the data cannot check.

The practical corollary
-------------------------

MAR is a statement about a **model**, not about the world: a mechanism that is non-ignorable given no
covariates can become ignorable once the variables driving selection are **included as predictors**.
This is the single most useful fact in the chapter — it is why survey analyses must include the design
variables (strata, cluster, weights' determinants), and why omitting them silently converts an
ignorable design into a biased analysis.

.. code-block:: python

   import pymc as pm
   # Non-response depends on age and region (both recorded) -> MAR *if* they are in the model.
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])   # X includes age, region, strata
       pm.Bernoulli("y", logit_p=X @ beta, observed=y_obs)
       # omit those columns and the same design becomes non-ignorable

Randomisation, then, is not a ritual: assigning treatment by a coin flip makes :math:`I` depend on
nothing but a known probability, guaranteeing ignorability **by design** rather than by assumption.

.. hint::

   **Related lessons:** :doc:`Bayesian inference requires a model for data collection <050-bayesian-inference-requires-a-model-for-data-collection>`  ·  :doc:`Sensitivity and the role of randomization <054-sensitivity-and-the-role-of-randomization>`  ·  :doc:`Observational studies <055-observational-studies>`  ·  :doc:`Multiple imputation <119-multiple-imputation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/data-collection-models-and-ignorability/ <https://insightful-data-lab.com/2025/11/11/data-collection-models-and-ignorability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
