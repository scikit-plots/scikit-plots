.. _bda-sample-surveys:

========================================================================
Sample surveys
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 052 of 144  ·  *intermediate*

:doc:`◀ Previous · Data-collection models and ignorability <051-data-collection-models-and-ignorability>`   ·   :doc:`Next · Designed experiments ▶ <053-designed-experiments>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Inference for a finite population
-----------------------------------

Survey inference has an unusual target. You do not want a parameter of an infinite superpopulation; you
want a **finite-population quantity** — the mean income of *these* :math:`N` households, of whom you
sampled :math:`n`. The Bayesian formulation is disarmingly direct: the unsampled values
:math:`y_{\text{mis}}` are simply **unknowns**, and the population mean is a function of them.

.. math::

   \bar{Y} = \frac{1}{N}\left( \sum_{i \in \text{sample}} y_i
             + \sum_{i \notin \text{sample}} y_i \right).

Draw :math:`y_{\text{mis}}` from its posterior predictive distribution, compute :math:`\bar{Y}` for each
draw, and you have the posterior for the population mean — uncertainty about the unsampled units
included, and the :math:`n/N` finite-population correction appearing automatically rather than being
bolted on.

Designs, and what they demand
-------------------------------

Real surveys are not simple random samples. **Stratified** designs sample strata at different rates;
**cluster** designs sample groups then units within them; **unequal probability** designs oversample
rare populations. Under each, :math:`I` depends on design variables — and by the ignorability lesson the
design is ignorable **exactly when those variables are in the model**. So:

* stratification → include stratum as a predictor (naturally, **hierarchically**);
* clustering → a group-level random effect, which is a hierarchical model;
* unequal probabilities → model whatever determined them.

Design-based practice handles this with **weights**; the model-based route puts the same information in
as **predictors**, and gains partial pooling for small strata for free.

Multilevel regression and poststratification
----------------------------------------------

The modern synthesis, **MRP**: fit a hierarchical regression of the outcome on demographic and
geographic cells, then **poststratify** — reweight the predicted cell means by the known population cell
counts from a census.

.. code-block:: python

   import numpy as np, pymc as pm
   with pm.Model():                              # 1. multilevel regression over cells
       a_state = pm.Normal("a_state", 0, pm.HalfNormal("s_state", 1), shape=n_states)
       a_age   = pm.Normal("a_age",   0, pm.HalfNormal("s_age", 1),   shape=n_ages)
       pm.Bernoulli("y", logit_p=a_state[state] + a_age[age], observed=y)
   # 2. poststratify: weight predicted cell means by census population counts
   theta_pop = (cell_pred * N_cells).sum() / N_cells.sum()

MRP produces stable estimates for small subgroups (a state with twelve respondents borrows from the
others) and corrects unrepresentative samples — the reason it can extract state-level estimates from
national polls, and even from famously non-representative online panels.

The limits
------------

MRP corrects for what it **models**. If non-response depends on something unmeasured — political
enthusiasm, say, not captured by age, race, education and region — the design remains **MNAR** and no
amount of poststratification fixes it. And poststratification needs **population cell counts**, which
constrains which variables you may adjust for. The honest summary: surveys are a missing-data problem,
and their difficulty is exactly the difficulty of knowing why people did not answer.

.. hint::

   **Related lessons:** :doc:`Data-collection models and ignorability <051-data-collection-models-and-ignorability>`  ·  :doc:`Designed experiments <053-designed-experiments>`  ·  :doc:`State-level opinons from national polls <110-state-level-opinons-from-national-polls>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/sample-surveys/ <https://insightful-data-lab.com/2025/11/11/sample-surveys/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
