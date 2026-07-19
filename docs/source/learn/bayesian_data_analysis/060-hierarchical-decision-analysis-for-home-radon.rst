.. _bda-hierarchical-decision-analysis-for-home-radon:

========================================================================
Hierarchical decision analysis for home radon
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 060 of 144  ·  *intermediate*

:doc:`◀ Previous · Multistage decision making: medical screening <059-multistage-decision-making-medical-screening>`   ·   :doc:`Next · Personal vs. institutional decision analysis ▶ <061-personal-vs-institutional-decision-analysis>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A decision for every county
-----------------------------

Radon is a radioactive soil gas that accumulates indoors and causes lung cancer. A homeowner may
**measure** their house (cheap, noisy) and, depending on the reading, **remediate** (expensive,
effective). The recommendation should differ by county — some have far higher radon — but many counties
have only a handful of measurements. This is the cancer-rate problem, now attached to a decision.

Hierarchical prediction first
-------------------------------

Model log radon concentration hierarchically: houses within counties, counties drawn from a population,
with a county-level predictor (soil uranium) explaining part of the variation. Partial pooling gives
every county a stable predictive distribution, and a data-poor county borrows from the rest:

.. math::

   \log y_{ij} \sim \mathrm{N}(\alpha_j + \beta \, x_{ij}, \; \sigma_y^2), \qquad
   \alpha_j \sim \mathrm{N}(\gamma_0 + \gamma_1 u_j, \; \sigma_\alpha^2),

with :math:`x_{ij}` the basement indicator and :math:`u_j` county uranium.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       g0, g1 = pm.Normal("g0", 0, 5), pm.Normal("g1", 0, 5)
       s_a = pm.HalfNormal("s_a", 1)
       a = pm.Normal("a", g0 + g1 * u, s_a, shape=n_counties)     # county intercepts
       b = pm.Normal("b", 0, 1)
       pm.Normal("y", a[county] + b * basement, pm.HalfNormal("s_y", 1), observed=log_radon)

Then the decision
-------------------

Remediation costs a known amount; radon exposure costs expected life-years, monetised. For each house
the actions are *do nothing*, *measure then decide*, or *remediate immediately*. Expected utility is
computed **over the posterior predictive distribution of that house's radon level** — which is exactly
what the hierarchical model supplies, uncertainty included.

The result is a **decision rule that varies by county**: in high-radon counties, remediate without
measuring; in low-radon counties, do nothing; in the intermediate band — where the predictive
distribution straddles the action threshold — **measure first**, because there the measurement changes
what you would do. That middle region is precisely where the expected value of information is positive.

Why the hierarchy earns its keep
----------------------------------

Use each county's raw mean and the small counties give absurd recommendations, exactly as raw cancer
rates gave absurd rankings. Pool completely and every county gets the same advice, wasting the real
variation. Partial pooling produces recommendations that are **stable where data are thin and
responsive where data are plentiful** — and because the decision uses the **predictive** distribution for
an individual house, it correctly accounts for within-county spread, not merely uncertainty about the
county mean. The caveat is the usual one: the answer depends on the monetary value placed on a life-year,
and that number should be stated, not smuggled.

.. hint::

   **Related lessons:** :doc:`Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>`  ·  :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/hierarchical-decision-analysis-for-home-radon/ <https://insightful-data-lab.com/2025/11/11/hierarchical-decision-analysis-for-home-radon/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
