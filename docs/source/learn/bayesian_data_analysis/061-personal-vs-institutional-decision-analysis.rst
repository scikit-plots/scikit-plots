.. _bda-personal-vs-institutional-decision-analysis:

========================================================================
Personal vs. institutional decision analysis
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 061 of 144  ·  *intermediate*

:doc:`◀ Previous · Hierarchical decision analysis for home radon <060-hierarchical-decision-analysis-for-home-radon>`   ·   :doc:`Next · Numerical integration ▶ <062-numerical-integration>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Whose utility?
----------------

Expected-utility maximisation presumes a utility function. But **whose**? A homeowner deciding whether
to remediate radon and a public-health agency deciding what to recommend nationally face the same
posterior and reach **different optimal actions** — not because one is irrational, but because their
utilities and their information genuinely differ.

Three ways they diverge
-------------------------

* **The value of a risk reduction varies between people.** The dollar amount an individual will pay to
  remove a given increment of lung-cancer risk is not universal; it varies with wealth, age, risk
  aversion and belief. An institution must adopt a single figure; the individual has their own.
* **Individuals condition on more.** A homeowner knows their basement, their habits, whether they smoke
  (radon risk is far higher for smokers). Their posterior for their **own** exposure is sharper than the
  agency's posterior for a house drawn from their county.
* **The decision spaces differ.** The individual chooses an action; the institution chooses a **rule**,
  applied to millions of houses. A rule optimal in aggregate is not optimal for most of the individuals
  it binds.

Aggregating individual decisions
----------------------------------

The subtle point is that a policy is not evaluated by its expected utility for a typical house, but by
the **aggregate consequences of the individual decisions it induces**. Set a recommendation threshold
and some households will remediate who need not, and others will not measure who should. The
distribution of outcomes across the population is the object of interest, and it must be simulated:

.. code-block:: python

   import numpy as np
   remediation_cost = 2000.0                               # dollars per house
   # posterior predictive radon for each of many houses, from the hierarchical model
   radon = ppc["radon"]                                    # (S draws, n houses)
   for threshold in [2.0, 4.0, 8.0]:                       # pCi/L action level
       remediate = radon.mean(axis=0) > threshold          # rule applied per house
       cost = remediate.sum() * remediation_cost
       exposure = radon[:, ~remediate].mean()              # residual risk borne by the rest
       print(threshold, cost, exposure)                    # the trade-off a policy makes

Where the two must disagree
-----------------------------

An institution's recommendation is rationally **more conservative** in some directions (it bears
reputational and legal cost for under-warning) and **less** in others (it cannot price an individual's
risk aversion). It also faces genuinely collective considerations — equity across households, the cost
of a programme, the credibility of future advice — which never appear in a personal utility.

The Bayesian contribution is not to dissolve this conflict but to **make it explicit**. The posterior is
shared; the utilities are not. When a personal and an institutional analysis disagree, the disagreement
should be traceable to a **named** difference in utility or information — and that is a far healthier
disagreement than one hidden inside a threshold nobody can justify.

.. hint::

   **Related lessons:** :doc:`Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`  ·  :doc:`Hierarchical decision analysis for home radon <060-hierarchical-decision-analysis-for-home-radon>`  ·  :doc:`Multistage decision making: medical screening <059-multistage-decision-making-medical-screening>`  ·  :doc:`Probability as a Measure of Uncertainty <005-probability-as-a-measure-of-uncertainty>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/personal-vs-institutional-decision-analysis/ <https://insightful-data-lab.com/2025/11/11/personal-vs-institutional-decision-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
