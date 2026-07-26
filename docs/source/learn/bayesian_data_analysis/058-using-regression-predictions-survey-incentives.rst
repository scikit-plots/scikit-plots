.. _bda-using-regression-predictions-survey-incentives:

========================================================================
Using regression predictions: survey incentives
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 058 of 144  ·  *intermediate*

:doc:`◀ Previous · Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`   ·   :doc:`Next · Multistage decision making: medical screening ▶ <059-multistage-decision-making-medical-screening>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Should you pay respondents?
-----------------------------

A survey organisation must choose whether to offer respondents an incentive, of what size, in what
form, and when. The stakes are concrete: incentives cost money but raise **response rates**, and a
higher response rate reduces both the number of calls needed and the nonresponse bias. This is a
cost–benefit problem, and Gelman, Stevens and Chan turned it into a worked Bayesian decision analysis.

Meta-analysis feeding a decision
----------------------------------

There is no single experiment that answers the question, so the analysis proceeds in two stages. First,
a **hierarchical meta-analysis** of many surveys' incentive experiments estimates the effect of incentive
value, timing and mode on response rate. The design variables matter:

* **prepaid** incentives (sent with the request) versus **postpaid** (paid on completion);
* the **value** in dollars, whose effect need not be linear;
* the survey's **burden** and mode.

The regression's output is a posterior for the **expected increase in response rate** as a function of
the incentive — with uncertainty, and with partial pooling across the studies, since the surveys are
exchangeable but not identical.

From response rate to utility
-------------------------------

The second stage converts that posterior into money. The utility of an incentive scheme is the
**net cost per respondent**, combining the incentive paid to everyone contacted, the interviewer time
saved by fewer callbacks, and the value placed on a marginal completed interview:

.. code-block:: python

   import numpy as np
   # posterior draws of the incentive's effect on response rate, from the meta-analysis
   d_rate = idata.posterior["beta_incentive"].values.ravel()

   def net_cost_per_respondent(incentive, d_rate, prepaid, base_rate=0.30,
                               call_cost=1.25, calls_per_contact=8):
       rate = base_rate + d_rate * incentive
       # prepaid: paid to everyone contacted; postpaid: paid only to respondents
       paid = incentive if prepaid else incentive / rate
       interviewing = call_cost * calls_per_contact / rate
       return paid + interviewing

   for inc in [0, 5, 10, 20]:
       c = net_cost_per_respondent(inc, d_rate, prepaid=True)
       print(inc, c.mean(), np.percentile(c, [2.5, 97.5]))   # posterior cost, with uncertainty

The lessons
-------------

Three, and they generalise. **The decision needs the whole posterior**, because cost is a nonlinear
function of the response rate and averaging the rate first would give the wrong answer. **Prepaid and
postpaid differ structurally**, not just in magnitude: a prepaid incentive is paid to *everyone
contacted*, a postpaid one only to *respondents*, so their cost curves diverge as the response rate
falls. And the analysis is honest about what it optimises — small incentives typically repay themselves,
but the recommendation depends on the dollar value assigned to a completed interview, which is a
**judgement**, stated openly rather than buried.

.. hint::

   **Related lessons:** :doc:`Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`  ·  :doc:`Hierarchical modeling applied to a meta-analysis <038-hierarchical-modeling-applied-to-a-meta-analysis>`  ·  :doc:`Sample surveys <052-sample-surveys>`  ·  :doc:`Conditional modeling <091-conditional-modeling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/using-regression-predictions-survey-incentives/ <https://insightful-data-lab.com/2025/11/11/using-regression-predictions-survey-incentives/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
