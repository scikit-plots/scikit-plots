.. _bda-regression-for-causal-inference-incumbency-and-voting:

========================================================================
Regression for causal inference: incumbency and voting
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 093 of 144  ·  *advanced*

:doc:`◀ Previous · Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`   ·   :doc:`Next · Goals of regression analysis ▶ <094-goals-of-regression-analysis>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Does holding office help you win?
-----------------------------------

The **incumbency advantage** is the extra vote share a candidate receives merely by being the sitting
representative. It cannot be randomised — nobody assigns incumbency by coin flip — so the question is
observational, and regression must carry the causal weight. Stage 7's warnings apply in full.

Why the naive measures fail
-----------------------------

Two classical estimators look reasonable and are **biased**. **Sophomore surge** compares a legislator's
vote share in their second election (as incumbent) with their first (as challenger). **Retirement
slump** compares a party's share before and after its incumbent retires. Each conflates the incumbency
effect with the fact that **incumbents are not a random sample**: they are the candidates who won, in
districts that favoured them. Gelman and King showed the bias directly — and showed that the information
in these comparisons, placed in a **regression framework**, yields an unbiased estimate.

The model
-----------

Model district :math:`i`'s vote share :math:`v_{it}` in election year :math:`t`, controlling for the
district's **previous** vote share (a proxy for partisan strength) and party, with an indicator
:math:`I_{it}` for whether an incumbent is running:

.. math::

   v_{it} = \beta_0 + \beta_1 \, v_{i,t-1} + \beta_2 \, P_{it} + \underbrace{\gamma}_{\text{incumbency}} I_{it}
            + \epsilon_{it} .

The coefficient :math:`\gamma` is the estimand. Conditioning on the lagged vote is what removes the
selection: districts where incumbents run are compared with **similar** districts holding open seats.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       b = pm.Normal("b", 0, 0.5, shape=3)
       gamma = pm.Normal("gamma", 0, 0.1)               # incumbency effect, on vote-share scale
       mu = b[0] + b[1] * v_lag + b[2] * party + gamma * incumbent
       pm.Normal("v", mu, pm.HalfNormal("sigma", 0.1), observed=v)
       idata = pm.sample()
   # posterior for gamma, by decade, shows the advantage growing over the century

What makes it causal, and what does not
-----------------------------------------

The regression **is** the causal claim only under the no-unmeasured-confounding assumption of Stage 7:
that, given the lagged vote and party, whether an incumbent runs is as good as random. That is an
assumption about **why** incumbents retire — health, scandal, redistricting, or anticipated defeat — and
the last of these breaks it. If incumbents strategically retire when they expect to lose, retirement is
informative about :math:`\epsilon_{it}`, and :math:`\gamma` is biased upward.

So the honest report has three parts: the estimate, the assumption it rests on, and a **sensitivity
analysis** for how strong a confounder would need to be to erase it. Regression supplies the arithmetic;
the design supplies the license.

.. hint::

   **Related lessons:** :doc:`Observational studies <055-observational-studies>`  ·  :doc:`Goals of regression analysis <094-goals-of-regression-analysis>`  ·  :doc:`Sensitivity and the role of randomization <054-sensitivity-and-the-role-of-randomization>`  ·  :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/regression-for-causal-inference-incumbency-and-voting/ <https://insightful-data-lab.com/2025/11/23/regression-for-causal-inference-incumbency-and-voting/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
