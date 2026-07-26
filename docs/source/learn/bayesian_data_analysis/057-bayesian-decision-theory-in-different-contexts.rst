.. _bda-bayesian-decision-theory-in-different-contexts:

========================================================================
Bayesian decision theory in diﬀerent contexts
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 057 of 144  ·  *intermediate*

:doc:`◀ Previous · Censoring and truncation <056-censoring-and-truncation>`   ·   :doc:`Next · Using regression predictions: survey incentives ▶ <058-using-regression-predictions-survey-incentives>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

From inference to action
--------------------------

A posterior is not a decision. To act you need one more ingredient: a **utility function** stating what
outcomes are worth. Bayesian decision theory then supplies the rule — among the available actions,
choose the one whose utility is greatest **on average over the posterior**.

The rule
----------

Let :math:`a` be an action, :math:`\theta` the unknown state, and :math:`U(a, \theta)` the utility of
taking :math:`a` when the truth is :math:`\theta`. The optimal action maximises **expected utility**:

.. math::

   a^{*} = \arg\max_{a} \; \mathrm{E}_{\theta \mid y}\bigl[U(a, \theta)\bigr]
         = \arg\max_{a} \int U(a, \theta) \; p(\theta \mid y) \; d\theta .

With posterior draws this is a one-line computation — the expectation becomes an average, and the
optimisation a search over the (usually few) candidate actions.

.. code-block:: python

   import numpy as np
   theta = idata.posterior["theta"].values.ravel()      # posterior draws
   actions = np.linspace(0, 1, 101)                     # candidate actions
   EU = [np.mean(utility(a, theta)) for a in actions]   # expected utility per action
   best = actions[int(np.argmax(EU))]

Note what the posterior does here: it **weights** each possible truth by its plausibility. The full
distribution matters, not just its mean — because utilities are usually **nonlinear**, and
:math:`\mathrm{E}[U(a, \theta)] \ne U(a, \mathrm{E}[\theta])`.

Why estimation is not decision
--------------------------------

Familiar Bayesian summaries are decisions in disguise, each optimal under a particular loss (negative
utility). Squared-error loss gives the **posterior mean**; absolute-error loss gives the **median**;
0–1 loss gives the **mode**. So the choice of point estimate is not a matter of taste but of the cost
of being wrong — and when those costs are **asymmetric** (a flood barrier too low versus too high) the
optimal action can lie far out in a tail, nowhere near any conventional estimate.

Contexts differ
-----------------

Three settings recur through this stage, and their utilities differ in kind. A **one-shot** choice: pick
:math:`a` once, as with a survey incentive. A **sequential** problem: act, observe, act again, where an
early action buys information for later ones. And a **hierarchical** problem, where decisions are made
for many units at once and shrinkage governs each. The final lesson adds the sharpest distinction of
all — **whose** utility, the individual's or the institution's, since the two can rationally disagree.

.. hint::

   **Related lessons:** :doc:`Using regression predictions: survey incentives <058-using-regression-predictions-survey-incentives>`  ·  :doc:`Multistage decision making: medical screening <059-multistage-decision-making-medical-screening>`  ·  :doc:`Summarizing Posterior Inference <013-summarizing-posterior-inference>`  ·  :doc:`Personal vs. institutional decision analysis <061-personal-vs-institutional-decision-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/bayesian-decision-theory-in-di%ef%ac%80erent-contexts/ <https://insightful-data-lab.com/2025/11/11/bayesian-decision-theory-in-di%ef%ac%80erent-contexts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
