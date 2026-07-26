.. _bda-multistage-decision-making-medical-screening:

========================================================================
Multistage decision making: medical screening
========================================================================

**Part 2 · Stage 7 · 🗳️ Data Collection & Decisions**  ·  Lesson 059 of 144  ·  *intermediate*

:doc:`◀ Previous · Using regression predictions: survey incentives <058-using-regression-predictions-survey-incentives>`   ·   :doc:`Next · Hierarchical decision analysis for home radon ▶ <060-hierarchical-decision-analysis-for-home-radon>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Decisions that buy information
--------------------------------

Screening is not one decision but a **sequence**. Test, and if the test is positive, decide whether to
run a second, more invasive test; then decide whether to treat. Each stage costs something and yields
**information** that improves the next choice. The value of a test lies not in the test but in the
**better decisions it enables**.

Working backwards
-------------------

The solution method is **backward induction**. At the final stage, the treatment decision is a plain
expected-utility maximisation given whatever has been learned. Step back: the value of *being* at that
stage is the expected utility of acting optimally there. So the earlier decision — whether to test —
compares the cost of testing against the improvement in the expected utility it produces:

.. math::

   \mathrm{EVSI} = \mathrm{E}_{y_{\text{new}}}\Bigl[\max_{a} \,
                   \mathrm{E}_{\theta \mid y, y_{\text{new}}} \, U(a, \theta) \Bigr]
                 - \max_{a} \, \mathrm{E}_{\theta \mid y} \, U(a, \theta) .

That quantity — the **expected value of sample information** — is the most one should ever pay for the
test. It is necessarily non-negative: information cannot hurt a decision-maker who is free to ignore it.

The screening structure
-------------------------

Screening makes the interplay of prevalence and accuracy vivid. Bayes' rule converts a positive test
into a posterior probability of disease, and when **prevalence is low**, even an accurate test leaves
that probability modest — the record-linkage lesson again, with lives at stake.

.. code-block:: python

   import numpy as np
   prev, sens, spec = 0.004, 0.95, 0.95            # rare disease, good test
   ppv = prev * sens / (prev * sens + (1 - prev) * (1 - spec))
   ppv                                              # ≈ 0.071 — most positives are false

   # stage 2: treat only if expected utility of treating exceeds not treating
   U_treat    = ppv * U_treat_sick + (1 - ppv) * U_treat_well     # side effects if well
   U_no_treat = ppv * U_none_sick  + (1 - ppv) * U_none_well
   act = "treat" if U_treat > U_no_treat else "wait"

Where it gets hard
--------------------

Three honest complications. Utilities must be **commensurable** — a life-year, a false-positive's
anxiety and a dollar placed on one scale, an ethical act, not a technical one. The state space grows
**combinatorially** with stages, so realistic problems are solved by simulation rather than by exact
backward induction. And the analysis assumes the model for test performance is right; a
**miscalibrated** sensitivity propagates through every stage. Screening policy is where Bayesian
decision theory is at its most useful and its most contested.

.. hint::

   **Related lessons:** :doc:`Bayesian decision theory in diﬀerent contexts <057-bayesian-decision-theory-in-different-contexts>`  ·  :doc:`Example — Calibration for Record Linkage <007-example-calibration-for-record-linkage>`  ·  :doc:`Hierarchical decision analysis for home radon <060-hierarchical-decision-analysis-for-home-radon>`  ·  :doc:`Personal vs. institutional decision analysis <061-personal-vs-institutional-decision-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/multistage-decision-making-medical-screening/ <https://insightful-data-lab.com/2025/11/11/multistage-decision-making-medical-screening/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
