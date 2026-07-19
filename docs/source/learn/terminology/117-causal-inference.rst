:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-causal-inference:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Causal Inference</b></div>`

==================
Causal Inference
==================

*Drawing cause-and-effect conclusions from data, not merely associations.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Causal inference** is the discipline of establishing whether one variable **actually
causes** a change in another, rather than merely **correlating** with it. The classic trap:
ice-cream sales and drownings rise together, but neither causes the other — hot weather
drives both. Causal inference is the toolkit for separating **causation from association**.

Why it matters
--------------

Decisions hinge on it. Did the campaign **cause** the sales lift, or would sales have risen
anyway? Did the treatment **cause** recovery, or was it natural? Did the policy **cause** the
change, or did something else? Acting on a spurious correlation wastes money and harms
people.

The gold standard and its alternatives
--------------------------------------

A **randomised controlled trial (RCT)** assigns treatment and control **at random**, which
balances confounders and makes the comparison causal. When randomisation is impossible
(ethics, cost), **observational** methods approximate it: **regression** adjusts for measured
confounders; **propensity-score matching** pairs treated and untreated units with similar
covariates; **difference-in-differences** compares before/after trends against an untreated
group; **instrumental variables** exploit a variable that moves treatment but not the
outcome directly; and **regression discontinuity** compares units just above and below a
treatment cutoff.

The counterfactual core
-----------------------

Underneath sits the **potential-outcomes (Rubin) model**: each unit has a treated outcome
:math:`Y_1` and an untreated outcome :math:`Y_0`, and the causal effect is
:math:`E[Y_1 - Y_0]`. The **fundamental problem** is that we never observe **both** for the
same unit — which is exactly why we need RCTs, DiD and the rest. **Directed acyclic graphs
(DAGs)** complement this by mapping confounders and mediators visually.

The pitfalls
------------

The enemies of causal claims are **confounding** (a third variable driving both),
**selection bias** (groups that differ systematically), **reverse causality** (the outcome
also influences the cause), and **unobserved variables** that can't be adjusted for if
they're never measured.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Causal Impact <112-causal-impact>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Regression Coefficient <090-regression-coefficient>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Frequentist <059-frequentist>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Causal Inference <https://insightful-data-lab.com/2025/08/24/causal-inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
