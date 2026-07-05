:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-uplift-score:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Uplift Score</b></div>`

==============
Uplift Score
==============

*A model's estimate of how much an action changes an individual's outcome.*

What it is
----------

An **uplift score** is the number an **uplift model** assigns to an individual, estimating the
**incremental impact** — positive or negative — of a treatment (a campaign, discount or medical
intervention) on that person's likelihood of acting. It answers: *how much more (or less) likely is
this person to act if we intervene than if we do not?*

The formula
-----------

At the individual level,

.. math::

   \text{Uplift Score}_i = P(\text{Outcome} \mid \text{Treatment}, i) - P(\text{Outcome} \mid \text{Control}, i).

A **positive** score means the treatment raises the chance of the desired outcome, a **negative**
score means it backfires, and a score **near zero** means the treatment barely matters.

A worked example
----------------

For a subscription campaign: **Customer A** has a treated probability of 0.40 versus 0.25 untreated —
an uplift of **+0.15**, a strong target. **Customer B** sits at 0.70 versus 0.68 — just **+0.02**,
barely worth the spend. **Customer C** is 0.10 versus 0.20 — an uplift of **-0.10**, meaning the
campaign actively *hurts* (a "Do-Not-Disturb" case).

How it's used
-------------

Uplift scores let you **rank and target** the highest-scoring customers — the *persuadables* — while
avoiding *sure things* and *lost causes*, and **excluding negative-uplift** customers who might churn
or unsubscribe if contacted. Aggregated across the population, the scores build the **uplift and Qini
curves** whose quality the **Qini coefficient** summarises.

----

**Mind map — connected ideas**

   :doc:`Uplift Models <205-uplift-models>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>`

----

**More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Uplift Score <https://insightful-data-lab.com/2025/08/23/uplift-score/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
