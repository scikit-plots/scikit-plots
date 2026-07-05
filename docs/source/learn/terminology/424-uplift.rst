:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-uplift:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Uplift</b></div>`

========
Uplift
========

*The incremental effect of an action on an individual's outcome.*

What it is
----------

**Uplift** measures the **incremental impact** of an action, treatment or intervention compared to
**not taking it**. In machine learning, **uplift modelling** — also called *incremental response* or
*true lift* modelling — predicts the **change in outcome probability** caused by applying a treatment
(say, sending a marketing offer). It is about **causal effect**, not mere correlation.

The formula
-----------

.. math::

   \text{Uplift}(x) = P(Y = 1 \mid T = 1, X = x) - P(Y = 1 \mid T = 0, X = x),

where :math:`Y` is the outcome (purchase, churn, click), :math:`T` is the treatment (1 = received,
0 = not), and :math:`X` are the individual's features.

A worked example
----------------

For a promotional email, compare each customer's treated and untreated purchase probability.
**Customer A**: 30% with the email versus 25% without — an uplift of **+5%**, worth targeting.
**Customer B**: 60% versus 60% — **0%**, the email is irrelevant. **Customer C**: 20% versus 30% — a
**-10%** uplift, meaning the email actively *reduces* the chance (a spam-sensitive recipient).

Why it matters
--------------

Uplift buys **targeting efficiency** — spending only on customers who change behaviour *because of*
the treatment — and **cost savings**, by skipping "sure things" who would act anyway and "sleeping
dogs" who react badly. Used across **marketing, healthcare, finance and policy**, it is measured with
the **uplift (Qini) curve**, its area **AUUC**, and the **Qini coefficient**.

----

**Mind map — connected ideas**

   :doc:`Uplift Models <205-uplift-models>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Qini Coefficient <397-qini-coefficient>`

----

**More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Uplift <https://insightful-data-lab.com/2025/08/17/uplift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
