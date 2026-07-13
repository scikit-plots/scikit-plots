:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-uplift-models:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Uplift Models</b></div>`

===============
Uplift Models
===============

*Models that predict the incremental effect of treating each individual.*

What it is
----------

**Uplift models** — also called incremental, net-lift or true-lift models — are predictive models
that estimate the **causal impact** of an action (a campaign, discount or treatment) on an
individual's behaviour, rather than the behaviour itself. They shift the question from *"who will
buy?"* to *"who will buy* **because of** *the campaign?"* — predicting incremental change, not raw
outcome.

Four kinds of customer
------------------------

Uplift modelling divides people by how they respond to intervention: **persuadables** act *only*
because of it (the target), **sure things** would act anyway, **lost causes** never act, and
**do-not-disturbs** react *negatively* if contacted. The goal is to reach persuadables, stop wasting
budget on sure things and lost causes, and avoid provoking do-not-disturbs.

Techniques
----------

Four families. The **two-model approach** trains separate treated- and control-group models and
subtracts their probabilities. **Class transformation** relabels the target to encode both outcome
and treatment, so a single classifier predicts uplift directly. **Uplift trees and forests** choose
splits that maximise the treatment-control difference. And **meta-learners** (T-, S- and X-learners)
build uplift on top of standard models within causal-ML frameworks.

Evaluation and trade-offs
---------------------------

Because uplift concerns *causal* effect, accuracy and AUC are not enough; models are judged by the
**Qini curve and coefficient**, the **uplift curve**, and **AUUC**. In a subscription campaign where
the treatment group buys at 20% against a control's 15%, the model tries to pinpoint which
individuals make up that **+5%**. The payoff is better spend, ROI and causal insight; the costs are a
need for **experimental (A/B) data**, greater model complexity, and harder interpretation.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Uplift <424-uplift>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Treatment Effect <072-treatment-effect>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Uplift Models <https://insightful-data-lab.com/2025/08/23/uplift-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
