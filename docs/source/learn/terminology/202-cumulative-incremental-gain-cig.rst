:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cumulative-incremental-gain-cig:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Cumulative Incremental Gain (CIG)</b></div>`

===================================
Cumulative Incremental Gain (CIG)
===================================

*The running sum of incremental gain as more of the ranked population is treated.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Cumulative incremental gain (CIG)** is an uplift-modelling measure of the **total additional
outcomes** — purchases, sign-ups, conversions — won by targeting customers **ranked by uplift score**,
relative to random or no targeting. It is the quantity plotted on the **Y-axis of a Qini curve**.

How it's built
---------------

Four steps: **rank** customers by predicted uplift score, highest to lowest; **split** them into
deciles or percentiles; for each segment compute the **incremental gain** as treatment responses minus
control responses; then take the **running sum** of those gains down to the segment of interest.

The formula
-----------

.. math::

   \text{CIG}(p) = \sum_{i=1}^{pN} \left( y_i^{\text{treat}} - y_i^{\text{control}} \right),

the treated-minus-untreated outcome accumulated over the top proportion :math:`p` of a population of
size :math:`N`.

A worked example
----------------

Across 10,000 customers, targeting the **top 20%** yields 400 treatment versus 300 control
conversions — a gain of **100**; extending to the **top 40%** adds 750 versus 550 — a further **200**.
The cumulative incremental gain at 40% is therefore **300 conversions**.

Reading the Qini curve
------------------------

Plotting CIG against the percentage of the population targeted gives the **Qini curve**. A **steep
early slope** means the model finds *persuadables* first and targeting pays off quickly; a **flat**
curve means the model is no better than random. The shape reveals the **optimal targeting fraction** —
where gain stops rising — and lets you **compare models** by how much incremental value each delivers.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cumulative Incremental Gain (CIG) <https://insightful-data-lab.com/2025/08/23/cumulative-incremental-gain-cig/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
