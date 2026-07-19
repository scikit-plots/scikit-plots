:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-uplift-curve:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Uplift Curve</b></div>`

==============
Uplift Curve
==============

*A curve showing cumulative incremental gain as more of the ranked population is treated.*

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

An **uplift curve** is a tool for **evaluating uplift models** (incremental-response models). It plots
the **incremental gain** — the extra benefit caused by the treatment — against the **proportion of the
population targeted**, showing how much improvement you gain by targeting the top fraction of customers
ranked by uplift score. It is closely related to, and often drawn as, the **Qini curve**.

Why not target everyone
-------------------------

Traditional models (like logistic regression) predict response probability, but in marketing you
**don't want to target everyone** likely to buy — some would buy anyway, and some are even negatively
influenced. The uplift curve reflects the model's job of isolating **persuadables**, those who change
behaviour because of the treatment.

How it's built
---------------

Rank customers by predicted uplift score from high to low, split them into buckets (top 10%, next
10%, …), and for each bucket compute the **treated-minus-control** difference in outcome rate. Plotting
the **cumulative** incremental gains against the percentage targeted gives the curve: the **X-axis** is
the fraction targeted, the **Y-axis** the incremental gain, with a **random baseline** line and the
**model curve** ideally above it.

Reading it, with an example
-----------------------------

A **steeper** curve means the model is better at finding the most-influenced customers; a curve **close
to the random line** adds little value; and the **area between the model curve and the random line**
serves as a performance metric, much like AUC. For an email renewal campaign, the curve might show
that targeting the **top 20%** by uplift score generates most of the incremental renewals, while
targeting everyone simply wastes resources.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Qini Curve <203-qini-curve>` · :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Uplift Models <205-uplift-models>` · :doc:`Uplift <424-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Uplift Curve <https://insightful-data-lab.com/2025/08/21/uplift-curve/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
