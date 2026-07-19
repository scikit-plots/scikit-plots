:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-incremental-gain:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Incremental Gain</b></div>`

==================
Incremental Gain
==================

*The improvement in outcome from targeting versus a baseline.*

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

**Incremental gain** is the **extra positive outcome** — conversions, sales, sign-ups — produced by
applying a treatment (a campaign, discount or intervention) **versus not applying it**, measured for a
**specific slice of the population**. It is the **building block** of the cumulative incremental gain
and the Qini curve: the step-by-step value added as you target more people.

The formula
-----------

For one segment,

.. math::

   \text{Incremental Gain} = \text{Responses}_{\text{treatment}} - \text{Responses}_{\text{control}},

or, in probability terms, :math:`\left(P(\text{Outcome} \mid \text{Treatment}) - P(\text{Outcome} \mid \text{Control})\right) \times n`, where :math:`n` is the number of individuals targeted in that
segment and the control response is what the same group would have done untreated.

A worked example
----------------

Rank 2,000 customers by uplift score and split into deciles of 200. In the **top decile**, the
treatment group converts at 18% (36 purchases) against a control of 10% (20) — an incremental gain of
**16**. The **second decile** converts at 14% (28) versus 12% (24) — a gain of **4**. Their running
total, the cumulative incremental gain after the top 20%, is **20 extra purchases**.

Where it fits
-------------

Incremental gain is the **local** effect in each segment; summing it as you move down the ranking
gives the **cumulative incremental gain**, and plotting that against the fraction targeted traces the
**Qini curve**. Reading it segment by segment shows **where uplift is strongest** and where to stop
targeting — once the gain flattens or turns negative, additional targeting destroys value.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Incremental Revenue <193-incremental-revenue>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Incremental Gain <https://insightful-data-lab.com/2025/08/23/incremental-gain/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
