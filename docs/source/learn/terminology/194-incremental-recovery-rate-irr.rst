:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-incremental-recovery-rate-irr:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Incremental Recovery Rate (IRR)</b></div>`

=================================
Incremental Recovery Rate (IRR)
=================================

*The added recovery (e.g. in collections) attributable to a treatment.*

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

The **incremental recovery rate** measures the **lift** an intervention produces in a **recovery** outcome —
reactivating lapsed customers, collecting overdue accounts, or winning back churned users — beyond what would
have recovered **anyway**. It is the treated group's recovery rate **minus** a control group's:

.. math::

   \text{IRR} = \text{recovery rate}_{\text{treated}} - \text{recovery rate}_{\text{control}}.

Why the baseline matters
------------------------

A raw recovery rate **overstates** a campaign's value, because some accounts **self-cure** or return without
any nudge. Subtracting a **holdout** control isolates the **causal** lift — the recoveries the intervention
**actually** caused — the same incrementality logic behind incremental sales and conversions.

Where it's used
---------------

IRR drives **targeting** and **budget** in collections, retention, and win-back — you focus effort on segments
with the **highest** incremental recovery, not the highest raw recovery, since some of those would come back
**for free**. It pairs naturally with **uplift** models that predict per-customer lift.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Uplift Random Forests <302-uplift-random-forests>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Incremental Recovery Rate (IRR) <https://insightful-data-lab.com/2025/08/23/incremental-recovery-rate-irr/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
