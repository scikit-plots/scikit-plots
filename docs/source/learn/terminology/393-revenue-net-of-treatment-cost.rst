:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-revenue-net-of-treatment-cost:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Revenue net of treatment cost</b></div>`

===============================
Revenue net of treatment cost
===============================

*Incremental revenue from a treatment after subtracting its cost.*

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

**Revenue net of treatment cost** is the extra revenue a treatment earns **after subtracting the cost
of running it**. It answers the blunt question: *after paying for the campaign, how much incremental
revenue did we really keep?*

The formula
-----------

.. math::

   \text{Net Revenue} = \text{Incremental Revenue} - \text{Treatment Cost},

where **incremental revenue** is treatment-group revenue minus control-group revenue, and **treatment
cost** covers marketing spend, incentives, delivery fees — any direct cost of the treatment.

A worked example
----------------

An email upsell campaign brings the treatment group to ``$120,000`` in revenue against the control's
``$100,000`` — an incremental revenue of ``$20,000``. If the campaign cost ``$5,000`` to run, then net
revenue is ``$15,000`` — the ``$20,000`` of incremental revenue minus the ``$5,000`` cost. The
headline ``$20K`` uplift is really **``$15K``** once costs are counted.

Why it matters
--------------

Uplift measured on revenue alone **overstates** the benefit. Net revenue sits closer to **ROI**
(though ROI further accounts for profit margin) and is the right lens for **comparing campaigns**: two
treatments with equal uplift are not equal if one costs far less to run.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Treatment Cost <192-treatment-cost>` · :doc:`ROI (Return on Investment) <191-roi-return-on-investment>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Gross Margin <043-gross-margin>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Revenue net of treatment cost <https://insightful-data-lab.com/2025/08/19/revenue-net-of-treatment-cost/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
