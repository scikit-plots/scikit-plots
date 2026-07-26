:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cohort-based-ltv-simple-version:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Cohort-Based LTV (Simple Version)</b></div>`

===================================
Cohort-Based LTV (Simple Version)
===================================

*Lifetime value estimated by tracking the revenue of customer cohorts over time.*

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

**Cohort-based LTV** estimates lifetime value from *observed* behaviour rather than a
churn assumption. Customers are grouped into **cohorts** by when they joined (the
January 2025 cohort, the February 2025 cohort, and so on), and you track how much
revenue each cohort generates month by month, then sum it — so the retention curve is
*measured*, not assumed.

The method
----------

1. **Define cohorts** — e.g. everyone acquired in January 2025.
2. **Track average revenue per customer** for each month after signup (month 0,
   month 1, and so on).
3. **Accumulate** that revenue until the cohort stabilises or churns out.
4. **Estimate the tail** — if the curve has flattened, take the total; if not, fit a
   simple decay (exponential or linear) to project the remaining months.

Formula
-------

Summing average per-customer revenue across months, optionally discounted to present
value:

.. math::

   \text{LTV} = \sum_{t=0}^{T} \frac{\text{Avg Revenue per Customer in month } t}{(1 + r)^t},

where :math:`T` is the months tracked and :math:`r` is an optional discount rate
(often dropped in a simple calculation).

Worked example
--------------

A 100-customer January cohort with average per-customer revenue of $100, 40, 38, 35,
32, 30 over months 0–5 gives

.. math::

   \text{LTV (6 months)} = 100 + 40 + 38 + 35 + 32 + 30 = 275 \text{ per customer},

with later months projected via a decay assumption if revenue is still falling.

Why it beats the churn shortcut
-------------------------------

Because it uses **real observed retention and spend**, cohort LTV captures the early
drop-off and long-tail loyalty that a flat :math:`1/\text{churn}` misses, and it
exposes differences by acquisition month, channel or segment — making it easy to see
whether **retention is improving or worsening** over time.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cohort <183-cohort>` · :doc:`Predictive LTV (pLTV) <040-predictive-ltv-pltv>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>` · :doc:`Retention <124-retention>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>` · :doc:`FTEs <147-ftes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cohort-Based LTV (Simple Version) <https://insightful-data-lab.com/2025/08/29/cohort-based-ltv-simple-version/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
