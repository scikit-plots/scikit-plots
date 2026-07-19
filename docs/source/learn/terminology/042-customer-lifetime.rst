:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-customer-lifetime:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Customer Lifetime</b></div>`

===================
Customer Lifetime
===================

*The expected duration of a customer's active relationship with a business.*

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

**Customer lifetime** is the average length of time a customer keeps buying from or
using a business before **churning**. It is measured in time units (months, years)
and is a key *input* to lifetime value (LTV) — LTV is the revenue or profit earned
across that span.

Formula
-------

If you know the churn rate (the share of customers lost per period), the expected
lifetime is its reciprocal:

.. math::

   \text{Customer Lifetime} \approx \frac{1}{\text{Churn Rate}}.

This follows from modelling churn as a constant per-period probability: the expected
number of periods until a customer leaves is :math:`1/\text{churn}`.

Worked example
--------------

At 5% monthly churn,

.. math::

   \text{Customer Lifetime} = \frac{1}{0.05} = 20 \text{ months}.

How it's used
-------------

It plugs straight into LTV and unit economics:

.. math::

   \text{LTV} = \text{ARPU} \times \text{Gross Margin} \times \text{Customer Lifetime},

and the resulting LTV is compared against CAC, with **LTV:CAC ≥ 3** as the healthy
benchmark.

Limitations
-----------

- **Constant-churn assumption** — real churn is highest early and falls for loyal
  cohorts, so :math:`1/\text{churn}` can mislead.
- **Sensitivity** — because lifetime is :math:`1/\text{churn}`, a small churn error
  swings it a lot (5% → 20 months, but 4% → 25 months).
- For retention-driven businesses, refine it with **survival analysis** or
  **cohort analysis** rather than the shortcut.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Churn <123-churn>` · :doc:`Retention <124-retention>` · :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>` · :doc:`LTV (Customer Lifetime Value) <373-ltv-customer-lifetime-value>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>` · :doc:`FTEs <147-ftes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Customer Lifetime <https://insightful-data-lab.com/2025/08/29/customer-lifetime/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
