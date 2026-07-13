:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-retention:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Retention</b></div>`

===========
Retention
===========

*The share of users who remain active over a given period.*

What it is
----------

**Retention** measures how well a business **keeps its customers or users over time** — the
**complement of churn**, and a direct signal of product value: if people keep coming back,
the product is working.

The metrics
-----------

The **customer retention rate (CRR)** strips out new sign-ups to measure how many existing
customers stayed:

.. math::

   \text{CRR} = \frac{E - N}{S} \times 100,

with :math:`S` customers at the start, :math:`E` at the end, and :math:`N` newly acquired.
**Cohort retention** tracks a starting group over time (100 sign up in week 1, 40 still
active in week 4 → **40% week-4 retention**), and **revenue retention** comes in two forms:
**gross (GRR)**, which ignores expansion, and **net (NRR)**, which adds upsells and
cross-sells.

Example
-------

Start with 100 paying customers, lose 10 and gain 20: the CRR counts only the kept ones,
:math:`(110 - 20)/100 = 90\%`. A mobile app with 300 of 1,000 installs still active at 30
days has **30% retention**.

Why it matters
--------------

Retention is **cheaper than acquisition**, drives **predictable recurring revenue**, and is
the clearest **indicator of product value** — which is why it sits at the heart of
**lifetime value** and growth. Teams raise it with onboarding, A/B tests and
recommendations, and predict who is about to drop with the same models used for churn.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Churn <123-churn>` · :doc:`Revenue per User (RPU / ARPU) <122-revenue-per-user-rpu-arpu>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`SaaS (Software as a Service) <034-saas-software-as-a-service>` · :doc:`Upselling <032-upselling>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Retention <https://insightful-data-lab.com/2025/08/24/retention/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
