:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-revenue-per-user-rpu-arpu:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Revenue per User (RPU / ARPU)</b></div>`

===============================
Revenue per User (RPU / ARPU)
===============================

*Average revenue generated per user over a period.*

What it is
----------

**Revenue per user (RPU)** — almost always called **average revenue per user (ARPU)** — is
the **average income a business earns per customer** over a period (monthly, quarterly,
yearly). A **unit-economics** metric, it answers how much revenue each unit of the user base
generates, and it is a staple of telecom, streaming, SaaS and mobile-gaming reporting.

The formula
-----------

.. math::

   \text{ARPU} = \frac{\text{total revenue in period}}{\text{active users in period}},

where revenue spans subscriptions, in-app purchases and ads, and "active users" is usually
**MAU** or **DAU** depending on the business.

The key variants
----------------

**ARPU** counts *all* users (free and paying); **ARPPU** counts only **paying** users —
crucial for freemium models; and ARPU can be **blended** (across everyone) or **segmented**
(by product, geography or cohort). A freemium game with ``$50,000`` revenue across 10,000
MAU has an **ARPU of** ``$5``, but if only 2,000 pay, its **ARPPU is** ``$25`` — revealing
that a small paying base carries the economics.

How it fits the other metrics
-----------------------------

ARPU is a **short-term snapshot** of revenue per user, complementing the **lifetime view**:
**CLV** is total expected revenue over the whole relationship, while **CAC** is the cost to
acquire a user. A healthy business keeps :math:`\text{CLV} > \text{CAC}`, and ideally
:math:`\text{ARPU} \times \text{retention} \gg \text{CAC}`.

The limitations
---------------

ARPU is an **average**, so it hides the **distribution** (one user at ``$500`` and one at
``$0`` both read as ``$50``), can **mask churn** (fewer customers paying more keeps ARPU flat
while the base shrinks), and means little alone — it must be paired with **CAC, churn and
CLV** to judge whether the business is actually healthy.

----

**Mind map — connected ideas**

   :doc:`LTV:CAC Ratio <037-ltv-cac-ratio>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`SaaS (Software as a Service) <034-saas-software-as-a-service>` · :doc:`Customer Segmentation <033-customer-segmentation>`

----

**More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Revenue per User (RPU / ARPU) <https://insightful-data-lab.com/2025/08/24/revenue-per-user-rpu-arpu/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
