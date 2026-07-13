:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-channel-specific-cac-customer-acquisition-cost:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Channel-Specific CAC (Customer Acquisition Cost)</b></div>`

==================================================
Channel-Specific CAC (Customer Acquisition Cost)
==================================================

*Acquisition cost computed separately for each marketing channel.*

What it is
----------

**Channel-specific CAC** is the cost to acquire a customer through a **single
channel** — Facebook Ads, Google Search, LinkedIn, SEO, events, each measured on its
own. Where blended CAC averages everything, this isolates the efficiency of each
channel.

Formula
-------

.. math::

   \text{CAC}_{\text{channel}} = \frac{\text{Total Spend on Channel}}{\text{Customers from Channel}},

using channel-specific ad spend, agency fees, tools and creative in the numerator and
the customers directly attributable to that channel in the denominator.

Worked example
--------------

In one month, three channels can look completely different:

- Google Ads — $30,000 / 200 customers = $150
- Facebook Ads — $20,000 / 250 customers = $80
- SEO — $10,000 / 500 customers = $20

Those gaps are **invisible** in a single blended number.

Why it's useful
---------------

It tells you **which channels are most cost-effective**, so you can scale the cheap
ones and fix the expensive ones, and — paired with LTV — gives a per-channel
:math:`\text{LTV:CAC}` to rank channels by true value, not just cost.

Challenges
----------

- **Attribution** — a customer may touch ads, then email, then organic search before
  converting; deciding which channel gets credit is the central difficulty.
- **Lag** — SEO and content investment can take months to pay off, inflating
  short-term CAC.
- **Hidden overhead** — brand-building and shared salaries don't map cleanly to one
  channel.

The data-science angle
----------------------

Getting channel CAC right is fundamentally an **attribution** problem: last-touch and
first-touch are crude, so teams use **multi-touch attribution**, **Shapley-value**
credit, or **marketing-mix modelling (MMM)** — a regression of conversions on
per-channel spend with adstock/lag terms — to assign credit across the journey.

Blended vs channel-specific
---------------------------

**Blended CAC** is the snapshot for board reports and financial health;
**channel-specific CAC** is the optimisation tool for deciding where the next dollar
goes.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`Paid CAC (Customer Acquisition Cost) <046-paid-cac-customer-acquisition-cost>` · :doc:`Fully Loaded CAC (Customer Acquisition Cost) <044-fully-loaded-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`LTV:CAC Ratio <037-ltv-cac-ratio>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>` · :doc:`FTEs <147-ftes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Channel-Specific CAC (Customer Acquisition Cost) <https://insightful-data-lab.com/2025/08/29/channel-specific-cac-customer-acquisition-cost/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
