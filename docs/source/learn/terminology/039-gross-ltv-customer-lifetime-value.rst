:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-gross-ltv-customer-lifetime-value:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Gross LTV (Customer Lifetime Value)</b></div>`

=====================================
Gross LTV (Customer Lifetime Value)
=====================================

*Total expected revenue from a customer before any cost deductions.*

What it is
----------

**Gross LTV** is the **total revenue** a customer is expected to generate over their
lifetime, *before* subtracting any cost of serving them. It is the top-line view of
customer value — "how much will this customer pay us in total?", not "how much profit
do we keep?".

Formula
-------

.. math::

   \text{Gross LTV} = \text{ARPU} \times \text{Customer Lifetime},

where **ARPU** is average revenue per user per period and **customer lifetime** is the
average time a customer stays — commonly estimated as :math:`1/\text{churn rate}`.

Worked example
--------------

With ARPU = $50/month and 5% monthly churn, the expected lifetime is
:math:`1/0.05 = 20` months, so

.. math::

   \text{Gross LTV} = 50 \times 20 = 1{,}000.

Each customer pays about $1,000 in total revenue over their average lifetime.

Gross vs Net LTV
----------------

Gross LTV counts **revenue only**; **Net (Contribution) LTV** multiplies it by gross
margin to get profit, e.g. at 70% margin :math:`1{,}000 \times 0.7 = 700`. Gross LTV
is the optimistic number; Net LTV is the one to compare against CAC.

When it's used (and its limits)
-------------------------------

Gross LTV gives a **quick, high-level** read and is handy for **early-stage**
companies that lack detailed cost data. Two caveats matter:

- It ignores cost of goods sold, so it overstates true value — finance teams prefer
  Net LTV.
- The :math:`1/\text{churn}` lifetime assumes a **constant churn rate**; real
  retention curves flatten over time, so survival analysis or cohort retention curves
  give a more accurate lifetime.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Net LTV (sometimes called Contribution LTV) <038-net-ltv-sometimes-called-contribution-ltv>` · :doc:`Revenue per User (RPU / ARPU) <122-revenue-per-user-rpu-arpu>` · :doc:`Churn <123-churn>` · :doc:`Retention <124-retention>` · :doc:`Predictive LTV (pLTV) <040-predictive-ltv-pltv>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Gross LTV (Customer Lifetime Value) <https://insightful-data-lab.com/2025/08/29/gross-ltv-customer-lifetime-value/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
