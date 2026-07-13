:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-ltv-cac-ratio:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>LTV:CAC Ratio</b></div>`

===============
LTV:CAC Ratio
===============

*The ratio of customer lifetime value to acquisition cost — a core unit-economics health metric.*

What it is
----------

The **LTV:CAC ratio** compares the lifetime value of a customer to the cost of
acquiring them. It answers the core unit-economics question: *for every $1 spent
acquiring a customer, how many dollars of lifetime value come back?* It tells you
whether growth is **profitable and scalable**.

Formula
-------

.. math::

   \text{LTV:CAC} = \frac{\text{LTV}}{\text{CAC}}.

By convention the numerator is **Net (Contribution) LTV** — revenue alone overstates
the case — and the denominator can be **Blended**, **Paid**, or **Fully-Loaded CAC**
depending on what question you are asking.

Worked example
--------------

Chaining the standard example: ARPU $50/month and 5% churn give a 20-month lifetime,
so Gross LTV :math:`= 50 \times 20 = 1{,}000`; at 70% gross margin, Net LTV
:math:`= 700`; with blended CAC $200,

.. math::

   \text{LTV:CAC} = \frac{700}{200} = 3.5.

Every $1 of acquisition spend returns $3.50 in profit contribution.

How to read it
--------------

- **< 1** — losing money on every customer; unsustainable.
- **1–2** — barely breaking even; not yet scalable.
- **3–4** — healthy, efficient growth (the classic **3:1** benchmark).
- **> 5** — strong, but may signal **under-investment** in growth (you could afford to
  spend more to acquire).

Why it matters
--------------

It is the headline KPI for SaaS, subscription and D2C businesses and a key lens for
investors judging scalability. It is best read **alongside the payback period** — a
great ratio still hurts cash flow if CAC takes too long to recover.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`LTV (Customer Lifetime Value) <373-ltv-customer-lifetime-value>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Net LTV (sometimes called Contribution LTV) <038-net-ltv-sometimes-called-contribution-ltv>` · :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>` · :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `LTV:CAC Ratio <https://insightful-data-lab.com/2025/08/29/ltvcac-ratio/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
