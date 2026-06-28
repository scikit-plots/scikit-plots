:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-net-ltv-sometimes-called-contribution-ltv:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Net LTV (sometimes called Contribution LTV)</b></div>`

=============================================
Net LTV (sometimes called Contribution LTV)
=============================================

*Customer lifetime value after subtracting variable and serving costs.*

What it is
----------

**Net LTV** (also **Contribution LTV**) is the **profit contribution** a customer
generates over their lifetime, *after* the direct cost of serving them. Where Gross
LTV is pure revenue, Net LTV adjusts for **gross margin**, answering "how much do we
actually keep?".

Formula
-------

.. math::

   \text{Net LTV} = \text{Gross LTV} \times \text{Gross Margin \%}
   = (\text{ARPU} \times \text{Customer Lifetime}) \times \text{Gross Margin \%},

with gross margin :math:`= (\text{Revenue} - \text{COGS}) / \text{Revenue}` and
customer lifetime :math:`\approx 1/\text{churn}`.

Worked example
--------------

Continuing the running example — ARPU $50/month, 5% churn (20-month lifetime) — gives
Gross LTV $1,000. At 70% gross margin:

.. math::

   \text{Net LTV} = 1{,}000 \times 0.7 = 700.

The customer pays $1,000 in revenue, but the business keeps $700 in profit
contribution.

Why it's the number that matters
--------------------------------

Net LTV is the **realistic** measure of customer value and the correct numerator for
unit economics: the healthy benchmark is

.. math::

   \frac{\text{Net LTV}}{\text{CAC}} \ge 3.

Investors and CFOs favour Net over Gross LTV because revenue without margin is not
sustainable. As with Gross LTV, prefer **retention-curve or predictive** lifetime
estimates over a flat :math:`1/\text{churn}` when the data allows.

----

**Mind map — connected ideas**

   :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>` · :doc:`LTV:CAC Ratio <037-ltv-cac-ratio>` · :doc:`Gross Margin <043-gross-margin>` · :doc:`Predictive LTV (pLTV) <040-predictive-ltv-pltv>` · :doc:`Revenue per User (RPU / ARPU) <122-revenue-per-user-rpu-arpu>`

----

**More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Net LTV (sometimes called Contribution LTV) <https://insightful-data-lab.com/2025/08/29/net-ltv-sometimes-called-contribution-ltv/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
