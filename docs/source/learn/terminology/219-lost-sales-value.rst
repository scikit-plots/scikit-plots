:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-lost-sales-value:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Lost Sales Value</b></div>`

==================
Lost Sales Value
==================

*Revenue forgone when demand cannot be met from available stock.*

What it is
----------

**Lost Sales Value** is the **monetary value of sales that could not be realised** because products
were out of stock or unavailable when customers wanted them. It puts the **dollar impact** on
stockouts, going beyond the percentages of stockout rate or fill rate. High lost-sales value means
serious revenue leakage; low means sound inventory and demand planning.

The formula
-----------

.. math::

   \text{Lost Sales Value} = \text{Unfulfilled Units} \times \text{Unit Selling Price},

or equivalently the total demand value (demand units times price) minus the actual sales value
(fulfilled units times price).

A worked example
----------------

A customer wants 500 units but only 450 are in stock, so 50 go unfilled. At ``$20`` each, the lost
sales value is 50 × 20 = ``$1,000`` — a thousand dollars of revenue forgone to the stockout.

Why it matters
--------------

It shows the money lost **directly**, drives **customer-loyalty** concerns, informs the **inventory
cost versus service level** trade-off, and prioritises the SKUs whose stockouts hurt most (typically
high-margin ones). Alongside its percentage cousins — stockout rate (how *often*) and fill rate (how
*much* is filled) — lost sales value answers how much revenue is actually lost: a snapshot might read
5% stockout, 95% fill, and ``$50,000`` lost.

----

**Mind map — connected ideas**

   :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>` · :doc:`SLA Breach Rate <207-sla-breach-rate>`

----

**More in Operations & Supply Chain**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Stockouts <401-stockouts>`

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Lost Sales Value <https://insightful-data-lab.com/2025/08/23/lost-sales-value/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
