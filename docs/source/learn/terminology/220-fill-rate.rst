:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-fill-rate:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Fill Rate</b></div>`

===========
Fill Rate
===========

*The fraction of demand satisfied directly from inventory.*

What it is
----------

**Fill Rate** is a supply-chain and inventory metric measuring the **percentage of customer demand
met immediately from available stock, without delay**. It answers: *of everything customers wanted,
how much did we deliver on time and in full, from stock?* A high fill rate means availability; a low
one means frequent shortages, backorders and delays.

The formulas
------------

It has three common forms. **Order fill rate** divides orders completely fulfilled by total orders;
**line fill rate** divides order lines completely fulfilled by total order lines; and **unit fill
rate** divides units delivered on the first shipment by total units ordered — each expressed as a
percentage,

.. math::

   \text{Fill Rate (unit)} = \frac{\text{Units delivered on first shipment}}{\text{Total Units Ordered}} \times 100\%.

A worked example
----------------

A week's demand is 1,000 units; the warehouse ships 970 immediately and backorders the remaining 30.
The fill rate is 970 / 1,000 = **97%** — 97% of demand satisfied instantly from stock.

Why it matters, and its mirror
--------------------------------

Fill rate drives **customer satisfaction** and **revenue**, signals forecasting and replenishment
health, and is benchmarked at **95-98%** across many industries. It is the exact **complement of the
stockout rate**: :math:`\text{Fill Rate} = 100\% - \text{Stockout Rate}`.

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>` · :doc:`SLA (Service Level Agreement) <208-sla-service-level-agreement>`

----

.. hint::
   **More in Operations & Supply Chain**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Stockouts <401-stockouts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Fill Rate <https://insightful-data-lab.com/2025/08/23/fill-rate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
