:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-backorder-rate:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Backorder Rate</b></div>`

================
Backorder Rate
================

*The share of orders that cannot be filled immediately from stock.*

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

**Backorder Rate** is a supply-chain KPI measuring the **proportion of customer demand that cannot be
filled immediately** and must be placed on backorder — **delayed** fulfillment, not necessarily lost.
A high rate points to frequent shortages and weak planning; a low one to efficient inventory and
satisfied customers.

The formula
-----------

It is computed by units or by orders,

.. math::

   \text{Backorder Rate} = \frac{\text{Backordered Units}}{\text{Total Units Ordered}} \times 100\%,

or, order-based, orders containing backordered items divided by total orders.

A worked example
----------------

Of 1,000 units ordered in a month, 920 ship immediately and 80 are backordered, giving 80 / 1,000 =
**8%** of demand delayed.

Where it sits
-------------

Backorder rate shapes **customer experience**, **revenue** (some backorders convert later, some
cancel), **forecasting accuracy** and **supply-chain health**, and it falls with better forecasting
(ARIMA, Prophet, LSTM), proper safety stock, reliable suppliers and ABC segmentation. In one full
snapshot — 920 shipped, 50 backordered then filled, 30 cancelled — the fill rate is 92%, the backorder
rate 5%, the stockout rate 3%, and those 30 cancellations become **lost sales value**.

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Fill Rate <220-fill-rate>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>` · :doc:`SLA (Service Level Agreement) <208-sla-service-level-agreement>`

----

.. hint::
   **More in Operations & Supply Chain**

   :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Stockouts <401-stockouts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Backorder Rate <https://insightful-data-lab.com/2025/08/23/backorder-rate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
