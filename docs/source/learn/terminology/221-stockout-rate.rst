:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-stockout-rate:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Stockout Rate</b></div>`

===============
Stockout Rate
===============

*The frequency with which an item is out of stock.*

What it is
----------

**Stockout Rate** is a supply-chain and inventory metric measuring **how often items are out of stock
when there is demand** — the percentage of demand that could not be filled because the product was
unavailable. It is a direct reading of **service-level** performance: a low rate means shelves stay
stocked, a high rate means customers keep hitting empty ones.

The formula
-----------

It is computed by demand or by orders,

.. math::

   \text{Stockout Rate} = \frac{\text{Unfulfilled Units}}{\text{Total Demand}} \times 100\%,

or, order-based, the share of orders that hit at least one stockout.

A worked example
----------------

A month's demand is 1,000 units; inventory covers 950 and 50 go unfilled. The stockout rate is
50 / 1,000 = **5%** — one order in twenty meets an empty shelf.

Why it matters, and reducing it
---------------------------------

Frequent stockouts **push customers to competitors**, forfeit sales, and flag weak forecasting, so the
metric captures the tension between **service level** and **holding cost**. It falls with better
**demand forecasting** (ARIMA, Prophet, LSTM), adequate **safety stock**, shorter vendor **lead
times**, **multi-location** inventory and **real-time tracking**. It is the exact **complement of the
fill rate**.

----

**Mind map — connected ideas**

   :doc:`Fill Rate <220-fill-rate>` · :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>`

----

**More in Operations & Supply Chain**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockouts <401-stockouts>`

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Stockout Rate <https://insightful-data-lab.com/2025/08/23/stockout-rate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
