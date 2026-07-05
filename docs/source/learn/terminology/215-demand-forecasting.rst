:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-demand-forecasting:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Demand Forecasting</b></div>`

====================
Demand Forecasting
====================

*Predicting future demand to guide inventory and production.*

What it is
----------

**Demand forecasting** predicts **future customer demand** for a product from **historical sales**, market
signals, seasonality and known upcoming events. It is the foundation of inventory planning — nearly every
replenishment decision depends on it.

How it's done
-------------

Methods range from **simple baselines** (moving averages) to **time-series** models (ARIMA, exponential
smoothing, Prophet) and **ML**; what matters is capturing **trend**, **seasonality** and demand
**variability** (its standard deviation), not just the average.

Why accuracy matters
--------------------

Forecast **error** propagates downstream — the **reorder point** and **safety stock** are both sized from
the forecast, so a biased or noisy forecast either **starves** shelves (stockouts) or **bloats** inventory
(holding cost). Better forecasts shrink the safety buffer needed for a given service level.

----

**Mind map — connected ideas**

   :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Long Lead Times <210-long-lead-times>`

----

**More in Operations & Supply Chain**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Stockouts <401-stockouts>`

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Demand Forecasting <https://insightful-data-lab.com/2025/08/23/demand-forecasting/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
