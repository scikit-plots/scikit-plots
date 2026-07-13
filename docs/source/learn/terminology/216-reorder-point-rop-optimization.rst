:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-reorder-point-rop-optimization:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Reorder Point (ROP) Optimization</b></div>`

==================================
Reorder Point (ROP) Optimization
==================================

*Setting the stock level at which to reorder to avoid stockouts.*

What it is
----------

The **reorder point** is the inventory level that **triggers a new order** — set so stock arrives just
before you run out. Under a **continuous-review** policy, when on-hand inventory falls to the ROP, a
replenishment order is placed.

The formula
-----------

.. math::

   \text{ROP} = \underbrace{\mu_D \times L}_{\text{demand during lead time}} \;+\; \underbrace{\text{SS}}_{\text{safety stock}},

where :math:`\mu_D` is average demand per period and :math:`L` is the lead time. The first term covers
**expected** usage while the order is in transit; the second buffers **variability**.

Optimizing it
-------------

A good ROP balances **stockout risk** against **holding cost**. It is tuned with **accurate demand
forecasts**, measured **demand and lead-time variability**, and a chosen **service level**; modern systems
recompute it in **real time** as those inputs drift. Drop the safety-stock term only when demand is very
stable and suppliers are reliable.

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Safety Stock <217-safety-stock>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Supplier Management <214-supplier-management>`

----

.. hint::
   **More in Operations & Supply Chain**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Safety Stock <217-safety-stock>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Stockouts <401-stockouts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Reorder Point (ROP) Optimization <https://insightful-data-lab.com/2025/08/23/reorder-point-rop-optimization/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
