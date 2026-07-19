:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-safety-stock:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📦&nbsp;&nbsp;<b>Safety Stock</b></div>`

==============
Safety Stock
==============

*Extra inventory held to buffer against demand or supply variability.*

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

**Safety stock** is the **buffer inventory** held to absorb **uncertainty** in demand and supply — the
cushion that keeps you selling when demand spikes or a shipment is late. It is the difference between a
naive "average" reorder level and a robust one.

The formula
-----------

The statistical form sizes it from the **service level** and demand **variability**:

.. math::

   \text{SS} = Z \times \sigma_D \times \sqrt{L},

where :math:`Z` is the service-level z-score (1.28 for 90\%, 1.65 for 95\%, 2.33 for 99\%), :math:`\sigma_D`
is the standard deviation of demand per period, and :math:`L` is the lead time.

The trade-off
-------------

More safety stock **raises** the service level (fewer stockouts) but **ties up capital** in holding cost —
so the **service level** is chosen by weighing stockout cost against carrying cost, often set higher for
critical or perishable items (via ABC / XYZ classing).

----

*Theme:* :ref:`Operations & Supply Chain <term-theme-ops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Long Lead Times <210-long-lead-times>`

----

.. hint::
   **More in Operations & Supply Chain**

   :doc:`Backorder Rate <218-backorder-rate>` · :doc:`Crew Overtime <398-crew-overtime>` · :doc:`Demand Forecasting <215-demand-forecasting>` · :doc:`Fill Rate <220-fill-rate>` · :doc:`Long Lead Times <210-long-lead-times>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Lost Sales Value <219-lost-sales-value>` · :doc:`Overstock % <400-overstock>` · :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>` · :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>` · :doc:`SKU <212-sku>` · :doc:`Slow-Moving SKUs <211-slow-moving-skus>` · :doc:`Stockout Rate <221-stockout-rate>` · :doc:`Stockouts <401-stockouts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Safety Stock <https://insightful-data-lab.com/2025/08/23/safety-stock/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
