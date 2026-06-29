:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-lagging-indicators:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Lagging Indicators</b></div>`

====================
Lagging Indicators
====================

*Metrics that confirm trends after they occur, such as revenue or churn.*

What it is
----------

A **lagging indicator** is a metric that **confirms the impact of a change after it has already
happened**. Lagging indicators measure **outcomes**, not early signals — in ML monitoring, they are
the **model-performance metrics**: loss, AUC, accuracy, calibration.

Characteristics
---------------

They are **reactive**, surfacing a problem only once it has occurred. They are a **direct measure**
of end results — model performance and business KPIs — which makes them the natural tools for
**validation and confirmation** rather than early warning.

Examples
--------

Three groups. **Performance metrics**: accuracy, precision/recall/F1, AUC, log loss, calibration
error. **Business KPIs after the fact**: CTR falling, fraud losses rising, churn climbing. And in a
**monitoring** context: an AUC drop that means drift *already* hurt predictions, or a loss spike
that follows a distribution change.

Why they matter
---------------

Lagging indicators **confirm whether the leading indicators actually mattered** — whether drift or a
data-quality issue translated into real damage. They are what go/no-go **retraining decisions** rest
on. In a fraud model, a leading indicator might be drift in ``transaction_type``; the **lagging**
indicator is AUC sliding from 0.87 to 0.72 — proof the model is now underperforming.

----

**Mind map — connected ideas**

   :doc:`Leading Indicators <169-leading-indicators>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Re-scoring <137-re-scoring>`

----

**More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Lagging Indicators <https://insightful-data-lab.com/2025/08/23/lagging-indicators/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
