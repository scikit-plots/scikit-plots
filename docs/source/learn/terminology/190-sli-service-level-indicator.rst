:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sli-service-level-indicator:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>SLI (Service Level Indicator)</b></div>`

===============================
SLI (Service Level Indicator)
===============================

*A measured signal such as latency or error rate used to track service health.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

A **Service Level Indicator (SLI)** is a **specific, measurable metric** that reflects a service's
actual performance against what was agreed or expected — the **"thermometer"** of how well a service
is running. It is the **raw measurement** used to judge whether a service is meeting its objective.

SLI vs SLO vs SLA
-----------------

Three layers stack up. The **SLI** is the metric measured — "percentage of requests completed within
300 ms". The **SLO** is the target for that metric — "99% of requests must complete within 300 ms".
The **SLA** is the formal contract, often with penalties — "if availability drops below 99%, the
provider credits the customer". SLI is the most granular layer; SLO and SLA build on it.

Common SLIs
-----------

They fall into families: **reliability** (uptime %, mean time between failures), **performance**
(latency, throughput in requests per second), and **quality** (error rate, data accuracy). The same
idea appears in operations as **on-time delivery %**, **fill rate**, or **stockout frequency**.

An example, and why it matters
--------------------------------

An e-commerce site might measure that **95%** of checkout requests finish in under two seconds (the
SLI) against an SLO target of **99%**, under an SLA that compensates customers if uptime falls below
**98%**. SLIs matter because they give **objective evidence** of performance, guide **where to
improve**, and form the measurable **foundation** of every SLO and SLA.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`SLOs (Service Level Objectives) <391-slos-service-level-objectives>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Model Stability <187-model-stability>` · :doc:`ROI (Return on Investment) <191-roi-return-on-investment>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `SLI (Service Level Indicator) <https://insightful-data-lab.com/2025/08/23/sli-service-level-indicator/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
