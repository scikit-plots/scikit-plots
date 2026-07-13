:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-slos-service-level-objectives:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>SLOs (Service Level Objectives)</b></div>`

=================================
SLOs (Service Level Objectives)
=================================

*Internal targets for reliability that a service aims to meet.*

What it is
----------

A **Service Level Objective (SLO)** is a **measurable target** for how a service should perform —
usually a percentage or number. It is the **goal** for reliability, performance or availability that a
team commits to.

The SLA, SLO, SLI hierarchy
-----------------------------

The three fit together. The **SLA** is the external **contract** with customers; the **SLO** is the
internal **target** that supports it; and the **SLI** is the actual **measurement** used to check
performance. The SLO sits in the middle — stricter than the SLA, expressed in terms the SLI can
measure.

Examples and error budgets
----------------------------

Typical SLOs cover **availability** ("99.9% uptime per month", SLI = uptime %), **latency** ("95% of
requests under 300 ms"), **error rate** ("under 0.1% of requests return 5xx in 30 days") and
**throughput** ("at least 5,000 requests per second at peak"). An SLO of 99.9% implies an **error
budget** of 0.1% allowed downtime — the slack that decides when to prioritise reliability over new
features, and that stops teams from the costly, unrealistic chase for 100%.

A worked example
----------------

Suppose an SLA requires **99.5%** uptime. The internal SLO is set at **99.9%** for margin, and the SLI
is measured uptime over the last 30 days. If uptime slips to **99.6%**, the SLA is still safe but the
**SLO is breached** — an early warning of reliability risk *before* any SLA penalty is triggered.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`SLI (Service Level Indicator) <190-sli-service-level-indicator>` · :doc:`SLA (Service Level Agreement) <208-sla-service-level-agreement>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Model Stability <187-model-stability>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `SLOs (Service Level Objectives) <https://insightful-data-lab.com/2025/08/19/slos-service-level-objectives/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
