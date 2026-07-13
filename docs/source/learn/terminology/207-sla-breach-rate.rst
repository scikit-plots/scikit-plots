:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sla-breach-rate:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>SLA Breach Rate</b></div>`

=================
SLA Breach Rate
=================

*The fraction of time a service fails to meet its agreed service level.*

What it is
----------

**SLA Breach Rate** is a **performance metric** measuring how often a service provider — a supplier,
IT team or internal group — **fails to meet the service levels agreed in the SLA**. Each **breach** is
a missed target (a late delivery, missed uptime, poor quality), and the rate reports the **percentage
of obligations not met** over a period. It is the exact **opposite of the SLA compliance rate**.

The formula
-----------

.. math::

   \text{SLA Breach Rate} = \frac{\text{Number of SLA breaches}}{\text{Total SLA obligations}} \times 100\%,

where **breaches** count the times performance fell below target and **obligations** count every
chance there was to meet it.

Worked examples
---------------

In a **supply chain**, 1,000 orders with 50 delivered late gives a breach rate of 50 / 1,000 =
**5%**. In an **IT support desk**, 2,000 tickets with 200 missing their resolution deadline gives
200 / 2,000 = **10%**.

Why it matters, and reducing it
---------------------------------

The rate drives **accountability**, signals **customer satisfaction**, triggers **contractual
penalties or credits**, and pinpoints weak areas. It falls with **better forecasting**, **real-time
monitoring and alerts**, **supplier collaboration**, **automated escalation**, and **realistic SLA
reviews**. Its mirror image is the compliance rate (100% minus the breach rate), and the common KPIs
it is measured against are on-time delivery, fill rate and uptime.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`SLA (Service Level Agreement) <208-sla-service-level-agreement>` · :doc:`SLOs (Service Level Objectives) <391-slos-service-level-objectives>` · :doc:`SLI (Service Level Indicator) <190-sli-service-level-indicator>` · :doc:`SLA Breaches <399-sla-breaches>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `SLA Breach Rate <https://insightful-data-lab.com/2025/08/23/sla-breach-rate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
