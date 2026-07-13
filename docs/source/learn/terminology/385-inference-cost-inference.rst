:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-inference-cost-inference:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Inference Cost (Inference $)</b></div>`

==============================
Inference Cost (Inference $)
==============================

*The monetary cost of serving model predictions.*

What it is
----------

**Inference cost** is the **ongoing** expense of **serving predictions** — the compute (and money) spent
every time the deployed model answers a request. It is usually tracked as the **cost per prediction** (or
per token), the fundamental **unit economic** of an ML product.

Why it dominates
----------------

Unlike **training**, which is paid **once**, inference runs **continuously** and **scales with adoption** —
every user request consumes compute, so cost grows with traffic. At scale it is the **larger** line item,
often **65–80%** of an AI budget, and it is where revenue meets the bill.

Bringing it down
----------------

It is driven by **model size**, **hardware**, and **utilization**, so it falls with **quantization**
(smaller, faster weights), **caching**, **batching**, and **right-sizing** capacity to demand — techniques
that can cut cost per prediction substantially. The target is a **declining** cost-per-prediction over time.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Compute budgets <383-compute-budgets>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Quantization <343-quantization>` · :doc:`Caching <342-caching>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Cloud Inference <153-cloud-inference>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Inference Cost (Inference $) <https://insightful-data-lab.com/2025/08/19/inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
