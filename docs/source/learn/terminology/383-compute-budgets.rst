:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-compute-budgets:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Compute budgets</b></div>`

=================
Compute budgets
=================

*Limits on the compute resources a workload may consume.*

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

A **compute budget** is the pool of **computational resources** — GPU / TPU hours, **FLOPs**, and the dollars
behind them — allocated to an ML system's **training** and **serving**. It caps how big a model you can
train and how much traffic you can serve.

The two halves
--------------

**Training** is a **one-time** cost that grows with model and data size (its FLOPs approximated by the
**6ND** rule — roughly 6 × parameters × tokens), while **inference** is an **ongoing** cost (about **2N**
FLOPs per forward pass) that scales with usage:

.. math::

   C_{\text{train}} \approx 6ND, \qquad C_{\text{inf}} \approx 2N \ \text{FLOPs per pass}.

In production, inference usually claims the **majority** of the budget.

Managing it
-----------

Teams set **budgets and alerts**, model **optimistic / expected / pessimistic** scenarios, and track unit
economics like **cost per prediction** and **GPU utilization**. Hidden drains — idle instances, failed runs,
oversized experiments — routinely add **20–40%** over the planned figure.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Quantization <343-quantization>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Compute budgets <https://insightful-data-lab.com/2025/08/19/compute-budgets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
