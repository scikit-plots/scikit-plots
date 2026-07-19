:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-continuous-retraining:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Continuous Retraining</b></div>`

=======================
Continuous Retraining
=======================

*Automatically retraining models on fresh data to counter drift.*

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

**Continuous retraining** is the practice of **regularly updating a model with new data** to keep
it accurate in production — also called online retraining or model refresh. It exists because
real-world data **changes over time**: distribution shift, new categories, seasonal patterns.

Why it's needed
---------------

Three pressures. **Drift** — feature distributions move (customer behaviour, fraud tactics) and
feature-target relationships evolve. **Business change** — new products, regulations or customer
segments. And **operational resilience** — keeping KPIs (AUC, calibration, accuracy) stable
rather than letting the model **decay** on stale data.

How it works
------------

A monitoring pipeline watches for **drift and KPI degradation**, then a **trigger** fires —
**scheduled** (weekly/monthly refresh) or **event-driven** (drift past a threshold, KPI below
target). The retraining pipeline pulls **new labelled data**, retrains or fine-tunes, **validates
on a fresh holdout**, compares against the current model (A/B or shadow deployment), and
**deploys only if it improves**.

Approaches and trade-offs
-------------------------

**Batch** retraining rebuilds from scratch periodically — simple but resource-heavy.
**Incremental / online** learning updates weights as data streams in. **Hybrid** keeps a frozen
base and fine-tunes on recent data. The payoff is **stability under drift** with less manual work;
the costs are needing **robust MLOps** (validation, reproducibility), **label availability** (no
labels, no retraining), and guarding against **catastrophic forgetting** when old data is dropped.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Drift Detection <138-drift-detection>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Data Drift <331-data-drift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Recalibration <159-recalibration>` · :doc:`Reweighting <160-reweighting>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Continuous Retraining <https://insightful-data-lab.com/2025/08/23/continuous-retraining/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
