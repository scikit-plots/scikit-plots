:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-monitoring-pipelines:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Monitoring Pipelines</b></div>`

======================
Monitoring Pipelines
======================

*Automated systems that track model and data health in production.*

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

A **monitoring pipeline** is the system of checks and data flows that **continuously tracks the
health and performance** of an ML model in production — a "control tower" whose job is to catch
**drift, degradation, anomalies and failures early**, before they cause silent harm.

What it watches
---------------

Four layers. **Data monitoring**: schema validation, missing values and outliers, **feature
drift** (PSI, KS test, MMD) and representation drift in embeddings. **Model performance**: AUC,
precision, recall, F1 and calibration for classifiers; MSE/RMSE/MAE/R² for regressors; business
metrics like CTR and fraud savings. **Operational**: latency, throughput, uptime, cost per
prediction. And **guardrails**: alerts when thresholds break (drift > 0.2, latency > 200ms),
triggering auto-retrain or rollback.

How it flows
------------

The cycle is **collect** (log predictions, inputs, metadata, eventual outcomes) → **aggregate**
(metrics over daily/weekly windows) → **compare** (against training baselines and SLAs) →
**alert** (flag anomalies and degraded KPIs) → **action** (retrain, adjust thresholds, or
investigate the data). Dashboards slice these signals by geo, device or cohort to separate
leading from lagging indicators.

Why it matters
--------------

A fraud model whose AUC quietly slips from 0.9 to 0.75, with latency spiking past 300ms, fails
**silently** without monitoring. Pipelines prevent that — and underwrite **fairness and
compliance** (no group disproportionately harmed), **accountability** to stakeholders, and the
feedback loop that drives continuous retraining.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Drift Detection <138-drift-detection>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Data Drift <331-data-drift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Re-scoring <137-re-scoring>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Monitoring Pipelines <https://insightful-data-lab.com/2025/08/23/monitoring-pipelines/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
