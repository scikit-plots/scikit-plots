:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-model-stability:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Model Stability</b></div>`

=================
Model Stability
=================

*The consistency of a model's predictions and performance over time.*

What it is
----------

**Model stability** is how **consistently** a model performs when faced with different data samples
(train versus validation), new production data, and **time-evolving** data subject to drift. A stable
model is reliable and predictable and does not swing wildly; an unstable one is sensitive to small
changes in data or training splits and gives inconsistent results.

Four dimensions
---------------

Stability has several faces: **performance** stability (accuracy, AUC, RMSE hold across datasets and
time), **prediction** stability (similar inputs give similar outputs after retraining), **feature**
stability (importances and coefficients stay consistent), and **temporal** stability (the model
resists drift as new data arrives).

Measuring it
------------

The tools are familiar: low **cross-validation variance** across folds, **retraining consistency**
across random seeds, the **Population Stability Index (PSI)** for train-versus-production distribution
shift, **feature-importance** consistency across runs, and ongoing **drift monitoring**. A credit
model with validation AUC **0.85** that falls to **0.70** on next year's production data is unstable
over time — concept drift has set in, and it needs retraining.

Improving it, and why it matters
----------------------------------

Stability improves with **robust feature engineering**, **regularisation**, **ensembles** (which cut
variance), **data-quality monitoring**, **drift-detection systems** (PSI, KS test), and a
**retraining schedule**. It matters because stakeholders need **trust**, regulated industries demand
it for **compliance**, and unstable models make **inconsistent business decisions** — approving and
denying similar loans at random.

----

**Mind map — connected ideas**

   :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

----

**More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Model Stability <https://insightful-data-lab.com/2025/08/23/model-stability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
