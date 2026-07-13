:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-model-kpis-key-performance-indicators:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Model KPIs (Key Performance Indicators)</b></div>`

=========================================
Model KPIs (Key Performance Indicators)
=========================================

*Metrics tracked to judge a deployed model's ongoing performance.*

What it is
----------

**Model KPIs** are the **key metrics that track a model's performance, reliability and impact**.
They span **technical performance** (loss, accuracy, AUC, calibration) and **business impact** (ROI,
churn reduction, revenue uplift) — answering both "does it predict well?" and "does it help the
business?".

The four families
-----------------

**Prediction quality**: for classifiers, accuracy, precision/recall/F1, ROC-AUC and PR-AUC, log
loss and calibration; for regressors, MSE/RMSE, MAE and R². **Drift and stability**: feature drift
(PSI, KS test), data-quality checks and representation drift. **Operational**: latency, throughput,
uptime and cost per prediction. **Business impact**: revenue uplift, churn reduction, fraud savings
and ROI.

Examples
--------

A **fraud model** might report a technical KPI of **AUC 0.92**, an operational KPI of **50ms**
latency, and a business KPI of ``$1.2M`` of fraud prevented last quarter. A **recommender** might
report **NDCG@10 of 0.65**, sub-100ms response time, and an **8% lift in click-through rate**.

Leading vs lagging
------------------

KPIs divide into two roles. **Leading indicators** like drift **warn of future trouble** before it
hits performance. **Lagging indicators** like AUC, calibration and loss **confirm actual impact**
after the fact. A healthy dashboard watches both.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Lagging Indicators <168-lagging-indicators>` · :doc:`Recalibrate Thresholds <165-recalibrate-thresholds>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Model KPIs (Key Performance Indicators) <https://insightful-data-lab.com/2025/08/23/model-kpis-key-performance-indicators/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
