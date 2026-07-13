:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-recalibrate-thresholds:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Recalibrate Thresholds</b></div>`

========================
Recalibrate Thresholds
========================

*Updating decision cut-offs as data or costs change.*

What it is
----------

To **recalibrate thresholds** is to **adjust the decision cutoffs or alert limits** of a model or
monitoring system so they stay valid as data or business needs change. In a **classifier**, that
means moving the probability threshold (the default 0.5) used to split positive from negative; in
**monitoring**, it means tuning alert limits on drift, anomaly rate or latency.

Why it's needed
---------------

Thresholds go stale for four reasons. **Data drift** shifts the distribution so the old cutoff no
longer fits. **Business shift** changes the relative cost of false positives versus false
negatives. **Model updates** alter the calibration curve and thus the optimal cutoff. And
**operational noise** makes metrics fluctuate more or less than before.

Examples
--------

A fraud model defaults to flagging when :math:`p > 0.5`; once the business decides missed fraud is
too costly, the threshold drops to **0.3** — more flags, higher recall, lower precision. A PSI
drift alert at **0.1** proves too jumpy against seasonality, so it is relaxed to **0.2**. And after
recalibrating an overconfident model (Platt scaling, isotonic regression), the decision cutoffs
must move with it.

How to do it
------------

Four approaches. **Empirical evaluation** — test candidate thresholds on a validation set.
**Cost-sensitive analysis** — pick the threshold that maximises expected business value.
**Periodic review** — revisit thresholds when retraining or after drift. And **dynamic thresholds**
that adapt automatically to recent performance.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Recalibration <159-recalibration>` · :doc:`Bayesian Correction <164-bayesian-correction>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Critical Value <087-critical-value>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Recalibrate Thresholds <https://insightful-data-lab.com/2025/08/23/recalibrate-thresholds/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
