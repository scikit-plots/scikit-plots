:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-caching:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Caching</b></div>`

=========
Caching
=========

*Storing computed results to serve repeated requests faster.*

What it is
----------

**Caching** stores the results of expensive computation and **reuses** them instead of recomputing —
trading **memory** for **speed** and **cost**. If the same work would produce the same answer, a cache
returns it instantly.

In model serving
----------------

The signature example is the **KV cache** in transformers, which keeps the **key/value** tensors of earlier
tokens so generating each new token doesn't re-process the whole sequence. It grows **linearly with sequence
length** and can exceed the **model weights** in memory — which is why it is often **quantized**. Beyond
that, **prediction caching** reuses answers to repeated requests and **feature caching** precomputes costly
features.

The trade-offs
--------------

A cache must handle **staleness** — cached values can go **out of date** when inputs or the model change — so
it needs **invalidation** and **eviction** policies, and it consumes **memory** that must be budgeted against
the speed it buys.

----

**Mind map — connected ideas**

   :doc:`Quantization <343-quantization>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Compute budgets <383-compute-budgets>`

----

**More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Caching <https://insightful-data-lab.com/2025/08/20/caching/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
