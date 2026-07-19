:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cloud-inference:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Cloud Inference</b></div>`

=================
Cloud Inference
=================

*Running model predictions on managed cloud infrastructure.*

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

**Cloud inference** is running a **trained model on cloud infrastructure** to make predictions
on new data. "Inference" means *using* a trained model rather than training one, and "cloud"
means it runs on AWS, GCP or Azure rather than on local or on-device hardware.

How it works
------------

Three stages. **Train** a model (locally or in the cloud) and save its weights. **Deploy** it to
a cloud service (SageMaker, Vertex AI, Azure ML) behind an **API endpoint** (REST or gRPC).
Then **infer**: the client sends input, the service loads the model, runs a forward pass and
returns the prediction.

Benefits and challenges
-----------------------

The upsides are **scalability** (autoscale to millions of requests), **low maintenance** (the
provider runs servers and GPUs), **flexibility** (serve and version many models), and
**accessibility** (any device can call the API). The costs are **latency** (a network round
trip), **OpEx** per prediction at scale, **privacy** obligations when sensitive data leaves the
device, and the need for **monitoring** to keep uptime and fairness in check.

Cloud vs edge
-------------

The contrast is with **edge / on-device** inference. Cloud runs on powerful servers (GPUs,
TPUs) and is easy to update but pays a network-latency cost; edge runs locally on a phone or
IoT device with **low latency** but limited compute and harder remote updates. Image-tagging
APIs, LLM chat services and fraud-scoring endpoints are all cloud inference.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Vertex AI <149-vertex-ai>` · :doc:`Quantization <343-quantization>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cloud Inference <https://insightful-data-lab.com/2025/08/24/cloud-inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
