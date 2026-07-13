:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-aws-sagemaker-endpoints:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>AWS SageMaker Endpoints</b></div>`

=========================
AWS SageMaker Endpoints
=========================

*Managed HTTPS endpoints that serve SageMaker model predictions.*

What it is
----------

An **AWS SageMaker endpoint** is a **fully managed API** for serving a trained model: you
deploy the model and SageMaker handles **infrastructure, scaling and serving** so there are no
servers to run. Input goes in, the endpoint runs inference, predictions come out.

The three types
---------------

**Real-time** endpoints are always-on HTTPS with low latency and autoscaling — for fraud
checks or chatbots. **Asynchronous** endpoints handle **large payloads or long jobs**: upload
to S3, the endpoint processes, results land back in S3 — good for video or big documents.
**Batch transform** runs **offline** over a whole S3 dataset with no live endpoint — ideal for
scheduled jobs like scoring every customer monthly.

Deploying one
-------------

The path is: train or **import** a model (any framework, or ONNX) and save artifacts to S3;
**register** it as a Model with its inference script; create an **endpoint configuration**
(instance type, scaling); deploy; then **invoke** via the SDK or REST:

.. code-block:: python

   import boto3
   sm = boto3.client("sagemaker-runtime")
   response = sm.invoke_endpoint(
       EndpointName="my-endpoint",
       ContentType="application/json",
       Body='{"data":[1.0, 2.0, 3.0]}',
   )

Why use them
------------

They are **fully managed** (no EC2 to babysit), **autoscaling** for traffic spikes,
**framework-flexible** (custom Docker too), **secure** (IAM, VPC, encryption), and **cost
controllable** — pay for instance time, or shift to async/batch to save when latency is not
critical.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Vertex AI <149-vertex-ai>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `AWS SageMaker Endpoints <https://insightful-data-lab.com/2025/08/24/aws-sagemaker-endpoints/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
