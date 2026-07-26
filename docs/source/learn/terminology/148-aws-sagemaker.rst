:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-aws-sagemaker:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧰&nbsp;&nbsp;<b>AWS SageMaker</b></div>`

===============
AWS SageMaker
===============

*Amazon's managed platform for building, training and deploying ML models.*

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

**Amazon SageMaker** is AWS's **end-to-end, fully managed ML platform** for **building,
training, deploying and monitoring** models at scale. It is the AWS answer to Google's Vertex
AI and Microsoft's Azure ML.

The pipeline it covers
----------------------

SageMaker spans the whole lifecycle. **Data prep** with Data Wrangler and a Feature Store
(over S3, Redshift, Athena). **Training** for any major framework (TensorFlow, PyTorch,
scikit-learn, XGBoost) on managed CPU/GPU instances, with distributed training and **automatic
hyperparameter tuning**. **Deployment** as real-time, asynchronous or batch endpoints, with
autoscaling. **MLOps** via Pipelines (CI/CD), Model Monitor (drift, bias, quality), Clarify
(bias and explainability) and Debugger. And **generative AI** through JumpStart's pre-trained
models.

Workflow, benefits, costs
-------------------------

The flow is: prepare data in S3 → train on managed infrastructure → auto-tune → deploy to an
endpoint → monitor and retrain. The upside is **scale, framework flexibility** (any Docker
container), deep **AWS integration** and a full **MLOps** toolkit. The downside is
**complexity** — many services and a steep curve — and **cost** that climbs fast if endpoints
run 24/7.

Versus the alternatives
-----------------------

Against **Vertex AI**, the two are close peers — both full end-to-end platforms, differing
mainly by cloud ecosystem (AWS vs GCP). Against the **OpenAI API**, the trade is control for
simplicity: SageMaker trains and serves *any* custom model, while an inference-only API is
faster to start but limited to the provider's models.

----

*Theme:* :ref:`ML Platforms & Tools <term-theme-platforms>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Vertex AI <149-vertex-ai>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>`

----

.. hint::
   **More in ML Platforms & Tools**

   :doc:`Google Experiments <100-google-experiments>` · :doc:`Kaggle <273-kaggle>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Vertex AI <149-vertex-ai>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `AWS SageMaker <https://insightful-data-lab.com/2025/08/24/aws-sagemaker/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
