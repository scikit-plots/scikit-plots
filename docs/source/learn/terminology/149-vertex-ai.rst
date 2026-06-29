:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-vertex-ai:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧰&nbsp;&nbsp;<b>Vertex AI</b></div>`

===========
Vertex AI
===========

*Google Cloud's managed platform for ML model development and deployment.*

What it is
----------

**Vertex AI** is **Google Cloud's managed ML platform** for building, training, deploying and
monitoring models end-to-end, unifying **data, training, inference and monitoring** in one
workflow. It is the GCP counterpart to AWS SageMaker and Azure ML.

The pipeline it covers
----------------------

**Training** for custom models (TensorFlow, PyTorch, scikit-learn, XGBoost) with distributed
runs across CPUs, GPUs and **TPUs**, plus built-in tuning. **Deployment** as real-time or batch
endpoints over Google Cloud Storage, with autoscaling. **Foundation models** — Google's PaLM,
Gemini, Imagen and Chirp — accessible through **Vertex AI Studio** without training from
scratch. **MLOps** with monitoring (drift, bias, feature skew), pipelines, **explainable AI**
and metadata versioning. And a **Feature Store** integrated with BigQuery and Dataflow.

Workflow, benefits, costs
-------------------------

The flow runs ingest (BigQuery, GCS) → prepare features → train (custom or **AutoML**) →
deploy → monitor for drift, performance and fairness. Its strengths are being **fully
managed**, **first-class access to Google's foundation models**, and tight **GCP integration**.
The costs mirror SageMaker's: it can be **expensive at scale** and brings **vendor lock-in** to
GCP.

Versus SageMaker
----------------

The two are near-mirror images — both end-to-end, both supporting any framework and full
MLOps. The real differentiators are **ecosystem** (BigQuery/GCS vs S3/Redshift) and
**foundation-model access** (Gemini and friends native to Vertex). The choice usually follows
whichever cloud an organisation already lives in.

----

**Mind map — connected ideas**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Quantization <343-quantization>`

----

**More in ML Platforms & Tools**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Google Experiments <100-google-experiments>` · :doc:`Kaggle <273-kaggle>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`TPU Clusters <347-tpu-clusters>`

----

*Theme:* :ref:`ML Platforms & Tools <term-theme-platforms>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Vertex AI <https://insightful-data-lab.com/2025/08/24/vertex-ai/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
