:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-openai-api-ml-api:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧰&nbsp;&nbsp;<b>OpenAI API (ML API)</b></div>`

=====================
OpenAI API (ML API)
=====================

*A hosted API for accessing OpenAI's models programmatically.*

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

The **OpenAI API** is a cloud **machine-learning API**: it exposes OpenAI's models — LLMs,
embeddings, fine-tuned variants — through simple **HTTP endpoints**. Rather than training and
hosting a model yourself, you **send a request, the cloud runs inference, and JSON comes
back** — inference-as-a-service.

What it offers
--------------

The surface is broad. **Chat and text generation** (GPT-family models) for Q&A, summarisation
and reasoning. An **embeddings** endpoint turning text into vectors for semantic search,
clustering and **RAG**. A **fine-tuning** endpoint to specialise a model on your data.
**Moderation** for unsafe content, **vision/multimodal** for images alongside text, and
**speech** both ways (text-to-speech and Whisper transcription). Plus **batch/async**
processing for volume.

Calling it
----------

You POST to a REST endpoint (or use an SDK) and read the response:

.. code-block:: python

   from openai import OpenAI
   client = OpenAI()

   response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": "Explain Bayesian inference simply."}],
   )

Benefits and limits
-------------------

The appeal is **no training or hosting**, **automatic scale**, and **simple integration** that
improves as OpenAI ships new models. The constraints: it is **cloud-only** (needs internet),
adds **network latency**, **costs per token** of input and output, and raises **privacy**
questions — sensitive data must clear HIPAA/GDPR review before it leaves your systems.

----

*Theme:* :ref:`ML Platforms & Tools <term-theme-platforms>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Vertex AI <149-vertex-ai>` · :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Embedding <173-embedding>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`

----

.. hint::
   **More in ML Platforms & Tools**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Google Experiments <100-google-experiments>` · :doc:`Kaggle <273-kaggle>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Vertex AI <149-vertex-ai>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `OpenAI API (ML API) <https://insightful-data-lab.com/2025/08/24/openai-api-ml-api/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
