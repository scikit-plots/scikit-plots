:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-onnx-open-neural-network-exchange:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧰&nbsp;&nbsp;<b>ONNX (Open Neural Network Exchange)</b></div>`

=====================================
ONNX (Open Neural Network Exchange)
=====================================

*An open format for exchanging trained models across frameworks.*

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

**ONNX (Open Neural Network Exchange)** is an **open-source, standardised format** for representing
machine-learning models — both deep-learning and traditional ML. It acts as a **universal
translator**: build a model in one framework (PyTorch, TensorFlow, scikit-learn) and **deploy it in
another** environment optimised for inference, without rewriting or retraining. Originally called
*Toffee* and built by the PyTorch team at Facebook, it was renamed ONNX in September 2017 and is now
backed by Microsoft, IBM, Intel, AMD, Arm, Qualcomm and others.

How it represents a model
---------------------------

An ONNX model is an **extensible computation graph** — a directed acyclic graph whose **nodes** are
**operators** (convolution, pooling, activation) with typed inputs and outputs. It defines a set of
**built-in operators** and **standard data types**, and serialises the network structure (layers,
connections) and parameters (weights, biases) in a **framework-agnostic** way using **Protocol
Buffers** as the container format. The focus is on **inference** (evaluation), not training.

The workflow
------------

The lifecycle is **train, export, run**. Train in any framework, **export** to a single ``.onnx`` file
(for example with ``torch.onnx.export``), then execute it with a runtime such as **ONNX Runtime**,
whose **execution providers** target CPUs, GPUs and specialised accelerators. The runtime applies
**graph optimisations** — node fusion, constant folding — that cut inference latency.

Why it matters
--------------

ONNX delivers **framework interoperability** (no ecosystem lock-in), **hardware optimisation** (one
``.onnx`` runs on NVIDIA GPUs, Intel CPUs or mobile NPUs via tools like OpenVINO and CoreML),
**faster inference**, and **simplified deployment** — one delivery format across cloud, edge, mobile
and even the browser (ONNX Runtime Web on WebGL/WebAssembly). Its main limitation is that some
**proprietary or very new operators** are not yet supported.

.. code-block:: python

   import torch
   import onnxruntime as ort

   # Export a trained PyTorch model to ONNX
   dummy = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy, "model.onnx",
                     input_names=["input"], output_names=["output"],
                     dynamic_axes={"input": {0: "batch_size"}})

   # Run inference with ONNX Runtime
   session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
   outputs = session.run(None, {"input": input_array})

----

*Theme:* :ref:`ML Platforms & Tools <term-theme-platforms>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Quantization <343-quantization>` · :doc:`Caching <342-caching>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

----

.. hint::
   **More in ML Platforms & Tools**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Google Experiments <100-google-experiments>` · :doc:`Kaggle <273-kaggle>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Vertex AI <149-vertex-ai>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `ONNX (Open Neural Network Exchange) <https://insightful-data-lab.com/2025/08/20/onnx-open-neural-network-exchange/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
