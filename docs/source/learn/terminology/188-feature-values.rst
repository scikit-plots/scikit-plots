:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-feature-values:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Feature Values</b></div>`

================
Feature Values
================

*The actual input feature values fed to a model at scoring time.*

What it is
----------

**Feature values** are the actual **numerical, categorical or textual values** describing an
observation. The distinction in terms: a **feature** is a variable — a column in the dataset — while
a **feature value** is the concrete entry for one observation, a single cell in a given row.

An example
----------

In a table predicting whether a customer buys, the **features** are Age, Gender and Income; the
**feature values** for one customer might be Age = 25, Gender = Male, Income = ``40,000``, and for
another Age = 32, Gender = Female, Income = ``55,000``. Each row supplies one set of feature values.

Types of value
--------------

They come in several kinds: **numerical** (Age = 25), **categorical** (Gender = Male/Female),
**binary** (Yes/No, 0/1), **textual** (reviews, turned into embeddings or bag-of-words), and
**derived** features engineered from raw data ("income per household member").

Their role, and why they matter
---------------------------------

Feature values are the **inputs** a model learns from to predict a target. A linear model writes the
prediction as

.. math::

   \hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b,

with the :math:`x_i` the feature values and the :math:`w_i` the learned weights. Because they drive
everything downstream, **data quality**, **scaling** (so no variable dominates), **engineering**, and
**interpretability** all hinge on getting feature values right.

----

**Mind map — connected ideas**

   :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Model Weights <155-model-weights>` · :doc:`Embedding <173-embedding>` · :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Model Stability <187-model-stability>`

----

**More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Ops Health Dashboard <206-ops-health-dashboard>`

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Feature Values <https://insightful-data-lab.com/2025/08/23/feature-values/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
