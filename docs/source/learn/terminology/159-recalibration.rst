:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-recalibration:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Recalibration</b></div>`

===============
Recalibration
===============

*Re-aligning predicted probabilities with observed frequencies after drift.*

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

**Recalibration** adjusts a model's **predicted probabilities** so they match the **true
likelihood** of outcomes. A model may output a 0.9 "probability of fraud", but if that score is
**over- or under-confident**, recalibration corrects the mismatch — without touching the model's
ranking.

Why it's needed
---------------

Many models (SVMs, neural nets, boosted trees) are **poorly calibrated**: their raw outputs are
useful for *ordering* cases but are not true probabilities. In high-stakes domains — medicine,
finance, fraud — decisions need **trustworthy probabilities**, not just a ranking. The target is
simple: of all cases scored **0.7**, about **70%** should actually be positive.

How it's done
-------------

Fit a small **calibration model on a validation set**, comparing predicted scores to true
outcomes. The common methods are **Platt scaling** (a logistic fit on the raw scores),
**isotonic regression** (a non-parametric monotonic mapping), **temperature scaling** (one
parameter on the logits, popular in deep learning), and **Bayesian** recalibration. A
**calibration curve** — predicted probability versus observed frequency — diagnoses the need:
deviation from the diagonal means miscalibration.

Examples and the contrast
-------------------------

A spam filter scores an email **0.9**, but only 70% of such emails are truly spam — Platt scaling
corrects it toward **0.72**. A mortality model predicts **0.2** where the real rate is **30%** —
recalibration nudges it to **0.3**. This is distinct from **threshold tuning**, which moves the
decision cutoff (say 0.5 → 0.3) to trade precision against recall; recalibration changes the
**probabilities themselves**. It works as a **post-processing** step (no retraining), though it
needs a reliable calibration set and may leave ranking metrics like AUC unchanged.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Re-scoring <137-re-scoring>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Recalibrate Thresholds <165-recalibrate-thresholds>` · :doc:`Drift Detection <138-drift-detection>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Recalibration <https://insightful-data-lab.com/2025/08/23/recalibration/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
