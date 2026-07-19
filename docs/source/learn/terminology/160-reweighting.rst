:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-reweighting:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚙️&nbsp;&nbsp;<b>Reweighting</b></div>`

=============
Reweighting
=============

*Adjusting sample or class weights to correct bias or distribution shift.*

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

**Reweighting** assigns **different weights to samples, features or loss terms** so that training
or evaluation reflects the **true importance, fairness or distribution** of the data. It leaves
the raw data untouched and instead changes **how much influence** each part has.

Where it's used
---------------

Several settings. **Class imbalance**: upweight rare positives (fraud, disease) so the model
can't ignore them. **Covariate shift / domain adaptation**: reweight training samples to match
the production distribution (an 80%-desktop training set toward 50%-mobile production).
**Fairness**: upweight underrepresented protected groups for equal contribution. Plus **loss
reweighting** (balancing terms in multi-task or adversarial training) and **importance sampling**
(correcting biased draws for unbiased estimates).

The math
--------

The ordinary average loss

.. math::

   L = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), y_i)

becomes, with weights,

.. math::

   L = \frac{1}{N} \sum_{i=1}^{N} w_i \, \ell(f(x_i), y_i),

where :math:`w_i` is the weight on sample :math:`i` — a larger :math:`w_i` gives that example
more influence.

Examples and trade-offs
-----------------------

On a **1%-fraud** dataset, an unweighted model predicts "not fraud" always; weighting fraud at 99
against 1 makes those cases count and **lifts recall**. A loan model that is 80% male can upweight
female applicants toward fairness. The benefits — handling imbalance and drift, improving
fairness, **keeping data intact** (no over/undersampling) — come with risks: **extreme weights**
can overfit the minority, and choosing weights well takes validation.

----

*Theme:* :ref:`MLOps, Serving & Monitoring <term-theme-mlops>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Recalibration <159-recalibration>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>`

----

.. hint::
   **More in MLOps, Serving & Monitoring**

   :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>` · :doc:`Caching <342-caching>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>` · :doc:`Compute budgets <383-compute-budgets>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Feature Values <188-feature-values>` · :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>` · :doc:`Inference Cost (Inference $) <385-inference-cost-inference>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Manual review minutes <384-manual-review-minutes>` · :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>` · :doc:`Model Stability <187-model-stability>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Reweighting <https://insightful-data-lab.com/2025/08/23/reweighting/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
