:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-log-odds:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Log-Odds</b></div>`

==========
Log-Odds
==========

*The logarithm of the odds, the natural scale for logistic models.*

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

The **log-odds** (or **logit**) is the natural logarithm of the **odds** of an event — the ratio of its
probability to its complement:

.. math::

   \text{logit}(p) = \log\!\left(\frac{p}{1 - p}\right), \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}.

Odds run from 0 (at :math:`p=0`) through 1 (at :math:`p=0.5`) to :math:`\infty` (at :math:`p=1`); taking
the log spreads them onto the full line, from :math:`-\infty` to :math:`+\infty`.

Why models use it
-----------------

A probability is trapped in :math:`[0,1]`, awkward to model with a **linear** function; the log-odds is
**unbounded**, so **logistic regression** (and the final layer of many classifiers) models the log-odds as
a **linear** combination of features — the raw "score" before conversion.

Back to probability
-------------------

The **sigmoid** :math:`\sigma` is the **inverse** of the logit — it maps a log-odds score :math:`z` back to
a probability. So the pipeline runs linear score → log-odds → sigmoid → **classification probability** →
threshold → class.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Classification Probability <231-classification-probability>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Log-Odds <https://insightful-data-lab.com/2025/08/21/log-odds/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
