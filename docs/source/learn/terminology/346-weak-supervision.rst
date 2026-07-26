:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-weak-supervision:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Weak Supervision</b></div>`

==================
Weak Supervision
==================

*Training from noisy, heuristic or partial labels instead of clean ones.*

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

**Weak supervision** trains models from **noisy, cheap, or imprecise** label sources instead of costly
**hand-labeling** — a direct answer to the training-data **bottleneck**. Rather than perfect ground truth, it
leans on many **imperfect** signals.

How it works
------------

Users write **labeling functions** — small snippets of **heuristics**, keyword rules, external knowledge, or
other models' outputs — that each **label** or **abstain**, often with **unknown** accuracy and **conflicting**
votes. A **label model** then **de-noises** and combines them, estimating each function's reliability to
produce **probabilistic** consensus labels — with **no** ground truth. Those labels train a downstream
classifier. This is the **Snorkel / data-programming** paradigm.

Its trade-off
-------------

Weak supervision makes labeling **dramatically** faster and its rules **interpretable** and easy to update, at
the cost of **noisier** labels than full annotation. Best practice keeps a small **hand-labeled** set to
**validate** quality and compare against fully supervised baselines.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Full Annotation <345-full-annotation>` · :doc:`Label Noise <354-label-noise>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Embedding <173-embedding>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Weak Supervision <https://insightful-data-lab.com/2025/08/20/weak-supervision/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
