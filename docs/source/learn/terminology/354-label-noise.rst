:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-label-noise:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Label Noise</b></div>`

=============
Label Noise
=============

*Errors in training labels that can mislead a model.*

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

**Label noise** is **incorrect or unreliable labels** in a dataset — training examples assigned the
wrong class or target, arising from human labellers, weak heuristics or imperfect sensors (a tweet "I
love this" marked **negative**, or a misrecorded diagnosis).

The three types
---------------

**Random (uniform) noise**: labels flipped independently of the features (say 10% at random).
**Class-conditional noise**: mislabeling depends on the class — "cat" is confused for "dog" more often
than for "car". **Feature-dependent (systematic) noise**: ambiguous or low-quality inputs are
mislabeled more — blurry dog photos read as cats.

Why it hurts
------------

It **degrades training** (models learn wrong patterns), **miscalibrates** probabilities, and
**distorts evaluation** (a noisy test set makes metrics meaningless). Tell-tale signs: training
accuracy never reaching 100%, a **stalling loss**, memorisation of noise with low validation
performance, and **high disagreement among annotators**.

Coping
------

**Clean** the data (multiple annotators, keep high-agreement labels); use **noise-robust losses** (MAE,
generalised cross-entropy) with regularisation and early stopping; **model the noise** with a
transition matrix; combine a small clean set with a large noisy one (**weak / semi-supervised**); and
always keep a **clean gold-standard evaluation set** with robust metrics like AUC.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Weak Supervision <346-weak-supervision>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Multiclass AUROC <022-multiclass-auroc>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Label Noise <https://insightful-data-lab.com/2025/08/20/label-noise/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
