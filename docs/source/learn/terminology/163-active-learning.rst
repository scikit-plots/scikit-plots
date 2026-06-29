:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-active-learning:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Active Learning</b></div>`

=================
Active Learning
=================

*Iteratively querying the most informative examples to label, cutting labelling cost.*

What it is
----------

**Active learning** trains a model **iteratively** and lets it **choose the most informative
examples to label**, instead of labelling everything. The goal is **high accuracy from far fewer
labels** — invaluable when annotation is expensive or slow, as with medical images or legal
documents.

The loop
--------

Start with a **small labelled set** and a large **unlabelled pool**. Train an initial model, use
a **query strategy** to pick the most valuable unlabelled samples, send them to an **oracle** (a
human expert) for labels, add them to the training set and retrain — repeating until the model is
good enough or the labelling budget runs out.

Query strategies
----------------

How to choose what to label. **Uncertainty sampling** picks the least confident cases (for binary
classification, predicted probability near 0.5). **Query by committee** trains several models and
picks where they **disagree** most. **Expected model change** chooses points that would most move
the model. **Diversity sampling** picks examples **unlike** the existing training data to cover
the input space.

Why it works
------------

Given 100,000 unlabelled emails at ``$2`` each to label, training on 1,000 and then querying the
500 most uncertain improves accuracy **faster than random labelling**. Active learning cuts
**annotation cost**, accelerates learning, and naturally **prioritises rare or uncertain cases** —
helping with imbalance for free.

----

**Mind map — connected ideas**

   :doc:`Reweighting <160-reweighting>` · :doc:`Continuous Retraining <161-continuous-retraining>` · :doc:`Ensemble <154-ensemble>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Bayesian Correction <164-bayesian-correction>`

----

**More in Model Training & Optimization**

   :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Active Learning <https://insightful-data-lab.com/2025/08/23/active-learning/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
