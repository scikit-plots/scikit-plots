:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-full-annotation:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Full Annotation</b></div>`

=================
Full Annotation
=================

*Labelling every example fully, the most costly supervision setting.*

What it is
----------

**Full annotation** means **labeling every example** in a dataset with its **ground-truth** target — the
complete, high-quality supervision that classic **supervised learning** assumes. Each image gets its boxes,
each sentence its tags.

The cost
--------

It is **manual, slow, and expensive** — often the most **tedious** part of an ML project — and requires
annotators following **guidelines**, whose **disagreements** become a data-quality issue (measured with
inter-annotator agreement). At scale, labeling everything is simply **infeasible**.

Why it persists
---------------

Despite the cost, fully annotated data gives the **strongest** signal and remains the **go-to** for
production and the **gold-standard benchmark**. Its expense is exactly what motivates **weak**, **semi-**,
and **self-supervised** learning, which trade some label quality for far less human effort.

----

**Mind map — connected ideas**

   :doc:`Weak Supervision <346-weak-supervision>` · :doc:`Label Noise <354-label-noise>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Manual review minutes <384-manual-review-minutes>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Full Annotation <https://insightful-data-lab.com/2025/08/20/full-annotation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
