:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-representation-shift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Representation Shift</b></div>`

======================
Representation Shift
======================

*A change in learned feature representations between training and serving.*

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

**Representation shift** occurs when the **internal representation** of data — learned embeddings,
feature vectors — **changes over time** between training and deployment, *even if the raw input
distribution looks similar*. It is a special case of distribution shift, but focused on the
**feature/embedding space** rather than the raw input.

Where it appears
----------------

Three places. In **neural networks and embeddings**, the learned mapping can change (through
retraining or new data), so downstream tasks built on the old space fail. In **preprocessing
pipelines**, steps like TF-IDF, PCA or scaling drift as the data changes — TF-IDF weights move as
new vocabulary dominates. And in **domain shift**, inputs that look similar can still drift in
embedding space — a face model trained on frontal faces, deployed on side profiles.

Why it matters
--------------

Downstream classifiers and regressors that **assume a stable representation degrade**;
**similarity search** (nearest-neighbour in embedding space) returns wrong results; and **fairness**
suffers if some groups' embeddings drift more than others.

Detecting it, with an example
------------------------------

Detection works on the embeddings themselves: **distance metrics** (MMD, energy distance, KL),
tracking cosine or Euclidean shifts; **visualisation** with t-SNE or UMAP to watch clusters move;
and **classifier two-sample tests**. The classic example: the word "mask" embedded mostly as
*cosmetic* in 2019 shifts toward *face covering* in 2020 — breaking any downstream sentiment or
topic model that relied on the old representation.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Embedding <173-embedding>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Leading Indicators <169-leading-indicators>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Energy Distance <176-energy-distance>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Representation Shift <https://insightful-data-lab.com/2025/08/23/representation-shift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
