:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-categorical-explosions:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Categorical Explosions</b></div>`

========================
Categorical Explosions
========================

*A surge in distinct categorical values that strains encoders and models.*

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

A **categorical explosion** happens when a categorical feature has a **very large number of unique
levels**, so that naive encoding — one-hot in particular — produces a **feature explosion**: the
dataset becomes enormously **wide and sparse**, straining storage, computation and model quality.

The problem in numbers
----------------------

A ``Zip Code`` field with **50,000** values becomes **50,000 binary columns** after one-hot encoding;
a ``Product ID`` with a million values becomes a **million columns**. The damage is fourfold: **high
dimensionality** (overfitting), **sparsity** (mostly zeros), **compute cost** (slow, memory-hungry
training), and **poor generalisation** to unseen categories.

Handling it
-----------

Six strategies replace naive one-hot. **Group** rare categories into "Other" or bucket by region.
**Frequency or target encoding** replaces a category with its count or mean target. The **hashing
trick** maps categories into a fixed number of buckets. **Entity embeddings** learn dense vectors for
each category during training. **Dimension reduction** (PCA, autoencoders) compresses the encoding.
And **domain knowledge** lowers granularity — "Product Category" instead of "Product ID".

Where it appears
----------------

The usual sources are **retail** (product and user IDs), **geography** (zip codes, GPS), **web data**
(URLs, session and device IDs) and **healthcare** (ICD-10 codes, tens of thousands of them).

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Embedding <173-embedding>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`Autoencoder <171-autoencoder>` · :doc:`Drift Detection <138-drift-detection>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>` · :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Categorical Explosions <https://insightful-data-lab.com/2025/08/23/categorical-explosions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
