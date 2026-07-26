:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-upsampling:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Upsampling</b></div>`

============
Upsampling
============

*Increasing minority-class examples to balance a dataset.*

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

**Upsampling** (random **oversampling**) rebalances an **imbalanced** dataset by **inflating the minority
class** — duplicating its examples until the classes are closer to even, so the classifier isn't overwhelmed
by the majority.

The risk
--------

Because it **repeats** existing points, upsampling can cause **overfitting** — the model learns patterns that
only exist in the **duplicated** samples rather than the true minority distribution. The fix is **SMOTE**,
which **interpolates** new synthetic points between a minority point and its nearest neighbors instead of
copying, so no example is an exact duplicate.

When to use it
--------------

Prefer upsampling when the dataset is **small** and discarding data would hurt. Critically, apply it to the
**training set only** — resampling the validation or test set causes **data leakage** and inflates your
metrics.

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Downsampling <368-downsampling>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Recall <423-recall>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Model Stability <187-model-stability>`

----

.. hint::
   **More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Upsampling <https://insightful-data-lab.com/2025/08/20/upsampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
