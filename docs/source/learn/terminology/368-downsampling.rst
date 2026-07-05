:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-downsampling:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Downsampling</b></div>`

==============
Downsampling
==============

*Reducing majority-class examples to balance a dataset.*

What it is
----------

**Downsampling** (random **undersampling**) rebalances an **imbalanced** dataset the opposite way — by
**removing majority-class** examples until the classes are closer to even. It keeps all the minority data and
thins out the majority.

The risk
--------

Discarding majority examples can cause **underfitting** — the model loses **informative** cases and may miss
the majority class's general pattern. In extreme imbalance you may throw away the vast bulk of the data
(99%+), damaging its representation.

When to use it
--------------

Prefer downsampling when data is **plentiful**, since it is **computationally efficient** (less data to train
on) and **avoids the overfitting** of duplication. As with upsampling, apply it to the **training set only**
to avoid **data leakage**.

----

**Mind map — connected ideas**

   :doc:`Upsampling <367-upsampling>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Recall <423-recall>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Model Stability <187-model-stability>`

----

**More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Downsampling <https://insightful-data-lab.com/2025/08/20/downsampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
