:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-random-undersampling:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Random Undersampling</b></div>`

======================
Random Undersampling
======================

*Drops random majority-class rows to balance classes — simple, but can discard useful information.*

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

**Random undersampling** balances an imbalanced dataset the simplest way possible:
by randomly deleting majority-class rows until the classes are closer in size. It is
the counterpart of oversampling, which adds minority rows rather than removing
majority ones.

Why it's used
-------------

When the majority class dominates (fraud, churn), a classifier can drift toward
always predicting it. Trimming the majority restores balance so the model gives the
minority class real weight — and, as a bonus, trains faster on the smaller set.

How it works
------------

With 10,000 majority and 1,000 minority rows, draw 1,000 of the majority at random
and keep all the minority, yielding a balanced 1,000 vs 1,000.

Trade-offs
----------

Advantages:

- Simple and fast.
- Balances the classes so the minority is not ignored.
- Less data means quicker training.

Disadvantages:

- **Information loss** — useful majority rows are thrown away.
- Risk of **underfitting** from training on fewer points.
- Poor when the minority class is tiny, since too much majority data is discarded.

Smarter alternatives
--------------------

- **Random oversampling** or **SMOTE** — grow the minority instead of shrinking the
  majority.
- **Tomek links / Edited Nearest Neighbours** — remove borderline or overlapping
  majority points rather than random ones.
- **Ensemble methods** — combine undersampling with bagging/boosting, e.g. a
  Balanced Random Forest.

Example
-------

.. code-block:: python

   from collections import Counter
   from imblearn.under_sampling import RandomUnderSampler

   print("before:", Counter(y))
   X_res, y_res = RandomUnderSampler(random_state=42).fit_resample(X, y)
   print("after: ", Counter(y_res))

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Oversampling <004-oversampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Class Weighting <002-class-weighting>` · :doc:`Subsampling <001-subsampling>`

----

.. hint::
   **More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Random Undersampling <https://insightful-data-lab.com/2025/08/30/random-undersampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
