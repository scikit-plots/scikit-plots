:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-oversampling:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Oversampling</b></div>`

==============
Oversampling
==============

*Rebalancing classes by replicating or synthesising additional minority-class examples.*

What it is
----------

**Oversampling** is a resampling strategy for class imbalance that *grows the
minority class* — by duplicating real samples or generating new ones — until the
classes are closer to balanced. It is the mirror image of undersampling, which
shrinks the majority class instead.

Why it's used
-------------

When one class is rare (fraud, disease, defects), a model can reach high accuracy
by almost always predicting the majority class while essentially ignoring the
minority. Oversampling raises the minority's presence in training so the model is
forced to learn it, usually improving minority-class recall and F1.

How it's done
-------------

- **Random oversampling** — duplicate existing minority rows at random until the
  counts match.
- **Synthetic oversampling** — create *new* minority points with methods such as
  SMOTE or ADASYN instead of exact copies.

Trade-offs
----------

Advantages:

- Stops the model from ignoring the minority class.
- Random oversampling is trivial to apply.
- Often lifts recall and F1 for the rare class.

Disadvantages:

- Duplicating rows can cause overfitting to those exact points.
- A larger training set means longer training.
- Synthetic methods can introduce unrealistic or noisy samples.

Example
-------

With 10,000 non-fraud and 1,000 fraud rows, oversampling brings the fraud class up
to 10,000 — by duplication (random) or by synthesis (SMOTE).

.. code-block:: python

   from collections import Counter
   from imblearn.over_sampling import RandomOverSampler, SMOTE

   X_dup, y_dup = RandomOverSampler(random_state=42).fit_resample(X, y)
   X_syn, y_syn = SMOTE(random_state=42).fit_resample(X, y)
   print(Counter(y_dup), Counter(y_syn))

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`Class Weighting <002-class-weighting>` · :doc:`Subsampling <001-subsampling>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>`

----

.. hint::
   **More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Oversampling <https://insightful-data-lab.com/2025/08/30/oversampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
