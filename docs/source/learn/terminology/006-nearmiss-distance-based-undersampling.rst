:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-nearmiss-distance-based-undersampling:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>NearMiss (Distance-based Undersampling)</b></div>`

=========================================
NearMiss (Distance-based Undersampling)
=========================================

*Undersamples the majority class by keeping the majority points nearest to the minority, chosen by distance heuristics.*

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

**NearMiss** is a family of *undersampling* methods that shrink the majority class
intelligently. Instead of dropping majority samples at random, NearMiss chooses
which ones to keep based on their **distance to minority samples**, retaining the
informative, hard-to-classify majority points near the boundary and discarding the
"easy" ones far away.

The three versions
------------------

- **NearMiss-1** — keep the majority samples whose *average distance to their
  :math:`k` nearest minority samples* is smallest (closest to the minority class).
- **NearMiss-2** — keep the majority samples whose *average distance to their
  :math:`k` farthest minority samples* is smallest.
- **NearMiss-3** — for each minority sample, keep a fixed number of its nearest
  majority samples, guaranteeing every minority point is surrounded.

Trade-offs
----------

Advantages:

- Preserves boundary-defining points instead of throwing data away blindly.
- Reduces majority-class bias and can generalise better than naive undersampling.

Disadvantages:

- Distance computations make it more expensive than random undersampling.
- Pruning "easy" samples too hard can overfit the difficult regions.
- Results are sensitive to the choice of :math:`k`.

Example
-------

.. code-block:: python

   from collections import Counter
   from imblearn.under_sampling import NearMiss

   print("before:", Counter(y))
   X_res, y_res = NearMiss(version=1).fit_resample(X, y)   # version 1, 2 or 3
   print("after: ", Counter(y_res))

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Random Undersampling <008-random-undersampling>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Subsampling <001-subsampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Oversampling <004-oversampling>`

----

.. hint::
   **More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `NearMiss (Distance-based Undersampling) <https://insightful-data-lab.com/2025/08/30/nearmiss-distance-based-undersampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
