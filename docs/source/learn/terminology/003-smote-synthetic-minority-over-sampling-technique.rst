:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-smote-synthetic-minority-over-sampling-technique:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>SMOTE (Synthetic Minority Over-sampling Technique)</b></div>`

====================================================
SMOTE (Synthetic Minority Over-sampling Technique)
====================================================

*Creates new minority examples by interpolating between nearest neighbours instead of duplicating existing rows.*

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

**SMOTE** (Synthetic Minority Over-sampling Technique), introduced by Chawla and
colleagues in 2002, fixes class imbalance by *synthesising* new minority-class
examples rather than duplicating existing ones. Where plain duplication just copies
points, SMOTE invents plausible new ones in the gaps between real minority samples.

Why not just duplicate
----------------------

Random oversampling repeats the same minority rows, so a classifier can memorise
those exact points and overfit. SMOTE instead places new points *along the lines
between* nearby minority samples, filling out the minority region and pushing the
decision boundary toward something more general.

The algorithm
-------------

For each minority sample :math:`x`:

1. find its :math:`k` nearest minority-class neighbours;
2. pick one neighbour :math:`x_{nn}` at random;
3. create a synthetic point on the segment joining them,

.. math::

   x_{\text{new}} = x + \delta\,(x_{nn} - x), \qquad \delta \sim \mathcal{U}(0, 1).

Because :math:`\delta` is uniform on :math:`[0, 1]`, the new point lands somewhere
between the two originals.

Variants
--------

- **Borderline-SMOTE** — synthesises only near the decision boundary, where
  mistakes are most likely.
- **SMOTE-Tomek / SMOTE-ENN** — pair SMOTE with a cleaning step that removes
  overlapping or noisy points.
- **ADASYN** — generates more synthetic points for the minority samples that are
  hardest to learn.

Trade-offs
----------

Advantages:

- Less overfitting than duplicating samples.
- Smoother, more general boundaries and better minority-class scores.

Disadvantages:

- Can create unrealistic points when the minority distribution is complex.
- May push synthetic points into majority territory, causing class overlap.
- More expensive than simple duplication.

Example
-------

With 100 minority and 1,000 majority samples, SMOTE generates 900 synthetic
minority points, giving a balanced 1,000 vs 1,000.

.. code-block:: python

   from collections import Counter
   from imblearn.over_sampling import SMOTE

   print("before:", Counter(y))
   X_res, y_res = SMOTE(k_neighbors=5, random_state=42).fit_resample(X, y)
   print("after: ", Counter(y_res))

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`Class Weighting <002-class-weighting>` · :doc:`Subsampling <001-subsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>`

----

.. hint::
   **More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `SMOTE (Synthetic Minority Over-sampling Technique) <https://insightful-data-lab.com/2025/08/30/smote-synthetic-minority-over-sampling-technique/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
