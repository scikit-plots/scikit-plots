:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-cluster-based-undersampling:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Cluster-based undersampling</b></div>`

=============================
Cluster-based undersampling
=============================

*Clusters the majority class and keeps representatives of each cluster, shrinking it while preserving its structure.*

What it is
----------

**Cluster-based undersampling** is a smarter way to shrink the majority class than
dropping rows at random. It first **clusters** the majority samples (typically with
K-means) and then keeps a **representative** from each cluster, so the reduced set
still covers the full spread of the majority class instead of leaving gaps by
chance.

How it works
------------

1. Split the data into majority and minority classes.
2. Cluster the majority class (commonly K-means).
3. From each cluster, keep representatives — the points nearest the centroid, or a
   fixed proportion of the cluster.
4. Combine those with the minority class to form a balanced dataset.

Trade-offs
----------

Advantages:

- Preserves the *structure* of the majority class, so less information is lost.
- Avoids the luck-of-the-draw problem of random undersampling.
- Often classifies better than naive undersampling.

Disadvantages:

- The clustering step adds cost.
- Results depend on the clustering method and the number of clusters :math:`K`.
- It still discards data, so cutting too far risks underfitting.

Example
-------

With 10,000 majority and 1,000 minority samples, cluster the majority into 1,000
clusters and keep one representative per cluster — a balanced 1,000 vs 1,000 that
still spans the majority distribution.

.. code-block:: python

   from collections import Counter
   from imblearn.under_sampling import ClusterCentroids

   print("before:", Counter(y))
   X_res, y_res = ClusterCentroids(random_state=42).fit_resample(X, y)
   print("after: ", Counter(y_res))

----

**Mind map — connected ideas**

   :doc:`Random Undersampling <008-random-undersampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Subsampling <001-subsampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>`

----

**More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Cluster-based undersampling <https://insightful-data-lab.com/2025/08/30/cluster-based-undersampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
