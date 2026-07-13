.. _dpa-clustering:

========================================================================
Clustering
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 27 of 56  ·  *intermediate*

:doc:`◀ Previous · Putting Similar Observations into Clusters <26-putting-similar-observations-into-clusters>`   ·   :doc:`Next · Recency, Frequency, and Monetary Value (RFM) ▶ <28-recency-frequency-and-monetary-value-rfm>`   ·   :doc:`↑ Section <index>`


Algorithms for groups
-----------------------

Turning the idea of clustering into practice means choosing an **algorithm** — a procedure that
actually finds the groups. Many exist, differing in how they define a cluster and how they search. The
most widely used, and the natural starting point, is **k-means**.

k-means
---------

**k-means** partitions the data into a pre-chosen number :math:`k` of clusters, each summarised by its
**centroid** (the mean of its members). It seeks to minimise the **within-cluster sum of squares** —
the total squared distance from points to their centroids — through a simple, repeating two-step loop:

1. **Assign** each observation to the **nearest** centroid;
2. **Update** each centroid to the mean of the points now assigned to it.

Repeat until assignments stop changing. In scikit-learn this is ``KMeans(n_clusters=k)``. It is fast
and intuitive, though it assumes roughly round, similarly-sized clusters and needs :math:`k` chosen in
advance.

Choosing k
------------

Since :math:`k` is an input, how many clusters should there be? Two common guides: the **elbow
method** plots the within-cluster sum of squares against :math:`k` and looks for the "elbow" where
adding clusters stops helping much; the **silhouette score** measures how well each point sits in its
cluster versus the nearest other one, rewarding tight, well-separated groups. Neither is automatic —
the "right" :math:`k` often depends on what is **useful** for the business.

Other approaches
------------------

k-means is not the only option. **Hierarchical** clustering builds a tree (dendrogram) of nested
groups, needing no :math:`k` up front and revealing structure at every scale. **Density-based**
methods like **DBSCAN** grow clusters from dense regions, handling odd shapes and marking outliers as
noise. Each embodies a different notion of what a cluster *is* — but all serve the same goal: similar
together, different apart.

.. hint::

   **Related lessons:** :doc:`Putting Similar Observations into Clusters <26-putting-similar-observations-into-clusters>`  ·  :doc:`Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`  ·  :doc:`RFM Analysis <29-rfm-analysis>`  ·  :doc:`Using Decision Trees to Explain Clustering Results <49-using-decision-trees-to-explain-clustering-results>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/clustering/ <https://insightful-data-lab.com/2026/01/14/clustering/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
