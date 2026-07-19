.. _dpa-putting-similar-observations-into-clusters:

========================================================================
Putting Similar Observations into Clusters
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 26 of 56  ·  *intermediate*

:doc:`◀ Previous · Partitioning Observations to Train Objective Models <25-partitioning-observations-to-train-objective-models>`   ·   :doc:`Next · Clustering ▶ <27-clustering>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Grouping without labels
-------------------------

Sometimes there is no outcome to predict — just a pile of observations and a hunch that they fall into
**natural groups**. **Clustering** is the task of finding those groups: partitioning observations so
that each group gathers items that are **alike**. It is **unsupervised** — unlike the classification
and regression of later stages, there are no labels to learn from, only the structure of the data
itself.

Similar within, different between
-----------------------------------

A good clustering has a simple signature: **high similarity within** each cluster and **low similarity
between** clusters. Members of a group should resemble one another; members of different groups should
not. This dual goal — tight, well-separated clusters — is what every clustering algorithm chases, and
what quality measures like the silhouette score reward.

Distance as similarity
------------------------

To make "alike" computable, similarity is usually expressed as **distance**: two observations are
similar if they are **close** in the feature space, typically by **Euclidean** distance. This is why
clustering is sensitive to **scale** — a feature measured in large units can dominate the distance —
so features are commonly **standardised** first. Close points cluster together; distant ones fall into
different groups.

Why segment?
--------------

The payoff is **segmentation**: dividing customers, products or observations into meaningful groups
that can be understood and treated differently. A business might discover a cluster of high-value
frequent buyers and another of occasional bargain-hunters, then tailor its approach to each. The next
lesson turns this idea into concrete algorithms; the RFM lessons that follow apply it to real customer
data.

.. hint::

   **Related lessons:** :doc:`Clustering <27-clustering>`  ·  :doc:`Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`  ·  :doc:`Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>`  ·  :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/putting-similar-observations-into-clusters/ <https://insightful-data-lab.com/2026/01/14/putting-similar-observations-into-clusters/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
