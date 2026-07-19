.. _dpa-using-decision-trees-to-explain-clustering-results:

========================================================================
Using Decision Trees to Explain Clustering Results
========================================================================

**Stage 7 · 🌳 Decision Trees**  ·  Lesson 49 of 56  ·  *advanced*

:doc:`◀ Previous · Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>`   ·   :doc:`Next · Assessing the Quality of Prediction Models ▶ <50-assessing-the-quality-of-prediction-models>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Explaining a black box
------------------------

The last lesson **profiled** individual clusters; this one generalises the idea into one of machine
learning's most useful tools. Clustering algorithms — and many models besides — are **opaque**: k-means
or a neural network produces outputs, but not an **explanation** of its logic. A **surrogate model** is
a deliberately **simple, interpretable** model trained to **mimic** the opaque one, so that its
transparent logic can stand in for the black box's hidden one.

The surrogate idea
--------------------

The recipe is general. Take whatever the opaque method produced — here, the **cluster label** for each
observation — and treat those outputs as the **target**. Then fit an interpretable model, most
naturally a **decision tree**, to reproduce them from the features. The tree learns the **rules** that
best explain the clustering as a whole, and because it is a tree, those rules can be **read**: a
compact, global description of *how* the clustering carved up the data. This is a **global surrogate**
— it approximates the whole model's behaviour, not just one prediction — and it is **model-agnostic**,
needing no knowledge of how the clustering worked inside.

Fidelity
----------

The obvious question is: can you **trust** the surrogate? A tree that mimics the clustering only
loosely would give a misleading explanation. The measure of trust is **fidelity** — how **faithfully**
the surrogate reproduces the original's outputs, typically the surrogate's **accuracy** at predicting
the black box's labels. **High** fidelity means the tree's rules genuinely reflect the clustering;
**low** fidelity means the explanation is unreliable. There is a real **tension** here: a deeper tree
fits the clustering more faithfully but is harder to read, so surrogates trade **fidelity against
interpretability**, and a shallow, high-fidelity tree is the happy case.

A general pattern
-------------------

This surrogate trick is far bigger than clustering. The **same** move — fit an interpretable model to a
complex model's outputs, then check its fidelity — explains **black-box classifiers, regressors and
ensembles** too, and it complements the per-prediction attributions of **Shapley values** from Stage 5.
It closes the trees stage on a fitting note: the decision tree, prized for its transparency, becomes a
**lens** for seeing into models that have none. With the workflow's models built and understood, the
final stage asks the essential question — **how good are they?**

.. hint::

   **Related lessons:** :doc:`Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>`  ·  :doc:`Clustering <27-clustering>`  ·  :doc:`How Shapley Values Work <37-how-shapley-values-work>`  ·  :doc:`The CART Algorithm <45-the-cart-algorithm>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/using-decision-trees-to-explain-clustering-results/ <https://insightful-data-lab.com/2026/01/16/using-decision-trees-to-explain-clustering-results/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced
