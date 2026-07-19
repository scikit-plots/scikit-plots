:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-class-weighting:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Class Weighting</b></div>`

=================
Class Weighting
=================

*Giving minority-class errors a larger weight in the loss so the model stops ignoring rare classes.*

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

**Class weighting** rebalances an imbalanced problem *without touching the data*.
Instead of duplicating or deleting rows, it changes the **loss function** so that
mistakes on rare classes cost more than mistakes on common ones. The dataset keeps
its original size; only the penalties change.

Why use it
----------

On imbalanced data a model can score well by simply always predicting the majority
class. Up-weighting the minority class forces the optimiser to take those examples
seriously. Compared with resampling, class weighting adds no synthetic points and
throws nothing away.

How it works
------------

Give each class :math:`c` a weight :math:`w_c` and scale every sample's loss by the
weight of its true class:

.. math::

   L_{\text{weighted}} = \sum_{i=1}^{N} w_{y_i}\,\ell\big(f(x_i),\, y_i\big),

where :math:`y_i` is the true label, :math:`\ell(\cdot)` is the per-sample loss
(for example cross-entropy) and :math:`w_{y_i}` is that class's weight. A standard
choice makes the weight inversely proportional to class frequency:

.. math::

   w_c = \frac{N}{K \cdot n_c},

with :math:`N` total samples, :math:`K` classes and :math:`n_c` samples in class
:math:`c` — so smaller classes receive larger weights.

Trade-offs
----------

Advantages:

- No information loss — every row is kept.
- No synthetic data (unlike SMOTE).
- One-line support in most libraries.

Disadvantages:

- Extreme weights (for very rare classes) can make training unstable.
- A tiny minority class is still hard to learn from, however it is weighted.

Examples
--------

scikit-learn — let the estimator set the weights automatically:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression

   # weights set to N / (K * n_c) per class
   model = LogisticRegression(class_weight="balanced")
   model.fit(X, y)
   # or pass them explicitly, e.g. class_weight={0: 1, 1: 10}

PyTorch — pass a weight tensor to the loss:

.. code-block:: python

   import torch
   import torch.nn as nn

   class_weights = torch.tensor([1.0, 10.0])   # minority class weighted higher
   criterion = nn.CrossEntropyLoss(weight=class_weights)

Weighting vs resampling
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Method
     - How it works
     - Trade-off
   * - Oversampling
     - Duplicate or synthesise minority samples
     - Improves recall, but risks overfitting
   * - Undersampling
     - Drop majority samples
     - Faster and balanced, but loses information
   * - Class weighting
     - Reweight the loss, data unchanged
     - No data loss, but unstable if weights are extreme

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Oversampling <004-oversampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`Subsampling <001-subsampling>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>`

----

.. hint::
   **More in Imbalanced Learning & Resampling**

   :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Subsampling <001-subsampling>` · :doc:`Upsampling <367-upsampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Class Weighting <https://insightful-data-lab.com/2025/08/30/class-weighting/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
