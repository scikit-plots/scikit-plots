:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-support-vector-machines-svms:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Support Vector Machines (SVMs)</b></div>`

================================
Support Vector Machines (SVMs)
================================

*Classifiers that find the maximum-margin boundary, optionally via kernels.*

What it is
----------

A **support vector machine** is a **supervised max-margin** algorithm for classification (and, as SVR,
regression). It finds the **optimal separating hyperplane** — the decision boundary that **maximizes the
margin**, the distance to the nearest points of each class. Developed from the work of Vapnik and
Chervonenkis, with the soft-margin form due to Cortes & Vapnik (1995).

.. math::

   \mathbf{w}^\top \mathbf{x} + b = 0 \quad\text{(the separating hyperplane)}

Margin and support vectors
--------------------------

The **support vectors** are the training points closest to the boundary — and they **alone** define it
(remove any other point and nothing changes). A **hard margin** separates the classes perfectly; a
**soft margin** tolerates some violations through **slack variables** :math:`\xi_i`, with the penalty
**C** trading margin width against misclassification (large ``C`` → stricter, narrower margin; small
``C`` → wider, more tolerant). The objective minimizes

.. math::

   \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i,

where the per-point cost is the **hinge loss** :math:`\max(0,\, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))`.

The kernel trick
----------------

When data is not linearly separable, a **kernel** maps it into a higher-dimensional space where it is —
**without ever computing the coordinates**, using only pairwise dot products. Common kernels are
**linear**, **polynomial**, **RBF** (the most popular) and **sigmoid**; the choice trades accuracy
against complexity and compute.

When to use it
--------------

SVMs are strong on **high-dimensional** data (text, images, bioinformatics), **resilient to noise**, and
guard against overfitting, but are **expensive on very large datasets** and sensitive to kernel choice.

.. code-block:: python

   from sklearn.svm import SVC

   clf = SVC(kernel="rbf", C=1.0)   # RBF kernel, soft margin
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Discriminatory Power <185-discriminatory-power>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Support Vector Machines (SVMs) <https://insightful-data-lab.com/2025/08/21/support-vector-machines-svms/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
