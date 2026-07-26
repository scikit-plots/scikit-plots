:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-decision-trees:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Decision Trees</b></div>`

================
Decision Trees
================

*Models that split data on feature thresholds to reach predictions.*

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

A **decision tree** is a supervised-learning algorithm that splits data into branches by **feature
values**, forming a tree. Each **internal node** is a decision (a feature and a threshold) and each
**leaf** a prediction (a class label or a number) — think of it as a flowchart: ask questions, follow
branches, reach a prediction. It comes in two flavours: **classification trees** (discrete labels) and
**regression trees** (continuous values).

How it learns, and splitting
------------------------------

Training is recursive: start with all data at the **root**, evaluate candidate **splits** for each
feature, keep the split that best separates the data (minimises impurity), and repeat until a
**stopping rule** (max depth, minimum samples per leaf). Classification trees split by **Gini
impurity** or **entropy** (information gain); regression trees by **MSE reduction**. The Gini impurity
at a node is

.. math::

   G = 1 - \sum_{k} p_k^2,

where :math:`p_k` is the proportion of class :math:`k` at that node.

Strengths and weaknesses
--------------------------

Trees are **easy to interpret and visualise**, handle **mixed numeric and categorical** features,
capture **nonlinear boundaries and interactions**, and need no feature scaling. But a deep tree
**overfits**, is **unstable** (a small data change reshapes it), splits **greedily** (it can miss the
global optimum), and is **weaker alone** than an ensemble.

From one tree to many
-----------------------

**Pruning** cuts back branches to curb overfitting; **random forests** bag many trees; and
**gradient-boosted trees** (XGBoost, LightGBM, CatBoost) build trees sequentially to correct earlier
errors.

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.tree import DecisionTreeClassifier, export_text

   X, y = load_iris(return_X_y=True)
   tree = DecisionTreeClassifier(max_depth=3).fit(X, y)

   print(export_text(tree, feature_names=["sepal_length", "sepal_width",
                                          "petal_length", "petal_width"]))

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Post-hoc Explainability <339-post-hoc-explainability>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Uplift Random Forests <302-uplift-random-forests>` · :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Decision Trees <https://insightful-data-lab.com/2025/08/20/decision-trees/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
