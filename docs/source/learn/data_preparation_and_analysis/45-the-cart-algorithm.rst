.. _dpa-the-cart-algorithm:

========================================================================
The CART Algorithm
========================================================================

**Stage 7 · 🌳 Decision Trees**  ·  Lesson 45 of 56  ·  *advanced*

:doc:`◀ Previous · Motivation of Decision Trees: An Incremental Model of Decision-Making <44-motivation-of-decision-trees-an-incremental-model-of-decision-making>`   ·   :doc:`Next · Decision Trees as Piecewise Models and Their Predictive Structure ▶ <46-decision-trees-as-piecewise-models-and-their-predictive-structure>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Growing a tree from data
--------------------------

How is a tree actually built? The standard method is **CART** — **Classification And Regression
Trees** — introduced by Breiman and colleagues in **1984**. Given labelled data, CART grows the tree
**top-down** and **greedily**: starting from all the data at the root, it repeatedly finds the **best
single split** and divides the node, then recurses on each resulting piece.

Recursive binary splits
-------------------------

Every CART split is **binary** — a node divides into exactly **two** children. For a numeric feature
the split is a threshold ("distance ≤ 3 miles?"); for a categorical one, a grouping of categories. At
each node the algorithm searches **all features and all candidate split points** and picks the single
split that best **separates** the outcomes, then applies the same procedure to each child. This
**recursion** continues down every branch, growing the tree one split at a time.

Measuring impurity
--------------------

"Best" is defined by **impurity** — a measure of how **mixed** the outcomes are within a node. A node
of all-retained students is **pure** (impurity 0); a fifty-fifty mix is maximally impure. CART chooses
the split that most **reduces** impurity across the two children. For **classification** the usual
measure is the **Gini impurity** (or entropy); for **regression**, the node's **variance** (mean
squared error). Each split is the one that makes the children as **homogeneous** as possible, echoing
the "similar within" goal of clustering.

Knowing when to stop
----------------------

Left unchecked, CART will split until every leaf is pure — a tree that **memorises** the training data
and **overfits** badly, the danger from the partitioning lesson. Two remedies rein it in. **Stopping
rules** halt growth early (a maximum **depth**, or a minimum number of samples per leaf). More
principled is **pruning**: grow a large tree, then **cut back** the branches that add little predictive
value, using **cost-complexity pruning** to trade size against accuracy. In scikit-learn,
``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` implement CART with both controls. The next
lessons read what a fitted tree **means**.

.. hint::

   **Related lessons:** :doc:`Motivation of Decision Trees: An Incremental Model of Decision-Making <44-motivation-of-decision-trees-an-incremental-model-of-decision-making>`  ·  :doc:`Decision Trees as Piecewise Models and Their Predictive Structure <46-decision-trees-as-piecewise-models-and-their-predictive-structure>`  ·  :doc:`How CART Decision Trees Model Interactions <47-how-cart-decision-trees-model-interactions>`  ·  :doc:`Clustering <27-clustering>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/the-cart-algorithm/ <https://insightful-data-lab.com/2026/01/16/the-cart-algorithm/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced
