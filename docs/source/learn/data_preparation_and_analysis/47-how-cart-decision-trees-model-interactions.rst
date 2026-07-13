.. _dpa-how-cart-decision-trees-model-interactions:

========================================================================
How CART Decision Trees Model Interactions
========================================================================

**Stage 7 · 🌳 Decision Trees**  ·  Lesson 47 of 56  ·  *advanced*

:doc:`◀ Previous · Decision Trees as Piecewise Models and Their Predictive Structure <46-decision-trees-as-piecewise-models-and-their-predictive-structure>`   ·   :doc:`Next · Cluster Profiling Using Decision Trees ▶ <48-cluster-profiling-using-decision-trees>`   ·   :doc:`↑ Section <index>`


What is an interaction?
-------------------------

An **interaction** occurs when the effect of one feature on the outcome **depends on the value of
another**. Airport surcharges might make **distance** matter more for **airport** trips than for
others; a drug might help one age group and harm another. In each case you cannot describe the effect
of one variable **without knowing** the other — the two **interact**.

The regression chore
----------------------

Ordinary linear regression **cannot see** interactions on its own. Its form is strictly **additive** —
each feature contributes its coefficient times its value, independently — so the effect of one feature
is the **same** regardless of the others. To model an interaction you must **manually add** a product
term (:math:`x_1 \times x_2`), and you have to **know in advance** which interactions to include. Miss
one, and the model is blind to it.

Trees get them free
---------------------

Decision trees capture interactions **automatically**, as a byproduct of their structure. Because
splits are **nested**, a split on one feature can be **followed by different splits** on another in
different branches — so the effect of the second feature genuinely **differs** depending on the first.
A tree might split on trip type, then split on distance **only** in the airport branch: exactly an
interaction between type and distance, discovered without anyone specifying it. Each split is an effect
**conditional** on all the splits above it.

Depth and order
-----------------

This is why tree **depth** matters for expressiveness. A one-split (depth-1) tree captures only a
single feature's main effect; each additional level lets the tree condition on **one more** feature, so
deep trees can represent **high-order** interactions among many variables. The first split naturally
falls on the feature with the strongest overall (main) effect, with interactions emerging **below** it.
This effortless interaction modelling — together with the piecewise structure of the last lesson — is
what lets a single tree describe complex, realistic patterns, and it powers the cluster-explanation
uses that close this stage.

.. hint::

   **Related lessons:** :doc:`Decision Trees as Piecewise Models and Their Predictive Structure <46-decision-trees-as-piecewise-models-and-their-predictive-structure>`  ·  :doc:`The CART Algorithm <45-the-cart-algorithm>`  ·  :doc:`Multiple Linear Regression <32-multiple-linear-regression>`  ·  :doc:`Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/how-cart-decision-trees-model-interactions/ <https://insightful-data-lab.com/2026/01/16/how-cart-decision-trees-model-interactions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced
