.. _dpa-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach:

===========================================================================================
How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach
===========================================================================================

**Stage 3 · 🛒 Market Basket & Association Rules**  ·  Lesson 19 of 56  ·  *intermediate*

:doc:`◀ Previous · What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>`   ·   :doc:`Next · Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm> ▶`


Three measures of a rule
--------------------------

A rule like :math:`X \rightarrow Y` is only worth keeping if it is both **common** and **reliable**.
Three numbers, all computed from how often itemsets appear in the transactions, make "worth keeping"
precise: **support**, **confidence** and **lift**.

Support, confidence, lift
---------------------------

Each captures a different facet:

* **Support** — how **common** the combination is, the fraction of all transactions containing the whole itemset:

  .. math::

     \operatorname{support}(X \rightarrow Y) =
       \frac{\text{transactions containing } X \cup Y}{\text{all transactions}}.

* **Confidence** — how **reliable** the rule is, the share of :math:`X`-baskets that also contain :math:`Y`, estimating :math:`P(Y \mid X)`:

  .. math::

     \operatorname{confidence}(X \rightarrow Y) =
       \frac{\operatorname{support}(X \cup Y)}{\operatorname{support}(X)}.

* **Lift** — how **surprising** it is, confidence compared with :math:`Y`'s baseline frequency:

  .. math::

     \operatorname{lift}(X \rightarrow Y) =
       \frac{\operatorname{confidence}(X \rightarrow Y)}{\operatorname{support}(Y)}.

  Lift :math:`> 1` means :math:`X` and :math:`Y` occur together **more** than chance (positive
  association); :math:`= 1` means independent; :math:`< 1` means they repel.

The scale problem
-------------------

Why not simply score every possible rule? Because there are **too many**. With :math:`d` distinct
items there are :math:`2^d` possible itemsets and more possible rules still — for a supermarket with
thousands of products, an astronomically large number. Counting the support of every candidate by
brute force is hopeless.

The Apriori idea
------------------

The escape is one simple observation, the **Apriori principle** (also called **downward closure**):
*if an itemset is frequent, all of its subsets must be frequent too* — and, turned around, *if an
itemset is infrequent, every superset of it is infrequent as well*. That lets an algorithm **prune**
vast regions of the search: once the pair **{milk, caviar}** is rare, nothing containing it can be
common, so none of its extensions need be checked. The next lesson builds an algorithm around exactly
this.

.. seealso::

   **Related lessons:** :doc:`What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>`  ·  :doc:`Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm>`  ·  :doc:`association_rules: Generating Association Rules from Frequent Itemsets (mlxtend) <21-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend>`  ·  :doc:`Cross-Selling <22-cross-selling>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/14/how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach/ <https://insightful-data-lab.com/2026/01/14/how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach/>`__

.. tags:: purpose: reference, topic: data preparation, level: intermediate
