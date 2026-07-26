.. _dpa-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend:

==================================================================================
association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)
==================================================================================

**Stage 3 · 🛒 Market Basket & Association Rules**  ·  Lesson 21 of 56  ·  *intermediate*

:doc:`◀ Previous · Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm>`   ·   :doc:`Next · Cross-Selling ▶ <22-cross-selling>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

From itemsets to rules in code
--------------------------------

The Python library **mlxtend** implements the whole pipeline of the last two lessons in a few lines:
encode the transactions, mine frequent itemsets with **apriori**, then turn them into ranked rules with
**association_rules**. It is the standard tool for market-basket analysis in the scientific-Python
stack.

One-hot transactions
----------------------

The algorithms expect a **one-hot encoded** table — one row per transaction, one boolean column per
item, ``True`` where the item is in the basket. mlxtend's **TransactionEncoder** builds it from raw
lists of items:

.. code-block:: python

   from mlxtend.preprocessing import TransactionEncoder
   import pandas as pd

   transactions = [["bread", "milk", "eggs"], ["bread", "butter"], ["milk", "butter"]]
   te = TransactionEncoder()
   df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

Two functions
---------------

With the table ready, two calls do the work:

.. code-block:: python

   from mlxtend.frequent_patterns import apriori, association_rules

   items = apriori(df, min_support=0.5, use_colnames=True)
   rules = association_rules(items, metric="confidence", min_threshold=0.6)

**apriori** returns the frequent itemsets and their support; **association_rules** expands them into
rules and filters by the metric you choose (``"confidence"``, ``"lift"``, and others).

Reading the output
--------------------

The result is a tidy ``DataFrame``: each row a rule, with columns for **antecedents**, **consequents**,
**support**, **confidence** and **lift** (plus leverage and conviction). Sorting by **lift** surfaces
the most surprising, actionable pairings — the rules a shop would actually act on. The final lesson of
this stage puts them to use: **cross-selling**.

.. hint::

   **Related lessons:** :doc:`Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm>`  ·  :doc:`How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>`  ·  :doc:`What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>`  ·  :doc:`Cross-Selling <22-cross-selling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/association_rules-generating-association-rules-from-frequent-itemsets-mlxtend/ <https://insightful-data-lab.com/2026/01/14/association_rules-generating-association-rules-from-frequent-itemsets-mlxtend/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
