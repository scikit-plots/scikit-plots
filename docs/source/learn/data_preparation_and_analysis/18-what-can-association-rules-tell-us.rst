.. _dpa-what-can-association-rules-tell-us:

========================================================================
What Can Association Rules Tell Us?
========================================================================

**Stage 3 · 🛒 Market Basket & Association Rules**  ·  Lesson 18 of 56  ·  *intermediate*

:doc:`◀ Previous · Understanding Market Baskets and Ideal Customers <17-understanding-market-baskets-and-ideal-customers>`   ·   :doc:`Next · How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach ▶ <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

If this, then that
--------------------

An **association rule** is an "if-then" statement about items in a basket: **if** a customer buys some
set of items, **then** they are likely to buy another. Written
:math:`\{\text{bread}, \text{butter}\} \rightarrow \{\text{milk}\}`, it reads "baskets with bread and
butter tend also to contain milk". The left side is the **antecedent**, the right the **consequent**.

What a rule says
------------------

A rule captures a **regularity** in the data — a combination that shows up together more than you
might expect. On its own the arrow is just a candidate pattern; its **usefulness** depends on how often
it holds and how reliable it is, which the next lesson measures with **support**, **confidence** and
**lift**. For now, the point is the *shape* of the knowledge: compact, readable statements about what
accompanies what.

What they're good for
-----------------------

Rules turn into **decisions**. "Customers who buy X also buy Y" suggests **recommendations** ("you
might also like…"), **bundles**, **store layout**, and **targeted promotions**. This is the engine
behind **cross-selling** (Stage 3's closing lesson): using a known purchase to suggest a complementary
one, lifting basket size and revenue.

A familiar caution
--------------------

The caution from the start of this stage returns: a rule reports **association, not causation**. That
bread and butter travel with milk does not mean one **causes** the other — both may simply reflect a
weekly grocery run. Rules are superb at spotting **what** goes together and useful for acting on it,
but they do not, by themselves, explain **why**.

.. hint::

   **Related lessons:** :doc:`Understanding Market Baskets and Ideal Customers <17-understanding-market-baskets-and-ideal-customers>`  ·  :doc:`How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>`  ·  :doc:`Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm>`  ·  :doc:`Cross-Selling <22-cross-selling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/what-can-association-rules-tell-us/ <https://insightful-data-lab.com/2026/01/14/what-can-association-rules-tell-us/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
