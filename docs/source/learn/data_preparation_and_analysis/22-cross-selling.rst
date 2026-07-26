.. _dpa-cross-selling:

========================================================================
Cross-Selling
========================================================================

**Stage 3 · 🛒 Market Basket & Association Rules**  ·  Lesson 22 of 56  ·  *intermediate*

:doc:`◀ Previous · association_rules: Generating Association Rules from Frequent Itemsets (mlxtend) <21-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend>`   ·   :doc:`Next · Stratified Random Sampling ▶ <23-stratified-random-sampling>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Selling the complement
------------------------

**Cross-selling** is the practice of suggesting **complementary** products to a customer based on what
they are already buying — a bag and a lens for someone buying a camera, buns for someone buying hot
dogs. It is the direct **payoff** of the association rules this stage has built: the rules say what
goes with what, and cross-selling acts on it.

Cross-sell vs up-sell
-----------------------

It is worth distinguishing two moves. **Cross-selling** offers a **different, complementary** item
("customers who bought this also bought…"). **Up-selling** offers a **better version of the same**
item — a premium model or a larger size. Cross-selling widens the basket; up-selling deepens a single
choice. Association rules speak most directly to the former.

Rules in action
-----------------

The mechanics are exactly the mined rules. A rule :math:`\{\text{camera}\} \rightarrow \{\text{tripod}\}`
with high **confidence** and **lift** becomes a recommendation shown at checkout or on the product
page. Ranking candidate rules by lift surfaces the pairings that are **surprisingly** common — the
ones most likely to be genuine complements rather than two independently popular items.

Why it pays
-------------

The business case is simple: a relevant complement raises the **average order value** and improves the
customer's experience by anticipating a real need. Done well, cross-selling turns a single purchase
into a larger, more useful basket — which is why recommendation engines built on association rules are
everywhere in retail. It also sets up the next stage's question: not just *what* customers buy
together, but *which customers* to treat differently.

.. hint::

   **Related lessons:** :doc:`What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>`  ·  :doc:`How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>`  ·  :doc:`Understanding Market Baskets and Ideal Customers <17-understanding-market-baskets-and-ideal-customers>`  ·  :doc:`Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/cross-selling-2/ <https://insightful-data-lab.com/2026/01/14/cross-selling-2/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
