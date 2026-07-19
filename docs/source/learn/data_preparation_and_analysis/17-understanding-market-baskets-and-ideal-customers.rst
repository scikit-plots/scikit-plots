.. _dpa-understanding-market-baskets-and-ideal-customers:

========================================================================
Understanding Market Baskets and Ideal Customers
========================================================================

**Stage 3 · 🛒 Market Basket & Association Rules**  ·  Lesson 17 of 56  ·  *intermediate*

:doc:`◀ Previous · Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova>`   ·   :doc:`Next · What Can Association Rules Tell Us? ▶ <18-what-can-association-rules-tell-us>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What's in the basket
----------------------

A **market basket** is just the set of items a customer buys **together** in one transaction — the
contents of a single shopping cart or receipt. **Market basket analysis** is the study of these
baskets across many customers, looking for the products that tend to **appear together**. It is one of
the oldest and most intuitive forms of data mining, born in retail.

Why baskets matter
--------------------

Knowing what goes with what is directly **actionable**. If bread and butter sell together, a shop can
place them nearby, bundle them in a promotion, or recommend one when the other is added to a cart. The
patterns hidden in baskets drive **product placement, recommendations, promotions and cross-selling**
— turning a pile of receipts into merchandising decisions.

The ideal customer
--------------------

Basket analysis also sharpens the idea of an **"ideal customer"**. By seeing which combinations of
purchases mark high-value or loyal shoppers, a business can recognise and target customers who look
like its best ones. What people **buy together** becomes a signature of **who they are** and what they
might want next.

From baskets to rules
-----------------------

To act on baskets at scale, the co-occurrence patterns are written as **association rules** — precise
"if this, then that" statements — and mined automatically from transaction data. The next lessons make
that idea exact: what a rule is, how its strength is measured, and how the **Apriori** algorithm finds
the good ones among astronomically many possibilities.

.. hint::

   **Related lessons:** :doc:`What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>`  ·  :doc:`How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>`  ·  :doc:`Cross-Selling <22-cross-selling>`  ·  :doc:`Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/understanding-market-baskets-and-ideal-customers/ <https://insightful-data-lab.com/2026/01/14/understanding-market-baskets-and-ideal-customers/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
