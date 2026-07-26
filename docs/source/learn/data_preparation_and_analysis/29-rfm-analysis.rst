.. _dpa-rfm-analysis:

========================================================================
RFM Analysis
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 29 of 56  ·  *intermediate*

:doc:`◀ Previous · Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`   ·   :doc:`Next · Creating Segments of Observations for Business Reasons (RFM) ▶ <30-creating-segments-of-observations-for-business-reasons-rfm>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

From raw values to scores
---------------------------

RFM analysis turns the three raw numbers into a **comparable score**. Raw values are awkward to
compare directly — is 30 days "recent"? is $500 "high"? — because the answer depends on your business.
The fix is to score each customer **relative to the others**, so the numbers become ranks on a common
1–5 scale.

Quintile scoring
------------------

The standard method is **quintiles**. For each dimension, sort all customers and split them into **five
equal groups**, then assign a score from **1 to 5**. The top 20% on **Frequency** and **Monetary**
score 5; for **Recency** the scale is **inverted** — the most recent 20% (fewest days) score 5.
Because quintiles are defined by your **own** data distribution, the scores adapt automatically to any
business, with each score level holding about 20% of customers.

The RFM code
--------------

The three scores are combined into a single **RFM code** — usually just concatenated, like
:math:`R{=}5,\ F{=}4,\ M{=}5 \rightarrow` "**545**". This gives :math:`5 \times 5 \times 5 = 125`
possible codes, from **555** (the best — recent, frequent, high-spending) down to **111** (the worst).
Sorting customers by their code, or by a **weighted** combination if one dimension matters more, ranks
the whole base at a glance. Because behaviour drifts, the scores are **recomputed** regularly.

In Python
-----------

The computation is a short ``pandas`` routine: group the orders **by customer**, then aggregate to the
three values — recency from the latest ``InvoiceDate``, frequency from a count of orders, monetary
from the sum of spend. ``pandas.qcut`` cuts each into quintile scores in one call. The result is a tidy
table, one row per customer with an R, F and M score — ready to be grouped into the segments of the
next lesson.

.. hint::

   **Related lessons:** :doc:`Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`  ·  :doc:`Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`  ·  :doc:`Stratified Random Sampling <23-stratified-random-sampling>`  ·  :doc:`Putting Similar Observations into Clusters <26-putting-similar-observations-into-clusters>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/rfm-analysis/ <https://insightful-data-lab.com/2026/01/14/rfm-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
