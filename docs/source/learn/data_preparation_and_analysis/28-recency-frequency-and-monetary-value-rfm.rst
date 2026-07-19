.. _dpa-recency-frequency-and-monetary-value-rfm:

========================================================================
Recency, Frequency, and Monetary Value (RFM)
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 28 of 56  ·  *intermediate*

:doc:`◀ Previous · Clustering <27-clustering>`   ·   :doc:`Next · RFM Analysis ▶ <29-rfm-analysis>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Three questions about a customer
----------------------------------

How valuable is a customer? **RFM** answers with three simple questions, each read straight from
transaction history: **how recently** did they buy, **how often** do they buy, and **how much** do
they spend? These three numbers — **Recency, Frequency and Monetary value** — summarise a customer's
behaviour compactly enough to rank an entire database.

The three dimensions
----------------------

Each dimension is one measurement per customer:

* **Recency** — days since their **last** purchase. Fewer days is better: recent buyers are far likelier to buy again.
* **Frequency** — the **number** of purchases in a chosen window. More is better: repeat buying signals habit and loyalty.
* **Monetary** — the **total spend** over that window. More is better: it captures the customer's economic value.

Why all three
---------------

No single dimension tells the whole story. A **big spender** who has not bought in two years is a
**churn risk**, not a star; a **frequent** buyer with tiny orders is loyal but low-margin. Combined,
the three give a **holistic** view that any one alone would distort. (Of the three, **recency** tends
to predict future behaviour best, and monetary least.)

Simple and proven
-------------------

RFM's great virtue is **simplicity**. It needs only data every business already has — an order history
— and no elaborate modelling. The technique dates back to **direct-mail** marketing in the last
century, and it endures because it works: it reliably surfaces the roughly 20% of customers who drive
most of the revenue. The next lesson turns these three raw numbers into scores.

.. hint::

   **Related lessons:** :doc:`RFM Analysis <29-rfm-analysis>`  ·  :doc:`Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`  ·  :doc:`Cross-Selling <22-cross-selling>`  ·  :doc:`Clustering <27-clustering>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/recency-frequency-and-monetary-value-rfm/ <https://insightful-data-lab.com/2026/01/14/recency-frequency-and-monetary-value-rfm/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
