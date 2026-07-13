.. _dpa-creating-segments-of-observations-for-business-reasons-rfm:

========================================================================
Creating Segments of Observations for Business Reasons (RFM)
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 30 of 56  ·  *intermediate*

:doc:`◀ Previous · RFM Analysis <29-rfm-analysis>`   ·   :doc:`Next · Least Squares Regression ▶ <31-least-squares-regression>`   ·   :doc:`↑ Section <index>`


From codes to segments
------------------------

125 RFM codes are too many to act on. The final step is to group them into a handful of **named
segments** — typically **six to ten** — each describing a recognisable kind of customer and, crucially,
each calling for a **different business response**. This is where scores become **decisions**.

A common taxonomy
-------------------

A widely used starting taxonomy names segments by their RFM profile:

* **Champions** — high on all three (555, 554): the best customers.
* **Loyal Customers** — buy consistently and often.
* **Potential Loyalists** — recent buyers with growing frequency.
* **New Customers** — recent, but few purchases so far.
* **At Risk** — once frequent and high-spending, but **lapsing** (low recency).
* **Can't Lose Them** — high past value, gone quiet.
* **Hibernating / Lost** — low on everything, long inactive.

Each segment, an action
-------------------------

The point of naming segments is the **"so what"**. **Champions** get rewards, early access and
referral asks — **not** blanket discounts that erode margin. **At Risk** and **Can't Lose Them** get
**win-back** campaigns and personal outreach, ideally *before* they fully churn. **Hibernating / Lost**
get a re-permission push or are **suppressed** to save budget. One message for a new buyer and a
ten-year loyalist would waste both; segmentation lets each be treated for **who they are**.

Segments as clusters
----------------------

Notice this is the **clustering** idea from earlier in the stage, made concrete: customers are grouped
by similarity — here, similarity in RFM space — so that each group can be understood and served
differently. Whether the groups come from a rule-based RFM taxonomy or from an algorithm like k-means,
the goal is the one this stage began with: turn a mass of observations into **meaningful segments** a
business can act on. With customers understood, the course turns next to **predicting** outcomes —
starting with regression.

.. hint::

   **Related lessons:** :doc:`Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`  ·  :doc:`RFM Analysis <29-rfm-analysis>`  ·  :doc:`Clustering <27-clustering>`  ·  :doc:`Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/creating-segments-of-observations-for-business-reasons-rfm/ <https://insightful-data-lab.com/2026/01/14/creating-segments-of-observations-for-business-reasons-rfm/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
