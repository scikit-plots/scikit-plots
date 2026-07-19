:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-mean-average-precision-map:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>Mean Average Precision (MAP)</b></div>`

==============================
Mean Average Precision (MAP)
==============================

*The mean of average-precision scores across queries.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Mean Average Precision** is the **mean**, across all users, of each user's **Average Precision (AP)**. AP
averages the **precision** measured at **every** rank where a relevant item appears — the area under that
user's **precision–recall** curve:

.. math::

   \text{MAP} = \frac{1}{|U|}\sum_{u \in U} \text{AP}_u.

It rolls per-user ranking quality into one number.

What AP rewards
---------------

Because AP recomputes precision at each **relevant** position, it **emphasizes** getting relevant items
**early** — a relevant item at rank 1 lifts every later precision term, while one at rank 10 lifts few. So MAP
is strongly **order-sensitive**, rewarding front-loaded relevance.

How it compares
---------------

MAP works with **binary** relevance (relevant or not), where **NDCG** handles **graded** relevance; MAP
summarizes the **whole** precision–recall trade-off, where **Hit Rate** only checks for any hit. Reported at a
cutoff (MAP@K), it is a standard top-N metric for search and recommendation.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Mean Average Precision (MAP) <https://insightful-data-lab.com/2025/08/19/mean-average-precision-map/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
