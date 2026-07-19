:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-dcg-discounted-cumulative-gain:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>DCG (Discounted Cumulative Gain)</b></div>`

==================================
DCG (Discounted Cumulative Gain)
==================================

*A ranking metric that rewards relevant items more when ranked higher.*

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

**Discounted Cumulative Gain** scores a **ranked list** by summing each item's **graded relevance**,
discounted by how **far down** it sits — so a relevant result near the top counts far more than the same
result buried lower:

.. math::

   \text{DCG}_p = \sum_{i=1}^{p} \frac{\mathrm{rel}_i}{\log_2(i+1)}.

The **logarithmic** discount encodes that users examine top results most.

Why it beats precision
----------------------

Unlike binary **precision / recall**, DCG uses **multi-level** relevance (say 0–3) **and** position, capturing
both *how relevant* each item is and *where* it was ranked — exactly what matters for **search** and
**recommendation**.

Normalizing it
--------------

Raw DCG isn't comparable across queries with different numbers of relevant items, so
:math:`\text{NDCG} = \text{DCG} / \text{IDCG}` divides by the **ideal** DCG (the best possible ordering),
giving a **0-to-1** score where **1** is a perfect ranking. It is the standard offline ranking metric.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Kaggle <273-kaggle>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `DCG (Discounted Cumulative Gain) <https://insightful-data-lab.com/2025/08/22/dcg-discounted-cumulative-gain/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
