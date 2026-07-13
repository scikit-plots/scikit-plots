:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-ndcg-normalized-discounted-cumulative-gain:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>NDCG (Normalized Discounted Cumulative Gain)</b></div>`

==============================================
NDCG (Normalized Discounted Cumulative Gain)
==============================================

*DCG normalised by the ideal ordering, bounded between 0 and 1.*

What it is
----------

**Normalized Discounted Cumulative Gain** measures **ranking quality** using both **graded relevance** and
**position**. It sums each item's relevance with a **logarithmic** discount for lower ranks (DCG), then
divides by the **ideal** ordering's score (IDCG):

.. math::

   \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K} \in [0, 1].

A perfect ranking scores **1**.

Why it's powerful
-----------------

Unlike Hit Rate, NDCG rewards putting **highly** relevant items **near the top** and handles **multi-level**
relevance (a 5-star match beats a 3-star one). Normalizing by IDCG makes it **comparable** across users with
different numbers of relevant items — the reason it is the **default** offline ranking metric for
recommenders and search.

In recsys
---------

Computed **per user** then **averaged**, NDCG@K captures the personalized-ordering quality that drives
engagement. It is the recommender-system application of the same **DCG** used in information retrieval, so IR
and recsys share this yardstick.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `NDCG (Normalized Discounted Cumulative Gain) <https://insightful-data-lab.com/2025/08/19/ndcg-normalized-discounted-cumulative-gain/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
