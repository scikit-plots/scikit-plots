:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-ranking-algorithms:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>Ranking Algorithms</b></div>`

====================
Ranking Algorithms
====================

*Methods that order items by relevance or predicted value.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

A **ranking algorithm** **orders a set of items** — documents, products, ads, songs — so
the **most relevant or useful appear at the top**. Given a **query** (a search term, a user
profile), it scores items by predicted relevance, utility, or likelihood of interaction
(click, purchase, watch) and sorts by that score. Search engines, recommenders, ad systems
and social feeds are all ranking problems, and better ranking translates directly into user
satisfaction and revenue.

Classical IR
------------

The oldest rankers are lexical. **TF-IDF** weights a term by how often it appears in a
document against how rare it is across the corpus; **BM25** refines this with
**term-frequency saturation** and **document-length** normalisation, and remains a strong
search baseline.

Learning to rank
----------------

**Learning to rank (LTR)** trains a model from labelled relevance data, in three paradigms
by what the loss looks at: **pointwise** scores each item alone (regression or
classification — predict a click probability with logistic regression or gradient-boosted
trees); **pairwise** learns from comparisons ("is A better than B?", e.g. RankNet); and
**listwise** optimises the whole ordering at once (LambdaMART, ListNet), often against a
ranking metric like **NDCG**.

Neural rankers
--------------

The newest models use **embeddings** (Word2Vec, BERT, Transformers) to capture *semantic*
match rather than exact words — **DSSM** projects queries and documents into a shared space,
and **BERT-based rankers** (monoBERT, ColBERT) read query and document in context, sharply
improving relevance at higher compute cost.

How ranking is judged
---------------------

Ranking has its own metrics: **NDCG** (rewards relevant items near the top), **MAP**
(average precision across queries), **MRR** (rank of the first relevant item), **CTR**, and
**precision@k / recall@k**. The hard parts are **personalisation** (every user's "best"
differs), **position bias** (higher slots get clicks regardless of quality), **scale**
(rank billions of items fast), and **fairness** to minority items.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`Embedding <173-embedding>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Ranking Algorithms <https://insightful-data-lab.com/2025/08/24/ranking-algorithms/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
