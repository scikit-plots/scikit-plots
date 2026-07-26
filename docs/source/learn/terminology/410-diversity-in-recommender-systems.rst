:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-diversity-in-recommender-systems:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Diversity (in Recommender Systems)</b></div>`

====================================
Diversity (in Recommender Systems)
====================================

*How varied the items within a recommendation list are.*

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

**Diversity** measures how **varied** the items **within** a single recommendation list are — the opposite of
ten near-identical suggestions. A diverse list spans a user's **multiple** interests rather than hammering
one.

How it's measured
-----------------

The standard gauge is **intra-list dissimilarity** — the average **pairwise** distance between recommended
items (often 1 − **cosine similarity** of their features), captured by **Intra-List Diversity**. At the
catalog level, **Gini** or **entropy** across all recommendations measures aggregate diversity.

Why it matters
--------------

Diversity improves the **experience** — it hedges against a wrong guess about intent and keeps lists
interesting — but there's an **accuracy-diversity** trade-off, since the most "accurate" items are often
**similar**. Good systems tune diversity **without** dumping relevance.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Coverage <411-coverage>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Diversity (in Recommender Systems) <https://insightful-data-lab.com/2025/08/19/diversity-in-recommender-systems/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
