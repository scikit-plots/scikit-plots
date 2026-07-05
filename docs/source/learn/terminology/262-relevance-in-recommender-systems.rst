:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-relevance-in-recommender-systems:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Relevance in Recommender Systems</b></div>`

==================================
Relevance in Recommender Systems
==================================

*How well a recommended item matches a user's interests.*

What it is
----------

**Relevance** is whether a recommended item actually **matches the user's tastes and needs** — an item they
would find useful and want to engage with. It is the property that **accuracy** metrics (precision, recall,
NDCG, MAP) are built to measure.

The traditional goal
--------------------

Recommend **as many relevant items as possible**, maximizing accuracy. For a long time this was the sole
objective of recommender systems.

Not enough alone
----------------

A perfectly relevant list can still be **boring** — ten near-identical hits the user already knows. So
relevance is balanced against **novelty**, **diversity** and **coverage**, and modern novelty / diversity
metrics are made **relevance-aware** (rewarding items that are novel **and** relevant) so a system is not
credited for surfacing surprising-but-useless items. The aim is **relevant *and* diverse**.

----

**Mind map — connected ideas**

   :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>`

----

**More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Relevance in Recommender Systems <https://insightful-data-lab.com/2025/08/22/relevance-in-recommender-systems/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
