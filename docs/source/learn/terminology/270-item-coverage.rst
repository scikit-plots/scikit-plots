:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-item-coverage:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Item Coverage</b></div>`

===============
Item Coverage
===============

*The share of items that the system is able to recommend.*

What it is
----------

**Item coverage** is the fraction of individual items the recommender is **able to recommend at all** —
for which it can produce a prediction or place in a list. It asks *how many products can the system
reach?*, item by item:

.. math::

   \text{Item Coverage} = \frac{|\text{items the system can recommend}|}{|\text{total items}|}.

What limits it
--------------

Items with **no or few interactions** — new or niche products (the **cold-start** problem, data
**sparsity**) — may be impossible to recommend, dragging item coverage down. **Content-based** signals or
**hybrid** models raise it by letting the system reason about **unseen** items from their features.

Recommendability
----------------

Item coverage is about **recommendability** (can this item ever surface?), whereas **catalog coverage** is
about how much of the catalog **actually** surfaces in practice.

----

**Mind map — connected ideas**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`User Coverage <269-user-coverage>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Diminishing Utility <271-diminishing-utility>`

----

**More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Item Coverage <https://insightful-data-lab.com/2025/08/22/item-coverage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
