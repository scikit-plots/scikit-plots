:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-catalog-coverage:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Catalog Coverage</b></div>`

==================
Catalog Coverage
==================

*The share of the item catalog that ever gets recommended.*

What it is
----------

**Catalog coverage** is the proportion of the entire item catalog that a recommender actually **surfaces**
to users — the share of products that ever appear in someone's recommendations. It measures the
**breadth** of the system's reach, not the quality of any one list.

The formula
-----------

.. math::

   \text{Catalog Coverage} = \frac{|\text{distinct items recommended}|}{|\text{total items in catalog}|},

computed as an **aggregate** over all users and a time window (often expressed as a percentage).

Why it matters
--------------

**Low** coverage signals **popularity bias** — the system funnels everyone toward the same few hits,
starving the **long tail** and neglecting niche tastes; **high** coverage means diverse, inclusive
recommendations. Because recommending popular items is often **accurate but narrow**, coverage is reported
**alongside** accuracy — a good system scores well on both.

----

**Mind map — connected ideas**

   :doc:`Item Coverage <270-item-coverage>` · :doc:`User Coverage <269-user-coverage>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>`

----

**More in Recommender Systems**

   :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Catalog Coverage <https://insightful-data-lab.com/2025/08/22/catalog-coverage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
