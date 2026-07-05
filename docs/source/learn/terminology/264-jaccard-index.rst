:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-jaccard-index:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Jaccard index</b></div>`

===============
Jaccard index
===============

*Intersection over union of two sets, a similarity measure.*

What it is
----------

The **Jaccard index** (Jaccard similarity coefficient) measures the overlap between two **sets** — the size
of their **intersection** over the size of their **union**. For items, the sets are typically the **users
who liked** each item, or their **tags / features**.

The formula
-----------

.. math::

   J(A, B) = \frac{|A \cap B|}{|A \cup B|},

ranging 0 to 1 — **0** for disjoint sets, **1** for identical ones; the complement :math:`1 - J` is the
**Jaccard distance**.

When to use it
--------------

It is the natural choice for **binary** (like / dislike, present / absent) data, where magnitudes don't
matter — only which elements are shared. Contrast with **cosine**, which works on real-valued vectors.

----

**Mind map — connected ideas**

   :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Catalog Coverage <268-catalog-coverage>`

----

**More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Jaccard index <https://insightful-data-lab.com/2025/08/22/jaccard-index/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
