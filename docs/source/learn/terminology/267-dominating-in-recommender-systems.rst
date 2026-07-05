:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-dominating-in-recommender-systems:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Dominating in Recommender Systems</b></div>`

===================================
Dominating in Recommender Systems
===================================

*When a few popular items crowd out the rest of recommendations.*

What it is
----------

A **multi-criteria** notion. One item **dominates** another (**Pareto dominance**) when it is **at least as
good on every criterion** and **strictly better on at least one**:

.. math::

   a \succ b \iff \big(\forall k:\ a_k \ge b_k\big) \ \wedge\ \big(\exists k:\ a_k > b_k\big).

If a hotel is cheaper, closer **and** higher-rated than another, it dominates it.

The skyline
-----------

The items **not dominated** by any other form the **skyline** (the **Pareto frontier**) — the only
candidates worth recommending under multiple objectives, since every dominated item is beaten outright by
something in the skyline.

Why it matters
--------------

Multi-criteria recommenders (price, distance, rating, recency) use dominance to **prune** clearly-inferior
options before ranking. Dominance also names a **failure mode** — when a few **popular** items dominate
everyone's lists, crowding out the long tail and harming exposure fairness.

----

**Mind map — connected ideas**

   :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>`

----

**More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Dominating in Recommender Systems <https://insightful-data-lab.com/2025/08/22/dominating-in-recommender-systems/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
