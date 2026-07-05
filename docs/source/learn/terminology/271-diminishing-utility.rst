:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-diminishing-utility:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Diminishing Utility</b></div>`

=====================
Diminishing Utility
=====================

*The decreasing marginal value of additional similar or lower-ranked items.*

What it is
----------

The principle of **diminishing marginal utility** applied to recommendation: **each additional similar item
adds less value** than the one before. The third action movie in a row is not three times as useful as the
first — its **marginal** contribution shrinks.

Why it matters
--------------

This is the economic reason to **diversify**. Because utility from redundant items is **submodular**
(diminishing returns), a **varied** list delivers more total value than a list of near-duplicates — so
**marginal relevance** matters more than absolute relevance.

How it's used
-------------

Diversification methods like **Maximal Marginal Relevance (MMR)** and submodular utility-maximization
objectives encode diminishing utility directly — each pick is scored by its relevance **minus** its
redundancy with what's already chosen, so once a genre or topic is covered, further items of the same kind
are penalized.

----

**Mind map — connected ideas**

   :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>`

----

**More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Diminishing Utility <https://insightful-data-lab.com/2025/08/22/diminishing-utility/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
