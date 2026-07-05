:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-self-information-of-popularity:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Self-Information of Popularity</b></div>`

================================
Self-Information of Popularity
================================

*An information-theoretic weighting that rewards recommending less popular items.*

What it is
----------

A **novelty** measure borrowed from **information theory**. The **self-information** (surprisal) of
recommending an item is the **negative base-2 logarithm** of its **popularity** — the probability that a
random user has interacted with it. Rare events carry more information, so **rare items score high**.

The formula
-----------

.. math::

   \text{self-information}(i) = -\log_2\!\left(\frac{\text{count}(i)}{|U|}\right),

where :math:`\text{count}(i)` is the number of users who consumed item :math:`i` and :math:`|U|` is the
total number of users. A metric averages this over a top-:math:`N` list and across users.

What it captures
----------------

**Popular** items (high probability) have **low** self-information — they are unsurprising; **long-tail**
items have **high** self-information — they are novel. It quantifies how much a recommendation tells the
user something **new**.

----

**Mind map — connected ideas**

   :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>`

----

**More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`User Coverage <269-user-coverage>`

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Self-Information of Popularity <https://insightful-data-lab.com/2025/08/22/self-information-of-popularity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
