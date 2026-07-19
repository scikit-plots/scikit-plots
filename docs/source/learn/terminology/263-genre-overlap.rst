:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-genre-overlap:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Genre Overlap</b></div>`

===============
Genre Overlap
===============

*The degree to which recommended items share genres, a diversity signal.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Genre overlap** measures how similar two items are by **how many genres (or categories) they share** — a
**metadata-based** similarity for items that carry categorical labels, like movies, music or books. It is
the domain-specific counterpart to **cosine** or **Jaccard** similarity when the features are **genres**.

How it's computed
-----------------

Often as the **Jaccard** of the two items' genre sets — the shared genres over the total distinct genres:

.. math::

   \text{overlap}(i, j) = \frac{|G_i \cap G_j|}{|G_i \cup G_j|}.

Two movies both tagged :math:`\{\text{action}, \text{thriller}\}` overlap fully; an action film and a
documentary don't overlap at all.

Where it's used
---------------

It serves as the **similarity kernel** for **intra-list similarity / diversity** (a list of same-genre
items has high overlap → low diversity), and it underpins **calibrated** recommendation, where the **genre
mix** of a list is kept aligned with the user's historical tastes.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Embedding Similarity <320-embedding-similarity>` · :doc:`Catalog Coverage <268-catalog-coverage>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Genre Overlap <https://insightful-data-lab.com/2025/08/22/genre-overlap/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
