:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cosine-similarity-of-item-features:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Cosine Similarity of Item Features</b></div>`

====================================
Cosine Similarity of Item Features
====================================

*Similarity as the cosine of the angle between two items' feature vectors.*

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

**Cosine similarity** measures how alike two items are by the **cosine of the angle** between their
**feature vectors** — representations built from metadata (one-hot genres, tags, text) or learned
**embeddings**. It captures **orientation**, not magnitude, so it is invariant to vector length.

The formula
-----------

.. math::

   \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\,\|\mathbf{B}\|},

ranging from :math:`-1` to :math:`1` (0 to 1 for non-negative features) — **1** means identical direction
(very similar), **0** means unrelated (orthogonal).

Where it's used
---------------

It powers **content-based filtering** and **item-item** similarity (recommend items close to those a user
liked), and it is the usual kernel for computing **intra-list similarity / diversity**.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Jaccard index <264-jaccard-index>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Embedding Similarity <320-embedding-similarity>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Embedding <173-embedding>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cosine Similarity of Item Features <https://insightful-data-lab.com/2025/08/22/cosine-similarity-of-item-features/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
