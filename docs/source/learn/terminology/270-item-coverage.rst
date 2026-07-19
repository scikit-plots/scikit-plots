:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-item-coverage:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Item Coverage</b></div>`

===============
Item Coverage
===============

*The share of items that the system is able to recommend.*

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

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`User Coverage <269-user-coverage>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Diminishing Utility <271-diminishing-utility>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Item Coverage <https://insightful-data-lab.com/2025/08/22/item-coverage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
