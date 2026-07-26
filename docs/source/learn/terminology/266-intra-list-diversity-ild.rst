:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-intra-list-diversity-ild:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Intra-List Diversity (ILD)</b></div>`

============================
Intra-List Diversity (ILD)
============================

*The average dissimilarity among items within a single recommendation list.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Intra-list diversity** measures how **varied** the items within a **single** recommendation list are —
the antidote to lists of near-identical products. It is defined from the **intra-list similarity (ILS)**,
the **average pairwise similarity** of all items in the list; diversity is its complement.

The formula
-----------

.. math::

   \mathrm{ILD} = \frac{2}{|L|\,(|L|-1)} \sum_{i < j} \big(1 - \mathrm{sim}(i, j)\big),

with :math:`\mathrm{sim}` a **cosine** (over embeddings / features) or **Jaccard** (over tags / genres)
similarity. High ILS → similar items → **low** diversity; low ILS → varied items → **high** diversity.

Why it matters
--------------

Accuracy alone rewards recommending ten versions of the same hit; **diversity** captures whether a list
actually **broadens** what the user sees. It trades off against relevance — the art is a **diverse yet
relevant** list.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Intra-List Diversity (ILD) <https://insightful-data-lab.com/2025/08/22/intra-list-diversity-ild/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
