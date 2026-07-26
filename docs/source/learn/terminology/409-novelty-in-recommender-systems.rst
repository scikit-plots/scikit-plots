:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-novelty-in-recommender-systems:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Novelty (in Recommender Systems)</b></div>`

==================================
Novelty (in Recommender Systems)
==================================

*How unfamiliar or unexpected recommended items are to the user.*

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

**Novelty** measures how **new** or **unfamiliar** a recommender's suggestions are to a user — surfacing items
they're **unlikely to have already seen** rather than obvious mainstream hits. It is a **beyond-accuracy**
objective: a technically accurate list of things the user already knows adds little value.

How it's measured
-----------------

Novelty is usually tied to **popularity** — the less popular an item, the more novel — so metrics use
**average recommendation popularity** (lower means more novel) or the **self-information** (the negative log
of an item's popularity). Recommending from the **long tail** raises novelty.

Why it matters
--------------

Novelty drives **discovery** and helps monetize the **long tail**, but pushed too far it sacrifices
**relevance** — nobody wants *random* items. The art is balancing the **accuracy-novelty** trade-off;
combined with relevance and surprise, novelty becomes **serendipity**.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Coverage <411-coverage>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Novelty (in Recommender Systems) <https://insightful-data-lab.com/2025/08/19/novelty-in-recommender-systems/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
