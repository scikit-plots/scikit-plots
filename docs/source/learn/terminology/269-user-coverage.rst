:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-user-coverage:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>User Coverage</b></div>`

===============
User Coverage
===============

*The share of users for whom the system can make useful recommendations.*

What it is
----------

**User coverage** is the fraction of the user base for whom the recommender can produce **useful
recommendations** — *how many users can the system serve?* A system with excellent recommendations for
only a subset of users has limited reach:

.. math::

   \text{User Coverage} = \frac{|\text{users served}|}{|\text{total users}|}.

What limits it
--------------

**New users** with little or no history (the user-side **cold-start** problem) may get no personalized
recommendations; **sparse** or atypical users are also hard to serve. Fallbacks — **popularity** lists,
onboarding questionnaires, **content-based** profiles — extend coverage to these users.

Why it matters
--------------

Coverage and accuracy trade off — it is easy to look accurate by only serving **well-understood** users —
so user coverage keeps the evaluation **honest** about the whole population.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`Long-Tail Items <260-long-tail-items>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Hit Rate (HR) <412-hit-rate-hr>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `User Coverage <https://insightful-data-lab.com/2025/08/22/user-coverage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
