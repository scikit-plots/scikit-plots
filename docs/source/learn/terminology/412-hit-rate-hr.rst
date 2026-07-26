:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-hit-rate-hr:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎁&nbsp;&nbsp;<b>Hit Rate (HR)</b></div>`

===============
Hit Rate (HR)
===============

*The share of users for whom a relevant item appears in the top-N list.*

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

**Hit Rate** is the simplest top-N recommendation metric — it asks whether **at least one** relevant item
appears in a user's top-**K** list. Each user scores **1** if there's any hit and **0** otherwise, and the
metric is the **average** across users:

.. math::

   \text{HR@}K = \frac{\#\{\text{users with} \ge 1 \text{ relevant item in top } K\}}{|U|}.

What it captures
----------------

HR measures **coverage of intent** at the coarsest level — did we surface *something* the user wanted? — which
is exactly right for feeds, "you might also like" rows, and any setting where a **single** good hit is a win.
It is intuitive and easy to explain to stakeholders.

Its limits
----------

HR is **binary** and **position-blind** — it doesn't care **where** in the list the hit landed or **how
many** relevant items were found, so a hit at rank 1 and a hit at rank 10 score the same. It also **rises**
mechanically with **K**, so always report the cutoff (Hit@5 vs Hit@10) and pair it with a **ranking** metric.

----

*Theme:* :ref:`Recommender Systems <term-theme-recsys>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Recall <423-recall>`

----

.. hint::
   **More in Recommender Systems**

   :doc:`Catalog Coverage <268-catalog-coverage>` · :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>` · :doc:`Diminishing Utility <271-diminishing-utility>` · :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>` · :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>` · :doc:`Genre Overlap <263-genre-overlap>` · :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>` · :doc:`Item Coverage <270-item-coverage>` · :doc:`Jaccard index <264-jaccard-index>` · :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>` · :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>` · :doc:`Self-Information of Popularity <261-self-information-of-popularity>` · :doc:`User Coverage <269-user-coverage>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Hit Rate (HR) <https://insightful-data-lab.com/2025/08/19/hit-rate-hr/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
