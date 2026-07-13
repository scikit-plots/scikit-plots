:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-team-draft-interleaving-tdi:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>Team Draft Interleaving (TDI)</b></div>`

===============================
Team Draft Interleaving (TDI)
===============================

*Interleaving where rankers alternately draft items into one list for fair comparison.*

What it is
----------

**Team-draft interleaving (TDI)** is an **online method for comparing two ranking
algorithms** A and B. Like balanced interleaving it merges both into one list shown to
users, but it fills the list by a **draft pick** — exactly like two captains choosing
players for a team — deciding which algorithm contributes the next slot.

Why use it
----------

**A/B testing** splits traffic and needs large samples; **balanced interleaving** can favour
one ranker depending on overlap and order. TDI **randomises the draft** so each algorithm
gets an equal, unbiased chance to place its items.

How it works
------------

Take the top-k from A and B; **randomly pick who drafts first**; then **alternate**: each
algorithm in turn adds its **highest-ranked item not already in the list**, until the
interleaving is full. Show it, and **attribute each click to whichever algorithm drafted
that item** — attribution is **deterministic and unambiguous**.

Example
-------

With A = [A1, A2, A3, A4] and B = [B1, B2, B3, B4], a draft might yield
[A1, B1, A2, B2, A3, B3, A4, B4]. A click on **B2** credits **algorithm B**, because B
drafted it.

Strengths and limits
--------------------

TDI is **fair** (randomised drafting removes systematic bias), **efficient** (fewer users
than A/B testing to detect a difference), gives **clear click ownership**, and is
**sensitive** to small gaps — which made it an industry standard. Its limits: it compares
**only two** algorithms at once, still assumes **clicks equal relevance** (noisy), and needs
care with **ties and overlapping** results.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Team Draft Interleaving (TDI) <https://insightful-data-lab.com/2025/08/24/team-draft-interleaving-tdi/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
