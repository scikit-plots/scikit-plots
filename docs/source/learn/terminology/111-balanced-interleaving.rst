:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-balanced-interleaving:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>Balanced Interleaving</b></div>`

=======================
Balanced Interleaving
=======================

*An early interleaving scheme that merges two ranked lists to attribute clicks.*

What it is
----------

**Balanced interleaving** is the simplest **online method for comparing two ranking
algorithms**: it mixes their results into one combined list shown to users, then reads off
which algorithm's items draw more clicks. Both rankers get a **fair, equal chance** to place
items, so the comparison happens *within a single session* rather than across split traffic.

Why use it
----------

Classic **A/B testing** sends half the users to A and half to B, which needs large samples
and time. Interleaving shows **both at once**, making the comparison **faster and more
sensitive** — small quality gaps surface with far fewer interactions.

How it works
------------

Take the top-k from A and B and build the list by **strict alternation** — first from A,
then B, then A, ensuring neither dominates by position. Show the combined list, **credit
each click to the algorithm that contributed that item**, and the ranker with more credited
clicks wins.

Example
-------

With A = [A1, A2, A3, A4] and B = [B1, B2, B3, B4], the interleaving is
[A1, B1, A2, B2, A3, B3, A4, B4]. If the user clicks A1, B2 and A3, then A scores **2** and
B scores **1** — A wins this impression.

Strengths and limits
--------------------

It is **efficient, fair and sensitive**, but it compares **only two** algorithms, assumes
**clicks track relevance** (which is noisy), and needs care so that **position bias** in the
alternation doesn't quietly favour one side. **Team-draft** and **probabilistic
interleaving** were developed to address exactly these weaknesses.

----

**Mind map — connected ideas**

   :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>`

----

**More in Ranking & Interleaving**

   :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Balanced Interleaving <https://insightful-data-lab.com/2025/08/24/balanced-interleaving/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
