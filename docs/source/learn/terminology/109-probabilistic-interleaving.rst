:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probabilistic-interleaving:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>Probabilistic Interleaving</b></div>`

============================
Probabilistic Interleaving
============================

*An online method that probabilistically mixes two rankers' results to compare them.*

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

**Probabilistic interleaving** is an **online method for comparing ranking algorithms** by
blending their results into one list shown to users. Unlike **balanced interleaving**
(strict alternation) or **team-draft interleaving** (a draft pick), it builds the combined
list **probabilistically** — each algorithm defines a distribution over its ranking (higher
positions get more weight, e.g. a softmax over rank), and items are **sampled** from those
distributions.

Why use it
----------

Full **A/B testing** of rankers needs huge traffic and spends users on the worse variant;
balanced interleaving can be biased when lists overlap; team-draft works only for two
algorithms. Probabilistic interleaving is **flexible, scalable to many algorithms, and
statistically principled**.

How it works
------------

Build a rank-based probability distribution for each algorithm, **sample** items to form the
interleaved list, show it, and then **attribute clicks probabilistically** — rather than a
click belonging deterministically to one algorithm, the credit is **shared in proportion**
to how strongly each ranked the clicked item. Comparing **expected credit** across
algorithms reveals the winner.

Example
-------

With A = [A1, A2, A3] and B = [B1, A2, B3], a sampled interleaving might be
[A1, B1, A2, B3, A3]. If the user clicks **A2** — ranked highly by *both* — team-draft
would hand the whole click to one side, but probabilistic interleaving **splits the credit**
between A and B.

Strengths and costs
-------------------

It handles **more than two** algorithms, stays **fair when rankings overlap** (shared items
share credit), reduces bias, and is often **more sensitive** (detecting differences with
fewer clicks). The price is **complexity**: probabilistic attribution is harder to explain,
and the probability function (e.g. softmax temperature) needs careful design.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probabilistic Interleaving <https://insightful-data-lab.com/2025/08/24/probabilistic-interleaving/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
