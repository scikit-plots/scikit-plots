:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-interleaving-tests:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔎&nbsp;&nbsp;<b>Interleaving Tests</b></div>`

====================
Interleaving Tests
====================

*Online ranker comparison that blends two result lists and attributes clicks.*

What it is
----------

**Interleaving** compares **two rankers** — say two search algorithms — by **blending their result lists
into one** list shown to a **single user**, then crediting each **click** to whichever ranker supplied that
item. The same user effectively judges both at once.

Why it's powerful
-----------------

Because each user sees **both** rankers' results (a within-user comparison), interleaving is **far more
sensitive** than an A/B test — it reaches a reliable conclusion from **far fewer interactions**, so **fewer
users** are exposed to a possibly worse ranker. **Multileaving** extends the idea to compare **many** rankers
simultaneously.

Where it's used
---------------

Interleaving is a staple of **search and recommendation** evaluation and **learning-to-rank**, where ranking
quality differences are subtle and A/B tests would need huge traffic to detect them.

----

*Theme:* :ref:`Ranking & Interleaving <term-theme-ranking>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`A/B/n Test <114-a-b-n-test>`

----

.. hint::
   **More in Ranking & Interleaving**

   :doc:`Balanced Interleaving <111-balanced-interleaving>` · :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>` · :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>` · :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>` · :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>` · :doc:`Ranking Algorithms <108-ranking-algorithms>` · :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>` · :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Interleaving Tests <https://insightful-data-lab.com/2025/08/19/interleaving-tests/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
