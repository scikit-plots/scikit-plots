:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sequential-testing-also-called-sequential-analysis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Sequential Testing (also called sequential analysis)</b></div>`

======================================================
Sequential Testing (also called sequential analysis)
======================================================

*Analysing results as data arrive while controlling error from repeated looks.*

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

**Sequential testing** (sequential analysis) **monitors an experiment continuously** and lets you **stop as
soon as** the evidence is conclusive — rather than waiting for a fixed sample size. It is built for the
streaming data of modern experimentation platforms.

The problem it solves
---------------------

Repeatedly checking a **fixed-horizon** test and stopping when it looks good — **peeking** — badly
**inflates the Type I error**. Sequential methods like the **sequential probability ratio test (SPRT)** and
group-sequential designs keep the false-positive rate controlled **at any time**, so early stopping is
**valid**.

The payoff
----------

Because it can end as soon as a winner (or a dead end) is clear, sequential testing **cuts the average
sample size** and **deployment time**, exposing **fewer users** to an inferior variant — at the price of
slightly more conservative thresholds to preserve error control.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Sequential Testing (also called sequential analysis) <https://insightful-data-lab.com/2025/08/19/sequential-testing-also-called-sequential-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
