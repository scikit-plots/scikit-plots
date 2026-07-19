:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-a-b-testing:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>A/B Testing</b></div>`

=============
A/B Testing
=============

*A randomised experiment comparing two variants to measure an effect.*

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

An **A/B test** is a **controlled experiment** that **randomly assigns** users to two variants — **A**
(control) and **B** (treatment) — and measures which performs better on a chosen **metric** (conversion
rate, time on page, retention). Randomization is what lets you read the difference as **causal**.

How it's run
------------

You fix the **metric**, use a **power analysis** to set the **sample size**, pick a **statistical test**
(t-test, chi-square), and choose a **significance level α**. When the data are in, the test decides whether
B's effect is **real** or noise.

Its discipline
--------------

The classic A/B test is **fixed-horizon** — you must wait for the pre-planned sample before deciding.
**Peeking** early and stopping when it looks significant **inflates false positives**, which is exactly the
failure that **sequential** methods are designed to fix.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Interleaving Tests <379-interleaving-tests>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `A/B Testing <https://insightful-data-lab.com/2025/08/19/a-b-testing/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
