:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-multivariate-test-mvt:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Multivariate Test (MVT)</b></div>`

=========================
Multivariate Test (MVT)
=========================

*An experiment varying several elements at once to estimate combined effects.*

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

A **multivariate test (MVT)** varies **several page elements at once** and tests their
**combinations**, to learn not just which single change helps but **how elements interact**.
Where A/B compares whole versions, MVT decomposes the page into factors.

How it works
------------

Choose the elements to vary — say **headline, button colour, image** — and form the
**combinations**. With two options each, that is :math:`2 \times 2 \times 2 = 8` versions;
users are split across them and the metric measured per combination.

Full vs fractional factorial
----------------------------

A **full factorial** MVT tests **every** combination, giving the cleanest read on **main
effects** (each variable alone) and **interaction effects** (how they combine) — but the
count explodes. A **fractional factorial** tests a **chosen subset**, cutting traffic and
complexity at the cost of resolving some interactions.

Why interactions matter
-----------------------

The payoff is the interaction. A landing-page test might find a strong headline worth
**+10%**, a green button **+5%** and a lifestyle image **+2%** individually — yet the
**combination** of all three lifts conversion **+25%**, more than the parts suggest. That
synergy is invisible to separate A/B tests.

The cost
--------

MVT needs **much larger samples** (variants multiply fast), is **harder to design and
analyse**, and carries the same **multiple-comparisons** false-positive risk. Use it to
**fine-tune many elements** once you have the traffic; use A/B for big, single-design
changes.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Multivariate Test (MVT) <https://insightful-data-lab.com/2025/08/24/multivariate-test-mvt/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
