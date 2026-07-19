:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-online-experimentation-platforms:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Online Experimentation Platforms</b></div>`

==================================
Online Experimentation Platforms
==================================

*Systems that randomise, run and analyse controlled experiments on live traffic.*

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

An **online experimentation platform** is the **infrastructure that lets a company run
controlled experiments — A/B tests, multivariate tests, bandits — on a live digital
product**, usually built internally at scale. It gives product managers, data scientists
and engineers a way to randomly assign users to variants, log outcomes automatically,
analyse them with sound statistics, and enforce guardrails so experiments don't harm
users.

Core components
---------------

- **Assignment & randomisation** — bucket users (by cookie, device or account) into
  variants, with *stable* assignment so a user always sees the same one.
- **Data collection & logging** — track clicks, conversions, retention and revenue
  consistently across pipelines.
- **Metrics framework** — predefined primary KPIs plus **guardrail metrics** (engagement
  should rise, latency must not).
- **Statistical engine** — frequentist (p-values, fixed-horizon or sequential), Bayesian
  (posteriors, credible intervals), and variance reduction (CUPED) or stratified
  sampling.
- **Dashboard** — automated significance, effect sizes and time series.
- **Governance / guardrails** — flag harmful launches and enforce GDPR/HIPAA compliance.

Modern capabilities
-------------------

The leading platforms add **sequential testing** (safe peeking), **Bayesian analysis**,
**multi-armed bandits** (adaptive traffic), **heterogeneous treatment-effect / CATE**
analysis by segment, **ML-driven personalisation**, and **scale** to hundreds of
simultaneous tests.

Who builds them
---------------

Commercial tools include **Optimizely**, **Adobe Target**, **VWO** and feature-flag
platforms like **Split.io / LaunchDarkly** (Google Optimize was retired in favour of
GA4). The big platforms run their own: **Microsoft's ExP** (Bing, Office, Windows),
**Airbnb's ERF**, **Uber's** and **Netflix's** XP infrastructure, **LinkedIn's XLNT**,
and **Meta's PlanOut**.

Hard problems
-------------

Running thousands of concurrent experiments raises real challenges: **statistical
validity** at scale (many simultaneous tests), **interference / contamination** (users
caught in several experiments), **metric definition** (does the KPI track product
health?), sheer **scale**, and **ethics** (avoiding harmful or unfair experiments). The
payoff is **evidence-based decisions** instead of intuition, lower launch risk, and
continuous optimisation.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Optimizely <069-optimizely>` · :doc:`Google Experiments <100-google-experiments>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Bandit Algorithms <113-bandit-algorithms>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Online Experimentation Platforms <https://insightful-data-lab.com/2025/08/25/online-experimentation-platforms/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
