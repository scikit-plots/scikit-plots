:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-optimizely:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Optimizely</b></div>`

============
Optimizely
============

*A commercial online experimentation and A/B-testing platform.*

What it is
----------

**Optimizely** is a commercial **experimentation and digital-experience platform**.
Originally known for website A/B testing, it has grown into a full **experimentation +
feature-management** suite used by marketers, product managers and data scientists to
test and personalise user experiences.

Core capabilities
-----------------

- **Experimentation** — A/B, multivariate and multi-page tests across web, mobile and
  **server-side**, with random assignment and a no-code WYSIWYG editor for simple tests.
- **Feature management** — **feature flags** to toggle features without redeploying,
  gradual percentage rollouts, and targeting (e.g. only Premium users).
- **Personalisation** — audience-targeting rules and custom segments.
- **Statistics** — default **frequentist with sequential-testing adjustments** (so
  monitoring doesn't inflate error), Bayesian methods in some tiers, plus **variance
  reduction (CUPED-style)** in enterprise plans.
- **Integration** — connects to Google Analytics, Segment, Salesforce, Amplitude and
  warehouses like Snowflake and BigQuery.

A typical workflow
------------------

State a hypothesis → configure the experiment → split traffic (say 50/50) → track
conversions and events → monitor significance in real time → end when conclusive (with
**sequential-testing corrections**) → roll the winner out behind a **feature flag**.

Strengths and limits
--------------------

It is **user-friendly** (non-technical teams can run tests), spans **client- and
server-side**, has **built-in feature flags**, and brings real statistical rigour. The
costs: it is a **pricey SaaS** next to open-source options, results live or die by
**metric definition**, and the most complex experimentation needs (Google/Netflix scale)
may still want bespoke internal platforms. Among peers, VWO is cheaper but lighter for
engineering, Adobe Target is enterprise-grade but complex, and LaunchDarkly leads on
feature flagging but leans on external analytics.

----

**Mind map — connected ideas**

   :doc:`Google Experiments <100-google-experiments>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Sequential Settings <058-sequential-settings>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Optimizely <https://insightful-data-lab.com/2025/08/25/optimizely/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
