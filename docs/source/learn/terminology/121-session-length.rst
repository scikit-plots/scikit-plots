:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-session-length:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Session Length</b></div>`

================
Session Length
================

*The duration of a user's single visit or session.*

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

**Session length** is the **time a user spends in a single session** with an app, site or
product — from opening it to closing it or timing out from inactivity — usually in seconds
or minutes.

The formula
-----------

Per session it is simply

.. math::

   \text{Session Length} = \text{end time} - \text{start time},

and across many sessions the **average session length** is total time spent divided by the
number of sessions. Analytics platforms compute both automatically.

Why it's tracked
----------------

It is a core **engagement** signal — longer sessions often mean users find value — and it
feeds **retention** (consistently short sessions hint at usability problems),
**monetisation** (in ad models, more time means more impressions), and **UX** decisions. A
12-minute news session, a 2.5-minute site average (2,500 minutes over 1,000 sessions), or a
45-minute gaming session each read differently.

The interpretation trap
-----------------------

Longer is **not always better**. A short session can mean the user **found what they needed
instantly** (checking the weather), and the "right" length is **product-specific** — a few
minutes for a news site, 30–60 for video streaming. Beware **idle time**, too: an
open-but-unused app can inflate the number. Session length is meaningful only **read against
industry, product type and user intent**.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Revenue per User (RPU / ARPU) <122-revenue-per-user-rpu-arpu>` · :doc:`Churn <123-churn>` · :doc:`Retention <124-retention>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`SaaS (Software as a Service) <034-saas-software-as-a-service>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Session Length <https://insightful-data-lab.com/2025/08/24/session-length/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
