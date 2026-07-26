:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-predictive-ltv-pltv:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💼&nbsp;&nbsp;<b>Predictive LTV (pLTV)</b></div>`

=======================
Predictive LTV (pLTV)
=======================

*A model-based forecast of a customer's future lifetime value.*

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

**Predictive LTV (pLTV)** forecasts a customer's (or cohort's) lifetime value *before*
their lifecycle is complete. Where historical LTV looks backward, pLTV is
**forward-looking**: it predicts future revenue and retention from early signals using
statistical or machine-learning models, answering "how much will this customer be
worth, given what we know now?".

Why it matters
--------------

It lets you judge **CAC vs LTV early** (no waiting years), optimise marketing in
**real time** (how much to bid for an ad impression), and spot **high-value customers
early** for targeted retention.

Methods
-------

- **Rule-based** — use early behaviour (first-week spend, first-month activity) as a
  proxy: "users who spend $20+ in week 1 are 3× more valuable at 12 months".
- **Cohort extrapolation** — fit a decay curve (exponential, Pareto, Weibull, BG/NBD)
  to historical cohorts and apply it to newer ones.
- **Probabilistic models** — **Pareto/NBD**, **BG/NBD** and **Gamma-Gamma** estimate
  purchase frequency and monetary value separately; standard in marketing analytics.
- **Machine learning** — regression/ML models (gradient-boosted trees, random
  forests, neural nets) predict revenue over a horizon from rich features.

Formula (conceptual)
--------------------

.. math::

   \text{pLTV}_i = \sum_{t=1}^{T} \mathbb{E}\!\left[\text{Revenue}_{i,t} \mid X_i\right],

summing the model's expected revenue for customer :math:`i` over horizon :math:`T`,
given features :math:`X_i` (behaviour, demographics, channel, engagement).

Worked example
--------------

Two-week-old customers: A visits 10 times and spends $50; B visits twice and spends
$5. Historical cohorts show high-activity users average a $600 twelve-month LTV and
low-activity users $60 — so pLTV(A) ≈ $600 and pLTV(B) ≈ $60, flagging A for priority
retention even this early.

Pitfalls and edge cases
-----------------------

- Needs a solid **data history** to train on.
- **Model drift** — predictions degrade as behaviour and market shift, so monitor and
  retrain.
- Always **validate** predicted vs actual LTV as cohorts mature.

----

*Theme:* :ref:`Business & Growth Analytics <term-theme-growth>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`LTV (Customer Lifetime Value) <373-ltv-customer-lifetime-value>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Churn <123-churn>` · :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>`

----

.. hint::
   **More in Business & Growth Analytics**

   :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>` · :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>` · :doc:`Cannibalization <392-cannibalization>` · :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>` · :doc:`Churn <123-churn>` · :doc:`Cohort <183-cohort>` · :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>` · :doc:`Cross-Selling <031-cross-selling>` · :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>` · :doc:`Customer Lifetime <042-customer-lifetime>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Predictive LTV (pLTV) <https://insightful-data-lab.com/2025/08/29/predictive-ltv-pltv/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
