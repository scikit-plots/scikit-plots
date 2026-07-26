:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-google-experiments:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧰&nbsp;&nbsp;<b>Google Experiments</b></div>`

====================
Google Experiments
====================

*Google's online experimentation / A-B testing tooling.*

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

"**Google Experiments**" is an umbrella for several Google A/B-testing and
experimentation products that have come and gone:

- **Google Optimize** — a free website-testing tool tied to Google Analytics (A/B,
  multivariate and split-URL tests), **deprecated in 2023**.
- **Google Ads Experiments** — still active; lets advertisers split campaign traffic to
  test bids, keywords, creatives and audiences, comparing conversions, CPC and ROAS.
- **GA4 + third-party platforms** — Google now recommends pairing GA4 (event and
  conversion tracking) with external experimentation tools (Optimizely, VWO) that handle
  randomisation, stopping rules and statistics.
- **Vertex AI "experiments"** — a *different* meaning: tracking ML model versions,
  hyperparameters and metrics, not A/B testing.

The statistics
--------------

The reporting engine differed by product: **Google Ads Experiments** uses frequentist
methods with adjusted confidence intervals, while the legacy **Optimize** ran a
**Bayesian** engine that reported a **"probability to beat baseline"** instead of
p-values — e.g. "variant B has a 95% probability of being better than A", which
non-technical users found far easier to act on.

Examples
--------

An Ads experiment splitting traffic 50/50 to test a higher-bid strategy might show 12%
more conversions at significance after two weeks → adopt it. A legacy Optimize website
test of a red vs blue call-to-action might report "red has a 96% probability of beating
blue".

The takeaway
------------

With Optimize retired, web and product experimentation on Google's stack now means
**GA4 plus an external platform** (Optimizely, VWO, LaunchDarkly or custom infra); Ads
Experiments remain the built-in option, but only for ad-campaign settings, not
full-site UX.

----

*Theme:* :ref:`ML Platforms & Tools <term-theme-platforms>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Optimizely <069-optimizely>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>`

----

.. hint::
   **More in ML Platforms & Tools**

   :doc:`AWS SageMaker <148-aws-sagemaker>` · :doc:`Kaggle <273-kaggle>` · :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>` · :doc:`OpenAI API (ML API) <150-openai-api-ml-api>` · :doc:`TPU Clusters <347-tpu-clusters>` · :doc:`Vertex AI <149-vertex-ai>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Google Experiments <https://insightful-data-lab.com/2025/08/25/google-experiments/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
