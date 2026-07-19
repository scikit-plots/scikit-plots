:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-causal-impact:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Causal Impact</b></div>`

===============
Causal Impact
===============

*The estimated effect of an intervention, often via a counterfactual time-series model.*

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

**Causal impact** is the **overall effect of an intervention** — a campaign, launch or
policy — **on an outcome**. In data science the term usually points to **Google's
CausalImpact** (Brodersen et al., 2015): a **Bayesian structural time-series** model that
estimates what *would* have happened without the intervention and compares it to what
actually did.

Why it's needed
---------------

In an RCT or A/B test the causal effect is easy — you have treatment and control groups. But
in **observational or time-series** settings there may be **no untreated control**: run a
nationwide ad campaign and there is no parallel country that didn't see it. The fix is to
model the **counterfactual** — the outcome you'd have seen without the intervention — and
measure the gap.

The method and formula
----------------------

CausalImpact fits a model on the **pre-intervention** period (using historical data and
**control covariates**), forecasts the **counterfactual** for the post period, and
subtracts:

.. math::

   \text{Causal Impact} = Y_{\text{observed, post}} - Y_{\text{predicted, counterfactual}},

reporting the difference **with Bayesian credible intervals** rather than a single number.

Example
-------

A firm launches a TV campaign in July. Trained on January–June sales, the model forecasts a
**counterfactual of 50,000 units**; actual sales come in at **60,000** — an estimated impact
of **+10,000 units**, with a credible interval of roughly [7,000, 13,000].

Strengths, limits, and effect vs impact
---------------------------------------

It handles **time series** naturally, yields **full posterior distributions**, uses
**covariates** to sharpen accuracy, and needs no RCT — but it **assumes the model captures
the dynamics**, is **sensitive to the choice of controls**, and wants a **clear start date**.
Note the level distinction: the **causal effect** is micro (the change per unit — a drug
lowers blood pressure 5 mmHg), while the **causal impact** is macro (the aggregate — a
nationwide rollout prevents 10,000 hospitalisations).

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Causal Impact <https://insightful-data-lab.com/2025/08/24/causal-impact/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
