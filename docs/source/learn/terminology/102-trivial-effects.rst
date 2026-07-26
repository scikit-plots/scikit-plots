:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-trivial-effects:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Trivial Effects</b></div>`

=================
Trivial Effects
=================

*Effects too small to matter in practice even if statistically detectable.*

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

A **trivial effect** is a result that is **statistically significant**
(:math:`p < \alpha`) yet **so small in magnitude that it has little or no practical
importance**. The difference is *real* — not chance — but too tiny to matter.

Why they appear
---------------

Two causes. First, **large samples**: with enough data, even a minuscule difference clears
the significance bar — in an A/B test on millions of users, a 0.05% conversion bump can be
"significant" yet meaningless. Second, **fixating on p-values**: a p-value says whether an
effect exists, not how big it is, so reading it without an **effect size** invites
over-interpretation.

Examples
--------

A drug that lowers blood pressure by **0.5 mmHg** versus standard care, tested on 10,000
patients, can show :math:`p < 0.001` — significant, but clinically trivial. A landing page
that moves conversion from 10.00% to **10.05%** across 1,000,000 users gives
:math:`p < 0.01`, yet a 0.05% lift is not worth deploying.

How to avoid the trap
---------------------

Always report an **effect size** (Cohen's d, a difference in proportions) next to the
p-value; read the **confidence interval** (a tight band hugging zero signals triviality);
judge **practical relevance**; and fix a **minimum detectable effect** in advance — "we act
only if conversion improves by at least +1%."

The link to power and n
-----------------------

Because larger samples raise power, they make even **trivial** effects detectable. The
discipline is to size a study to catch effects **worth caring about**, not the smallest
detectable ones — significance is necessary for a finding to matter, but never sufficient.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Statistical Significance <096-statistical-significance>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Sample size <103-sample-size>` · :doc:`Statistical Power <348-statistical-power>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Trivial Effects <https://insightful-data-lab.com/2025/08/24/trivial-effects/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
