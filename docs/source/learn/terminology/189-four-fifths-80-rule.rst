:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-four-fifths-80-rule:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Four-Fifths (80%) Rule</b></div>`

========================
Four-Fifths (80%) Rule
========================

*An adverse-impact guideline flagging a group selection rate below 80% of the top group's.*

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

The **four-fifths rule** — also called the **80% rule** — is the primary rule-of-thumb the U.S.
**EEOC** uses to flag **adverse (disparate) impact** in a selection process. It comes from the
**Uniform Guidelines on Employee Selection Procedures (UGESP, 1978)** and operates under **Title VII**
of the Civil Rights Act. The rule: the **selection rate** for any protected group (by race, sex or
ethnicity) that is **less than four-fifths (80%) of the rate of the highest-selected group** is
generally treated as evidence of adverse impact.

How the ratio is computed
---------------------------

Four steps. Compute each group's **selection rate** — the number selected divided by the number of
applicants. Identify the group with the **highest** rate; that becomes the **baseline**. Divide every
other group's rate by that baseline to get an **impact ratio**. If any ratio falls **below 0.80**,
the procedure is flagged for that group.

A worked example
----------------

An employer hires for a warehouse role: of 100 white applicants, 40 are hired (a **40%** rate); of 80
Black applicants, 24 are hired (a **30%** rate). The impact ratio is 30 / 40 = **0.75**. Because 75%
is below 80%, adverse impact is indicated against Black applicants and the procedure would be flagged
for review. Note the test compares **rates, not head-counts** — a smaller group can still clear the
bar if its rate is high enough.

A threshold, not a verdict
----------------------------

Falling below 80% does **not** prove discrimination — it triggers **investigation**. Because the raw
ratio ignores whether the gap is beyond chance, practitioners pair it with **significance tests**
(Fisher's exact, chi-square, or a standard-deviation analysis). Once adverse impact is indicated, the
employer must show the procedure is **job-related and consistent with business necessity**, offer a
**less-discriminatory alternative**, or **modify** it. Courts treat the rule as guidance, not a
definitive legal standard.

Why it matters for ML
---------------------

The rule governs **automated selection** exactly as it governs human decisions: an AI resume screener
or scoring model that selects one protected group at **under 80%** of the top group's rate carries
indicated adverse impact — **even with no discriminatory intent**. This is why the 80% ratio is the
legal touchstone behind fairness criteria like **demographic parity**, which compares
positive-outcome rates across groups.

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Statistical Significance <096-statistical-significance>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Selection Rate <390-selection-rate>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Four-Fifths (80%) Rule <https://insightful-data-lab.com/2025/08/23/four-fifths-80-rule/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
