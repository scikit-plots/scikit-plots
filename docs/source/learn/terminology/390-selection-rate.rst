:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-selection-rate:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Selection Rate</b></div>`

================
Selection Rate
================

*The fraction of a group that receives a positive decision.*

What it is
----------

The **selection rate** is the fraction of a group that receives a **favorable** decision — the share predicted
**positive**, :math:`P(\hat{Y} = 1)` for that group. If a lender approves 60% of one group and 40% of
another, those are the two selection rates.

What it drives
--------------

Selection rate is the quantity behind **demographic parity** — parity holds exactly when selection rates are
**equal** across groups. The **disparate-impact ratio** divides the **lowest** group's selection rate by the
**highest**; a ratio **below 0.8** (the four-fifths rule) flags potential **adverse impact**.

Where it's used
---------------

It is the core number in auditing **hiring**, **lending**, and admissions for bias, because it captures **who
gets the good outcome** without needing labels. But equal selection rates say **nothing** about whether the
**right** people were chosen — that is why it is paired with error-based metrics like **recall**.

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Fairness parity <372-fairness-parity>` · :doc:`Recall <423-recall>` · :doc:`Fair Lending laws <332-fair-lending-laws>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Selection Rate <https://insightful-data-lab.com/2025/08/19/selection-rate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
