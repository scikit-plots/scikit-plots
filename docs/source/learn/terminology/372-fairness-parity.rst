:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-fairness-parity:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Fairness parity</b></div>`

=================
Fairness parity
=================

*Equalising a chosen metric across protected groups.*

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

**Fairness parity** is the family of **group-fairness** criteria that demand some metric be **equal across**
demographic groups — that the model treat protected groups **comparably**. Which metric you equalize defines
the flavor of parity.

The main flavors
----------------

**Demographic (statistical) parity** equalizes the **positive-outcome rate** across groups; **equal
opportunity** equalizes the **true-positive rate** (recall) among those who **qualify**; **equalized odds**
equalizes **both** TPR and FPR; **predictive parity** equalizes **precision**. Each is measured as a
**difference** or a **ratio** between groups.

Why it's hard
-------------

The different parities **conflict** — impossibility results show you generally **can't** satisfy all at once
(the **COMPAS** debate turned on predictive parity holding while equalized odds failed). Enforcing any parity
also usually **costs accuracy**, so teams choose the criterion that fits the **harm** they most need to
prevent. The **four-fifths (80%) rule** is a common legal threshold.

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Selection Rate <390-selection-rate>` · :doc:`Recall <423-recall>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fair Lending laws <332-fair-lending-laws>` · :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Selection Rate <390-selection-rate>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Fairness parity <https://insightful-data-lab.com/2025/08/19/fairness-parity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
