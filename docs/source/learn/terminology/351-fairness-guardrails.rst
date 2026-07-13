:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-fairness-guardrails:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Fairness Guardrails</b></div>`

=====================
Fairness Guardrails
=====================

*Automated checks that enforce fairness constraints on outputs.*

What it is
----------

**Fairness guardrails** are automated checks on a model's **disparity across groups** that **block a
deployment** or **trigger retraining** when a fairness metric exceeds an agreed limit. They bake **equity**
and **compliance** into the release process.

How they're enforced
--------------------

A pre-deployment **fairness audit** requires disparity measures — such as the **demographic-parity
difference** or **equalized-odds** gap — to stay **below a threshold** (for example **≤ 0.05**); if **DPD or
EO exceeds** it, the release is **halted** and the model is **retrained**. Post-deployment, real-time
monitoring watches for **bias spikes**.

Why they matter
---------------

Fairness can **degrade** as populations drift, and regulated domains (lending, hiring, healthcare) demand
**auditable** guarantees. Guardrails provide a **hard gate** and an **audit trail**, rather than relying on a
one-time fairness check that goes stale.

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Latency Guardrails <350-latency-guardrails>` · :doc:`Selection Rate <390-selection-rate>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Statistical Tests <328-statistical-tests>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Selection Rate <390-selection-rate>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Fairness Guardrails <https://insightful-data-lab.com/2025/08/20/fairness-guardrails/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
