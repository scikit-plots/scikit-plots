:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-o-brienfleming-obf-method:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎰&nbsp;&nbsp;<b>O'Brien–Fleming (OBF) Method</b></div>`

==============================
O'Brien–Fleming (OBF) Method
==============================

*A group-sequential boundary that is very strict early and relaxes toward the final analysis.*

What it is
----------

The **O'Brien–Fleming (OBF) method** is a **group-sequential design** that controls the
overall Type I error :math:`\alpha` across multiple interim analyses with a distinctive
spending shape: **very strict early, lenient late**. Early on you need overwhelming
evidence to stop; by the final look the threshold is essentially the usual
:math:`\alpha`.

The spending shape
------------------

For :math:`\alpha = 0.05` (two-sided) with 4 looks, the OBF critical z-values are roughly

- 25% — :math:`z \approx 3.47` (:math:`p \approx 0.0005`)
- 50% — :math:`z \approx 2.45` (:math:`p \approx 0.014`)
- 75% — :math:`z \approx 2.00` (:math:`p \approx 0.045`)
- 100% — :math:`z \approx 1.98` (:math:`p \approx 0.048`)

so the early bar is extreme and the final bar is almost a normal :math:`\alpha = 0.05`
test. This is the opposite philosophy to Pocock's flat :math:`z \approx 2.41`.

Example
-------

A heart-drug trial with 4 interim looks: at 25% an observed :math:`p = 0.002`
(:math:`z \approx 3.1`) is **not** below the OBF bound (:math:`p \approx 0.0005`), so
continue; at 50%, :math:`p = 0.009` (:math:`z \approx 2.6`) clears the bound
(:math:`\approx 0.014`) → **stop early for efficacy**.

Strengths and trade-offs
------------------------

OBF gives **strong early protection against false positives** (random noise can't stop
the trial prematurely) while still permitting an early stop for a genuinely large effect,
and its **final test barely loses power** versus fixed-horizon. The downside: it **rarely
stops early** unless the effect is huge, so you often collect nearly the full sample.
That conservatism is exactly why it suits **safety-critical** domains like medicine,
whereas Pocock fits exploratory or business A/B tests where stopping early saves money.

----

*Theme:* :ref:`Sequential Methods & Bandits <term-theme-bandits>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Pocock Method <077-pocock-method>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Frequentist <059-frequentist>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`

----

.. hint::
   **More in Sequential Methods & Bandits**

   :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `O'Brien–Fleming (OBF) Method <https://insightful-data-lab.com/2025/08/25/obrien-fleming-obf-method/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
