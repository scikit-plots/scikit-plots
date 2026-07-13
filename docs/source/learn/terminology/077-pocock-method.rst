:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-pocock-method:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎰&nbsp;&nbsp;<b>Pocock Method</b></div>`

===============
Pocock Method
===============

*A group-sequential boundary using a constant significance threshold at every interim look.*

What it is
----------

The **Pocock method** is a **group-sequential design** that allows several interim
analyses while keeping the overall Type I error at :math:`\alpha`. Its defining choice:
**every look — interim and final — uses the same significance threshold** (the same
critical value), in contrast to O'Brien–Fleming, which varies it.

How the threshold is set
------------------------

The constant cutoff is chosen so the *overall* error across all looks still equals
:math:`\alpha`. For :math:`\alpha = 0.05` (two-sided) with 4 looks at 25/50/75/100%, the
Pocock critical value is about :math:`z \approx \pm 2.41` (a nominal :math:`p \approx
0.017`) **at every look** — stricter than 1.96 because four chances to reject must share
the budget.

Example
-------

A checkout-flow A/B test plans 40,000 users (20,000 per arm) with interim looks every
10,000. The Pocock cutoff is :math:`z \ge 2.41` at every stage: an observed
:math:`z = 2.6` at the first look means **stop early, B wins**; :math:`z = 2.0` means
continue.

Trade-offs vs OBF
-----------------

Pocock is **easy to communicate** ("same threshold every time") and **stops early more
readily** for moderate effects, which saves resources. The cost: because it spends
:math:`\alpha` evenly, the **final test is stricter** than a plain :math:`\alpha = 0.05`
(≈ 0.017 with 4 looks), so it is **less powerful** if the trial runs to completion and
may need a slightly larger sample. Rule of thumb: **Pocock when early stopping is likely**
(business/exploratory A/B), **O'Brien–Fleming when the trial will probably run to the
end** (safety-critical medicine).

----

*Theme:* :ref:`Sequential Methods & Bandits <term-theme-bandits>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Frequentist <059-frequentist>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>`

----

.. hint::
   **More in Sequential Methods & Bandits**

   :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Pocock Method <https://insightful-data-lab.com/2025/08/25/pocock-method/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
