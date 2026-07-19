:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-fixed-horizon-testing:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Fixed-Horizon Testing</b></div>`

=======================
Fixed-Horizon Testing
=======================

*Testing where the sample size is fixed in advance and analysed only at the end.*

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

**Fixed-horizon testing** is the general statistical principle behind the classic
experiment: **decide the sample size** :math:`n` **(or end time) in advance, collect data
to that point, and run the hypothesis test exactly once.** The "horizon" is that
pre-committed stopping point. Applied to an A/B test specifically, this *is* the
**traditional A/B test**.

Why the horizon matters
-----------------------

Fixing it up front is what keeps the statistics honest: the **Type I error stays at**
:math:`\alpha`, **peeking is ruled out** (checking early and stopping on significance
inflates false positives), and the resulting **p-values and confidence intervals remain
valid** under their assumptions.

The procedure
-------------

State :math:`H_0` and :math:`H_1`; choose :math:`\alpha` (say 0.05); run an **a-priori
power analysis** to size the sample; fix the horizon; collect to it; test once; decide.

Example
-------

To detect a **+10% lift** (5% → 5.5%) at :math:`\alpha = 0.05` and **power 0.80**, an
a-priori power analysis calls for **≈ 7,850 users per variant**. You stop at that horizon
and run a single two-proportion z-test: :math:`p \le 0.05` rejects :math:`H_0`, otherwise
you fail to reject.

Strengths and limits
--------------------

It is **simple, widely accepted, and preserves error guarantees**. The price is
**rigidity**: no early stop even when the result is already obvious, wasted samples when
an effect is large, and no continuous monitoring. **Sequential and adaptive** methods
(α-spending, Bayesian updating, bandits) trade some of that simplicity for the ability to
stop early.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Standard Error (SE) <084-standard-error-se>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Fixed-Horizon Testing <https://insightful-data-lab.com/2025/08/25/fixed-horizon-testing/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
