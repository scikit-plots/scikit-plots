:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-traditional-a-b-test-fixed-horizon-a-b-test:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Traditional A/B Test (Fixed-Horizon A/B Test)</b></div>`

===============================================
Traditional A/B Test (Fixed-Horizon A/B Test)
===============================================

*A test analysed once at a pre-committed sample size in order to control error rates.*

What it is
----------

A **traditional A/B test** — equivalently a **fixed-horizon A/B test** — is the classical
approach: **predefine a sample size or duration, collect data until that point, and run
the hypothesis test exactly once at the end**. No interim decisions.

The procedure
-------------

1. State the hypotheses (:math:`H_0`: :math:`CR_A = CR_B`).
2. Choose a significance level :math:`\alpha` (typically 0.05).
3. Run an **a-priori power analysis** to find the required sample size :math:`n`.
4. **Fix the horizon** — e.g. "stop at 10,000 users per variant."
5. Collect data to that horizon.
6. Run the test (a **two-proportion z-test** for conversion rates).
7. Reject or fail to reject :math:`H_0`.

The defining feature is **no peeking**: because the analysis happens once, at a
pre-committed sample size, the Type I error stays at :math:`\alpha`.

Example
-------

To test a new button colour with a 5% baseline and a **minimum detectable lift** of +10%,
an a-priori power analysis might call for **~8,000 users per group**. You run until each
arm hits 8,000, then run a two-proportion z-test, and only then decide whether B beats A.

Strengths and limits
--------------------

It is **rigorous, widely understood, and easy to explain**, and it controls Type I error
cleanly when its assumptions hold. The cost is **inflexibility**: you must wait for the
full horizon, which **wastes traffic** when one variant is clearly better early, and it
can't adapt in real time. The modern alternatives relax exactly this — **sequential
testing** (interim looks via α-spending), **Bayesian A/B** (probability of superiority,
peek freely), and **multi-armed bandits** (shift traffic to the winner as you learn).

----

**Mind map — connected ideas**

   :doc:`Type I Error <080-type-i-error>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Treatment Effect <072-treatment-effect>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Traditional A/B Test (Fixed-Horizon A/B Test) <https://insightful-data-lab.com/2025/08/25/traditional-a-b-test-fixed-horizon-a-b-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
