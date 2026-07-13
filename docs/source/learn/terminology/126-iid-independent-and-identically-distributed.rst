:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-iid-independent-and-identically-distributed:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>IID (Independent and Identically Distributed)</b></div>`

===============================================
IID (Independent and Identically Distributed)
===============================================

*An assumption that samples are mutually independent and share one distribution.*

What it is
----------

**IID** — **independent and identically distributed** — is the bedrock assumption of most
statistics and machine learning. Two conditions: each observation is **independent** (knowing
one tells you nothing about another) and all observations are **identically distributed**
(drawn from the same :math:`P(X)`). Compactly,

.. math::

   X_1, X_2, \dots, X_n \sim \text{i.i.d. } P(X).

Ten fair-coin flips are the canonical case — each flip independent, each with the same
:math:`P(H) = 0.5`.

The two halves
--------------

**Independence** fails when one sample carries information about another. **Identical
distribution** fails when the underlying distribution shifts across the sample. Both can
break separately: data can be dependent but identically distributed, or independent but
drifting.

When it holds and when it doesn't
---------------------------------

The clean case is **random sampling** from a fixed population (survey respondents drawn at
random). It breaks in **time series** (today's stock price depends on yesterday's — not
independent), under a **changing population** (early vs late customers differ — not
identical), and with **grouped data** (several records from the same patient are correlated).

Why it matters
--------------

IID is what makes the math tractable — the **law of large numbers** and the **central limit
theorem** lean on it, and linear/logistic regression, hypothesis tests and freshly
initialised neural nets all assume it. Violating it yields **biased estimates and
overconfident predictions**. In ML the training set is usually assumed IID but often isn't
(autocorrelation, drift, leakage), which is why **time-series CV, grouped CV and domain
adaptation** exist to cope.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>` · :doc:`Time Series <010-time-series>` · :doc:`Probability <025-probability>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `IID (Independent and Identically Distributed) <https://insightful-data-lab.com/2025/08/24/iid-independent-and-identically-distributed/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
