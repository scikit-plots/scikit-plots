:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-outlier:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Outlier</b></div>`

=========
Outlier
=========

*An observation far from the bulk of the data.*

What it is
----------

An **outlier** is an **extreme, atypical** value that sits far from the bulk of the data. Outliers arise
from genuine rare events, measurement or data-entry **errors**, or a mixture of populations — and spotting
them matters because they can **distort** an analysis.

What they break
---------------

Outliers hit statistics that use **every** value hardest — the **mean**, the **variance / standard
deviation**, and **squared-error** losses like **MSE**, which square the large residual — while **robust**
measures like the **median** barely move. This gap between mean and median is itself a **signal** of outliers
or skew.

How they're handled
-------------------

Outliers are **detected** (z-scores, the **IQR** rule, distance- or model-based methods), then
**investigated** — a true error is corrected or removed, but a genuine extreme is often **kept** and handled
with **robust** methods or transforms rather than silently discarded.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Z-Score <097-z-score>` · :doc:`Standard Error (SE) <084-standard-error-se>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Outlier <https://insightful-data-lab.com/2025/08/21/outlier/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
