:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-regression-coefficient:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Regression Coefficient</b></div>`

========================
Regression Coefficient
========================

*The estimated effect of a predictor on the response in a regression model.*

What it is
----------

A **regression coefficient** :math:`\beta` quantifies the **relationship between a
predictor** :math:`X` **and the outcome** :math:`Y` in a regression model: how much
:math:`Y` changes when :math:`X` rises by **one unit**, holding all other predictors
fixed.

Simple and multiple regression
------------------------------

In **simple** linear regression, :math:`Y = \beta_0 + \beta_1 X + \varepsilon`, where
:math:`\beta_0` is the **intercept** (the value of :math:`Y` at :math:`X = 0`) and
:math:`\beta_1` the **slope**. In **multiple** regression,
:math:`Y = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k + \varepsilon`, each
:math:`\beta_i` is a **partial** coefficient — the effect of :math:`X_i` *controlling for*
the other predictors.

Reading a coefficient
---------------------

Its **sign** gives direction (positive: :math:`Y` rises with :math:`X`; negative: it
falls), its **magnitude** the strength of the effect, and its **p-value** whether it
differs significantly from 0 (a non-significant coefficient may not contribute). For
example, ``Salary = 30000 + 2000 × years`` says each extra year adds about 2,000 in
salary; ``Price = 50000 + 100 × sqft + 20000 × garage`` says each square foot adds 100
*holding garage fixed*, and a garage adds 20,000 *holding size fixed*.

Standardised and logistic
-------------------------

**Unstandardised** coefficients are in original units (dollars, cm); **standardised** ones
(:math:`\beta^*`) are in standard-deviation units, so effects on different scales can be
compared. In **logistic** regression the coefficients are in **log-odds**; exponentiating
gives an **odds ratio** — e.g. :math:`\beta = 0.7` gives :math:`e^{0.7} \approx 2.0`, so a
one-unit increase roughly **doubles the odds** of the event.

Why it matters
--------------

Regression coefficients are the **parameters of interest** in regression — read in
context, alongside p-values, confidence intervals and effect sizes, and (in multiple
models) always as *partial* effects.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Sample Mean <089-sample-mean>` · :doc:`Frequentist <059-frequentist>` · :doc:`Standard Error (SE) <084-standard-error-se>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Regression Coefficient <https://insightful-data-lab.com/2025/08/25/regression-coefficient/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
