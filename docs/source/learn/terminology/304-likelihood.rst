:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-likelihood:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Likelihood</b></div>`

============
Likelihood
============

*How probable observed data are under a model's parameters.*

What it is
----------

The **likelihood** is the probability of the **observed data given a model's parameters** —
:math:`\mathcal{L}(\theta) = P(\text{data} \mid \theta)`. The twist is perspective: it is read as a function
of the **parameters** :math:`\theta` (with the data fixed), asking *which parameters make what we saw most
probable?*

Maximum likelihood
------------------

**MLE** picks the parameters that **maximize** the likelihood (in practice the **log**-likelihood, since
sums are easier and more stable than products):

.. math::

   \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; \mathcal{L}(\theta).

It is the dominant engine of statistical **inference** — logistic regression, and most classifiers, are fit
this way.

The connection
--------------

Minimizing **cross-entropy** (like **binary cross-entropy**) is exactly **maximizing likelihood** — BCE is
the **negative log-likelihood** of the Bernoulli model. Likelihood **ratios** also underlie optimal
**decision** rules, tying estimation and decision-making together.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Probability Forecasts <235-probability-forecasts>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Correlation <305-correlation>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Likelihood <https://insightful-data-lab.com/2025/08/21/likelihood/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
