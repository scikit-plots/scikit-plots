:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-true-conversion-rate:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>True Conversion Rate</b></div>`

======================
True Conversion Rate
======================

*The unknown underlying probability that a user converts, estimated from observed conversions.*

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

The **true conversion rate** :math:`p` is the **actual probability that a user in the
whole population converts** (clicks, buys, signs up). It is a **population parameter** —
fixed but unknown. What an experiment actually measures is the **sample conversion rate**
:math:`\hat{p}`, an *estimate* of :math:`p`.

Parameter vs estimate
---------------------

.. math::

   p = \frac{\text{conversions in the population}}{\text{users in the population}},
   \qquad
   \hat{p} = \frac{x}{n},

where :math:`x` is conversions in the sample and :math:`n` the sample size. We can rarely
see the whole population, so we work with :math:`\hat{p}` and quantify its uncertainty.

Example
-------

1,000 users see version A and 50 convert, so :math:`\hat{p}_A = 50/1000 = 0.05` (5%). The
true rate might be :math:`p = 0.052`, but we never observe it exactly — only estimate it.

Confidence interval for p
-------------------------

Because :math:`\hat{p}` carries sampling error, a **Wald confidence interval** brackets
the likely range of :math:`p`:

.. math::

   \text{CI} = \hat{p} \pm z_{\alpha/2}\, \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}.

With :math:`\hat{p} = 0.05, n = 1000` and 95% confidence, the standard error is
:math:`\sqrt{0.05 \times 0.95 / 1000} \approx 0.0069`, giving
:math:`0.05 \pm 1.96 \times 0.0069 \approx [0.036, 0.064]` — we're 95% confident the true
rate lies between **3.6% and 6.4%**.

Why it matters
--------------

In A/B testing we never know either group's true rate; we estimate both with
:math:`\hat{p}` and use a **two-proportion z-test** to judge whether the *observed*
difference is real evidence of a difference in the **true** conversion rates — the actual
quantity of interest.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Frequentist <059-frequentist>` · :doc:`A/B Testing <380-a-b-testing>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `True Conversion Rate <https://insightful-data-lab.com/2025/08/25/true-conversion-rate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
