:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-bayesian-sequential-testing:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Bayesian Sequential Testing</b></div>`

=============================
Bayesian Sequential Testing
=============================

*Continuously monitoring an experiment in a Bayesian framework, valid to stop at any time.*

What it is
----------

**Bayesian sequential testing** evaluates evidence **continuously (or at interim looks)
using Bayesian updating**, instead of frequentist p-values against a fixed
:math:`\alpha`. At every step you compute a **posterior probability** or **Bayes factor**
and decide to **stop for** :math:`H_1`, **stop for** :math:`H_0`, or **keep sampling** —
with **no pre-specified sample size**.

The two evidence metrics
------------------------

- **Posterior probability** — via Bayes' theorem,
  :math:`P(H \mid \text{data}) = P(\text{data} \mid H)\,P(H) / P(\text{data})`.
- **Bayes factor** — :math:`\text{BF} = P(\text{data} \mid H_1)/P(\text{data} \mid H_0)`;
  :math:`\text{BF} > 1` favours :math:`H_1`, with :math:`\text{BF} > 10` and
  :math:`\text{BF} < 1/10` as strong-evidence thresholds.

Decision rules
--------------

- **Stop for efficacy** — :math:`P(H_1 \mid \text{data}) > 0.95` (or
  :math:`\text{BF} > 10`).
- **Stop for futility** — :math:`P(H_0 \mid \text{data}) > 0.95` (or
  :math:`\text{BF} < 1/10`).
- **Continue** — evidence still inconclusive.

The key property: **continuous monitoring does not inflate the Type I error rate**, so
peeking is allowed.

Example
-------

An A/B test with equal priors: after 5,000 visitors the posterior that B beats A is
**0.93** → keep sampling; after 8,000 it reaches **0.97** → stop and conclude B is
better.

Vs frequentist sequential
-------------------------

The upside: no fixed horizon (stop anytime), intuitive **probability statements** ("97%
chance B is better"), prior information can be folded in, and error is controlled
**without α-spending** corrections. The downside: it needs a **prior** (a subjective
choice that can sway results) and more **computation**. Where SPRT and group-sequential
methods speak in p-values and likelihood ratios with α-spending, Bayesian sequential
testing speaks in posteriors and Bayes factors and treats multiple looks as a non-issue.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Frequentist <059-frequentist>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Bayesian Sequential Testing <https://insightful-data-lab.com/2025/08/25/bayesian-sequential-testing/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
