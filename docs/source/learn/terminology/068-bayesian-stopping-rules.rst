:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-bayesian-stopping-rules:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Bayesian Stopping Rules</b></div>`

=========================
Bayesian Stopping Rules
=========================

*Criteria for ending an experiment based on posterior quantities, such as probability of being best.*

What it is
----------

A **Bayesian stopping rule** is a **predefined criterion — based on posterior
probabilities, Bayes factors, or credible intervals — for when to stop collecting data**
in an experiment. Where a frequentist test fixes the sample size to control the Type I
error rate :math:`\alpha`, a Bayesian rule watches the **strength of evidence** and stops
once it is decisive. In a sentence: *stop when the posterior probability of a hypothesis
is high enough.*

Three kinds of rule
-------------------

- **Posterior-probability threshold** — stop and accept :math:`H_1` once
  :math:`P(H_1 \mid \text{data}) > 0.95`, or stop for **futility** if
  :math:`P(H_0 \mid \text{data}) > 0.95`.
- **Bayes-factor threshold** — stop when the Bayes factor
  :math:`\text{BF} = P(\text{data} \mid H_1) / P(\text{data} \mid H_0)` reaches a strong
  level on Jeffreys' scale (:math:`\text{BF} > 10` for :math:`H_1`,
  :math:`\text{BF} < 1/10` for :math:`H_0`; otherwise keep sampling).
- **Credible-interval rule** — stop when the posterior credible interval for the effect
  **excludes zero** (e.g. a 95% interval of :math:`[0.01, 0.03]` for a conversion-rate
  difference).

Why the peeking is allowed
--------------------------

The headline advantage: **continuous monitoring does not inflate false positives** the
way repeated frequentist peeking does. The results are also directly interpretable —
"there is a 97% probability that B beats A" — priors can be folded in, and overwhelming
evidence lets you stop early and save traffic.

Worked example
--------------

Control converts at 5%; the treatment shows 5.6% after 5,000 users, and the posterior
gives :math:`P(p_B > p_A \mid \text{data}) = 0.97`. With a "stop if posterior > 0.95"
rule, the decision is to **stop and roll out B**.

The catch
---------

Stopping rules depend on the **prior** (which can sway the result), are more
**computationally intensive** (often Monte Carlo / MCMC), and ask stakeholders to think
in posterior probabilities rather than the p-values they may know better. Versus
frequentist fixed-horizon or α-spending designs, the trade is intuitive probability
statements and peeking freedom for prior-sensitivity and compute.

----

**Mind map — connected ideas**

   :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Posterior Probability <073-posterior-probability>` · :doc:`Frequentist <059-frequentist>` · :doc:`A/B Testing <380-a-b-testing>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Bayesian Stopping Rules <https://insightful-data-lab.com/2025/08/25/bayesian-stopping-rules/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
