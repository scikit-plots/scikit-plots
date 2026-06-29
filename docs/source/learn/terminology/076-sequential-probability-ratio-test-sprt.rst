:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-sequential-probability-ratio-test-sprt:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎰&nbsp;&nbsp;<b>Sequential Probability Ratio Test (SPRT)</b></div>`

==========================================
Sequential Probability Ratio Test (SPRT)
==========================================

*A sequential test that accumulates the likelihood ratio and stops when it crosses preset bounds.*

What it is
----------

The **sequential probability ratio test (SPRT)**, due to Abraham **Wald**, is a
hypothesis test that **evaluates data as it arrives** rather than fixing the sample size
in advance. It compares two **simple** hypotheses,

.. math::

   H_0 : \theta = \theta_0 \quad \text{vs} \quad H_1 : \theta = \theta_1,

and after each observation decides to **accept** :math:`H_0`, **accept** :math:`H_1`, or
**keep sampling**.

The running likelihood ratio
----------------------------

After :math:`n` observations, accumulate the likelihood ratio

.. math::

   \Lambda_n = \frac{L(\text{data}_{1:n} \mid H_1)}{L(\text{data}_{1:n} \mid H_0)}.

Decision boundaries
-------------------

Two boundaries are set from the desired Type I (:math:`\alpha`) and Type II
(:math:`\beta`) error rates:

.. math::

   A = \frac{1 - \beta}{\alpha}, \qquad B = \frac{\beta}{1 - \alpha}.

Then stop when :math:`\Lambda_n \ge A` (accept :math:`H_1`) or :math:`\Lambda_n \le B`
(accept :math:`H_0`); while :math:`B < \Lambda_n < A`, continue.

Worked example
--------------

For conversion rates :math:`H_0 : p = 0.10` vs :math:`H_1 : p = 0.12` with
:math:`\alpha = 0.05, \beta = 0.20`,

.. math::

   A = \frac{1 - 0.20}{0.05} = 16, \qquad B = \frac{0.20}{0.95} \approx 0.21.

Update :math:`\Lambda_n` as conversions come in; cross 16 → conclude :math:`H_1`, drop
below 0.21 → conclude :math:`H_0`, otherwise keep going.

Strengths and limits
--------------------

On average the SPRT needs **fewer samples** than a fixed-horizon test, stops as soon as
evidence is decisive, and holds the chosen :math:`\alpha` and :math:`\beta`. The catch:
it's built for **simple** hypotheses (fixed :math:`\theta_0, \theta_1`), is awkward for
**composite** ones (e.g. :math:`p \le 0.10`), and needs real-time monitoring. Wald
introduced it for **WWII quality control** (accepting or rejecting munitions lots with
fewer inspections); today it appears in clinical trials, sequential A/B testing and
industrial QC. Versus group-sequential designs (which look at fixed checkpoints), the
SPRT is genuinely **continuous**.

----

**Mind map — connected ideas**

   :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Frequentist <059-frequentist>` · :doc:`Group Sequential Testing <079-group-sequential-testing>`

----

**More in Sequential Methods & Bandits**

   :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`

----

*Theme:* :ref:`Sequential Methods & Bandits <term-theme-bandits>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Sequential Probability Ratio Test (SPRT) <https://insightful-data-lab.com/2025/08/25/sequential-probability-ratio-test-sprt/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
