:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-two-proportion-z-test:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Two-Proportion Z-Test</b></div>`

=======================
Two-Proportion Z-Test
=======================

*A hypothesis test for whether two groups' success proportions differ, using a normal approximation.*

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

The **two-proportion z-test** decides whether an outcome's **rate differs significantly
between two independent groups** — the standard test behind A/B experiments (is control's
5% really below treatment's 6.2%, or just noise?).

Hypotheses
----------

The null is **equality**, :math:`H_0 : p_1 = p_2`; the alternative is
:math:`H_1 : p_1 \neq p_2` (two-tailed) or a one-sided version.

The statistic
-------------

.. math::

   z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}},

where :math:`\hat{p}_i = x_i / n_i` are the group proportions and
:math:`\hat{p} = (x_1 + x_2)/(n_1 + n_2)` is the **pooled** proportion — the shared rate
*assumed under* :math:`H_0`, used to build the standard error. Compare :math:`z` to a
critical value (:math:`\pm 1.96` at :math:`\alpha = 0.05`) or convert it to a p-value.

Worked example
--------------

Control: 100 of 1,000 → :math:`\hat{p}_1 = 0.10`. Treatment: 130 of 1,000 →
:math:`\hat{p}_2 = 0.13`. Pooled :math:`\hat{p} = 230/2000 = 0.115`; standard error
:math:`\sqrt{0.115 \times 0.885 \times 0.002} \approx 0.01425`; so
:math:`z = (0.10 - 0.13)/0.01425 \approx -2.11`, a two-tailed :math:`p \approx 0.035`.
Since :math:`p < 0.05`, reject :math:`H_0` — treatment converts significantly higher.

Assumptions
-----------

Independent samples, binary (success/failure) observations, and samples large enough for
the normal approximation (:math:`np \ge 5` and :math:`n(1 - p) \ge 5`). It powers A/B
tests, clinical recovery-rate comparisons and survey yes/no contrasts alike.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Proportion <091-proportion>` · :doc:`Z-Score <097-z-score>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Statistical Significance <096-statistical-significance>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Two-Proportion Z-Test <https://insightful-data-lab.com/2025/08/24/two-proportion-z-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
