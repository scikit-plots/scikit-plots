:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-post-hoc-power-analysis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Post Hoc Power Analysis</b></div>`

=========================
Post Hoc Power Analysis
=========================

*Computing achieved power after a study from the observed effect — widely criticised as uninformative.*

What it is
----------

**Post-hoc power analysis** computes the **statistical power of a test after the study is
finished**, plugging in the **observed** sample size :math:`n`, the **observed** effect
size :math:`\delta`, and the chosen :math:`\alpha`. It asks: *given what we actually saw,
what was the probability we could have detected an effect?*

Why people run it
-----------------

Usually to interpret a **non-significant** result ("was it real-but-missed, or genuinely
null?"), to satisfy a journal asking about sensitivity, or to judge older studies in a
meta-analysis. The calculation is the a-priori one with the *observed* effect size
substituted in:

.. math::

   \text{Power} = P(\text{reject } H_0 \mid \delta_{\text{observed}}, n, \alpha).

Example
-------

If :math:`H_0` is a 10% conversion rate, the treatment shows a tiny 10.2%, with 1,000 per
group at :math:`\alpha = 0.05`, post-hoc power might be only **12%** — the study was
**underpowered** to detect so small a lift.

The tautology problem
---------------------

The deep flaw: post-hoc power is a **deterministic function of the p-value**, so it adds
nothing. A non-significant result *always* yields low post-hoc power, and a significant
one *always* high — it merely restates the test. Worse, it invites the fallacy
"non-significant + low power ⇒ :math:`H_0` is true," when it only means "this study wasn't
sensitive enough."

Report this instead
-------------------

Rather than post-hoc power, report the **observed effect size** (Cohen's d, a difference
in proportions, an odds ratio) and a **confidence interval** for the effect — these convey
the strength and precision of the result without the circularity.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Frequentist <059-frequentist>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Post Hoc Power Analysis <https://insightful-data-lab.com/2025/08/24/post-hoc-power-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
