:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-significance-level:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Significance Level (α)</b></div>`

========================
Significance Level (α)
========================

*The tolerated false-positive probability, fixed before testing.*

What it is
----------

The **significance level** :math:`\alpha` is the **threshold probability** for rejecting
the null hypothesis :math:`H_0` — the **maximum risk of a Type I error** (rejecting a true
:math:`H_0`) you are willing to accept. Common choices are :math:`\alpha = 0.05` (the
default), :math:`0.01` (stricter, stronger evidence demanded) and :math:`0.10` (more
lenient).

The decision rule
-----------------

Compute a test statistic and its **p-value**, then compare: if :math:`p \le \alpha`,
**reject** :math:`H_0`; if :math:`p > \alpha`, **fail to reject**. So :math:`\alpha` is
simply the **decision cutoff** fixed in advance.

What α is
---------

It is the **long-run false-positive rate**: at :math:`\alpha = 0.05`, about 5 in 100 tests
of a *true* null will wrongly reject it. Choose it by stakes — **0.01** in medicine,
genetics and other high-stakes settings; **0.05** as a general balance; **0.10** in
exploratory work where missing a real effect costs more than a false alarm.

Tied to the confidence level
----------------------------

Significance and confidence are complements: the **confidence level is** :math:`1 - \alpha`.
An :math:`\alpha = 0.05` test corresponds to **95% confidence** — across repeated
experiments, 95% of the resulting intervals would contain the true parameter.

What it is not
--------------

Three cautions: :math:`\alpha` is **not** the probability that :math:`H_0` is true; it is
chosen **before** the data, not tuned after; and clearing it means **statistically**
significant, not **practically** important — for that you still need an effect size.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Type I Error <080-type-i-error>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Statistical Significance <096-statistical-significance>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Critical Value <087-critical-value>` · :doc:`Effect Size (δ) <106-effect-size>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Significance Level (α) <https://insightful-data-lab.com/2025/08/24/significance-level-%ce%b1/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
