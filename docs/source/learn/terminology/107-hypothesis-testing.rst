:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-hypothesis-testing:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Hypothesis Testing</b></div>`

====================
Hypothesis Testing
====================

*A framework for deciding between a null and an alternative using data.*

What it is
----------

**Hypothesis testing** is a formal procedure for **deciding about a population parameter
from sample data while controlling the risk of error**. It is best read not as proof but
as **risk-managed decision-making**.

The two hypotheses
------------------

The **null** :math:`H_0` is the status quo — no effect, zero difference, zero coefficient.
The **alternative** :math:`H_1` is a departure from it, either **two-sided** (:math:`\neq`)
or **one-sided** (:math:`>` or :math:`<`). The test never *proves* :math:`H_1`; it asks
whether the data are **inconsistent with** :math:`H_0`.

The logic
---------

A proof-by-contradiction: **assume** :math:`H_0`, ask how likely data this extreme (or
more) would be under it, and if that likelihood is tiny, **reject** :math:`H_0`. The
machinery is a **test statistic**,

.. math::

   \text{test statistic} = \frac{\text{observed effect} - \text{effect under } H_0}{\text{standard error}},

which under :math:`H_0` follows a **known sampling distribution** (normal for z, t for
t-tests, :math:`\chi^2`, F) — that is what lets us turn it into a **p-value**, the
probability under :math:`H_0` of data as extreme or more.

The two errors
--------------

Decisions can fail two ways: a **Type I error** (reject a true :math:`H_0`, rate
:math:`\alpha`) and a **Type II error** (fail to reject a false one, rate :math:`\beta`),
with **power** :math:`= 1 - \beta`. The two trade off — tightening :math:`\alpha` raises
:math:`\beta`. A **two-sided** test is more conservative; a **one-sided** test has more
power but needs strong prior justification.

What "fail to reject" means
---------------------------

Crucially, failing to reject :math:`H_0` is **not** evidence that :math:`H_0` is true or
that no effect exists — only that the data are **insufficient against** :math:`H_0` at the
chosen :math:`\alpha`. It could be a true null, a **small** effect, or simply **too little
power**. And because large samples make trivial effects significant, always read the
p-value **alongside an effect size and a confidence interval** — significance for whether,
effect size for how much.

----

**Mind map — connected ideas**

   :doc:`Statistical Significance <096-statistical-significance>` · :doc:`Significance Level (α) <105-significance-level>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Effect Size (δ) <106-effect-size>`

----

**More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Hypothesis Testing <https://insightful-data-lab.com/2025/08/24/hypothesis-testing/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner
