:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-auuc-area-under-the-uplift-curve:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>AUUC (Area Under the Uplift Curve)</b></div>`

====================================
AUUC (Area Under the Uplift Curve)
====================================

*A summary of uplift-model quality as the area under its uplift curve.*

What it is
----------

**AUUC (Area Under the Uplift Curve)** is the **cumulative incremental gain** of an uplift model,
integrated over the **entire population** — the total area under the uplift curve (incremental gain
against population proportion). It is an **overall** measure of how well the model ranks individuals
by uplift.

How it's computed
------------------

Sort customers by predicted uplift score (descending) and partition them into bins such as deciles.
For each bin, compute the uplift as the treatment response rate minus the control rate,

.. math::

   \text{Uplift}_k = \frac{y^{T}_k}{n^{T}_k} - \frac{y^{C}_k}{n^{C}_k},

then plot cumulative uplift against the fraction of the population targeted; **AUUC is the area under
that curve**.

The formula
-----------

.. math::

   \text{AUUC} = \int_0^1 U(x) \, dx,

where :math:`U(x)` is the cumulative uplift at population fraction :math:`x`. A strong model traces a
steep curve — targeting the top 20% might capture 80% of all achievable incremental responses — while
random targeting hugs the baseline near zero.

AUUC versus the Qini coefficient
---------------------------------

The two are close cousins. **AUUC** is a **raw** area, so it depends on dataset size and response
rate — much like raw accuracy. The **Qini coefficient** is a **normalised** AUUC, scaled between
random and perfect targeting, which makes it **comparable across datasets** — much like AUC. Both are
core metrics for evaluating uplift models.

----

**Mind map — connected ideas**

   :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Uplift Curve <303-uplift-curve>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Uplift <424-uplift>`

----

**More in Causal Inference & Uplift**

   :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `AUUC (Area Under the Uplift Curve) <https://insightful-data-lab.com/2025/08/19/auuc-area-under-the-uplift-curve/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
