:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-cumulative-uplift:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Cumulative Uplift</b></div>`

===================
Cumulative Uplift
===================

*The running incremental effect as more of the targeted population is treated.*

What it is
----------

**Cumulative uplift** is the **running total of the incremental effect** — extra responses,
conversions or purchases — gained by targeting customers ranked by uplift score. It says how much
total benefit has accrued up to a given proportion of the population, and it is the quantity on the
**Y-axis of the Qini curve**.

The formula
-----------

For the top :math:`k\%` of customers ranked by uplift score,

.. math::

   \text{Cumulative Uplift}(k) = \sum_{i=1}^{kN} \left( y_i^{\text{treatment}} - y_i^{\text{control}} \right),

summing the treated-minus-untreated outcome over those customers, where :math:`N` is the total count.

A worked example
----------------

Rank 1,000 customers (split treatment/control) by uplift score and walk down the deciles. The **top
10%** gives 18 treatment versus 12 control conversions — **+6**, so cumulative uplift is **6**. The
**top 20%** adds 32 versus 24 — **+8**, bringing it to **14**. The **top 30%** adds **+10**, reaching
**24**. At 100% the final cumulative uplift equals the **total incremental benefit**.

Why it matters, and its cousins
---------------------------------

The curve shows **where uplift concentrates** (the top 20% may hold most of the benefit), fixes the
**optimal targeting cutoff** (stop where it flattens), and **evaluates models** — steep early is
good, flat like the random baseline is bad. Its family: **incremental gain** is the per-group step,
**cumulative uplift** (the cumulative incremental gain) the running sum, **total incremental benefit**
the final value, and the **Qini curve** their plot.

----

**Mind map — connected ideas**

   :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Uplift Models <205-uplift-models>`

----

**More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Cumulative Uplift <https://insightful-data-lab.com/2025/08/23/cumulative-uplift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
