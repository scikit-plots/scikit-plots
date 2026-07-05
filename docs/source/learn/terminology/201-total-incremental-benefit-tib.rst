:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-total-incremental-benefit-tib:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Total Incremental Benefit (TIB)</b></div>`

=================================
Total Incremental Benefit (TIB)
=================================

*The total added benefit from treating a targeted population versus not.*

What it is
----------

**Total incremental benefit (TIB)** is the **overall net gain** from applying a treatment compared to
**not applying it at all** — the absolute improvement an uplift strategy delivers across the whole
population. In uplift modelling it is the **final value of the cumulative incremental gain curve**, at
100% of the targeted population.

The formula
-----------

.. math::

   \text{TIB} = \sum_{i=1}^{N} \left( y_i^{\text{treat}} - y_i^{\text{control}} \right),

summing the treated-minus-untreated outcome over the whole population :math:`N`. Expressed in money,
it becomes **incremental conversions × profit per conversion**.

A worked example
----------------

A promotion runs on 10,000 customers: the treatment group (5,000) makes 700 purchases (14%), the
control group (5,000) makes 500 (10%). That is **200 incremental conversions**; at ``$50`` profit
each, the total incremental benefit is **``$10,000``** — 200 extra purchases the campaign genuinely
caused.

Why it matters
--------------

TIB puts a single number on a campaign's **business value**. It feeds ROI directly —

.. math::

   \text{ROI} = \frac{\text{TIB} - \text{Campaign Cost}}{\text{Campaign Cost}},

and serves as a **benchmark** for comparing uplift models and a basis for **budget allocation**, since
it reflects the true added impact rather than gross outcomes.

----

**Mind map — connected ideas**

   :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`ROI (Return on Investment) <191-roi-return-on-investment>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Treatment Cost <192-treatment-cost>`

----

**More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Total Incremental Benefit (TIB) <https://insightful-data-lab.com/2025/08/23/total-incremental-benefit-tib/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
