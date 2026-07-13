:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-treatment-cost:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Treatment Cost</b></div>`

================
Treatment Cost
================

*The cost of applying an intervention to a unit in an experiment or campaign.*

What it is
----------

In **causal ML and uplift modeling**, **treatment cost** is the expense of applying an intervention
to one individual, customer or unit — the **per-unit cost** of sending a coupon, serving an ad,
granting a discount, or administering a drug. Summed over everyone treated, it becomes the **total
cost of the campaign**.

The formula
-----------

Per unit it is simply the direct cost of the intervention per customer; in total,

.. math::

   \text{Total Treatment Cost} = \text{Cost per unit} \times \text{Number of Treated Customers}.

A worked example
----------------

A retailer sends a ``$5`` coupon to 2,000 customers, so the per-unit cost is ``$5`` and the total
treatment cost is 2,000 × 5 = ``$10,000``. If that campaign produces ``$15,000`` in incremental
revenue, the **net incremental profit** is 15,000 − 10,000 = ``$5,000``.

Why it matters for uplift and ROI
-----------------------------------

Treatment cost is what turns raw uplift into *profit*. Incremental revenue is the extra money earned;
treatment cost is what was spent to earn it; the difference is **incremental profit**, and the ratio
is the **ROI of treatment**. Tracking it prevents targeting customers where **cost exceeds benefit**,
enables **profit-based** (not conversion-only) uplift modeling, and forces attention on **net value**
rather than gross outcomes.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`ROI (Return on Investment) <191-roi-return-on-investment>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Treatment Cost <https://insightful-data-lab.com/2025/08/23/treatment-cost/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
