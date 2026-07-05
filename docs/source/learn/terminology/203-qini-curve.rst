:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-qini-curve:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Qini Curve</b></div>`

============
Qini Curve
============

*An uplift-evaluation curve plotting incremental gain against the fraction targeted.*

What it is
----------

The **Qini curve** is the **visual evaluation tool** for uplift models (causal or
incremental-response models). It shows the **incremental benefit** gained by targeting customers
**ranked by predicted uplift score** — the uplift-modelling analogue of an ROC curve, measuring how
well the model finds **persuadables**. The higher and more "bowed upward" it is, the better.

How it's built
---------------

Rank customers by uplift score from highest to lowest, **incrementally target** the top *x*%, and for
each portion compute the **incremental response** — treatment responses minus control responses. The
**cumulative** incremental response is what gets plotted.

Axes and baselines
------------------

The **X-axis** is the proportion of the population targeted (0% to 100%); the **Y-axis** is the
cumulative incremental response. A **random-targeting diagonal** marks the uplift you would expect from
targeting at random, and the **model curve** ideally sits well above it.

Reading it, with an example
-----------------------------

Targeting the top 20% might show a treatment purchase rate of 18% against a control of 12% — a **+6%**
incremental lift on those customers — and so on at 40%, 60%. A **steep early rise** means the model
concentrates persuadables at the top; a **flat, near-diagonal** line means it is no better than
random. The area between the model curve and the diagonal is the **Qini coefficient**.

----

**Mind map — connected ideas**

   :doc:`Uplift Score <204-uplift-score>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Uplift Models <205-uplift-models>`

----

**More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Qini Curve <https://insightful-data-lab.com/2025/08/23/qini-curve/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
