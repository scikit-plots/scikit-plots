:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-qini-coefficient:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Qini Coefficient</b></div>`

==================
Qini Coefficient
==================

*A normalised summary of the Qini curve measuring uplift-model performance.*

What it is
----------

The **Qini coefficient** is a single-number **performance metric for uplift models** — their
equivalent of AUC. Where AUC asks how well a classifier predicts *who will buy*, the Qini coefficient
asks how well an uplift model ranks people by *who will buy because of the treatment*.

How it's computed
-----------------

Sort customers by predicted uplift (descending), split them into deciles or percentiles, and plot the
**Qini curve** — cumulative incremental response (treatment minus control) against the proportion
targeted. The coefficient is the **normalised area between the model's curve and the random-targeting
diagonal**.

The formula
-----------

.. math::

   \text{Qini} = \frac{\int_0^1 \left( G_{\text{model}}(x) - G_{\text{random}}(x) \right) dx}{\int_0^1 \left( G_{\text{perfect}}(x) - G_{\text{random}}(x) \right) dx}.

It ranges from **0 to 1** — 0.5 and up is decent, and close to 1 is excellent separation.

A worked example, and cousins
-------------------------------

Targeting the top 20%, a random selection might yield +100 purchases while the model yields **+300** —
the model curve sits above random, and the coefficient quantifies that gap. It is the **normalised**
form of the **AUUC** (raw area under the uplift curve), which makes it comparable across datasets:
ROC-AUC is to classification what the Qini coefficient is to uplift.

----

**Mind map — connected ideas**

   :doc:`Qini Curve <203-qini-curve>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Uplift Models <205-uplift-models>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>`

----

**More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Curve <203-qini-curve>`

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Qini Coefficient <https://insightful-data-lab.com/2025/08/19/qini-coefficient/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
