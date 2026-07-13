:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-causal-ml-causal-machine-learning:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Causal ML (Causal Machine Learning)</b></div>`

=====================================
Causal ML (Causal Machine Learning)
=====================================

*ML methods that estimate causal effects rather than predictive associations.*

What it is
----------

**Causal ML (causal machine learning)** is the branch of machine learning that estimates
**cause-and-effect relationships**, not merely correlations or predictions. Where traditional ML asks
*"what is likely to happen?"*, causal ML asks *"what will happen* **because of** *this
intervention?"* — making it essential for policy evaluation, medical treatment, marketing and any
decision where an action *changes* the outcome.

The counterfactual problem
----------------------------

Everything rests on **treatment versus control** — the intervention against no intervention. The
difficulty is the **counterfactual problem**: for any individual we observe only *one* outcome
(treated or not), never the other, so the effect must be **estimated**. That effect comes in three
grains: the **ATE** (average treatment effect across the population,
:math:`\text{ATE} = \mathbb{E}[Y(1) - Y(0)]`), the **CATE** (conditional on a subgroup), and the
**ITE** (for a single individual).

Methods
-------

The gold standard is **experimental** — a randomised controlled trial (RCT) assigns treatment at
random, eliminating confounding. When experiments are impossible, **observational** methods step in:
propensity-score matching, inverse-propensity weighting, and doubly-robust estimators that combine
regression with weighting. **ML extensions** add **meta-learners** (S-, T-, X- and R-learners),
**causal trees and forests** for heterogeneous effects, and deep models (Dragonnet, TARNet) — with
libraries such as ``CausalML``, ``EconML`` and ``DoWhy``.

An example
----------

In a marketing email campaign, traditional ML predicts **who will buy**; causal ML predicts **who
will buy because of the email**. If the treatment group buys at 15% and the control at 10%, the
**ATE is a 5% uplift** — and causal ML then estimates **CATE/ITE** to reveal which segments or
individuals respond most.

Benefits and challenges
-----------------------

The payoff is real: it captures **true causal effect** rather than correlation, targets only those
who benefit, and supports **counterfactual reasoning** ("what if?"). The costs are structural — it
needs a **treatment-control design** (experiments or strong assumptions), is **sensitive to
confounding** in observational data, and is harder to explain and validate than standard ML.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Uplift Models <205-uplift-models>` · :doc:`Uplift <424-uplift>` · :doc:`Random Targeting Strategy <196-random-targeting-strategy>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Causal ML (Causal Machine Learning) <https://insightful-data-lab.com/2025/08/23/causal-ml-causal-machine-learning/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
