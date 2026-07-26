:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-random-targeting-strategy:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Random Targeting Strategy</b></div>`

===========================
Random Targeting Strategy
===========================

*A random-selection baseline used to benchmark uplift models.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

A **random targeting strategy** selects customers (or units) **at random** for an intervention — a
campaign, treatment or policy — instead of using a predictive or uplift model. In uplift modelling it
is the **baseline** against which a model is judged: if a model cannot beat random targeting, it is
not useful.

The diagonal baseline
-----------------------

Treat a random proportion of the whole population — 20%, 50%, 100% — and, because the choice is
random, the **treatment effect spreads evenly**. On a Qini or uplift curve (proportion targeted on
the X-axis, cumulative incremental gain on the Y-axis) this traces a **straight diagonal line**:
uplift accumulates linearly with the fraction treated.

A worked example
----------------

With 10,000 customers and an average treatment effect of **+5%**, random targeting of 20% (2,000)
yields about **100** incremental conversions; 40% yields about **200**; and 100% yields about **500**
— the campaign's total incremental benefit. The line from (0, 0) to (100%, 500) is the random
baseline.

Role, pros and cons
---------------------

That line is the **benchmark**: a good model's curve starts steep — finding persuadables first — and
stays **above** it throughout; a curve that hugs the diagonal adds no value. Random targeting is
**simple** and **fair** (hence its use in A/B testing), but it **wastes budget** on sure things, lost
causes and do-not-disturbs, delivers lower ROI, and cannot adapt to customer heterogeneity.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Qini Curve <203-qini-curve>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Uplift Models <205-uplift-models>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Random Targeting Strategy <https://insightful-data-lab.com/2025/08/23/random-targeting-strategy/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
