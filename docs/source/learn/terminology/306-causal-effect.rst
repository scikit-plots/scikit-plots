:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-causal-effect:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Causal Effect</b></div>`

===============
Causal Effect
===============

*The change in an outcome caused by an intervention, all else equal.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

A **causal effect** is the change in an outcome **directly caused** by a change in a treatment,
**holding everything else constant** — *if X changes, how much does Y change because of X, and not
because of other factors?* This is strictly stronger than **correlation**, which shows only
association; causation means X **produces** the change in Y.

The potential-outcomes view
-----------------------------

The **Rubin causal model** frames it with potential outcomes: :math:`Y(1)` is a unit's outcome if
treated and :math:`Y(0)` its outcome if not, so the causal effect for that unit is
:math:`Y(1) - Y(0)`. Because we can only ever observe **one** of the two for any individual — the
**fundamental problem of causal inference** — we estimate the **average causal effect** instead,
:math:`\text{ACE} = \mathbb{E}[Y(1) - Y(0)]`.

Identifying it
--------------

Every method exists to **control for confounders** — factors that influence both treatment and
outcome. The toolkit runs from **randomised controlled trials** and **matching / stratification**
through **regression adjustment**, **instrumental variables** and **difference-in-differences**, up to
**causal graphs (DAGs)** for reasoning about confounding formally.

Why it matters
--------------

A causal effect is the **true impact** of one variable on another — what would change *if we
intervened*. It tests mechanisms in **science**, tells **policy** whether an intervention actually
works, and measures the real business impact of **marketing, pricing and product** changes.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Uplift <424-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Causal Effect <https://insightful-data-lab.com/2025/08/21/causal-effect/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
