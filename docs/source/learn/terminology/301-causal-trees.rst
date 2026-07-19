:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-causal-trees:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Causal Trees</b></div>`

==============
Causal Trees
==============

*Decision trees that partition data by differences in treatment effect.*

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

**Causal trees** estimate **heterogeneous treatment effects** — how an intervention's impact **varies** across
subgroups — by **recursively partitioning** the feature space into **leaves** within which the treatment
effect (treated-vs-control outcome difference) is roughly **constant**. Introduced by **Athey and Imbens**,
they adapt decision trees from **prediction** to **causal** estimation.

The honesty trick
-----------------

Unlike a standard tree, a causal tree uses **honest** estimation — one part of the data **chooses** the
splits, a **separate** part **estimates** the effect in each leaf. This prevents the tree from **overfitting**
the same data it split on, giving effect estimates you can build **confidence intervals** around.

What it's for
-------------

Causal trees answer **who benefits** — the core question of **uplift modeling**, personalized medicine, and
policy targeting. Extended to **causal forests** for stability, and to **Bayesian** versions (causal BART)
that return a full **posterior** over each unit's effect for uncertainty-aware decisions.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Uplift Random Forests <302-uplift-random-forests>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Bayesian Inference. <375-bayesian-inference>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Causal Trees <https://insightful-data-lab.com/2025/08/21/causal-trees/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
