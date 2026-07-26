:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-uplift-random-forests:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Uplift Random Forests</b></div>`

=======================
Uplift Random Forests
=======================

*An ensemble of trees that estimates individual-level treatment effects.*

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

An **uplift random forest** is a modified random forest built to **estimate the causal effect** of a
treatment directly, not merely to predict an outcome. Instead of modelling :math:`P(Y \mid X)`, it
models the **treatment-versus-control difference**,

.. math::

   \Delta(X) = P(Y = 1 \mid T = 1, X) - P(Y = 1 \mid T = 0, X).

How it works
------------

The trees are **uplift trees**: each split is chosen to **maximise the difference in treatment effect**
between its branches, rather than to maximise class purity. Many such trees are then **averaged in an
ensemble**, exactly as in an ordinary random forest, for stability — and each individual receives an
estimated uplift, the incremental probability change caused by the treatment.

Why use it
-----------

It **handles nonlinear relationships and feature interactions** automatically, **reduces variance**
compared with a single uplift tree, and delivers **individual-level treatment-effect** predictions —
learning how features *modify* the treatment effect, not just how they drive the outcome.

Applications
------------

It powers **marketing** (targeting customers who respond *because of* a campaign), **personalised
medicine** (patients who benefit most from a drug), and **policy** (subgroups most positively
affected). It requires **both treated and control data** — an A/B setup. In an email sign-up campaign,
a standard forest predicts the probability of signing up, while the uplift forest predicts the *extra*
probability caused by the email — separating loyal always-signers (low uplift) from persuadables
(high) and negative reactors (negative uplift).

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Uplift Models <205-uplift-models>` · :doc:`Uplift <424-uplift>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Uplift Random Forests <https://insightful-data-lab.com/2025/08/21/uplift-random-forests/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
