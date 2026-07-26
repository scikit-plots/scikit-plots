.. _dpa-how-shapley-values-work:

========================================================================
How Shapley Values Work
========================================================================

**Stage 5 · 📈 Regression**  ·  Lesson 37 of 56  ·  *intermediate*

:doc:`◀ Previous · Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`   ·   :doc:`Next · Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds ▶ <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A fair division problem
-------------------------

The feature-importance question has a surprisingly deep answer borrowed from **economics**. Imagine
several players cooperating to earn a joint payout — how should the winnings be split **fairly**,
according to each player's real contribution? Lloyd **Shapley** solved this in 1953, and the **Shapley
value** he defined turns out to be exactly what is needed to attribute a model's **prediction** to its
features.

Marginal contributions
------------------------

The idea rests on **marginal contribution**. A player's contribution to a group (a **coalition**) is
how much the payout **grows** when they join it: :math:`v(S \cup \{i\}) - v(S)` for a coalition
:math:`S`. But that depends on **who is already there** — a player may add a lot to a small group and
little to a large one. The Shapley value resolves this by **averaging** a player's marginal
contribution over **every** possible coalition (equivalently, every order in which players could join).

The formula
-------------

Written out, the Shapley value of feature :math:`i` among :math:`n` features is that weighted average:

.. math::

   \phi_i = \sum_{S \subseteq N \setminus \{i\}}
            \frac{|S|!\,(n - |S| - 1)!}{n!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr].

The weights count the orderings, so each coalition is credited correctly. What makes the Shapley value
special is that it is the **unique** scheme satisfying four fairness axioms: **efficiency** (the parts
sum to the whole), **symmetry** (equal contributors get equal shares), **dummy** (a feature that
changes nothing gets zero), and **additivity**.

Explaining predictions
------------------------

In machine learning the analogy is exact: the **features are the players** and the **prediction is the
payout**. The Shapley value of a feature is how much it pushed *this* prediction **above or below** the
average prediction — a fair, model-agnostic attribution that works for any model, regression or
classification. The popular **SHAP** framework (2017) builds on this, decomposing a prediction into
feature contributions that **sum** to the output. The catch is cost: exact values require all
:math:`2^n` coalitions, so in practice they are **approximated**. Unlike the greedy selection of
earlier lessons, Shapley values weigh every feature **fairly against all others**.

.. hint::

   **Related lessons:** :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`  ·  :doc:`Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`  ·  :doc:`Multiple Linear Regression <32-multiple-linear-regression>`  ·  :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/how-shapley-values-work/ <https://insightful-data-lab.com/2026/01/16/how-shapley-values-work/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: intermediate
