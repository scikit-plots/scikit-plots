:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-shap-shapley-additive-explanations:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔍&nbsp;&nbsp;<b>SHAP (SHapley Additive exPlanations)</b></div>`

======================================
SHAP (SHapley Additive exPlanations)
======================================

*Attributing a prediction to features using Shapley values from game theory.*

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

**SHAP (SHapley Additive exPlanations)** is a unified framework for explaining the predictions of
**any** machine-learning model, built on **Shapley values** from cooperative game theory (Lloyd
Shapley, 1953). Treat each feature as a **player** in a game and the prediction as the **payout**;
SHAP assigns each feature a **fair share** of the contribution — how much it pushed the prediction up
or down from a baseline.

The additive decomposition
----------------------------

A prediction is decomposed **exactly** into per-feature contributions,

.. math::

   \hat{y} = \phi_0 + \sum_{i=1}^{M} \phi_i,

where :math:`\phi_0` is the **baseline** (the average prediction when no features are known) and
:math:`\phi_i` is the **SHAP value** of feature :math:`i`.

A worked example
----------------

A loan-approval probability of **0.8** against a baseline of **0.5** might break down as Income
**+0.2**, Employment history **+0.1**, Debt ratio **0** and Age **0**, so
:math:`0.8 = 0.5 + 0.2 + 0.1`. Income and employment history raised the approval; the other features
were neutral.

Strengths, visuals, limits
----------------------------

SHAP is **consistent** (a fair allocation), works both **locally and globally**, and is
**model-agnostic or model-specific** (tree models, deep nets, linear), with **force**, **summary** and
**dependence** plots. Its costs: exact Shapley values are **exponential** in the number of features
(SHAP uses approximations), explanations can be **misused** out of context, and **correlated
features** are hard to attribute fairly.

In practice
-----------

.. code-block:: python

   import shap
   import xgboost as xgb
   from sklearn.datasets import fetch_california_housing

   X, y = fetch_california_housing(return_X_y=True, as_frame=True)
   model = xgb.XGBRegressor().fit(X, y)

   explainer = shap.Explainer(model, X)
   shap_values = explainer(X)

   shap.summary_plot(shap_values, X)   # global feature importance
   shap.plots.force(shap_values[0])    # local explanation for one row

----

*Theme:* :ref:`Explainability & Governance <term-theme-xai>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Post-hoc Explainability <339-post-hoc-explainability>` · :doc:`Feature Values <188-feature-values>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Deep Ensembles <335-deep-ensembles>`

----

.. hint::
   **More in Explainability & Governance**

   :doc:`Basel III <333-basel-iii>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Fair Lending laws <332-fair-lending-laws>` · :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>` · :doc:`Post-hoc Explainability <339-post-hoc-explainability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `SHAP (SHapley Additive exPlanations) <https://insightful-data-lab.com/2025/08/20/shap-shapley-additive-explanations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
