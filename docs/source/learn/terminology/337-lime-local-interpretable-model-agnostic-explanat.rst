:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-lime-local-interpretable-model-agnostic-explanations:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔍&nbsp;&nbsp;<b>LIME (Local Interpretable Model-agnostic Explanations)</b></div>`

========================================================
LIME (Local Interpretable Model-agnostic Explanations)
========================================================

*Explaining a prediction by fitting a simple local surrogate model.*

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

**LIME (Local Interpretable Model-agnostic Explanations)** explains an **individual prediction** of
any ML model by approximating the black box **locally** — around the instance of interest — with a
simpler, interpretable model such as linear regression. It answers: *why did the model predict this
for this example?*

How it works
------------

Six steps. Take the instance to explain; create **perturbed samples** by slightly varying its
features; collect the black-box **predictions** for those samples; **weight** each sample by proximity
to the original; fit a **simple interpretable model** (linear or tree) on that local neighbourhood;
and read its **coefficients** as the feature contributions.

A worked example
----------------

For a **denied** loan, perturbing income, age and debt and fitting a local linear model might yield
Income **-0.4**, high debt **+0.3** and employment length **+0.1** — low income plus high debt pushed
the decision toward denial.

Strengths, limits, and SHAP
-----------------------------

LIME is **model-agnostic**, sharply **local** (one prediction at a time) and **human-friendly**, but
it is **unstable** (different perturbations give different explanations), only **locally faithful**,
**computationally expensive**, and shaky under **correlated features**. Against SHAP: LIME fits
**local surrogate** models (faster, less stable) while SHAP uses **game theory** (local *and* global,
more stable, an exact decomposition).

In practice
-----------

.. code-block:: python

   import lime.lime_tabular
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier

   X, y = load_iris(return_X_y=True)
   model = RandomForestClassifier().fit(X, y)

   explainer = lime.lime_tabular.LimeTabularExplainer(
       X,
       feature_names=["f1", "f2", "f3", "f4"],
       class_names=["setosa", "versicolor", "virginica"],
       discretize_continuous=True,
   )
   exp = explainer.explain_instance(X[0], model.predict_proba, num_features=2)
   exp.show_in_notebook()

----

*Theme:* :ref:`Explainability & Governance <term-theme-xai>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Post-hoc Explainability <339-post-hoc-explainability>` · :doc:`Feature Values <188-feature-values>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Deep Ensembles <335-deep-ensembles>`

----

.. hint::
   **More in Explainability & Governance**

   :doc:`Basel III <333-basel-iii>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Fair Lending laws <332-fair-lending-laws>` · :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`Post-hoc Explainability <339-post-hoc-explainability>` · :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `LIME (Local Interpretable Model-agnostic Explanations) <https://insightful-data-lab.com/2025/08/20/lime-local-interpretable-model-agnostic-explanations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
