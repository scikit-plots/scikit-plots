:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-post-hoc-explainability:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔍&nbsp;&nbsp;<b>Post-hoc Explainability</b></div>`

=========================
Post-hoc Explainability
=========================

*Interpreting an already-trained model rather than building it interpretable.*

What it is
----------

**Post-hoc explainability** means explaining a model's behaviour **after training**, without changing
its internal structure — making **black-box models** (deep nets, ensembles, gradient boosting)
interpretable. *Post-hoc* means *after the fact*: you do not train the model to be interpretable, you
**analyse its outputs afterward**. It contrasts with **intrinsically interpretable** models (linear
regression, small decision trees) that are transparent by design.

Why it matters
--------------

Many high-performing models are **opaque**, yet users, regulators and businesses need to know **why** a
prediction was made — for **debugging**, **trust and transparency**, and **compliance** (finance,
healthcare, GDPR / AI Act).

The techniques
--------------

The toolkit spans **feature importance** (global permutation or gain-based, and local),
**surrogate models** (a simple tree approximating the black box), the local methods **LIME** and
**SHAP**, **visualisations** — partial-dependence plots, ICE plots, and saliency maps / Grad-CAM for
images — and **counterfactual explanations**. For a black-box loan model, SHAP might flag low income
and short employment, while a counterfactual says "two more years of employment would flip the
decision."

Limitations
-----------

Post-hoc explanations are **approximations** of the true model logic, risk being **misleading** (the
faithfulness problem), can be **computationally expensive**, and are **diagnostic only** — not a
substitute for fair training practices.

----

*Theme:* :ref:`Explainability & Governance <term-theme-xai>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>` · :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Feature Values <188-feature-values>` · :doc:`Discriminatory Power <185-discriminatory-power>`

----

.. hint::
   **More in Explainability & Governance**

   :doc:`Basel III <333-basel-iii>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`Fair Lending laws <332-fair-lending-laws>` · :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>` · :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Post-hoc Explainability <https://insightful-data-lab.com/2025/08/20/post-hoc-explainability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
