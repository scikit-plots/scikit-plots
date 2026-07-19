:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-fair-lending-laws:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔍&nbsp;&nbsp;<b>Fair Lending laws</b></div>`

===================
Fair Lending laws
===================

*Regulations prohibiting discrimination in credit decisions.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What they are
-------------

**Fair lending laws** are the US statutes that **prohibit discrimination** in credit — chiefly the **Equal
Credit Opportunity Act (ECOA)**, covering all credit transactions, and the **Fair Housing Act**, covering
mortgages and housing. They bar decisions based on **protected characteristics** like race, sex, age,
religion, or national origin.

Two theories of harm
--------------------

Enforcement rests on **disparate treatment** (treating applicants differently **on purpose**) and **disparate
impact** (a **facially neutral** rule that disproportionately **burdens** a protected group, even with **no**
intent). Crucially, **statistical disparities alone** can trigger scrutiny before any intent is shown.

Why they matter for ML
----------------------

A credit model can discriminate through **proxies** — zip code or surname standing in for race — so compliance
demands **fairness testing**, sensitivity analysis, and ongoing **monitoring**. It also requires
**adverse-action** explanations specific enough to tell a rejected applicant **why**, which is hard for
**opaque** models — a driver of explainable AI in finance.

----

*Theme:* :ref:`Explainability & Governance <term-theme-xai>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`Basel III <333-basel-iii>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>` · :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>`

----

.. hint::
   **More in Explainability & Governance**

   :doc:`Basel III <333-basel-iii>` · :doc:`Counterfactual Explanations <336-counterfactual-explanations>` · :doc:`High-Stakes Domains <334-high-stakes-domains>` · :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>` · :doc:`Post-hoc Explainability <339-post-hoc-explainability>` · :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Fair Lending laws <https://insightful-data-lab.com/2025/08/20/fair-lending-laws/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
