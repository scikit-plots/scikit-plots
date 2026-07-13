:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-target-variable:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Target Variable</b></div>`

=================
Target Variable
=================

*The quantity a supervised model is trained to predict.*

What it is
----------

The **target variable** (:math:`y`) — also called the **dependent**, **response**, or **outcome**
variable, or the **label** — is the quantity a **supervised** model is trained to **predict** from the
input **features** (the independent variables). It is the "correct answer" that must be **observed** in
the training data.

Types
-----

It can be **continuous** (a regression target, e.g. a price) or **categorical** (a classification target,
e.g. spam / not-spam), and also **ordinal** or **multi-label**. Its type **determines the problem** and
which models fit.

Why it matters
--------------

The algorithm only ever learns a **function mapping features to target**, so a **well-defined** target is
decisive: without a labeled target, supervised learning cannot proceed, and a poorly chosen or **biased**
target propagates straight into the model's behavior.

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Feature Values <188-feature-values>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Regression Coefficient <090-regression-coefficient>` · :doc:`Point Forecasts <233-point-forecasts>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Label Noise <354-label-noise>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Target Variable <https://insightful-data-lab.com/2025/08/22/target-variable/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
