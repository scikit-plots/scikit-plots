:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-linear-models:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Linear Models</b></div>`

===============
Linear Models
===============

*Models predicting from a weighted sum of features.*

What it is
----------

A **linear model** predicts from a **weighted sum** of the input features, optionally passed through a link
function:

.. math::

   \hat{y} = \mathbf{w}^\top \mathbf{x} + b.

Its defining trait is that it is **linear in the parameters**, which makes it simple, fast, and highly
**interpretable**.

Both tasks
----------

The family spans **regression** (**linear regression**, ridge, lasso) and **classification** (**logistic
regression**, linear SVM), where the linear combination is squashed by a **sigmoid** or **softmax** into
probabilities. In every case the learned **weights** show each feature's direction and strength.

Strengths and limits
--------------------

Linear models are **data-efficient**, **cheap** to train and serve, and **transparent** — but they can only
capture **linear** relationships unless you add **interactions** or feature transforms. They are the natural
**baseline** against which more complex models must justify themselves.

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Classification Models <294-classification-models>` · :doc:`Regression Models <309-regression-models>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Linear Models <https://insightful-data-lab.com/2025/08/20/linear-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
