:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-regression-models:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Regression Models</b></div>`

===================
Regression Models
===================

*Models that predict continuous numeric outcomes.*

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

**Regression models** predict a **continuous number** — a price, a temperature, a demand — rather than a
class. They learn a function mapping features to a **real-valued** output, fitting a curve or surface through
the data.

The landscape
-------------

The simplest is **linear regression** (a weighted sum of features), extending to **polynomial**,
**regularized** (ridge, lasso), tree-based (**random forests**, gradient boosting), and **neural**
regressors. The same algorithm family often has both a classification and a regression form.

How they're judged
------------------

Regression is scored by how far predictions land from the truth — **MSE / RMSE**, **MAE**, and **R²** — and
trained to minimize a distance-based **loss**. Because those errors use magnitudes, regression is
**sensitive to outliers**, which is why robust losses and metrics exist.

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Classification Models <294-classification-models>` · :doc:`Linear Models <341-linear-models>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Outlier <307-outlier>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Regression Models <https://insightful-data-lab.com/2025/08/21/regression-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
