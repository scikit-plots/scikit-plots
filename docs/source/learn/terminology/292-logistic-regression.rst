:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-logistic-regression:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Logistic Regression</b></div>`

=====================
Logistic Regression
=====================

*A linear model mapping features to a probability via the logistic function.*

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

**Logistic regression** is a **linear model for binary classification** that predicts the **probability**
of the positive class. Despite *regression* in the name it **classifies**: it outputs a probability, then
a **threshold** (usually 0.5) assigns the label.

The sigmoid and log-odds
------------------------

It passes a linear combination through the **sigmoid**, squashing any real number into :math:`(0, 1)`:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \theta_0 + \boldsymbol{\theta}^\top \mathbf{x}.

Equivalently, the **log-odds** (logit) — the natural log of the odds :math:`p/(1-p)` — is **linear in the
features**, which is what makes the coefficients interpretable:

.. math::

   \log \frac{p}{1 - p} = \theta_0 + \boldsymbol{\theta}^\top \mathbf{x}.

Fitting by maximum likelihood
-----------------------------

Coefficients are chosen to **maximize the likelihood** of the observed labels — equivalently, to minimize
the **log loss** (negative log-likelihood), a **convex** objective solved by gradient-based methods:

.. math::

   \mathcal{L} = -\sum_i \big[\, y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i) \,\big].

Assumptions and multiclass
--------------------------

It assumes a **linear log-odds** relationship, **few extreme outliers**, and enough data; it extends to
several classes via **one-vs-rest** or **multinomial (softmax)**. Simple, fast and interpretable, it is a
workhorse for spam, fraud and medical diagnosis.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression

   clf = LogisticRegression(max_iter=1000)
   clf.fit(X_train, y_train)
   proba = clf.predict_proba(X_test)[:, 1]   # P(class 1)

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Linear Models <341-linear-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Discriminatory Power <185-discriminatory-power>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Logistic Regression <https://insightful-data-lab.com/2025/08/21/logistic-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
