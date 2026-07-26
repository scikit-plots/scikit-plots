:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-classification-models:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Classification Models</b></div>`

=======================
Classification Models
=======================

*Models that assign inputs to discrete categories.*

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

**Classification models** predict a **discrete category** — spam or not, which digit, which disease. The
output is a **class label** (often via a probability over classes), and the model learns a **decision
boundary** that separates the classes in feature space.

The landscape
-------------

They range from **linear** ones (**logistic regression**, linear SVM) to **non-linear** ones (**decision
trees**, random forests, **neural networks**, kernel SVMs). Tasks split into **binary** (two classes),
**multiclass** (one of many), and **multilabel** (several at once).

How they're judged
------------------

Because the target is categorical, classification uses metrics like **accuracy**, **precision / recall**,
**F1**, and **AUC** — not squared error — and its **loss functions** are typically **cross-entropy** rather
than a distance. The right metric depends on **class balance** and error costs.

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Regression Models <309-regression-models>` · :doc:`Linear Models <341-linear-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Multiclass Classification <311-multiclass-classification>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Classification Models <https://insightful-data-lab.com/2025/08/21/classification-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
