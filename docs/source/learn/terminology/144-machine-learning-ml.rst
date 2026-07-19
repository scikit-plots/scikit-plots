:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-machine-learning-ml:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Machine Learning (ML)</b></div>`

=======================
Machine Learning (ML)
=======================

*Algorithms that learn patterns from data rather than being explicitly programmed.*

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

**Machine learning (ML)** is the branch of AI in which computers **learn patterns from data**
rather than following hand-written rules. You supply **examples**, the model **learns the
relationship between inputs (features) and outputs (labels)**, and once trained it
**predicts** on new, unseen data.

The core idea
-------------

Formally, ML fits a function

.. math::

   y = f(X) + \varepsilon,

where :math:`X` are the input features, :math:`y` the output, :math:`f` the function learned
from data, and :math:`\varepsilon` irreducible noise. Learning means estimating :math:`f`.

The kinds of learning
---------------------

**Supervised** learning uses labelled data — **regression** for continuous targets (house
price), **classification** for categories (spam or not). **Unsupervised** learning works on
unlabelled data — **clustering** (customer segmentation) and **dimensionality reduction** (PCA,
embeddings). **Semi-supervised** mixes a little labelled with much unlabelled data (costly
medical labels). **Reinforcement learning** has an agent learn from rewards by acting in an
environment. And **self-supervised** learning predicts part of the input from the rest (masked
words) — the engine behind modern LLMs.

Workflow and an example
-----------------------

The lifecycle is collect → clean → choose a model → train → evaluate → deploy → **monitor and
retrain**. Train a model on thousands of houses — 1,000 sqft and 3 rooms sold for ``$250,000`` —
and it learns that price rises with size and rooms, so a new 1,200 sqft, 4-room house is
predicted at roughly ``$300,000``. ML matters because it **automates pattern discovery** at a
scale and complexity beyond hand-coded rules.

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Customer Segmentation <033-customer-segmentation>` · :doc:`Embedding <173-embedding>` · :doc:`Regression Coefficient <090-regression-coefficient>` · :doc:`Medical AI <145-medical-ai>`

----

.. hint::
   **More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Neural Networks <287-neural-networks>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Machine Learning (ML) <https://insightful-data-lab.com/2025/08/24/machine-learning-ml/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
