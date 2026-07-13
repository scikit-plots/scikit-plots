:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-deep-ensembles:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Deep Ensembles</b></div>`

================
Deep Ensembles
================

*Averaging several independently trained networks for accuracy and uncertainty.*

What it is
----------

A **deep ensemble** trains several neural networks **independently** and **aggregates** their
predictions, improving both **accuracy** and **uncertainty estimation**. Each member shares the
architecture but starts from a **different random initialisation** (and data order), so the networks
settle into different modes of the loss landscape; averaging their outputs cancels errors and reduces
variance:

.. math::

   \bar{p}(y \mid x) = \frac{1}{M} \sum_{m=1}^{M} p_{\theta_m}(y \mid x).

Why it works, and uncertainty
-------------------------------

Because independently-initialised networks explore **different functions**, their **disagreement** is
informative. Deep ensembles decompose predictive uncertainty into **aleatoric** (data noise) and
**epistemic** (model) components and produce well-calibrated **predictive intervals** — rivalling
**Bayesian neural networks** while being far simpler to implement.

Calibration and cost
--------------------

They usually still need **calibration** (for example temperature scaling), especially under
**distribution shift**. The main drawback is that cost grows **linearly** with the number of members
:math:`M`, which motivates efficient variants such as **BatchEnsemble**, **snapshot ensembles**, and
spreading members over time.

Where it's used
---------------

Deep ensembles shine wherever **reliable confidence** matters as much as the point prediction —
climate downscaling, **robotic perception** and safe human-robot interaction, and **low-data transfer
learning**.

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Decision Trees <340-decision-trees>` · :doc:`Post-hoc Explainability <339-post-hoc-explainability>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Model Stability <187-model-stability>` · :doc:`Uplift Random Forests <302-uplift-random-forests>` · :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>`

----

.. hint::
   **More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Ensemble <154-ensemble>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Deep Ensembles <https://insightful-data-lab.com/2025/08/20/deep-ensembles/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
