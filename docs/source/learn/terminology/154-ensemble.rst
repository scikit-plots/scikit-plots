:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-ensemble:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🏋️&nbsp;&nbsp;<b>Ensemble</b></div>`

==========
Ensemble
==========

*Combining several models' predictions to improve accuracy and robustness.*

What it is
----------

An **ensemble** combines **multiple models into one stronger predictor**. The intuition is the
**wisdom of the crowd** — a group of diverse or individually "weak" models, averaged together,
is usually more accurate and more stable than any one of them.

Why use one
-----------

Ensembles **reduce variance** (steadier, less noise-sensitive predictions), can **reduce bias**
(capturing more complex patterns), and **improve robustness** (one model's errors offset by
others). They routinely win competitions like Kaggle for exactly these reasons.

The five strategies
-------------------

**Bagging** (bootstrap aggregating) trains models on different random samples and averages or
votes — **random forest** is bagged decision trees. **Boosting** trains models
**sequentially**, each fixing the last one's errors (AdaBoost, gradient boosting, XGBoost,
LightGBM, CatBoost). **Stacking** trains base models and a **meta-learner** to combine them.
**Voting** takes a majority (hard) or averages probabilities (soft). **Blending** is stacking
with the meta-learner trained on a **holdout set** rather than CV folds.

Trade-offs
----------

The wins — **higher accuracy**, **resistance to overfitting**, complementary patterns from
different algorithms — come at a price: **more compute**, **harder interpretability** than a
single model, and **deployment complexity** (several models mean more latency and OpEx).

----

**Mind map — connected ideas**

   :doc:`Re-scoring <137-re-scoring>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>` · :doc:`Cloud Inference <153-cloud-inference>` · :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>` · :doc:`Model Weights <155-model-weights>`

----

**More in Model Training & Optimization**

   :doc:`Active Learning <163-active-learning>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Early Stopping <140-early-stopping>` · :doc:`Epochs <141-epochs>` · :doc:`FLOPs <156-flops>` · :doc:`Full Annotation <345-full-annotation>` · :doc:`Hyperparameter <142-hyperparameter>` · :doc:`Label Noise <354-label-noise>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Logit Space <291-logit-space>` · :doc:`Logits <420-logits>` · :doc:`Loss Functions <289-loss-functions>` · :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`

----

*Theme:* :ref:`Model Training & Optimization <term-theme-training>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Ensemble <https://insightful-data-lab.com/2025/08/24/ensemble/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
