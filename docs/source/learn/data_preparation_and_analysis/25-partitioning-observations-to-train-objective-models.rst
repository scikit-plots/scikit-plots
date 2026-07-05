.. _dpa-partitioning-observations-to-train-objective-models:

========================================================================
Partitioning Observations to Train Objective Models
========================================================================

**Stage 4 · 🧩 Sampling, Partitioning & Segmentation**  ·  Lesson 25 of 56  ·  *intermediate*

:doc:`◀ Previous · Linear Congruential Random Number Generator (LCG) <24-linear-congruential-random-number-generator-lcg>`   ·   :doc:`Next · Putting Similar Observations into Clusters <26-putting-similar-observations-into-clusters> ▶`


The temptation to cheat
-------------------------

How well does a model perform? The tempting answer — measure it on the **same data** you trained it on
— is also the **wrong** one. A model can score brilliantly on data it has already seen and still fail
on anything new. To judge a model **objectively**, you must test it on data it has **never
encountered**. That is what partitioning provides.

Train and test
----------------

The basic move is to split the observations into two disjoint sets. The **training set** is used to
**fit** the model; the **test set** (or holdout) is set aside and used **once**, at the end, to
estimate how the model will do on **unseen** data. Common splits give training the larger share —
80/20 or 70/30, sometimes 90/10 when data is plentiful. A third **validation** set (or
**cross-validation**) is used when tuning, so the test set stays untouched until the final verdict.

Overfitting
-------------

The problem the split guards against is **overfitting**: a model so flexible it memorises the training
data's **noise** along with its signal. Such a model looks excellent in training and disappoints in
deployment, because it learned the sample rather than the pattern. A held-out test set exposes this at
once — training accuracy soars while test accuracy stalls. A related danger is **data leakage**, where
information from the test set seeps into training and silently **inflates** the score.

Doing it right
----------------

In practice the split is one line — ``train_test_split`` in scikit-learn, ideally **stratified**
(previous lessons) so both parts stay representative. The discipline is non-negotiable and echoes this
course's standing warning: **report performance on the test set, never the training set**. Every
evaluation metric in Stage 8 assumes the model is being judged on data it has never seen.

.. seealso::

   **Related lessons:** :doc:`Stratified Random Sampling <23-stratified-random-sampling>`  ·  :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`  ·  :doc:`Linear Congruential Random Number Generator (LCG) <24-linear-congruential-random-number-generator-lcg>`  ·  :doc:`Binary Classification Model Evaluation and Threshold Optimization <53-binary-classification-model-evaluation-and-threshold-optimization>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/14/partitioning-observations-to-train-objective-models/ <https://insightful-data-lab.com/2026/01/14/partitioning-observations-to-train-objective-models/>`__

.. tags:: purpose: reference, topic: data preparation, level: intermediate
