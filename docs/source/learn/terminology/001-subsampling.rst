:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-subsampling:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧪&nbsp;&nbsp;<b>Subsampling</b></div>`

=============
Subsampling
=============

*Selecting a representative subset of rows, often to speed up training or to rebalance class frequencies.*

What it is
----------

**Subsampling** means working with a *subset* of the original data — or, for a
signal, with *fewer samples per second* — instead of the whole thing. It is used
for three broad reasons: to make training and analysis faster, to rebalance
classes, and (in signal processing) to lower a signal's sampling rate.

In machine learning
-------------------

When a dataset is very large or heavily imbalanced, subsampling trades a little
information for a lot of speed or balance:

- **Random subsampling** — draw a random subset of rows (like bootstrapping, but
  *without* replacement).
- **Undersampling the majority class** — shrink the over-represented class so it
  no longer dwarfs the minority class.
- **Cross-validation subsampling** — select different subsets per fold to estimate
  performance.

For example, a 1,000,000-row dataset might be cut to a 100,000-row subsample so
models train in a fraction of the time.

In signal processing
---------------------

Here subsampling is **downsampling** — reducing the sampling rate of a signal, for
instance taking 44.1 kHz audio down to 22.05 kHz. The catch is *aliasing*:
high-frequency content that the lower rate can no longer represent folds back and
masquerades as lower frequencies. The remedy is to apply a **low-pass filter
first**, removing those high frequencies before dropping samples.

Trade-offs
----------

Advantages:

- Faster training and inference, since there is simply less data.
- Lower storage and compute cost.
- Applied to the majority class, it helps balance class proportions.

Disadvantages:

- **Information loss** — discarding data can lower accuracy.
- If the draw is not *stratified*, it can shift the class distribution by accident.
- For signals, dropping samples without filtering injects aliasing noise.

Examples
--------

Subsample a dataset with scikit-learn:

.. code-block:: python

   from sklearn.utils import resample

   X_sub, y_sub = resample(X, y, n_samples=10_000, replace=False, random_state=42)

Downsample a signal with SciPy (its ``resample`` applies an anti-aliasing filter
internally):

.. code-block:: python

   import scipy.signal as sps

   signal_sub = sps.resample(signal, len(signal) // 2)

How it relates to nearby terms
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Term
     - Context
     - What it does
   * - Undersampling
     - Imbalanced classification
     - Removes majority-class samples
   * - Oversampling
     - Imbalanced classification
     - Adds minority-class samples
   * - Subsampling (ML)
     - General
     - Takes a subset for speed or balance
   * - Subsampling (DSP)
     - Signals
     - Lowers the sampling rate (downsampling)

----

**Mind map — connected ideas**

   :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Class Weighting <002-class-weighting>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`Downsampling <368-downsampling>`

----

**More in Imbalanced Learning & Resampling**

   :doc:`Class Weighting <002-class-weighting>` · :doc:`Cluster-based undersampling <007-cluster-based-undersampling>` · :doc:`Downsampling <368-downsampling>` · :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>` · :doc:`Oversampling <004-oversampling>` · :doc:`Random Undersampling <008-random-undersampling>` · :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>` · :doc:`Upsampling <367-upsampling>`

----

*Theme:* :ref:`Imbalanced Learning & Resampling <term-theme-imbalance>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Subsampling <https://insightful-data-lab.com/2025/08/30/subsampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
