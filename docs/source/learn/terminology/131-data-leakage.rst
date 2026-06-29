:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-data-leakage:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧷&nbsp;&nbsp;<b>Data Leakage</b></div>`

==============
Data Leakage
==============

*When information from outside the training set leaks in, inflating performance.*

What it is
----------

**Data leakage** is when **information from outside the training data slips into training**,
giving the model unfair access to **future or hidden knowledge**. The signature is a model
that looks **excellent in validation** but **collapses on real, unseen data**.

The four types
--------------

**Target leakage**: a feature encodes the answer — predicting loan default from
``debt_collected_after_default``, which only exists *because* of default. **Train-test
contamination**: test information bleeds in via preprocessing — e.g. scaling with a mean and
standard deviation computed over the **whole** dataset instead of the training fold alone.
**Temporal leakage**: using **future** data to predict the past — forecasting January's price
with March's trading volume. **Group leakage**: the **same group** (patient, user, session)
lands in both train and test, so the model just **recognises the group**.

How to prevent it
-----------------

Five guards: fit **preprocessing on the training fold only** and apply it to the rest; **drop
features that wouldn't exist at prediction time**; use **time-aware splits** for temporal
data; use **group-aware CV** (``GroupKFold``, ``StratifiedGroupKFold``) to keep groups
intact; and **monitor after deployment** — a sharp drop from validation to production is the
classic leakage tell.

Why it's dangerous
------------------

Leakage manufactures a **false sense of performance**, masking **overfitting** and poor
generalisation, and in regulated domains like finance and healthcare it can turn into a
**compliance problem** when the model fails on the data that matters.

----

**Mind map — connected ideas**

   :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>` · :doc:`Data Drift <331-data-drift>`

----

**More in Validation & Cross-Validation**

   :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Cross-Validation (CV) <136-cross-validation-cv>` · :doc:`Evaluation Set <355-evaluation-set>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`k-fold cross-validation <135-k-fold-cross-validation>` · :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>` · :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>` · :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>` · :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

----

*Theme:* :ref:`Validation & Cross-Validation <term-theme-validation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Data Leakage <https://insightful-data-lab.com/2025/08/24/data-leakage/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
