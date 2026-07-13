:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-normalize-in-feature-engineering:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Normalize (in Feature Engineering)</b></div>`

====================================
Normalize (in Feature Engineering)
====================================

*Rescaling features to a common range or distribution.*

What it is
----------

**Normalizing** (feature scaling) rescales numeric features to a **comparable range** so that features with
large magnitudes don't **dominate** the ones with small magnitudes. It changes a feature's **range**, not its
data type.

The two workhorses
------------------

**Min-max scaling** maps values to **[0, 1]**; **standardization (Z-score)** centers to **mean 0, standard
deviation 1**:

.. math::

   x' = \frac{x - \min(x)}{\max(x) - \min(x)} \quad\text{(min-max)}, \qquad x' = \frac{x - \mu}{\sigma} \quad\text{(Z-score)}.

Min-max suits bounded data; Z-score suits Gaussian-ish data and methods like **PCA**. Min-max is
**outlier-sensitive**, so **robust scaling** (median and **IQR**) is used when outliers are present.

When and when not
-----------------

Normalize for **scale-sensitive** models (KNN, SVM, neural nets); it **speeds convergence** and prevents
large-value bias. Don't normalize **one-hot** or categorical columns (they're already 0/1 and it destroys
their meaning), and fit the scaler on the **training set only** to avoid leakage.

----

*Theme:* :ref:`Data Preparation & Features <term-theme-features>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>` · :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>` · :doc:`Outlier <307-outlier>` · :doc:`Z-Score <097-z-score>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in Data Preparation & Features**

   :doc:`Advanced Sorting in Spreadsheets <431-advanced-sorting-in-spreadsheets>` · :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>` · :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Normalize (in Feature Engineering) <https://insightful-data-lab.com/2025/08/20/normalize-in-feature-engineering/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
