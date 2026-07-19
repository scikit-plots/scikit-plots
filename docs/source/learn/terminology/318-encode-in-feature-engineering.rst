:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-encode-in-feature-engineering:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Encode (in Feature Engineering)</b></div>`

=================================
Encode (in Feature Engineering)
=================================

*Converting categorical or text data into numeric form for models.*

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

**Encoding** converts **categorical** (non-numeric) data into a **numerical** form, because most ML
algorithms operate only on numbers. The trick is to add the numbers **without inventing meaning** that isn't
there.

The methods
-----------

**One-hot encoding** turns a category into several **binary** columns (exactly one 1) — right for **nominal**
categories with **low cardinality**, since it implies no order. **Ordinal / label** encoding assigns
integers, valid only when categories are genuinely **ordered**. For **high-cardinality** features,
**frequency**, **target**, or learned **embedding** encodings avoid the column explosion of one-hot.

Getting it wrong
----------------

Label-encoding a **nominal** variable (red = 1, blue = 2, green = 3) falsely tells the model green > blue — a
fake ordering that distance- and gradient-based models will believe. Match the encoding to whether the
category is **ordered**, and to its **cardinality**.

----

*Theme:* :ref:`Data Preparation & Features <term-theme-features>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>` · :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>` · :doc:`Embedding <173-embedding>` · :doc:`Feature Values <188-feature-values>` · :doc:`Outlier <307-outlier>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in Data Preparation & Features**

   :doc:`Advanced Sorting in Spreadsheets <431-advanced-sorting-in-spreadsheets>` · :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>` · :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Encode (in Feature Engineering) <https://insightful-data-lab.com/2025/08/20/encode-in-feature-engineering/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
