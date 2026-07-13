:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-frozen-encoder:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧬&nbsp;&nbsp;<b>Frozen Encoder</b></div>`

================
Frozen Encoder
================

*A pretrained encoder whose weights stay fixed while downstream layers train.*

What it is
----------

A **frozen encoder** is a pretrained model — such as BERT, ResNet or a sentence
transformer — that you reuse purely as a *feature extractor*: its weights are held
fixed ("frozen") and only the small new layers you stack on top (a classifier or
regression head) are trained on your task. The encoder turns inputs into
representations; your head learns to use them.

Why freeze
----------

- **Efficiency** — far fewer trainable parameters means faster training and lower
  memory use.
- **Small-data settings** — freezing prevents a large model from overfitting a
  small labelled set while still borrowing the general knowledge baked into the
  pretrained weights.
- **A staged transfer-learning strategy** — freeze first and train only the head;
  later, optionally unfreeze some layers to adapt the representation more closely.

The frozen-to-fine-tuned spectrum
---------------------------------

- **Fully frozen** — only the head trains.
- **Partially frozen** — unfreeze the top few layers (common with transformers),
  leaving lower, more general layers fixed.
- **Fully fine-tuned** — update every encoder weight on your data (most flexible,
  most data-hungry, easiest to overfit).

Mental model
------------

Treat a frozen encoder exactly like precomputed features. Using fixed Word2Vec or
GloVe vectors and training a small model on top is the same idea: you trust the
representation and spend your limited data learning only the task-specific part.

Where it's used
---------------

- **NLP** — embed text with a frozen BERT, then train a logistic-regression head
  for sentiment.
- **Computer vision** — use a frozen ImageNet-pretrained ResNet50 and train a new
  head for, say, medical images.
- **Recommender systems** — keep large pretrained user/item embeddings fixed and
  train only a lightweight ranking layer on your own interactions.

----

*Theme:* :ref:`Representations & Embeddings <term-theme-repr>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Embedding <173-embedding>` · :doc:`Autoencoder <171-autoencoder>`

----

.. hint::
   **More in Representations & Embeddings**

   :doc:`Autoencoder <171-autoencoder>` · :doc:`Embedding <173-embedding>` · :doc:`Embedding Similarity <320-embedding-similarity>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Frozen Encoder <https://insightful-data-lab.com/2025/08/23/frozen-encoder/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
