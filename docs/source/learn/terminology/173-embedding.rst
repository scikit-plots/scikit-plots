:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-embedding:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧬&nbsp;&nbsp;<b>Embedding</b></div>`

===========
Embedding
===========

*A dense vector representation that places similar items near each other.*

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

An **embedding** represents a complex object — a word, a sentence, an image, a
user, a product — as a dense vector of numbers in a continuous space, chosen so
that *similar objects land near each other*. Instead of treating categories as
opaque, unrelated symbols, an embedding gives every object coordinates whose
geometry encodes meaning.

Why it matters
--------------

Raw data is high-dimensional and awkward for models: a one-hot encoding of a
100,000-word vocabulary is a 100,000-long vector that is almost entirely zeros and
says nothing about how words relate. An embedding replaces that with a short dense
vector (commonly 50–1000 dimensions) that machine-learning models can compare,
cluster and retrieve over efficiently — and in which *distance carries semantic
meaning*.

Geometry of meaning
-------------------

Because related objects sit close together, relationships often appear as simple
vector arithmetic. The classic word-embedding analogy is that the vector for
*king*, minus *man*, plus *woman*, lands very close to the vector for *queen*:

.. math::

   v(\text{king}) - v(\text{man}) + v(\text{woman}) \approx v(\text{queen}).

How embeddings are learned
--------------------------

- **Supervised** — train on a labelled task and read off an internal layer
  (for example, a fine-tuned transformer for sentiment).
- **Self-supervised** — learn structure from raw data with no labels
  (Word2Vec, GloVe, or an autoencoder's latent code).
- **Contrastive** — pull matching pairs together and push mismatched pairs apart
  (SimCLR for images, CLIP for image–text pairs).

Where it's used
---------------

- **NLP** — word, sentence and document vectors for search and classification.
- **Computer vision** — face recognition and image retrieval.
- **Recommender systems** — represent users and items, then recommend by
  nearest-neighbour lookup in the shared space.
- **Clustering and visualisation** — project embeddings to 2-D (t-SNE, UMAP) to
  reveal structure.
- **Transfer learning** — reuse pretrained embeddings as a strong starting point
  for new tasks.

----

*Theme:* :ref:`Representations & Embeddings <term-theme-repr>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Embedding Similarity <320-embedding-similarity>` · :doc:`Autoencoder <171-autoencoder>` · :doc:`Frozen Encoder <172-frozen-encoder>`

----

.. hint::
   **More in Representations & Embeddings**

   :doc:`Autoencoder <171-autoencoder>` · :doc:`Embedding Similarity <320-embedding-similarity>` · :doc:`Frozen Encoder <172-frozen-encoder>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Embedding <https://insightful-data-lab.com/2025/08/23/embedding/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
