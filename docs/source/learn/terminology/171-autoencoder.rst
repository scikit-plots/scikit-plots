:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-autoencoder:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧬&nbsp;&nbsp;<b>Autoencoder</b></div>`

=============
Autoencoder
=============

*A neural network trained to reconstruct its input through a compressed latent code.*

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

An **autoencoder** is a neural network trained to copy its input to its output
through a deliberate *bottleneck*. Because the network must pass everything it
knows about an example through a representation that is too small to hold the raw
input, it is forced to discover and keep only the structure that actually matters
and to discard noise and redundancy. The small middle representation — the
**latent code** — is the part we usually care about.

How it works
------------

An autoencoder has three pieces:

- an **encoder** :math:`f_\theta` that maps an input :math:`x` to a low-dimensional
  code :math:`z`;
- a **bottleneck** (the latent space) that holds :math:`z`, whose dimension is
  much smaller than the input;
- a **decoder** :math:`g_\phi` that reconstructs an approximation
  :math:`\hat{x}` of the original from :math:`z`.

Encoder and decoder are trained *together* to minimise a reconstruction loss —
typically mean squared error for continuous data or cross-entropy for binary or
categorical data:

.. math::

   z = f_\theta(x), \qquad \hat{x} = g_\phi(z), \qquad
   \min_{\theta,\phi}\; L\big(x,\; g_\phi(f_\theta(x))\big).

Nothing about the target requires labels — the input *is* the target — so an
autoencoder learns in a fully self-supervised way.

Common variants
---------------

- **Denoising** — corrupt the input and ask the network to reconstruct the clean
  version, which forces robust features.
- **Sparse** — penalise the code so that most latent units are inactive for any
  given input.
- **Variational (VAE)** — make the latent space *probabilistic*, which regularises
  it and turns the decoder into a generator of new samples.
- **Convolutional** — build encoder and decoder from convolutional layers, the
  natural choice for images.

Where it's used
---------------

- **Non-linear dimensionality reduction** — a more flexible alternative to PCA.
- **Denoising** of images, audio or text.
- **Anomaly detection** — examples the model reconstructs badly (high error) are
  flagged as unusual, which is useful for fraud, intrusion and fault detection.
- **Representation learning** — the latent code becomes a feature vector for
  downstream models.

Worked example
--------------

Feed a 28×28 handwritten-digit image (784 pixels) through an encoder that
compresses it to a 32-dimensional code, then a decoder that expands it back to
784 pixels. Trained well, the reconstruction looks almost identical to the
original, yet the 32-number code captures the *essence* of the digit — enough to
cluster, search or detect outliers in a fraction of the original space.

----

*Theme:* :ref:`Representations & Embeddings <term-theme-repr>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Embedding <173-embedding>` · :doc:`Embedding Similarity <320-embedding-similarity>` · :doc:`Frozen Encoder <172-frozen-encoder>`

----

.. hint::
   **More in Representations & Embeddings**

   :doc:`Embedding <173-embedding>` · :doc:`Embedding Similarity <320-embedding-similarity>` · :doc:`Frozen Encoder <172-frozen-encoder>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Autoencoder <https://insightful-data-lab.com/2025/08/23/autoencoder/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
