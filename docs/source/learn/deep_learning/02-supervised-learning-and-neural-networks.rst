.. _dl-supervised-learning-and-neural-networks:

========================================================================
Supervised Learning and Neural Networks
========================================================================

**Stage 1 · 🧠 Introduction to Deep Learning**  ·  Lesson 02 of 17  ·  *beginner*

:doc:`◀ Previous · What is a Neural Network? <01-what-is-a-neural-network>`   ·   :doc:`Next · Why Deep Learning is Taking Off <03-why-deep-learning-is-taking-off> ▶`


Learning a mapping
--------------------

Nearly all the commercial value of neural networks comes from **supervised learning**: you are given
labelled examples — inputs :math:`x` paired with correct outputs :math:`y` — and you learn the
**function that maps** :math:`x \to y`. House features → price, an ad and a user → click-or-not, an
image → the object it contains: each is the same problem, *learn* :math:`x \to y` *from examples*.

Matching the architecture
--------------------------

The **input type** dictates the network. A **standard (fully connected)** network suits **tabular**
features (home prices, ad click-through). **Convolutional** networks (**CNNs**) suit **images** —
computer vision, photo tagging. **Recurrent** networks (**RNNs**) suit **sequences** — audio → text,
language translation. Rich inputs (autonomous driving from image + radar) often use
**hybrid / custom** designs. Same learning principle, different wiring.

Structured vs unstructured
----------------------------

Data splits into two kinds. **Structured** data is table-like: rows and columns where **each feature
has a clear meaning** (a house's size, a user's age). **Unstructured** data is raw **audio, images,
text**, where the features are pixels or waveform samples or words and **no single feature means much
on its own**. Classical algorithms handle structured data well but **struggle on unstructured** data.

Why it matters
----------------

This is exactly where neural networks changed the game: they made computers **far better at
unstructured data** — speech, vision, language — unlocking applications that were out of reach a
decade ago. Much of the economic value, meanwhile, still comes from **structured** data inside
companies, where accurate predictions on large databases translate directly into money.

.. seealso::

   **Related lessons:** :doc:`What is a Neural Network? <01-what-is-a-neural-network>`  ·  :doc:`Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>`  ·  :doc:`Why Deep Learning is Taking Off <03-why-deep-learning-is-taking-off>`  ·  :doc:`Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2025/04/07/supervised-learning-and-neural-networks/ <https://insightful-data-lab.com/2025/04/07/supervised-learning-and-neural-networks/>`__

.. tags:: purpose: reference, topic: deep learning, level: beginner
