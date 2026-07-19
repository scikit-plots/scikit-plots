.. _dl-what-is-a-neural-network:

========================================================================
What is a Neural Network?
========================================================================

**Stage 1 · 🧠 Introduction to Deep Learning**  ·  Lesson 01 of 17  ·  *beginner*

:doc:`Next · Supervised Learning and Neural Networks ▶ <02-supervised-learning-and-neural-networks>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A single neuron
----------------

The atom of a neural network is the **neuron**. It takes inputs, forms a **weighted sum** plus a
**bias**, and passes the result through a **nonlinear activation**:

.. math::

   z = w_1 x_1 + \dots + w_n x_n + b = \mathbf{w}^{\!\top}\mathbf{x} + b, \qquad a = g(z).

Andrew Ng's opening example fits **house size → price** with one neuron whose activation is a
**ReLU** (*rectified linear unit*), :math:`g(z) = \max(0, z)` — flat at zero for negative inputs,
rising linearly after. That single unit is already a tiny predictor.

Stacking into a network
------------------------

Real problems have many inputs. Stack neurons side by side into a **layer**, feed one layer's
outputs into the next, and you have a **network** — Ng's Lego-brick analogy. With features like
**size, number of bedrooms, zip code, wealth** feeding a **hidden layer** that feeds a **price**
output, the middle units can come to represent intermediate ideas such as **family size**,
**walkability** or **school quality**. Every input connects to every hidden unit
(**fully connected**, or **dense**).

Learning the features
----------------------

The crucial point: **you never specify those intermediate concepts**. You supply only input–output
pairs :math:`(x, y)` — sizes and prices — and **gradient descent** discovers whatever hidden-unit
features best predict the target. Given enough units, a network can approximate very complex
mappings; the rest of this course builds that learning machinery from a single neuron up.

Not really a brain
--------------------

The "a neural network is like the brain" line is a **loose analogy** — handy for a first mental
picture, but easily **oversold**. An artificial neuron is a small piece of arithmetic (a weighted
sum and a nonlinearity), not a biological cell. It is more honest to picture a network as a
**flexible, trainable function approximator**.

.. hint::

   **Related lessons:** :doc:`Supervised Learning and Neural Networks <02-supervised-learning-and-neural-networks>`  ·  :doc:`Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>`  ·  :doc:`Why Deep Learning is Taking Off <03-why-deep-learning-is-taking-off>`  ·  :doc:`Computation Graph <11-computation-graph>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/what-is-a-neural-network/ <https://insightful-data-lab.com/2025/04/07/what-is-a-neural-network/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: beginner
