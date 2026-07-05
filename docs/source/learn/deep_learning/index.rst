:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _deep-learning-index:

:raw-html:`<div style="text-align:center"><strong>` 🧠 Deep Learning
|br| From a single neuron to deep networks
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Deep Learning
======================================================================

**Deep learning** stacks simple, differentiable units into networks trained end-to-end by
**gradient descent**. This course follows the ground-up path the source corpus takes — view
**logistic regression as a single neuron**, learn the **computation graph** and
**backpropagation** that train it, then **vectorise** it into the efficient building block that
scales to deep networks — as an ordered, self-contained course of 17 lessons.

Read it at any depth:

* **newcomers** — what a neural network is, and a single neuron;
* **practitioners** — the sigmoid model, the cross-entropy cost, and gradient descent;
* **researchers / engineers** — the computation graph, backpropagation, and full vectorisation with ``numpy``.

.. warning::

   These lessons build the mathematics **from scratch** in ``numpy`` to show what a framework
   does under the hood. In production you would use a framework (**PyTorch** / **Keras**) — its
   automatic differentiation computes exactly these gradients for you.

.. note::

   Follow the lessons in order with **Next ▶**, or jump in by stage below. Snippets use
   real ``numpy`` / ``PyTorch`` / ``Keras`` calls and connect to scikit-plots' ``visualkeras``
   architecture views. This course pairs with the
   :ref:`Terminology reference <terminology-index>` (classification metrics for evaluating networks).

======================================================================

🧠 Stage 1 — Introduction to Deep Learning
------------------------------------------------------------------------

*What a neural network is, where supervised deep learning applies, and why it took off.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 01 · What is a Neural Network?
      :link: dl-what-is-a-neural-network
      :link-type: ref

      How stacked neurons map inputs to outputs, learning the mapping from examples by gradient descent.

   .. grid-item-card:: 02 · Supervised Learning and Neural Networks
      :link: dl-supervised-learning-and-neural-networks
      :link-type: ref

      Learning input→output mappings from labelled data, and which network suits structured, image or sequence inputs.

   .. grid-item-card:: 03 · Why Deep Learning is Taking Off
      :link: dl-why-deep-learning-is-taking-off
      :link-type: ref

      The three drivers — data scale, compute, and algorithms — that made deep networks practical.

   .. grid-item-card:: 04 · Geoffrey Hinton Interview
      :link: dl-geoffrey-hinton-interview
      :link-type: ref

      Perspectives from a pioneer of backpropagation on how the field's key ideas emerged.

🔵 Stage 2 — Logistic Regression as a Neuron
------------------------------------------------------------------------

*A single sigmoid unit — the binary-classification model, its loss and its cost function.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 05 · Binary Classification and Logistic Regression (Neural Network Basics)
      :link: dl-binary-classification-and-logistic-regression-neural-network-basics
      :link-type: ref

      Framing a yes/no prediction, the notation for one training example, and stacking data into matrices.

   .. grid-item-card:: 06 · Logistic Regression (Binary Classification Model)
      :link: dl-logistic-regression-binary-classification-model
      :link-type: ref

      A single neuron: a linear score passed through the sigmoid to output a probability.

   .. grid-item-card:: 07 · Logistic Regression – Loss Function and Cost Function
      :link: dl-logistic-regression-loss-function-and-cost-function
      :link-type: ref

      The per-example cross-entropy loss and the averaged cost that training minimises.

📉 Stage 3 — Derivatives & the Computation Graph
------------------------------------------------------------------------

*The calculus that trains a neuron: gradients, the computation graph and the chain rule.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 08 · Gradient Descent in Logistic Regression
      :link: dl-gradient-descent-in-logistic-regression
      :link-type: ref

      Iteratively stepping the parameters downhill along the cost gradient toward a minimum.

   .. grid-item-card:: 09 · Derivatives
      :link: dl-derivatives
      :link-type: ref

      The slope of a function — the building block of every gradient used to train a network.

   .. grid-item-card:: 10 · More Derivative Examples
      :link: dl-more-derivative-examples
      :link-type: ref

      Worked derivatives that build intuition for how small input changes move the output.

   .. grid-item-card:: 11 · Computation Graph
      :link: dl-computation-graph
      :link-type: ref

      Breaking a computation into elementary steps — the structure backpropagation runs on.

   .. grid-item-card:: 12 · Derivatives with a Computation Graph
      :link: dl-derivatives-with-a-computation-graph
      :link-type: ref

      Applying the chain rule backward through the graph to get every partial derivative efficiently.

⚙️ Stage 4 — Backprop & Vectorization
------------------------------------------------------------------------

*Backpropagation across a whole training set, then vectorising it for speed with numpy.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 13 · Logistic Regression Gradient Descent
      :link: dl-logistic-regression-gradient-descent
      :link-type: ref

      Backpropagation for one example: the derivatives of the loss w.r.t. each weight and the bias.

   .. grid-item-card:: 14 · Gradient Descent on m Training Examples
      :link: dl-gradient-descent-on-m-training-examples
      :link-type: ref

      Averaging the single-example gradients over the whole training set for one update step.

   .. grid-item-card:: 15 · Vectorization in Logistic Regression
      :link: dl-vectorization-in-logistic-regression
      :link-type: ref

      Replacing explicit loops with matrix operations so the whole batch computes at once.

   .. grid-item-card:: 16 · More Vectorization Examples
      :link: dl-more-vectorization-examples
      :link-type: ref

      Further numpy patterns — broadcasting and whole-array functions — that remove Python loops.

   .. grid-item-card:: 17 · Vectorizing Logistic Regression
      :link: dl-vectorizing-logistic-regression
      :link-type: ref

      A fully vectorised forward and backward pass over all examples — the gateway to deep networks.

.. toctree::
   :hidden:
   :maxdepth: 1

   01-what-is-a-neural-network
   02-supervised-learning-and-neural-networks
   03-why-deep-learning-is-taking-off
   04-geoffrey-hinton-interview
   05-binary-classification-and-logistic-regression-neural-network-basics
   06-logistic-regression-binary-classification-model
   07-logistic-regression-loss-function-and-cost-function
   08-gradient-descent-in-logistic-regression
   09-derivatives
   10-more-derivative-examples
   11-computation-graph
   12-derivatives-with-a-computation-graph
   13-logistic-regression-gradient-descent
   14-gradient-descent-on-m-training-examples
   15-vectorization-in-logistic-regression
   16-more-vectorization-examples
   17-vectorizing-logistic-regression

.. tags:: purpose: reference, topic: deep learning
