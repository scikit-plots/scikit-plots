.. _dl-vectorizing-logistic-regression:

========================================================================
Vectorizing Logistic Regression
========================================================================

**Stage 4 · ⚙️ Backprop & Vectorization**  ·  Lesson 17 of 17  ·  *intermediate*

:doc:`◀ Previous · More Vectorization Examples <16-more-vectorization-examples>`   ·   :doc:`↑ Section <index>`


The whole step, no loops
--------------------------

Everything in this stage now combines into a **single** gradient-descent iteration with **no Python
loop over the data** — not over the :math:`m` examples, not over the :math:`n` features. Both the
forward pass and the gradient computation become a handful of matrix operations on
:math:`X \in \mathbb{R}^{n_x \times m}` and :math:`Y \in \mathbb{R}^{1 \times m}`.

Forward, then backward
------------------------

The forward pass predicts all examples at once; the backward pass forms all gradients at once:

.. code-block:: python

   Z = np.dot(w.T, X) + b           # (1, m)   all pre-activations
   A = sigmoid(Z)                   # (1, m)   all predictions
   dZ = A - Y                       # (1, m)   the clean "prediction - truth"
   dw = (1 / m) * np.dot(X, dZ.T)   # (n_x, 1) averaged weight gradient
   db = (1 / m) * np.sum(dZ)        # scalar   averaged bias gradient
   w = w - alpha * dw
   b = b - alpha * db

The vectorised :math:`\mathrm{d}Z = A - Y` carries the per-example result :math:`\mathrm{d}z = a - y`
across the whole set, and ``np.dot(X, dZ.T)`` sums :math:`x^{(i)}\,\mathrm{d}z^{(i)}` over all examples
in one product.

One loop remains
------------------

This is **one** step of gradient descent. To actually **train**, you repeat it — and that outer loop
over **iterations** (epochs) is the **one loop you cannot vectorise away**, because each step depends
on the parameters the previous step produced. Everything *inside* the step, though, is loop-free.

The gateway to deep networks
------------------------------

That compact block is the whole of logistic regression — a single sigmoid neuron, trained by
vectorised gradient descent. A deep network is the **same pattern repeated**: stack more layers, run
the forward pass and backpropagation through each, and reuse exactly these ideas — the sigmoid (or
ReLU) activation, the cross-entropy cost, the computation graph, and vectorisation. With this stage
complete, you have built every piece a neural network is made of.

.. hint::

   **Related lessons:** :doc:`Vectorization in Logistic Regression <15-vectorization-in-logistic-regression>`  ·  :doc:`More Vectorization Examples <16-more-vectorization-examples>`  ·  :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`  ·  :doc:`What is a Neural Network? <01-what-is-a-neural-network>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/vectorizing-logistic-regression/ <https://insightful-data-lab.com/2025/04/07/vectorizing-logistic-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
