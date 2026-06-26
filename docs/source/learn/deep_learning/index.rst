:html_theme.sidebar_secondary.remove:

..
   ##################################################################
   learn/deep_learning/index.rst
   ==================================================================
   Deep-learning hub, framed for scikit-plots users.  Source context
   (framing only, re-expressed here):
   https://insightful-data-lab.com/category/deep-learning/ (17 posts)
   The source tracks the deeplearning.ai specialization curriculum.
   ------------------------------------------------------------------
   Extensions: sphinx_design, sphinx_tags (bottom), sphinx_copybutton.
   Underlines: = section (overline)  - subsection  ^ subsubsection
   ##################################################################

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _deep-learning-index:

:raw-html:`<div style="text-align:center"><strong>` 🧠 Deep Learning
|br| From a single neuron to deep architectures
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Deep Learning
======================================================================

Deep learning stacks simple, differentiable building blocks into networks
trained end-to-end by **gradient descent**. This hub follows the same
ground-up path as the source corpus: start by viewing **logistic
regression as one neuron**, learn the **computation graph** and
**backpropagation** that train it, then scale up to deep networks, the
techniques that make them work, and the major architectures.

Three levels run through the page:

* **newcomers** — a neuron, a loss, and how gradient descent learns;
* **practitioners** — forward/backward propagation, activations,
  regularization and optimization;
* **researchers / engineers** — CNNs and sequence models, and visualising
  them with scikit-plots' ``visualkeras`` integration.

.. note::

   Open a dropdown for detail and follow **See also** links. Snippets use
   real ``numpy`` / ``PyTorch`` / ``Keras`` calls. This page pairs with
   the :ref:`Terminology reference <terminology-index>` (classification
   metrics for evaluating networks) and the
   :ref:`Data Preparation & Analysis hub <data-preparation-and-analysis-index>`.

----------------------------------------------------------------------

.. _dl-discovery:

Discovery at a Glance
----------------------------------------------------------------------

.. tab-set::
   :class: sd-width-100

   .. tab-item:: 🟢 Start Here — One Neuron
      :sync: level-foundations

      The smallest network, and how it learns.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🔘 Logistic Regression as a Neuron
            :link: dl-neuron
            :link-type: ref
            :class-card: sd-border-1

            A weighted sum, a sigmoid, a probability — the atom of every
            deep net.

         .. grid-item-card:: 🕸️ Computation Graph & Backprop
            :link: dl-compgraph
            :link-type: ref
            :class-card: sd-border-1

            How derivatives flow backward through a chain of operations.

         .. grid-item-card:: ⬇️ Gradient Descent & Vectorization
            :link: dl-gradient-descent
            :link-type: ref
            :class-card: sd-border-1

            The update rule that does the learning — written without
            slow Python loops.

   .. tab-item:: 🔵 Core — Neural Networks
      :sync: level-core

      Stacking neurons into layers.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: ➡️ Forward Propagation
            :link: dl-forward
            :link-type: ref
            :class-card: sd-border-1

            Layer by layer from input to prediction.

         .. grid-item-card:: ⚡ Activation Functions
            :link: dl-activations
            :link-type: ref
            :class-card: sd-border-1

            The nonlinearities (ReLU, sigmoid, tanh) that give depth its
            power.

         .. grid-item-card:: 📉 Loss & Optimization
            :link: dl-loss-optim
            :link-type: ref
            :class-card: sd-border-1

            What the network minimises, and the optimisers that get it
            there.

   .. tab-item:: 🔴 Advanced — Make It Work & Scale It
      :sync: level-advanced

      Generalization techniques and the major architectures.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🛡️ Regularization & Tuning
            :link: dl-regularization
            :link-type: ref
            :class-card: sd-border-1

            Dropout, weight decay, batch norm — and the hyperparameters
            that matter.

         .. grid-item-card:: 🖼️ Convolutional Networks
            :link: dl-cnn
            :link-type: ref
            :class-card: sd-border-1

            Weight sharing for images — convolution, pooling, deep
            backbones.

         .. grid-item-card:: 🔁 Sequence Models
            :link: dl-sequence
            :link-type: ref
            :class-card: sd-border-1

            RNNs, LSTMs and attention for ordered data.

----------------------------------------------------------------------

.. _dl-foundations:

Part 1 — From Logistic Regression to a Neuron
----------------------------------------------------------------------

.. dropdown:: Logistic Regression as a Single Neuron
   :color: primary
   :icon: dot-fill
   :name: dl-neuron
   :open:

   **What is it?**

   A single neuron computes a weighted sum of its inputs, adds a bias, and
   passes the result through a **sigmoid** to produce a probability:

   .. math::

      \hat{y} = \sigma(w^\top x + b), \qquad
      \sigma(z) = \frac{1}{1 + e^{-z}}

   This *is* logistic regression — and the exact unit that, stacked and
   repeated, becomes a deep network. Training minimises **binary
   cross-entropy**:

   .. math::

      \mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}
      \big[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\big]

   .. seealso::

      :ref:`dl-compgraph` · :ref:`terminology-classification-types`

.. dropdown:: Computation Graph & Backpropagation
   :color: primary
   :icon: git-branch
   :name: dl-compgraph

   **What is it?**

   A **computation graph** represents a calculation as nodes
   (operations) and edges (values). **Backpropagation** applies the chain
   rule along this graph *in reverse* to compute the gradient of the loss
   with respect to every parameter efficiently — reusing intermediate
   results rather than recomputing them.

   This reverse-mode automatic differentiation is what frameworks like
   PyTorch and TensorFlow implement under the hood.

   .. seealso::

      :ref:`dl-gradient-descent` · :ref:`dl-forward`

.. dropdown:: Gradient Descent & Vectorization
   :color: primary
   :icon: arrow-down
   :name: dl-gradient-descent

   **The update rule**

   Gradient descent nudges each parameter against the gradient of the
   loss, scaled by a learning rate :math:`\alpha`:

   .. math::

      w := w - \alpha \frac{\partial \mathcal{L}}{\partial w}, \qquad
      b := b - \alpha \frac{\partial \mathcal{L}}{\partial b}

   **Vectorization** replaces per-example Python loops with matrix
   operations over the whole batch — the key to practical speed:

   .. code-block:: python

      import numpy as np

      Z = W @ X + b                 # all examples at once
      A = 1 / (1 + np.exp(-Z))      # sigmoid, vectorized
      dW = (A - Y) @ X.T / m        # gradient over the batch

   .. seealso::

      :ref:`dl-loss-optim`

----------------------------------------------------------------------

.. _dl-networks:

Part 2 — Neural Networks
----------------------------------------------------------------------

.. dropdown:: Forward Propagation
   :color: info
   :icon: arrow-right
   :name: dl-forward

   **What is it?**

   In a network, each layer applies a linear transform followed by a
   nonlinearity, feeding the next layer:

   .. math::

      a^{[l]} = g\big(W^{[l]} a^{[l-1]} + b^{[l]}\big)

   where :math:`a^{[0]} = x` is the input and the final layer's output is
   the prediction. Stacking layers lets the network compose simple
   features into complex ones.

   .. seealso::

      :ref:`dl-activations` · :ref:`dl-compgraph`

.. dropdown:: Activation Functions
   :color: info
   :icon: zap
   :name: dl-activations

   **Why they matter**

   Without a nonlinearity between layers, any stack collapses to a single
   linear map. Common choices:

   * **ReLU** :math:`g(z) = \max(0, z)` — the default for hidden layers;
     cheap and avoids vanishing gradients.
   * **Sigmoid** — squashes to :math:`(0, 1)` for binary output.
   * **Tanh** — zero-centred :math:`(-1, 1)`.
   * **Softmax** — multiclass output probabilities that sum to 1.

   .. seealso::

      :ref:`dl-loss-optim`

.. dropdown:: Loss & Optimization
   :color: info
   :icon: graph-down
   :name: dl-loss-optim

   **What is it?**

   The **loss** quantifies prediction error (cross-entropy for
   classification, MSE for regression). **Optimizers** extend plain
   gradient descent to train faster and more stably:

   * **Mini-batch SGD** — gradients on small batches.
   * **Momentum** — accumulates past gradients to smooth the path.
   * **Adam** — adaptive per-parameter step sizes; a strong default.

   **PyTorch**

   .. code-block:: python

      import torch
      import torch.nn as nn

      model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
      loss_fn = nn.BCEWithLogitsLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   .. seealso::

      :ref:`dl-regularization`

----------------------------------------------------------------------

.. _dl-advanced:

Part 3 — Improving & Scaling Networks
----------------------------------------------------------------------

.. dropdown:: Regularization, Normalization & Tuning
   :color: secondary
   :icon: shield
   :name: dl-regularization

   **Generalization techniques**

   * **L2 / weight decay** — penalise large weights.
   * **Dropout** — randomly zero units during training to prevent
     co-adaptation.
   * **Batch normalization** — standardise layer inputs to stabilise and
     speed up training.
   * **Early stopping** — halt when validation loss stops improving.

   **Hyperparameters that matter most** — learning rate (first), then
   batch size, network width/depth, and regularization strength. Tune on a
   validation split, never the test set.

   .. code-block:: python

      model = nn.Sequential(
          nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(),
          nn.Dropout(0.3), nn.Linear(64, 1),
      )

   .. seealso::

      :ref:`dl-cnn` · :ref:`terminology-calibration`

.. dropdown:: Convolutional Neural Networks (CNNs)
   :color: secondary
   :icon: image
   :name: dl-cnn

   **What is it?**

   CNNs exploit spatial structure by **sharing weights**: a small kernel
   slides across the image (convolution), detecting the same feature
   everywhere, while **pooling** downsamples for translation tolerance.
   Stacking these yields deep backbones (VGG, ResNet, EfficientNet).

   scikit-plots' ``visualkeras`` integration renders these architectures
   as layered diagrams — useful for documentation and review.

   .. code-block:: python

      from tensorflow import keras
      model = keras.Sequential([
          keras.layers.Conv2D(32, 3, activation="relu",
                              input_shape=(28, 28, 1)),
          keras.layers.MaxPooling2D(),
          keras.layers.Flatten(),
          keras.layers.Dense(10, activation="softmax"),
      ])

   .. seealso::

      :ref:`dl-sequence`

.. dropdown:: Sequence Models (RNNs, LSTMs, Attention)
   :color: secondary
   :icon: list-ordered
   :name: dl-sequence

   **What is it?**

   For ordered data (text, audio, time series), **recurrent** networks
   carry a hidden state across steps. **LSTMs / GRUs** add gating to learn
   long-range dependencies, while **attention** / transformer layers let
   every position attend to every other — now the dominant approach for
   language.

   .. seealso::

      :ref:`ts-forecasting` · :ref:`terminology-classification-types`

----------------------------------------------------------------------

.. _dl-skplt-map:

Map to scikit-plots (visualkeras) & Frameworks
----------------------------------------------------------------------

Verified architecture-visualisation galleries and framework docs:

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: visualkeras — Dense network
      :link: https://scikit-plots.github.io/dev/auto_examples/visualkeras/plot_dl_ann_dense.html

      Layered diagram of a fully-connected classifier.

   .. grid-item-card:: visualkeras — Conv1D + Dense
      :link: https://scikit-plots.github.io/dev/auto_examples/visualkeras/plot_dl_ann_conv_dense.html

      Spam classification network, visualised.

   .. grid-item-card:: visualkeras — Autoencoder (CNN)
      :link: https://scikit-plots.github.io/dev/auto_examples/visualkeras/plot_dl_cnn_autoencoder.html

      Encoder/decoder convolutional architecture.

   .. grid-item-card:: visualkeras — ResNetV2
      :link: https://scikit-plots.github.io/dev/auto_examples/visualkeras/plot_dl_cnn_resnetv2.html

      Deep residual backbone diagram.

   .. grid-item-card:: PyTorch
      :link: https://pytorch.org/docs/stable/index.html

      Tensors, autograd and ``torch.nn``.

   .. grid-item-card:: Keras
      :link: https://keras.io/

      High-level model building on TensorFlow / JAX / PyTorch.

----------------------------------------------------------------------

.. _dl-sources:

Sources
----------------------------------------------------------------------

Verified during preparation of this page; resolvable at build date.

**Source context (framing only, re-expressed in our own words)**

* Deep Learning category (17 posts):
  https://insightful-data-lab.com/category/deep-learning/

**Official documentation (API calls used above)**

* PyTorch: https://pytorch.org/docs/stable/index.html
* Keras: https://keras.io/
* TensorFlow: https://www.tensorflow.org/

**scikit-plots (this project)**

* visualkeras examples:
  https://scikit-plots.github.io/dev/auto_examples/index.html
* Terminology reference: :ref:`terminology-index`

**Standard reference**

* Goodfellow, Bengio & Courville, *Deep Learning* (free):
  https://www.deeplearningbook.org/

..
   ##################################################################
   Tags — bottom of page, project controlled vocabulary only.
   `domain: neural network` already exists in the tag index.
   ##################################################################

.. tags::
   purpose: reference,
   domain: neural network,
   model-type: classification,
   level: beginner,
   level: intermediate,
   level: advanced
