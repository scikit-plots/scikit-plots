#!/usr/bin/env python3
# ======================================================================
# term_content.py  —  full, self-contained bodies for terminology pages.
# ----------------------------------------------------------------------
# CONTENT maps an EXACT inventory title -> an RST body (str). The body is
# inserted verbatim into terminology/NNN-<slug>.rst by build_terminology.py,
# between the lead gloss and the "Related" links. Bodies are original
# explanations written in our own words (the source posts are reference
# and framing only). Headings inside a body use '-' (subsection) and '^'
# (subsubsection); the page title above already uses '=' (overline).
#
# Grow this file batch by batch; titles not present here fall back to a
# short "summary" page until their full write-up lands. Keys must match
# term_inventory.tsv exactly (the generator fails fast otherwise).
# ======================================================================

CONTENT: dict[str, str] = {}

# ----------------------------------------------------------------------
# Theme: Representations & Embeddings  (repr)
# ----------------------------------------------------------------------

CONTENT["Autoencoder"] = r"""
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
"""

CONTENT["Embedding"] = r"""
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
"""

CONTENT["Frozen Encoder"] = r"""
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
"""

CONTENT["Embedding Similarity"] = r"""
What it is
----------

**Embedding similarity** is how we measure the closeness of two objects once they
have been turned into embedding vectors: similar objects should have vectors that
are close, so a similarity (or distance) score on the vectors becomes a proxy for
how alike the *objects* are. It is the operation that makes embeddings useful in
practice — search, recommendation and clustering are all "find the nearby
vectors" at heart.

Similarity measures
-------------------

cosine similarity
^^^^^^^^^^^^^^^^^^

The most common choice for semantic vectors. It measures the angle between two
vectors and ignores their magnitude, so it compares *direction* (meaning) rather
than length:

.. math::

   \operatorname{cosine}(u, v) = \frac{u \cdot v}{\lVert u\rVert\,\lVert v\rVert}
   \quad\in\; [-1, 1],

where 1 means identical direction, 0 means unrelated and −1 means opposite.

other measures
^^^^^^^^^^^^^^

- **Dot product** — like cosine but sensitive to magnitude; common inside neural
  networks (attention, matrix factorisation).
- **Euclidean distance (L2)** — straight-line distance; smaller is more similar.
- **Manhattan distance (L1)** — sum of absolute coordinate differences.
- **Jaccard similarity** — for sparse sets rather than dense vectors.
- **Mahalanobis distance** — accounts for feature covariance.

Why it matters
--------------

Reducing "are these two things alike?" to a number on vectors is what powers
semantic search, "people who liked this also liked that" recommendations, face and
image retrieval, and clustering or anomaly detection in embedding space.

In practice
-----------

.. code-block:: python

   import numpy as np
   from sklearn.metrics.pairwise import cosine_similarity

   u = np.array([[0.1, 0.8, 0.5]])
   v = np.array([[0.2, 0.7, 0.4]])

   score = cosine_similarity(u, v)[0, 0]
   print(f"cosine similarity: {score:.2f}")   # ~0.99  ->  very similar
"""


# ----------------------------------------------------------------------
# Theme: Imbalanced Learning & Resampling  (imbalance)
# ----------------------------------------------------------------------

CONTENT["Subsampling"] = r"""
What it is
----------

**Subsampling** means working with a *subset* of the original data — or, for a
signal, with *fewer samples per second* — instead of the whole thing. It is used
for three broad reasons: to make training and analysis faster, to rebalance
classes, and (in signal processing) to lower a signal's sampling rate.

In machine learning
-------------------

When a dataset is very large or heavily imbalanced, subsampling trades a little
information for a lot of speed or balance:

- **Random subsampling** — draw a random subset of rows (like bootstrapping, but
  *without* replacement).
- **Undersampling the majority class** — shrink the over-represented class so it
  no longer dwarfs the minority class.
- **Cross-validation subsampling** — select different subsets per fold to estimate
  performance.

For example, a 1,000,000-row dataset might be cut to a 100,000-row subsample so
models train in a fraction of the time.

In signal processing
---------------------

Here subsampling is **downsampling** — reducing the sampling rate of a signal, for
instance taking 44.1 kHz audio down to 22.05 kHz. The catch is *aliasing*:
high-frequency content that the lower rate can no longer represent folds back and
masquerades as lower frequencies. The remedy is to apply a **low-pass filter
first**, removing those high frequencies before dropping samples.

Trade-offs
----------

Advantages:

- Faster training and inference, since there is simply less data.
- Lower storage and compute cost.
- Applied to the majority class, it helps balance class proportions.

Disadvantages:

- **Information loss** — discarding data can lower accuracy.
- If the draw is not *stratified*, it can shift the class distribution by accident.
- For signals, dropping samples without filtering injects aliasing noise.

Examples
--------

Subsample a dataset with scikit-learn:

.. code-block:: python

   from sklearn.utils import resample

   X_sub, y_sub = resample(X, y, n_samples=10_000, replace=False, random_state=42)

Downsample a signal with SciPy (its ``resample`` applies an anti-aliasing filter
internally):

.. code-block:: python

   import scipy.signal as sps

   signal_sub = sps.resample(signal, len(signal) // 2)

How it relates to nearby terms
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Term
     - Context
     - What it does
   * - Undersampling
     - Imbalanced classification
     - Removes majority-class samples
   * - Oversampling
     - Imbalanced classification
     - Adds minority-class samples
   * - Subsampling (ML)
     - General
     - Takes a subset for speed or balance
   * - Subsampling (DSP)
     - Signals
     - Lowers the sampling rate (downsampling)
"""

CONTENT["Class Weighting"] = r"""
What it is
----------

**Class weighting** rebalances an imbalanced problem *without touching the data*.
Instead of duplicating or deleting rows, it changes the **loss function** so that
mistakes on rare classes cost more than mistakes on common ones. The dataset keeps
its original size; only the penalties change.

Why use it
----------

On imbalanced data a model can score well by simply always predicting the majority
class. Up-weighting the minority class forces the optimiser to take those examples
seriously. Compared with resampling, class weighting adds no synthetic points and
throws nothing away.

How it works
------------

Give each class :math:`c` a weight :math:`w_c` and scale every sample's loss by the
weight of its true class:

.. math::

   L_{\text{weighted}} = \sum_{i=1}^{N} w_{y_i}\,\ell\big(f(x_i),\, y_i\big),

where :math:`y_i` is the true label, :math:`\ell(\cdot)` is the per-sample loss
(for example cross-entropy) and :math:`w_{y_i}` is that class's weight. A standard
choice makes the weight inversely proportional to class frequency:

.. math::

   w_c = \frac{N}{K \cdot n_c},

with :math:`N` total samples, :math:`K` classes and :math:`n_c` samples in class
:math:`c` — so smaller classes receive larger weights.

Trade-offs
----------

Advantages:

- No information loss — every row is kept.
- No synthetic data (unlike SMOTE).
- One-line support in most libraries.

Disadvantages:

- Extreme weights (for very rare classes) can make training unstable.
- A tiny minority class is still hard to learn from, however it is weighted.

Examples
--------

scikit-learn — let the estimator set the weights automatically:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression

   # weights set to N / (K * n_c) per class
   model = LogisticRegression(class_weight="balanced")
   model.fit(X, y)
   # or pass them explicitly, e.g. class_weight={0: 1, 1: 10}

PyTorch — pass a weight tensor to the loss:

.. code-block:: python

   import torch
   import torch.nn as nn

   class_weights = torch.tensor([1.0, 10.0])   # minority class weighted higher
   criterion = nn.CrossEntropyLoss(weight=class_weights)

Weighting vs resampling
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Method
     - How it works
     - Trade-off
   * - Oversampling
     - Duplicate or synthesise minority samples
     - Improves recall, but risks overfitting
   * - Undersampling
     - Drop majority samples
     - Faster and balanced, but loses information
   * - Class weighting
     - Reweight the loss, data unchanged
     - No data loss, but unstable if weights are extreme
"""

# ----------------------------------------------------------------------
# MINDMAP: title -> curated cross-topic links (rendered as a mind-map row).
# Targets must be exact inventory titles (the generator fails fast otherwise).
# ----------------------------------------------------------------------
MINDMAP: dict[str, list[str]] = {
    "Subsampling": [
        "Oversampling", "Random Undersampling",
        "SMOTE (Synthetic Minority Over-sampling Technique)",
        "Class Weighting", "Low-pass Filtering", "Downsampling",
    ],
    "Class Weighting": [
        "Oversampling", "SMOTE (Synthetic Minority Over-sampling Technique)",
        "Random Undersampling", "Subsampling", "Cluster-based undersampling",
    ],
    "Autoencoder": ["Embedding", "Embedding Similarity", "Frozen Encoder"],
    "Embedding": ["Embedding Similarity", "Autoencoder", "Frozen Encoder"],
    "Frozen Encoder": ["Embedding", "Autoencoder"],
    "Embedding Similarity": ["Embedding", "Autoencoder"],
}


CONTENT["SMOTE (Synthetic Minority Over-sampling Technique)"] = r"""
What it is
----------

**SMOTE** (Synthetic Minority Over-sampling Technique), introduced by Chawla and
colleagues in 2002, fixes class imbalance by *synthesising* new minority-class
examples rather than duplicating existing ones. Where plain duplication just copies
points, SMOTE invents plausible new ones in the gaps between real minority samples.

Why not just duplicate
----------------------

Random oversampling repeats the same minority rows, so a classifier can memorise
those exact points and overfit. SMOTE instead places new points *along the lines
between* nearby minority samples, filling out the minority region and pushing the
decision boundary toward something more general.

The algorithm
-------------

For each minority sample :math:`x`:

1. find its :math:`k` nearest minority-class neighbours;
2. pick one neighbour :math:`x_{nn}` at random;
3. create a synthetic point on the segment joining them,

.. math::

   x_{\text{new}} = x + \delta\,(x_{nn} - x), \qquad \delta \sim \mathcal{U}(0, 1).

Because :math:`\delta` is uniform on :math:`[0, 1]`, the new point lands somewhere
between the two originals.

Variants
--------

- **Borderline-SMOTE** — synthesises only near the decision boundary, where
  mistakes are most likely.
- **SMOTE-Tomek / SMOTE-ENN** — pair SMOTE with a cleaning step that removes
  overlapping or noisy points.
- **ADASYN** — generates more synthetic points for the minority samples that are
  hardest to learn.

Trade-offs
----------

Advantages:

- Less overfitting than duplicating samples.
- Smoother, more general boundaries and better minority-class scores.

Disadvantages:

- Can create unrealistic points when the minority distribution is complex.
- May push synthetic points into majority territory, causing class overlap.
- More expensive than simple duplication.

Example
-------

With 100 minority and 1,000 majority samples, SMOTE generates 900 synthetic
minority points, giving a balanced 1,000 vs 1,000.

.. code-block:: python

   from collections import Counter
   from imblearn.over_sampling import SMOTE

   print("before:", Counter(y))
   X_res, y_res = SMOTE(k_neighbors=5, random_state=42).fit_resample(X, y)
   print("after: ", Counter(y_res))
"""

CONTENT["Oversampling"] = r"""
What it is
----------

**Oversampling** is a resampling strategy for class imbalance that *grows the
minority class* — by duplicating real samples or generating new ones — until the
classes are closer to balanced. It is the mirror image of undersampling, which
shrinks the majority class instead.

Why it's used
-------------

When one class is rare (fraud, disease, defects), a model can reach high accuracy
by almost always predicting the majority class while essentially ignoring the
minority. Oversampling raises the minority's presence in training so the model is
forced to learn it, usually improving minority-class recall and F1.

How it's done
-------------

- **Random oversampling** — duplicate existing minority rows at random until the
  counts match.
- **Synthetic oversampling** — create *new* minority points with methods such as
  SMOTE or ADASYN instead of exact copies.

Trade-offs
----------

Advantages:

- Stops the model from ignoring the minority class.
- Random oversampling is trivial to apply.
- Often lifts recall and F1 for the rare class.

Disadvantages:

- Duplicating rows can cause overfitting to those exact points.
- A larger training set means longer training.
- Synthetic methods can introduce unrealistic or noisy samples.

Example
-------

With 10,000 non-fraud and 1,000 fraud rows, oversampling brings the fraud class up
to 10,000 — by duplication (random) or by synthesis (SMOTE).

.. code-block:: python

   from collections import Counter
   from imblearn.over_sampling import RandomOverSampler, SMOTE

   X_dup, y_dup = RandomOverSampler(random_state=42).fit_resample(X, y)
   X_syn, y_syn = SMOTE(random_state=42).fit_resample(X, y)
   print(Counter(y_dup), Counter(y_syn))
"""

CONTENT["Low-pass Filtering"] = r"""
What it is
----------

A **low-pass filter (LPF)** lets *low-frequency* content through while attenuating
*high-frequency* content. In practice that means smoothing a signal and removing
high-frequency noise while keeping the slow-moving structure.

Frequency-domain view
---------------------

An ideal low-pass filter keeps everything below a **cutoff frequency** :math:`f_c`
and removes everything above it:

.. math::

   H(f) = \begin{cases} 1 & |f| \le f_c \\ 0 & |f| > f_c \end{cases}

Real filters approximate this brick wall with a smoother roll-off.

Time-domain view
----------------

Equivalently, low-pass filtering is **convolution with a smoothing kernel** — a
moving average or a Gaussian window — which is exactly the smoothing used in
time-series analysis.

Common filter types
-------------------

- **Ideal** — perfect sharp cutoff (theoretical only).
- **Butterworth** — flat passband, smooth roll-off.
- **Chebyshev** — sharper cutoff at the cost of passband ripple.
- **Digital FIR / IIR** — the workhorses of practical DSP.
- **Moving average** — the simplest crude low-pass filter.

Where it's used
---------------

Removing hiss from audio, blurring images, extracting long-term trends from noisy
time series, isolating frequency bands in communications, and cleaning ECG/EEG
signals in biomedicine.

Example
-------

Daily stock prices wobble with short-term noise; a low-pass filter strips the
wobble and leaves the longer-term trend visible.

.. code-block:: python

   import numpy as np
   from scipy.signal import butter, filtfilt

   t = np.linspace(0, 1, 500)
   signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)   # 5 Hz + 50 Hz
   noisy = signal + 0.3*np.random.randn(len(t))

   b, a = butter(N=4, Wn=0.1)        # cutoff at 0.1 x Nyquist
   clean = filtfilt(b, a, noisy)     # zero-phase filtering
"""

CONTENT["NearMiss (Distance-based Undersampling)"] = r"""
What it is
----------

**NearMiss** is a family of *undersampling* methods that shrink the majority class
intelligently. Instead of dropping majority samples at random, NearMiss chooses
which ones to keep based on their **distance to minority samples**, retaining the
informative, hard-to-classify majority points near the boundary and discarding the
"easy" ones far away.

The three versions
------------------

- **NearMiss-1** — keep the majority samples whose *average distance to their
  :math:`k` nearest minority samples* is smallest (closest to the minority class).
- **NearMiss-2** — keep the majority samples whose *average distance to their
  :math:`k` farthest minority samples* is smallest.
- **NearMiss-3** — for each minority sample, keep a fixed number of its nearest
  majority samples, guaranteeing every minority point is surrounded.

Trade-offs
----------

Advantages:

- Preserves boundary-defining points instead of throwing data away blindly.
- Reduces majority-class bias and can generalise better than naive undersampling.

Disadvantages:

- Distance computations make it more expensive than random undersampling.
- Pruning "easy" samples too hard can overfit the difficult regions.
- Results are sensitive to the choice of :math:`k`.

Example
-------

.. code-block:: python

   from collections import Counter
   from imblearn.under_sampling import NearMiss

   print("before:", Counter(y))
   X_res, y_res = NearMiss(version=1).fit_resample(X, y)   # version 1, 2 or 3
   print("after: ", Counter(y_res))
"""

MINDMAP.update({
    "SMOTE (Synthetic Minority Over-sampling Technique)": [
        "Oversampling", "Random Undersampling", "Class Weighting",
        "Subsampling", "NearMiss (Distance-based Undersampling)",
    ],
    "Oversampling": [
        "SMOTE (Synthetic Minority Over-sampling Technique)", "Random Undersampling",
        "Class Weighting", "Subsampling", "Cluster-based undersampling",
    ],
    "Low-pass Filtering": [
        "Subsampling", "Downsampling", "Signal Processing", "Time Series",
    ],
    "NearMiss (Distance-based Undersampling)": [
        "Random Undersampling", "Cluster-based undersampling", "Subsampling",
        "SMOTE (Synthetic Minority Over-sampling Technique)", "Oversampling",
    ],
})


CONTENT["Cluster-based undersampling"] = r"""
What it is
----------

**Cluster-based undersampling** is a smarter way to shrink the majority class than
dropping rows at random. It first **clusters** the majority samples (typically with
K-means) and then keeps a **representative** from each cluster, so the reduced set
still covers the full spread of the majority class instead of leaving gaps by
chance.

How it works
------------

1. Split the data into majority and minority classes.
2. Cluster the majority class (commonly K-means).
3. From each cluster, keep representatives — the points nearest the centroid, or a
   fixed proportion of the cluster.
4. Combine those with the minority class to form a balanced dataset.

Trade-offs
----------

Advantages:

- Preserves the *structure* of the majority class, so less information is lost.
- Avoids the luck-of-the-draw problem of random undersampling.
- Often classifies better than naive undersampling.

Disadvantages:

- The clustering step adds cost.
- Results depend on the clustering method and the number of clusters :math:`K`.
- It still discards data, so cutting too far risks underfitting.

Example
-------

With 10,000 majority and 1,000 minority samples, cluster the majority into 1,000
clusters and keep one representative per cluster — a balanced 1,000 vs 1,000 that
still spans the majority distribution.

.. code-block:: python

   from collections import Counter
   from imblearn.under_sampling import ClusterCentroids

   print("before:", Counter(y))
   X_res, y_res = ClusterCentroids(random_state=42).fit_resample(X, y)
   print("after: ", Counter(y_res))
"""

CONTENT["Random Undersampling"] = r"""
What it is
----------

**Random undersampling** balances an imbalanced dataset the simplest way possible:
by randomly deleting majority-class rows until the classes are closer in size. It is
the counterpart of oversampling, which adds minority rows rather than removing
majority ones.

Why it's used
-------------

When the majority class dominates (fraud, churn), a classifier can drift toward
always predicting it. Trimming the majority restores balance so the model gives the
minority class real weight — and, as a bonus, trains faster on the smaller set.

How it works
------------

With 10,000 majority and 1,000 minority rows, draw 1,000 of the majority at random
and keep all the minority, yielding a balanced 1,000 vs 1,000.

Trade-offs
----------

Advantages:

- Simple and fast.
- Balances the classes so the minority is not ignored.
- Less data means quicker training.

Disadvantages:

- **Information loss** — useful majority rows are thrown away.
- Risk of **underfitting** from training on fewer points.
- Poor when the minority class is tiny, since too much majority data is discarded.

Smarter alternatives
--------------------

- **Random oversampling** or **SMOTE** — grow the minority instead of shrinking the
  majority.
- **Tomek links / Edited Nearest Neighbours** — remove borderline or overlapping
  majority points rather than random ones.
- **Ensemble methods** — combine undersampling with bagging/boosting, e.g. a
  Balanced Random Forest.

Example
-------

.. code-block:: python

   from collections import Counter
   from imblearn.under_sampling import RandomUnderSampler

   print("before:", Counter(y))
   X_res, y_res = RandomUnderSampler(random_state=42).fit_resample(X, y)
   print("after: ", Counter(y_res))
"""

MINDMAP.update({
    "Cluster-based undersampling": [
        "Random Undersampling", "NearMiss (Distance-based Undersampling)",
        "Subsampling", "Oversampling",
        "SMOTE (Synthetic Minority Over-sampling Technique)",
    ],
    "Random Undersampling": [
        "Oversampling", "SMOTE (Synthetic Minority Over-sampling Technique)",
        "NearMiss (Distance-based Undersampling)", "Cluster-based undersampling",
        "Class Weighting", "Subsampling",
    ],
})


# ----------------------------------------------------------------------
# Theme: Signal Processing & Time Series  (signal)
# ----------------------------------------------------------------------

CONTENT["Signal Processing"] = r"""
What it is
----------

**Signal processing** is the discipline of representing, transforming and pulling
information out of *signals* — any quantity that carries information about a
phenomenon, usually as a function of time or space. A signal can be
**continuous-time**, written :math:`x(t)`, or **discrete-time** (sampled), written
:math:`x[n]`. Almost any sequential or spatial measurement is a signal: audio
waveforms, image pixel grids (2-D signals), biomedical traces (ECG, EEG, EMG),
radio / Wi-Fi / 5G transmissions, and sensor streams (accelerometer, gyroscope,
temperature).

What you do with a signal
-------------------------

The recurring goals are:

- **Filtering** — remove noise or isolate a band of interest.
- **Transformation** — move the signal to a domain where the task is easier (most
  often the frequency domain).
- **Feature extraction** — derive compact descriptors (e.g. spectral features for
  speech recognition).
- **Compression** — shrink the data while preserving what matters (MP3, JPEG).
- **Restoration** — undo degradation (denoising, deblurring).

Three ways to look at a signal
------------------------------

time domain
^^^^^^^^^^^

Work directly on the samples in order — smoothing with a moving average, detecting
peaks, computing energy. Intuitive, but it hides periodic structure.

frequency domain
^^^^^^^^^^^^^^^^

Decompose the signal into the sinusoids that compose it using the **Fourier
transform**. For sampled data this is the **Discrete Fourier Transform**, computed
efficiently by the **FFT**:

.. math::

   X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}.

This is where "mostly 5 Hz with a little 50 Hz noise" becomes visible and you can
filter accordingly.

time-frequency domain
^^^^^^^^^^^^^^^^^^^^^^

When the frequency content *changes over time* (speech, music), one global
spectrum is not enough. The **Short-Time Fourier Transform (STFT)** and the
**wavelet transform** produce a spectrogram — frequency content as a function of
time.

Other standard tools: the **Laplace transform** (continuous-time systems), the
**Z-transform** (discrete-time systems), and **FIR / IIR digital filters**.

The digital pipeline (DSP)
--------------------------

Modern signal processing is mostly digital, in four stages:

1. **Sampling** — analog to discrete samples. The **Nyquist-Shannon** theorem
   requires the sampling rate to exceed twice the highest frequency,
   :math:`f_s > 2 f_{\max}`, or information is lost.
2. **Quantization** — round continuous amplitudes to discrete levels (adding small
   quantization noise).
3. **Processing** — filter, transform, compress, analyse.
4. **Reconstruction** — convert back to continuous form if needed.

Pitfalls and edge cases
-----------------------

- **Aliasing** — sampling too slowly makes high frequencies masquerade as low
  ones; always low-pass filter *before* downsampling.
- **Spectral leakage** — the FFT assumes the segment repeats exactly, so a
  non-integer number of cycles smears energy across bins. Apply a **window**
  (Hann, Hamming) to reduce it.
- **Edge effects** — filters need warm-up samples; the very start and end of a
  filtered signal are unreliable (use zero-phase ``filtfilt`` or drop the
  transient).
- **Non-stationarity** — a single global spectrum misleads when the statistics
  drift; prefer time-frequency methods.

Worked example — find the dominant frequency
--------------------------------------------

.. code-block:: python

   import numpy as np

   fs = 500                                   # sampling rate (Hz)
   t = np.arange(0, 2, 1 / fs)
   x = np.sin(2 * np.pi * 5 * t) + 0.3 * np.random.randn(t.size)

   spectrum = np.fft.rfft(x)                  # FFT of a real signal
   freqs = np.fft.rfftfreq(t.size, d=1 / fs)
   dominant = freqs[np.argmax(np.abs(spectrum))]
   print(f"dominant frequency ~ {dominant:.1f} Hz")   # ~5.0 Hz

Connection to data science and ML
---------------------------------

Signal processing is the front end of many ML pipelines: speech becomes **MFCC**
features, images are filtered by **convolutions** (a CNN layer *is* a learned bank
of filters), and ECG / EEG become time-frequency features for classification.
Seen this way, CNNs, RNNs and Transformers are advanced, *learned*
signal-processing pipelines.
"""

CONTENT["Time Series"] = r"""
What it is
----------

A **time series** is a sequence of observations recorded in time order, usually at
equal spacing:

.. math::

   Y_t, \quad t = 1, 2, \dots, T,

where :math:`Y_t` is the value at time :math:`t`. The defining feature is that
**order matters** — rows are not exchangeable the way they are in ordinary tabular
data, because each value is related to those around it. Everyday examples include
daily stock prices, monthly unemployment, hourly temperature, 15-minute
electricity usage and a patient's heart rate over time.

The four components
-------------------

A series is often decomposed into structure plus randomness:

- **Trend** — long-run drift up or down (rising house prices).
- **Seasonality** — a repeating pattern of *fixed* period (sales peaking every
  December).
- **Cyclic** — wave-like swings of *no fixed* length (business cycles).
- **Noise** — the residual that trend and seasonality do not explain.

Additively, :math:`Y_t = \text{Trend}_t + \text{Seasonality}_t + \text{Noise}_t`
(a multiplicative form fits when the seasonal swing grows with the level).

Types
-----

- **Univariate** vs **multivariate** — one series, or several measured together
  (sales + temperature + promo spend).
- **Stationary** vs **non-stationary** — a stationary series has constant mean and
  variance with no trend or seasonality. Most real data is non-stationary;
  differencing or detrending is used to make it stationary, and a test such as the
  **Augmented Dickey-Fuller (ADF)** test checks it.

Modelling approaches
--------------------

- **Classical statistics** — AR, MA, ARMA, **ARIMA** (adds differencing for
  trend), **SARIMA** (adds seasonality) and **exponential smoothing**
  (Holt-Winters).
- **General ML on lag features** — reshape the series into a supervised table of
  lagged values and fit Random Forest or gradient boosting (XGBoost).
- **Deep learning** — **RNN / LSTM / GRU**, **Temporal Convolutional Networks**,
  and Transformer variants (Informer, TimesNet) for long or multivariate series.

Forecasting vs analysis
-----------------------

Two distinct goals: **forecasting** predicts future values (tomorrow's
temperature), while **analysis** explains structure — detecting trend, decomposing
seasonality, or flagging anomalies in a sensor stream.

Evaluation metrics
------------------

Forecasts are scored with **MAE**, **MSE**, **RMSE**, and percentage errors
**MAPE**, **sMAPE** and **WAPE**. Choose carefully: MAPE explodes when actual
values are near zero and penalises over- and under-prediction asymmetrically;
sMAPE and WAPE are more robust for intermittent or zero-heavy data.

Pitfalls and edge cases
-----------------------

- **Leakage from random splits** — the single biggest mistake. A shuffled
  train/test split lets the model peek at the future. Always split *by time*
  (train on the past, test on the later portion) with a temporal / rolling-window
  scheme.
- **Look-ahead features** — every feature must be computable from information
  available *at prediction time*; rolling statistics must not include the current
  or future point.
- **Autocorrelation** — residuals are usually correlated in time, which breaks the
  i.i.d. assumption behind ordinary error bars.
- **Non-stationarity and regime change** — relationships drift, so a model fit on
  old data can silently degrade.
- **Seasonality mismatch** — the model must capture the right period (weekly *and*
  yearly cycles can coexist).

Worked example — decompose, then split by time
----------------------------------------------

.. code-block:: python

   import pandas as pd
   from statsmodels.tsa.seasonal import seasonal_decompose

   # y: a pandas Series on a daily DatetimeIndex with weekly seasonality
   parts = seasonal_decompose(y, model="additive", period=7)
   trend, seasonal, resid = parts.trend, parts.seasonal, parts.resid

   # temporal hold-out: NEVER shuffle a time series
   split = int(len(y) * 0.8)
   train, test = y.iloc[:split], y.iloc[split:]
"""

MINDMAP.update({
    "Signal Processing": [
        "Low-pass Filtering", "Subsampling", "Downsampling",
        "Time Series", "Time Series Forecasting",
    ],
    "Time Series": [
        "Time Series Forecasting",
        "ARIMA (AutoRegressive Integrated Moving Average)",
        "Seasonality", "Temporal autocorrelation (Serial Correlation)",
        "Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)",
        "Signal Processing",
    ],
})


# ----------------------------------------------------------------------
# Theme: Classification & Averaging Metrics  (metrics)
# ----------------------------------------------------------------------

CONTENT["Multi-label Classification"] = r"""
What it is
----------

In **multi-label classification** each sample can carry *several* labels at once.
This is different from **multi-class** classification, where every sample gets
exactly one label from a set. Here the labels are **not mutually exclusive** — the
model makes an independent yes/no decision for every class:

.. math::

   f : X \;\rightarrow\; \{0, 1\}^K,

so for :math:`K` classes the output is a length-:math:`K` vector of binary
decisions. A movie can be *Action + Comedy*, a news story *Politics + Economy*, a
patient *diabetic + hypertensive*, an image *dog + car + tree*.

Multi-label vs multi-class
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Feature
     - Multi-class
     - Multi-label
   * - Labels per sample
     - Exactly one
     - One or more
   * - Class exclusivity
     - Mutually exclusive
     - Independent
   * - Output layer
     - Softmax (probabilities sum to 1)
     - Sigmoid (independent probability per class)
   * - Loss
     - Categorical cross-entropy
     - Binary cross-entropy per class

How it's modelled
-----------------

The network ends in a **sigmoid** per class rather than a single softmax, and each
output is thresholded independently. The loss is **binary cross-entropy** summed or
averaged over the :math:`K` labels — effectively :math:`K` coupled binary problems
sharing one backbone.

How it's scored
---------------

Because a prediction is a *set* of labels, the metrics differ from single-label
ones:

- **Per-label precision / recall / F1**, then aggregated with **micro**, **macro**
  or **weighted** averaging.
- **Hamming loss** — the fraction of individual label slots that are wrong.
- **Subset accuracy** — strict: 1 only if *every* label of the sample is correct,
  else 0.
- **Jaccard similarity** — intersection over union of predicted and true label
  sets.

Worked example
--------------

True labels for an image: :math:`\{\text{Cat}, \text{Dog}\}`; the model predicts
:math:`\{\text{Cat}, \text{Horse}\}`.

- Precision :math:`= 1/(1+1) = 0.5` (Cat right, Horse is a false positive).
- Recall :math:`= 1/(1+1) = 0.5` (Dog was missed — a false negative).
- Jaccard :math:`= |\{\text{Cat}\}| / |\{\text{Cat},\text{Dog},\text{Horse}\}| = 1/3 \approx 0.33`.

Pitfalls and edge cases
-----------------------

- **Subset accuracy is harsh** — one wrong label out of many zeroes the whole
  sample; report it alongside Hamming loss, not alone.
- **Per-label thresholds** — a single 0.5 cutoff is rarely optimal for every label;
  tune thresholds per class, especially under imbalance.
- **Label imbalance and correlation** — rare labels and co-occurring labels (Cat
  with Dog) are easy to under-predict; micro averaging will hide that, macro will
  expose it.
"""

CONTENT["Micro AUROC"] = r"""
What it is
----------

**AUROC** (Area Under the ROC Curve) is, for a binary task, the probability that
the model scores a randomly chosen positive higher than a randomly chosen negative
— a threshold-free measure of ranking ability. **Micro AUROC** is one of the two
standard ways to extend that single number to :math:`K` classes.

The multiclass problem
----------------------

With :math:`K` classes there is no single ROC curve. Two aggregation strategies
dominate:

- **Macro AUROC** — compute a one-vs-rest AUROC per class, then average them with
  equal weight.
- **Micro AUROC** — *pool* every one-vs-rest binary decision across all classes
  into one big set, then compute a single AUROC.

How micro works
---------------

Flatten all per-class one-vs-rest scores and labels into a single pool, count true
and false positives **globally**, and evaluate AUROC as if it were one binary
problem:

.. math::

   \text{AUROC}_{\text{micro}}
   = \text{AUROC}\!\left(\textstyle\bigcup_{i=1}^{K}\text{positives}_i,\;
     \bigcup_{i=1}^{K}\text{negatives}_i\right).

Because counts are pooled, **frequent classes dominate** the result.

Worked example
--------------

Three classes with one-vs-rest scores AUROC(A)=0.90, AUROC(B)=0.70, AUROC(C)=0.60:

- **Macro AUROC** :math:`= (0.90 + 0.70 + 0.60)/3 = 0.733`.
- **Micro AUROC** pools predictions, so if class A has far more samples the micro
  value is pulled toward 0.90 — the majority class's score.

When to use it (and the trap)
-----------------------------

Micro AUROC answers "how well does the model discriminate **overall, per sample**".
Its trap is imbalance: it can look excellent while rare classes are handled badly,
because their few samples barely move the pooled total. Macro AUROC weights every
class equally and exposes weak minorities. **Report both.**

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   # y_true: one-hot (n_samples, K); y_score: predicted probabilities (n_samples, K)
   micro = roc_auc_score(y_true, y_score, average="micro", multi_class="ovr")
   macro = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
"""

CONTENT["Micro F1"] = r"""
What it is
----------

The **F1 score** is the harmonic mean of precision and recall, rewarding a model
only when *both* are high:

.. math::

   F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}
              {\text{Precision} + \text{Recall}}.

**Micro F1** extends F1 to :math:`K` classes by pooling counts before computing the
score, rather than averaging per-class F1 values (which is macro F1).

How it's computed
-----------------

First form **global** precision and recall by summing true/false positives and
false negatives over all classes:

.. math::

   \text{Precision}_{\text{micro}} = \frac{\sum_i TP_i}{\sum_i (TP_i + FP_i)},
   \qquad
   \text{Recall}_{\text{micro}} = \frac{\sum_i TP_i}{\sum_i (TP_i + FN_i)},

then combine them with the F1 formula:

.. math::

   F_{1,\text{micro}} = \frac{2 \cdot \text{Precision}_{\text{micro}}
   \cdot \text{Recall}_{\text{micro}}}
   {\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}.

A key identity
--------------

In **single-label** (multi-class) problems, micro precision, micro recall and micro
F1 are all equal — and equal to plain **accuracy**. With one label per sample, a
false positive for one class is the same event as a false negative for another, so
the pooled denominators coincide. (In *multi-label* problems they can differ.)

Worked example
--------------

Three classes with TP = (40, 30, 10), FP = (10, 20, 20), FN = (10, 20, 30):

- :math:`\text{Precision}_{\text{micro}} = 80/130 \approx 0.615`
- :math:`\text{Recall}_{\text{micro}} = 80/140 \approx 0.571`
- :math:`F_{1,\text{micro}} \approx 0.592`

When to use it
--------------

Micro F1 measures overall, sample-weighted performance and lets majority classes
dominate — handy on imbalanced data when overall throughput matters. Use **macro
F1** when every class should count equally, including rare ones.

In code
-------

.. code-block:: python

   from sklearn.metrics import f1_score

   micro = f1_score(y_true, y_pred, average="micro")
   macro = f1_score(y_true, y_pred, average="macro")
"""

MINDMAP.update({
    "Multi-label Classification": [
        "Single-label Classification", "Micro F1", "Macro F1",
        "Micro AUROC", "Macro AUROC (Macro-Averaged AUROC)",
    ],
    "Micro AUROC": [
        "Macro AUROC (Macro-Averaged AUROC)", "One-vs-Rest (OvR) AUROC",
        "Micro F1", "Micro Recall", "Micro Precision",
    ],
    "Micro F1": [
        "Macro F1", "Micro Precision", "Micro Recall",
        "Micro AUROC", "Multi-label Classification",
    ],
})


CONTENT["Single-label Classification"] = r"""
What it is
----------

In **single-label classification** every sample is assigned **exactly one** label
from a set of :math:`K` mutually exclusive classes. Formally the classifier maps an
input to one label,

.. math::

   f : X \rightarrow Y, \qquad Y = \{1, 2, \dots, K\},

and typically picks the highest-probability class,

.. math::

   f(x) = \arg\max_{k}\, P(y = k \mid x).

This is the most common classification setting: spam vs not-spam, an image that is a
cat *or* a dog *or* a horse (never two at once), a single diagnosis from mutually
exclusive outcomes.

How it's modelled
-----------------

The output layer is a **softmax**, so the predicted class probabilities sum to one
and the classes compete — raising one lowers the others. Training uses **categorical
cross-entropy**.

How it's scored
---------------

Because there is one prediction per sample, the natural metrics are **accuracy**
plus **precision, recall and F1** (with micro / macro / weighted averaging for the
multi-class case) and **AUROC / AUPRC** via one-vs-rest. A useful identity: under
single-label evaluation, *micro* precision, recall and F1 all equal accuracy.

vs multi-label
--------------

The contrast is exclusivity. Single-label gives one label per sample (an image is
cat *or* dog); **multi-label** allows any subset (a news story tagged *politics* and
*economy*), uses independent sigmoids instead of a softmax, and needs set-based
metrics like Hamming loss and Jaccard.

Pitfalls and edge cases
-----------------------

- **The exclusivity assumption** — if samples can truly belong to several classes,
  forcing one label loses information; use multi-label instead.
- **Calibrated probabilities** — ``argmax`` discards confidence; keep the softmax
  scores if you need ranking, thresholds or AUROC.
- **Imbalance** — plain accuracy flatters a majority-heavy dataset; prefer macro
  metrics when minority classes matter.
"""

CONTENT["Micro Recall"] = r"""
What it is
----------

**Recall** answers: of all the items that are *actually* positive, how many did the
model catch?

.. math::

   \text{Recall} = \frac{TP}{TP + FN},

so it is sensitive to **false negatives** (missed positives). **Micro recall**
extends this to :math:`K` classes by pooling counts across classes rather than
averaging per-class recall (which is macro recall).

How it's computed
-----------------

Sum true positives and false negatives over every class, then divide:

.. math::

   \text{Recall}_{\text{micro}}
   = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FN_i)}.

This treats the whole multi-class problem as one pooled "caught vs missed"
question, so **majority classes dominate** the result.

The micro identity
------------------

In single-label problems, micro recall equals micro precision equals micro F1 (and
equals accuracy): once everything is pooled, the denominators coincide. The three
diverge only in the multi-label setting.

Worked example
--------------

Three classes with TP = (40, 30, 10), FN = (10, 20, 30):

- Recall(A)=0.80, Recall(B)=0.60, Recall(C)=0.25 → **Macro recall** = 0.55.
- **Micro recall** :math:`= 80/140 \approx 0.571`.

Macro weights every class equally (exposing class C's weak 0.25); micro is a global,
sample-weighted figure.

When recall matters most
------------------------

Favour recall when a **missed positive is costly** — disease screening, fraud
detection, safety alerts — where you would rather tolerate false alarms than let a
true case slip through.

In code
-------

.. code-block:: python

   from sklearn.metrics import recall_score

   micro = recall_score(y_true, y_pred, average="micro")
   macro = recall_score(y_true, y_pred, average="macro")
"""

CONTENT["Micro Precision"] = r"""
What it is
----------

**Precision** answers: of all the items the model *flagged* positive, how many were
right?

.. math::

   \text{Precision} = \frac{TP}{TP + FP},

so it is sensitive to **false positives** (false alarms). **Micro precision**
extends it to :math:`K` classes by pooling counts before dividing, rather than
averaging per-class precision (macro precision).

How it's computed
-----------------

.. math::

   \text{Precision}_{\text{micro}}
   = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FP_i)}.

Summing across classes first makes the metric a single global "of all predictions,
how many correct", so **frequent classes carry the most weight**.

The micro identity
------------------

As with micro recall and micro F1, in single-label classification micro precision
equals the other two (and accuracy), because the pooled denominators line up. They
differ only under multi-label evaluation.

Worked example
--------------

Three classes with TP = (40, 30, 10), FP = (10, 20, 20):

- Precision(A)=0.80, Precision(B)=0.60, Precision(C)=0.33 → **Macro precision** = 0.58.
- **Micro precision** :math:`= 80/130 \approx 0.615`.

The micro value sits near the large classes' contribution; macro surfaces class C's
weaker 0.33.

When precision matters most
---------------------------

Favour precision when a **false positive is costly** — spam filters (blocking real
mail), recommending a bad product, flagging an innocent transaction — where the
price of a wrong "yes" is high.

In code
-------

.. code-block:: python

   from sklearn.metrics import precision_score

   micro = precision_score(y_true, y_pred, average="micro")
   macro = precision_score(y_true, y_pred, average="macro")
"""

CONTENT["One-vs-Rest (OvR) AUROC"] = r"""
What it is
----------

AUROC is only defined for a binary problem, so to score a :math:`K`-class model we
reduce it to :math:`K` binary ones with **One-vs-Rest (OvR)**: for each class
:math:`i`, treat class :math:`i` as positive and *all other classes together* as
negative, and compute that binary AUROC.

.. math::

   \text{AUROC}_{\text{OvR}}(i) = \text{AUROC}(\text{class}_i \;\text{vs}\; \text{rest}).

Each value measures how well the model separates *that one class* from everything
else.

From per-class to a single number
---------------------------------

You can report the per-class AUROCs directly, or average them into **macro AUROC**:

.. math::

   \text{AUROC}_{\text{macro}}
   = \frac{1}{K} \sum_{i=1}^{K} \text{AUROC}_{\text{OvR}}(i).

(Pooling the OvR decisions instead of averaging gives *micro* AUROC; averaging, as
here, weights every class equally.)

Worked example
--------------

Three classes with AUROC(A vs rest)=0.83, AUROC(B vs rest)=0.76,
AUROC(C vs rest)=0.70:

- Per class, these show A is the easiest to separate and C the hardest.
- **Macro AUROC** :math:`= (0.83 + 0.76 + 0.70)/3 = 0.763`.

When it's useful
----------------

OvR AUROC gives a **per-class, threshold-free** read on discrimination — exactly
what you want to answer "how well does the model pick out *this* class from the
rest?". It is especially informative for imbalanced problems (e.g. a rare fraud
class), where a single global score can hide a weak minority.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   per_class = roc_auc_score(y_true, y_score, multi_class="ovr", average=None)
   macro = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
"""

MINDMAP.update({
    "Single-label Classification": [
        "Multi-label Classification", "Micro F1", "Macro F1",
        "One-vs-Rest (OvR) AUROC",
    ],
    "Micro Recall": ["Micro Precision", "Micro F1", "Macro Recall", "Micro AUROC"],
    "Micro Precision": ["Micro Recall", "Micro F1", "Macro Precision", "Micro AUROC"],
    "One-vs-Rest (OvR) AUROC": [
        "Macro AUROC (Macro-Averaged AUROC)", "Micro AUROC",
        "Macro F1", "Single-label Classification",
    ],
})


CONTENT["Macro AUROC (Macro-Averaged AUROC)"] = r"""
What it is
----------

AUROC measures ranking ability for a binary task — the probability a random positive
outscores a random negative, ranging from 0.5 (chance) to 1.0 (perfect). Since ROC
is inherently binary, a :math:`K`-class model is scored by reducing it with
**one-vs-rest (OvR)** (or one-vs-one). **Macro AUROC** averages the per-class OvR
AUROCs with **equal weight**:

.. math::

   \text{AUROC}_{\text{macro}}
   = \frac{1}{K} \sum_{i=1}^{K} \text{AUROC}(\text{class}_i \;\text{vs}\; \text{rest}).

Because every class counts the same, a rare class the model handles badly drags the
score down just as much as a common one.

Macro vs micro
--------------

**Macro** gives equal weight per class, so it is sensitive to **minority-class**
performance. **Micro** pools all OvR decisions into one global AUROC and is therefore
dominated by **majority classes**. They answer different questions — fairness across
classes vs overall sample-level discrimination — so reporting both is good practice.
The scale matches binary AUROC in every case.

Worked example
--------------

Three classes with AUROC(A vs rest)=0.85, AUROC(B vs rest)=0.72,
AUROC(C vs rest)=0.65:

.. math::

   \text{AUROC}_{\text{macro}} = \frac{0.85 + 0.72 + 0.65}{3} = 0.74.

If C is rare but poorly separated, macro AUROC reflects it; micro might not.

Pitfalls and edge cases
-----------------------

- **Undefined per-class AUROC** — if a class has no positive (or no negative)
  samples in the evaluation set, its OvR AUROC is undefined and breaks the average;
  guard against empty classes.
- **Equal weighting cuts both ways** — a single tiny, hard class can dominate the
  headline number; inspect the per-class AUROCs, not just the mean.
- **Weighted variant** — averaging the per-class AUROCs by class frequency gives a
  middle ground between macro and micro.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   macro = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
   per_class = roc_auc_score(y_true, y_score, multi_class="ovr", average=None)
"""

CONTENT["Macro F1"] = r"""
What it is
----------

F1 is the harmonic mean of precision and recall, high only when both are high:

.. math::

   F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}
              {\text{Precision} + \text{Recall}}.

For :math:`K` classes, compute an F1 *per class* (one-vs-rest), then **average them
with equal weight** to get macro F1:

.. math::

   F_{1,\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} F_{1,i}.

Each class contributes the same, whatever its size — so weak performance on a rare
class is fully visible.

Macro vs micro vs weighted
--------------------------

- **Macro** — equal weight per class; surfaces minority-class weakness.
- **Micro** — pool global TP / FP / FN first, then compute F1; dominated by majority
  classes (and equals accuracy in single-label problems).
- **Weighted** — average per-class F1 weighted by class frequency; a compromise that
  respects class sizes while staying per-class.

Worked example
--------------

Three classes:

- F1(A) = 0.80 (P=0.80, R=0.80)
- F1(B) = 0.48 (P=0.60, R=0.40)
- F1(C) = 0.33 (P=0.50, R=0.25)

.. math::

   F_{1,\text{macro}} = \frac{0.80 + 0.48 + 0.33}{3} = 0.54.

Class A is strong, but B and C pull the average down — exactly what you want when
every class matters.

Pitfalls and edge cases
-----------------------

- **Zero-division** — if a class is never predicted, its precision is undefined; by
  convention its F1 is taken as 0, which (correctly) penalises ignoring that class.
- **Rare-class leverage** — one small, hard class can dominate the macro average;
  read the per-class F1s alongside it.

In code
-------

.. code-block:: python

   from sklearn.metrics import f1_score

   macro = f1_score(y_true, y_pred, average="macro")
   weighted = f1_score(y_true, y_pred, average="weighted")
"""

CONTENT["Macro Recall"] = r"""
What it is
----------

Recall is the share of actual positives the model catches,
:math:`\text{Recall} = TP/(TP+FN)`. For :math:`K` classes, compute recall *per class*
(one-vs-rest) and take the **unweighted mean**:

.. math::

   \text{Recall}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{Recall}_i.

Every class counts equally regardless of how many samples it has.

A useful equivalence
--------------------

In single-label classification, **macro recall is exactly balanced accuracy** — the
average per-class hit rate. That makes it a go-to headline metric for imbalanced
problems, because it refuses to let a dominant class inflate the score.

Macro vs micro vs weighted
--------------------------

- **Macro** — equal weight per class; good when every class is equally important.
- **Micro** — global TP and FN pooled first; dominated by large classes.
- **Weighted** — per-class recall averaged by the number of true samples in each
  class.

Worked example
--------------

Three classes with Recall(A)=0.90, Recall(B)=0.60, Recall(C)=0.30:

.. math::

   \text{Recall}_{\text{macro}} = \frac{0.90 + 0.60 + 0.30}{3} = 0.60.

Even though C is rare, it carries the same weight as A.

Pitfalls and edge cases
-----------------------

- **Ignores false positives** — recall says nothing about precision, so a model that
  over-predicts a class can still score well; pair it with macro precision or
  macro F1.
- **Empty classes** — a class with no true samples has undefined recall and must be
  handled before averaging.

In code
-------

.. code-block:: python

   from sklearn.metrics import recall_score, balanced_accuracy_score

   macro = recall_score(y_true, y_pred, average="macro")
   # in single-label problems this equals:
   bal_acc = balanced_accuracy_score(y_true, y_pred)
"""

MINDMAP.update({
    "Macro AUROC (Macro-Averaged AUROC)": [
        "Micro AUROC", "One-vs-Rest (OvR) AUROC", "Macro F1", "Macro Recall",
    ],
    "Macro F1": [
        "Micro F1", "Macro Precision", "Macro Recall",
        "Macro AUROC (Macro-Averaged AUROC)",
    ],
    "Macro Recall": [
        "Macro Precision", "Macro F1", "Micro Recall",
        "Macro AUROC (Macro-Averaged AUROC)",
    ],
})


CONTENT["Macro Precision"] = r"""
What it is
----------

**Precision** is the share of the model's positive predictions that are correct,
:math:`\text{Precision} = TP/(TP+FP)`, so it is sensitive to **false positives**.
For :math:`K` classes, compute precision *per class* (one-vs-rest) and take the
**arithmetic mean** to get macro precision:

.. math::

   \text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{Precision}_i.

Every class counts equally, whatever its size, so a small class the model
over-flags pulls the score down as much as a large one.

Macro vs micro vs weighted
--------------------------

- **Macro** — equal weight per class; fair across classes.
- **Micro** — pool global TP and FP first; dominated by large classes (and equals
  accuracy in single-label problems).
- **Weighted** — per-class precision averaged by class frequency.

Worked example
--------------

Three classes with Precision(A)=0.80, Precision(B)=0.60, Precision(C)=0.40:

.. math::

   \text{Precision}_{\text{macro}} = \frac{0.80 + 0.60 + 0.40}{3} = 0.60.

If C is tiny, macro precision still penalises weak performance on it.

Pitfalls and edge cases
-----------------------

- **Zero-division** — a class the model never predicts has 0 in the denominator;
  its precision is undefined and conventionally set to 0, which penalises ignoring
  the class. Set ``zero_division`` explicitly to control this.
- **Pair it with recall** — precision rewards being conservative; a model that
  rarely predicts a class can post high precision while missing most of it.

In code
-------

.. code-block:: python

   from sklearn.metrics import precision_score

   macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
   weighted = precision_score(y_true, y_pred, average="weighted")
"""

CONTENT["Multiclass AUROC"] = r"""
What it is
----------

AUROC — the probability a random positive outscores a random negative, from 0.5
(chance) to 1.0 (perfect) — is defined only for a binary problem. **Multiclass
AUROC** is the umbrella for the strategies that extend it to :math:`K` classes by
reducing the problem to many binary ones and aggregating.

Two reductions
--------------

one-vs-rest (OvR)
^^^^^^^^^^^^^^^^^

For each class, score it against *all others* combined — :math:`K` binary AUROCs:

.. math::

   \text{AUROC}_{\text{macro}}
   = \frac{1}{K} \sum_{i=1}^{K} \text{AUROC}(\text{class}_i \;\text{vs}\; \text{rest}).

one-vs-one (OvO)
^^^^^^^^^^^^^^^^

Score *every pair* of classes and average over all :math:`\binom{K}{2}` pairs:

.. math::

   \text{AUROC}_{\text{ovo}}
   = \frac{2}{K(K-1)} \sum_{i<j} \text{AUROC}(\text{class}_i \;\text{vs}\; \text{class}_j).

OvO gives a more balanced picture under heavy imbalance, at :math:`O(K^2)` cost.

Two averagings
--------------

Independently of OvR / OvO, the per-binary results can be combined as **macro**
(equal weight per class — fair to minorities), **micro** (pool all decisions into
one global AUROC — sample-weighted, majority-driven), or **weighted** (by class
frequency).

Worked example
--------------

OvR AUROCs of 0.82, 0.75, 0.70 over three classes:

.. math::

   \text{AUROC}_{\text{macro}} = \frac{0.82 + 0.75 + 0.70}{3} = 0.7567.

Interpretation
--------------

All variants keep the binary AUROC scale (0.5 random, 1.0 perfect). Read **macro**
for fairness across classes, **micro** for overall sample-level discrimination, and
**OvO** for pairwise separability. Reporting more than one avoids being misled by a
single summary.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   ovr_macro = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
   ovo_macro = roc_auc_score(y_true, y_score, multi_class="ovo", average="macro")
"""

CONTENT["Gini Coefficient"] = r"""
What it is
----------

The **Gini coefficient** has two lives. In economics (Corrado Gini, 1912) it
measures **inequality** in a distribution such as income, via the **Lorenz curve** —
the cumulative share of income against the cumulative share of population. It is the
normalised area between the line of perfect equality and the Lorenz curve:

.. math::

   \text{Gini} = \frac{A}{A + B},

where :math:`A` is the area between the equality line and the Lorenz curve and
:math:`B` is the area under the Lorenz curve.

In machine learning
-------------------

For a binary classifier, Gini measures **discriminatory power** and is a simple
linear rescaling of AUROC:

.. math::

   \text{Gini} = 2 \cdot \text{AUROC} - 1.

So AUROC 0.5 (random) gives Gini 0, AUROC 1.0 (perfect) gives Gini 1, and an AUROC
below 0.5 gives a negative Gini. It carries **no information beyond AUROC** — the
same ranking quality on a stretched scale — but it is the convention in finance.

Why credit risk uses it
-----------------------

Credit-scoring models (loan default, churn, fraud) usually report Gini rather than
AUROC: a higher Gini means the model separates "goods" (non-defaulters) from "bads"
(defaulters) better, and regulatory frameworks (Basel II/III) often expect it in
model validation. The ML Lorenz curve simply replaces "income" with the predicted
score and "population" with cases sorted by that score.

Worked example
--------------

- Model A: AUROC 0.72 → :math:`\text{Gini} = 2(0.72) - 1 = 0.44`.
- Model B: AUROC 0.85 → :math:`\text{Gini} = 0.70`.

Model B ranks defaulters above non-defaulters far better; a Gini of 0.70 is
considered excellent in credit risk.

Rules of thumb and edge cases
-----------------------------

- Typical bands: 0.20–0.30 weak, 0.40–0.50 useful, 0.60–0.70 strong.
- **0.80+ is suspicious** — usually overfitting or target leakage, not a genuinely
  great model.
- A **negative Gini** means predictions are inverted (worse than random); flipping
  the score sign fixes it.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   gini = 2 * roc_auc_score(y_true, y_score) - 1
"""

MINDMAP.update({
    "Macro Precision": [
        "Micro Precision", "Macro Recall", "Macro F1",
        "Macro AUROC (Macro-Averaged AUROC)",
    ],
    "Multiclass AUROC": [
        "One-vs-Rest (OvR) AUROC", "Micro AUROC",
        "Macro AUROC (Macro-Averaged AUROC)", "Gini Coefficient",
    ],
    "Gini Coefficient": [
        "Multiclass AUROC", "One-vs-Rest (OvR) AUROC",
        "Macro AUROC (Macro-Averaged AUROC)",
    ],
})


# ----------------------------------------------------------------------
# Themes: Model Evaluation & Uncertainty (evaluation) + Probability &
# Statistics Foundations (probstats)
# ----------------------------------------------------------------------

CONTENT["Bootstrap Confidence Intervals (CIs)"] = r"""
What it is
----------

A **bootstrap confidence interval** estimates the uncertainty of a statistic — a
mean, median, regression coefficient, even an AUROC — by **resampling the data**
rather than relying on a parametric formula. The idea: when the population
distribution is unknown, approximate it by drawing from the sample you already have.
It shines when sample sizes are small, the data are non-normal, or no clean
standard-error formula exists.

The procedure
-------------

1. Start with the original sample of size :math:`n`.
2. **Resample with replacement** to build :math:`B` bootstrap samples (often
   :math:`B = 1000` or more), each of size :math:`n`.
3. Compute the statistic on each bootstrap sample.
4. The spread of those :math:`B` values is the **bootstrap distribution** of the
   statistic.
5. Read a confidence interval off that distribution.

Three ways to build the interval
--------------------------------

- **Percentile** — take the :math:`\alpha/2` and :math:`1-\alpha/2` quantiles
  directly (a 95% CI is the 2.5th–97.5th percentiles).
- **Basic (reverse percentile)** — reflect the percentile interval around the
  observed statistic to correct simple bias.
- **BCa (bias-corrected and accelerated)** — adjusts for both bias and skew in the
  bootstrap distribution; usually the most accurate and the default recommendation.

Worked example
--------------

For :math:`X = [5, 7, 9, 10, 12, 8, 6, 7, 9, 11]` the mean is 8.4. Draw 1000
resamples, take each mean, and read the 2.5th and 97.5th percentiles — about 7.2 and
9.6 — giving a **95% CI of [7.2, 9.6]**.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(42)
   boot = [rng.choice(X, size=len(X), replace=True).mean() for _ in range(1000)]
   lo, hi = np.percentile(boot, [2.5, 97.5])

Pitfalls and edge cases
-----------------------

- **Too few resamples** — small :math:`B` makes the interval itself noisy; prefer
  thousands.
- **A bad sample stays bad** — the bootstrap can only resample what you have; a tiny
  or unrepresentative sample yields a confident-looking but wrong interval.
- **Dependent data break it** — for time series or grouped data the plain bootstrap
  destroys the dependence structure; use a **block bootstrap** instead.
"""

CONTENT["Probability"] = r"""
What it is
----------

**Probability** is a number between 0 and 1 that expresses how likely an event is:
0 means impossible, 1 means certain, and values in between are degrees of
likelihood. A fair coin landing heads has probability 0.5.

The basic formula
-----------------

When every outcome is equally likely, probability is the ratio of favourable to
total outcomes:

.. math::

   P(E) = \frac{\text{number of favourable outcomes}}{\text{total number of outcomes}}.

Rolling a 4 on a fair die: :math:`P = 1/6 \approx 0.167`.

Three ways to assign it
-----------------------

- **Theoretical** — from the structure of the problem (a fair die, a fair coin).
- **Experimental (frequentist)** — from observed frequencies (flip a coin 100 times
  and count heads); it converges to the theoretical value as trials grow.
- **Subjective (Bayesian)** — a degree of belief (a forecaster's "70% chance of
  rain").

The core rules
--------------

- **Complement** — :math:`P(\text{not } A) = 1 - P(A)`.
- **Addition (OR)** — in general
  :math:`P(A \cup B) = P(A) + P(B) - P(A \cap B)`, which reduces to
  :math:`P(A) + P(B)` only when the events are mutually exclusive.
- **Multiplication (AND)** — in general
  :math:`P(A \cap B) = P(A)\,P(B \mid A)`, which reduces to :math:`P(A)\,P(B)` only
  when the events are independent. Rolling a 2 then a 5 on two dice:
  :math:`\tfrac{1}{6} \times \tfrac{1}{6} = \tfrac{1}{36}`.

Pitfalls and edge cases
-----------------------

- **Mutually exclusive is not independent** — exclusive events *cannot* co-occur
  (and are strongly dependent); independent events have no influence on each other.
  Confusing the two is the classic probability error.
- **Conditional probability** — :math:`P(B \mid A) = P(A \cap B)/P(A)` underlies the
  general AND rule and, ultimately, Bayes' theorem.
- **Normalisation** — probabilities over all mutually exclusive outcomes must sum to
  exactly 1.
"""

CONTENT["Mann–Whitney U Test (also called the Wilcoxon rank-sum test)"] = r"""
What it is
----------

The **Mann–Whitney U test** (equivalently the **Wilcoxon rank-sum test**) is a
**non-parametric** test for whether two *independent* groups come from the same
distribution. Unlike the t-test it makes **no normality assumption** — it works on
the **ranks** of the pooled data, so it is the natural choice for skewed or ordinal
data.

Assumptions
-----------

- The two samples are **independent**.
- The outcome is **ordinal or continuous**.
- Strictly, it tests *stochastic dominance* (whether one group tends to be larger);
  only when the two distributions have the **same shape** does it become a test of
  **medians**.

The U statistic
---------------

Pool both groups, rank everything, and sum the ranks of each group
(:math:`R_1, R_2`). Then

.. math::

   U_1 = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1, \qquad
   U_2 = n_1 n_2 + \frac{n_2(n_2 + 1)}{2} - R_2, \qquad
   U = \min(U_1, U_2).

For large samples :math:`U` is approximately normal, giving a z-based p-value.

Hypotheses
----------

- **H₀** — the two groups come from the same distribution.
- **H₁** — one group tends to produce larger (or smaller) values than the other.

Worked example
--------------

Group A = {88, 92, 100, 75, 85}, Group B = {60, 70, 65, 80, 72}. Rank all ten
values, sum the ranks, compute :math:`U`, and look up the p-value; :math:`p < 0.05`
means the groups differ significantly.

.. code-block:: python

   from scipy.stats import mannwhitneyu

   U, p = mannwhitneyu(group_a, group_b, alternative="two-sided")

The link to ROC-AUC
-------------------

The U statistic is **mathematically equivalent to AUROC**:

.. math::

   \text{AUC} = \frac{U}{n_1 n_2},

i.e. the probability that a randomly chosen value from one group outranks a randomly
chosen value from the other — exactly the definition of AUROC. Testing whether two
score distributions differ is the same computation as measuring a classifier's
ranking power.

Pitfalls and edge cases
-----------------------

- **Ties** — many tied values need a tie correction (most libraries apply one).
- **The "median" shortcut** — only valid under equal-shape distributions; otherwise
  report it as a test of stochastic dominance.
- **Paired data** — for *matched* samples use the Wilcoxon *signed-rank* test
  instead; this test is for *independent* groups.
"""

MINDMAP.update({
    "Bootstrap Confidence Intervals (CIs)": [
        "Subsampling", "Probability",
        "Mann–Whitney U Test (also called the Wilcoxon rank-sum test)",
    ],
    "Probability": [
        "Bootstrap Confidence Intervals (CIs)",
        "Mann–Whitney U Test (also called the Wilcoxon rank-sum test)",
    ],
    "Mann–Whitney U Test (also called the Wilcoxon rank-sum test)": [
        "Multiclass AUROC", "One-vs-Rest (OvR) AUROC",
        "Bootstrap Confidence Intervals (CIs)", "Probability",
    ],
})


# ----------------------------------------------------------------------
# Theme: Fairness & Calibration  (fairness)
# ----------------------------------------------------------------------

CONTENT["Predictive Parity (Calibration)"] = r"""
What it is
----------

**Predictive parity**, also called **calibration by group**, asks that a score mean
the same thing regardless of group: among everyone assigned the same predicted
probability, the actual positive rate is equal across groups. Formally, for a
predicted probability :math:`\hat{p}`,

.. math::

   P(Y=1 \mid \hat{P}=\hat{p}, A=a) = P(Y=1 \mid \hat{P}=\hat{p}, A=b)
   \quad \forall\, a, b,

with :math:`Y` the true outcome, :math:`\hat{P}` the score and :math:`A` the
protected attribute. In plain terms: if two people from different groups are both
scored at 70%, then about 70% of *each* group should actually turn out positive.

Where it sits among fairness criteria
-------------------------------------

This is the **sufficiency** criterion, :math:`Y \perp A \mid \hat{Y}` — the label is
independent of group once you condition on the score. It is the counterpart to the
**independence** criterion (demographic parity) and the **separation** criterion
(equalized odds / equal opportunity).

Example
-------

A loan model scores a set of applicants at 0.7 predicted default probability. If 70%
of group A but only 50% of group B actually default, the score 0.7 does not carry the
same meaning across groups — predictive parity is violated.

The catch: it conflicts with the others
---------------------------------------

When the two groups have **different base rates**, predictive parity and equalized
odds **cannot both hold** (except in degenerate cases). This is the *fairness
impossibility theorem* (Kleinberg–Chouldechova): independence, separation and
sufficiency are mutually incompatible whenever base rates differ, so you must choose
which to prioritise.

Limitations
-----------

- A model can be perfectly calibrated yet still distribute **errors** unevenly — it
  says nothing about false-positive or false-negative rates.
- Small groups look mis-calibrated from variance alone; check with enough data.

In code
-------

.. code-block:: python

   from sklearn.calibration import calibration_curve

   # compare reliability curves per group
   for a in groups:
       frac_pos, mean_pred = calibration_curve(y_true[A == a], y_score[A == a], n_bins=10)
"""

CONTENT["Equalized Odds (Fairness)"] = r"""
What it is
----------

**Equalized odds** requires a classifier to have the **same true-positive rate and
the same false-positive rate** across groups. Formally,

.. math::

   P(\hat{Y}=1 \mid Y=y, A=a) = P(\hat{Y}=1 \mid Y=y, A=b)
   \quad \forall\, y \in \{0,1\},\ \forall\, a, b.

So among those who truly are positive (:math:`Y=1`) every group is recognised at the
same rate, and among those who truly are negative (:math:`Y=0`) every group is
wrongly flagged at the same rate.

Where it sits among fairness criteria
-------------------------------------

This is the **separation** criterion, :math:`\hat{Y} \perp A \mid Y` — the prediction
is independent of group once you condition on the truth. It is the strictest of the
error-rate criteria: equalized odds = **equal opportunity** (equal TPR) *plus* equal
FPR.

Example
-------

A loan model approves qualified men at TPR 80% but qualified women at TPR 60%, and
wrongly approves unqualified men at FPR 20% but unqualified women at FPR 30%. Both
rates differ by group, so equalized odds is violated on both counts.

The catch
---------

Like the other criteria, equalized odds collides with **predictive parity** when base
rates differ (the impossibility theorem), and enforcing it can cost overall accuracy
— so practitioners often target *approximate* equalized odds within a tolerance.

Limitations
-----------

- Hard to satisfy exactly, especially with unequal base rates.
- Trades off against accuracy; usually relaxed rather than enforced exactly.

In code
-------

.. code-block:: python

   from fairlearn.metrics import equalized_odds_difference

   eod = equalized_odds_difference(y_true, y_pred, sensitive_features=A)  # 0 = parity
"""

CONTENT["Equal Opportunity (Fairness)"] = r"""
What it is
----------

**Equal opportunity** is the relaxation of equalized odds that equalises only the
**true-positive rate** across groups:

.. math::

   P(\hat{Y}=1 \mid Y=1, A=a) = P(\hat{Y}=1 \mid Y=1, A=b) \quad \forall\, a, b.

Among the people who *should* receive a positive outcome, every group has the same
chance of being correctly recognised — so it targets **false negatives not falling
disproportionately on a disadvantaged group**.

Where it sits among fairness criteria
-------------------------------------

It is a one-sided **separation** criterion: equalized odds asks for equal TPR *and*
equal FPR; equal opportunity keeps only the TPR condition. It sits between the
label-blind **demographic parity** and the full **equalized odds**.

Example
-------

Among genuinely qualified applicants, a loan model approves 80% of men but only 60%
of women. Qualified women are recognised less often — equal opportunity is violated,
even if overall approval rates happen to match.

The catch
---------

Because it uses ground-truth labels it is more practical than demographic parity, but
it ignores **false positives**, and when base rates differ across groups achieving it
can still reduce accuracy and may clash with predictive parity.

Limitations
-----------

- Says nothing about false-positive fairness (a group could be over-approved).
- Base-rate differences can force an accuracy trade-off.

In code
-------

.. code-block:: python

   from fairlearn.metrics import true_positive_rate, MetricFrame

   tpr_by_group = MetricFrame(metrics=true_positive_rate,
                              y_true=y_true, y_pred=y_pred,
                              sensitive_features=A).by_group
"""

CONTENT["Demographic Parity (Statistical Parity)"] = r"""
What it is
----------

**Demographic parity** (or **statistical parity**) asks that the model's positive
decisions be **independent of the protected attribute** — every group receives a
positive prediction at the same rate:

.. math::

   P(\hat{Y}=1 \mid A=a) = P(\hat{Y}=1 \mid A=b) \quad \forall\, a, b.

Crucially it looks only at the prediction :math:`\hat{Y}`, never at the true label
:math:`Y`.

Where it sits among fairness criteria
-------------------------------------

This is the **independence** criterion, :math:`\hat{Y} \perp A`. It is the simplest
and most label-blind of the three families — independence (here), **separation**
(equalized odds / equal opportunity) and **sufficiency** (predictive parity).

Measuring it
------------

Two common gap metrics, with :math:`a` the disadvantaged group:

.. math::

   \text{DPD} = P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=b), \qquad
   \text{DPR} = \frac{P(\hat{Y}=1 \mid A=a)}{P(\hat{Y}=1 \mid A=b)}.

The ratio connects to the legal **four-fifths (80%) rule**: a selection-rate ratio
below 0.8 is treated as evidence of adverse impact.

Example
-------

A hiring model marks 60% of men but only 40% of women as interview-worthy. The rates
differ, so demographic parity is violated (and the 0.40 / 0.60 ≈ 0.67 ratio fails the
four-fifths rule).

The catch
---------

Because it ignores the label, demographic parity can be satisfied only by
**approving unqualified members** of one group to match rates — which may raise risk
and clash with equal opportunity, equalized odds and predictive parity.

Limitations
-----------

- Ignores genuine differences in qualification (the true label).
- Conflicts with the error-rate and calibration criteria when base rates differ.

In code
-------

.. code-block:: python

   from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio

   dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=A)
   dpr = demographic_parity_ratio(y_true, y_pred, sensitive_features=A)
"""

MINDMAP.update({
    "Predictive Parity (Calibration)": [
        "Equalized Odds (Fairness)", "Equal Opportunity (Fairness)",
        "Demographic Parity (Statistical Parity)",
    ],
    "Equalized Odds (Fairness)": [
        "Equal Opportunity (Fairness)", "Demographic Parity (Statistical Parity)",
        "Predictive Parity (Calibration)",
    ],
    "Equal Opportunity (Fairness)": [
        "Equalized Odds (Fairness)", "Demographic Parity (Statistical Parity)",
        "Predictive Parity (Calibration)",
    ],
    "Demographic Parity (Statistical Parity)": [
        "Equal Opportunity (Fairness)", "Equalized Odds (Fairness)",
        "Predictive Parity (Calibration)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Growth, Monetization & Customer Analytics  (growth)
# ----------------------------------------------------------------------

CONTENT["Cross-Selling"] = r"""
What it is
----------

**Cross-selling** is recommending **complementary or related products** alongside
what a customer is already buying, to grow the basket. The goal is a larger basket
size, higher **average order value (AOV)**, and ultimately higher **lifetime value
(LTV)**. Amazon's "frequently bought together — laptop + bag + mouse" is the textbook
case; in banking it is offering a credit card to a checking-account customer.

Cross-selling vs upselling
--------------------------

The two are easy to confuse:

- **Cross-sell** — a *different but related* product (iPhone → AirPods or AppleCare).
- **Upsell** — a *higher-tier version of the same* product (iPhone → iPhone Pro).

Why it matters
--------------

It lifts revenue per customer, deepens **stickiness** (a customer using several
products churns less), and in SaaS opens entirely new use cases (adding analytics or
security modules to a core product).

Common strategies
-----------------

- **Bundling** — sell related items together ("buy 2, get 1").
- **Recommendations** — "customers who bought X also bought Y".
- **Add-ons and accessories** — warranties, training, complementary items.
- **Account expansion** — selling more products into one account (B2B SaaS).
- **Loyalty rewards** — discounts for adding products.

The data-science behind it
--------------------------

Cross-sell recommendations are driven by **market-basket analysis** (association
rules such as Apriori, mining "X implies Y" co-purchase patterns) and **recommender
systems** (collaborative filtering), while **propensity models** rank which customers
are most likely to accept a given offer. Best practice is to keep suggestions
*relevant* — use segmentation and purchase history rather than spamming every add-on.
"""

CONTENT["Upselling"] = r"""
What it is
----------

**Upselling** is persuading a customer to buy a **higher-value version** of what they
are considering, or to **upgrade** to a more premium tier. The goal is higher
**average revenue per user (ARPU)** and **lifetime value (LTV)**. Starbucks' "grande
instead of a tall" is the everyday version; in SaaS it is "upgrade from Basic to Pro
for advanced features".

Upselling vs cross-selling
--------------------------

- **Upsell** — a more expensive / higher-tier version of the *same* product
  (iPhone 15 → iPhone 15 Pro).
- **Cross-sell** — a *different complementary* product (iPhone 15 → AirPods).

Why it matters
--------------

Selling more to existing customers is far cheaper than acquiring new ones, so
upselling improves the **LTV:CAC ratio** directly. Done honestly it also raises
satisfaction, because the upgrade genuinely solves the customer's problem.

Common strategies
-----------------

- **Tiered pricing** — Basic / Pro / Enterprise.
- **Feature upgrades** — more storage, faster delivery, premium support.
- **Personalized recommendations** — based on observed usage.
- **Time-limited offers** — "upgrade today and save 20%".
- **Value framing** — lead with ROI and convenience, not just price.

The data-science behind it
--------------------------

The targeting problem is "who is most likely to upgrade, and who would upgrade *only*
if prompted?" — answered with **propensity-to-upgrade models** and, more precisely,
**uplift (incremental) modeling**, which estimates the *causal* lift of an offer so
budget goes to the persuadable, not the sure-things. Usage-based **segmentation**
surfaces the power users worth a premium pitch.
"""

CONTENT["Customer Segmentation"] = r"""
What it is
----------

**Customer segmentation** divides a customer base into **distinct groups** that share
characteristics, so marketing, sales and product can be tailored to each. It answers
the question: *which kinds of customers do we serve, and how should we treat them
differently?*

Why it matters
--------------

Segmentation sharpens **targeted marketing**, lifts **conversion and retention**,
focuses resources on the **most valuable** customers, and guides product decisions
toward real segment needs.

Common bases for segmentation
-----------------------------

- **Demographic** — age, gender, income, occupation.
- **Geographic** — country, region, climate.
- **Psychographic** — lifestyle, values, interests.
- **Behavioral** — purchase patterns, usage, loyalty, engagement.
- **Value-based** — profitability, margin, **LTV** (e.g. high-LTV repeat buyers vs
  one-time buyers).

Methods
-------

- **Rule-based** — simple filters (age < 30 → Segment A).
- **RFM analysis** — score customers by **Recency, Frequency, Monetary** value.
- **Clustering (ML)** — k-means or hierarchical clustering on behavioural and
  demographic features.
- **Predictive models** — classify which segment a *new* customer will belong to.

Example
-------

A streaming service might segment by demographics (students, professionals,
families), behaviour (binge-watchers, casual viewers, sports fans) and value
(high-LTV subscribers vs churn-prone low-LTV users) — then give each different
promotions: student discounts, live-event upsells, loyalty rewards.

Pitfalls and best practices
---------------------------

- Segments should be **measurable, actionable, and stable** over time.
- Avoid a sprawl of **tiny segments** that are impossible to act on.
- **Revalidate** regularly — customer behaviour drifts, so yesterday's clusters go
  stale.
- For k-means specifically, **scale features** first (otherwise large-magnitude
  fields dominate) and pick K with the elbow or silhouette method.
"""

MINDMAP.update({
    "Cross-Selling": [
        "Upselling", "Customer Segmentation", "LTV (Customer Lifetime Value)",
        "Relevance in Recommender Systems", "Retention",
    ],
    "Upselling": [
        "Cross-Selling", "Customer Segmentation", "Revenue per User (RPU / ARPU)",
        "LTV (Customer Lifetime Value)", "LTV:CAC Ratio",
    ],
    "Customer Segmentation": [
        "Cross-Selling", "Upselling", "LTV (Customer Lifetime Value)",
        "Churn", "Cohort",
    ],
})


# ----------------------------------------------------------------------
# Theme: Business Models & Valuation  (growth / platforms)
# ----------------------------------------------------------------------

CONTENT["SaaS (Software as a Service)"] = r"""
What it is
----------

**SaaS (Software as a Service)** is a delivery model where applications are hosted in
the cloud and accessed over the internet, almost always on a **subscription**. Rather
than buying software once and installing it on-premises, customers pay a recurring
monthly or yearly fee. Salesforce, Slack, Zoom, Dropbox and HubSpot are canonical
examples.

How it works
------------

The vendor hosts the software on its own cloud infrastructure; users log in through a
browser or app with no local install; pricing is a recurring subscription; and the
vendor handles **maintenance, updates, scaling and security**. That shifts cost and
operational burden from customer to vendor and turns one-off licence sales into a
predictable revenue stream.

Why the model is attractive
---------------------------

- **Recurring revenue** → predictable cash flow.
- **High gross margins** → typically 70–90%, since serving one more user costs little.
- **Scalability and global reach** → no physical distribution.
- **First-party data** → usage data that fuels product improvement and upsell.

Challenges
----------

- **High CAC** — SaaS is marketing- and sales-heavy to acquire each customer.
- **Churn** — losing subscribers directly erodes recurring revenue.
- **Continuous investment** — the product must keep improving to retain users.

The metrics that define a SaaS business
---------------------------------------

- **ARR / MRR** — annual / monthly recurring revenue.
- **Churn rate** — share of customers (or revenue) lost per period.
- **Net revenue retention (NRR)** — expansion and upsell minus churn; **above 100%**
  means the existing base grows even with no new customers.
- **Gross margin**, **CAC**, **LTV**, and the **LTV:CAC ratio** (healthy at ≥ 3).
- **Rule of 40** — growth % + profit margin % should clear 40%.

SaaS companies are usually valued on **revenue multiples (EV/Revenue)** rather than
profit, since many are still reinvesting for growth.

The data-science angle
----------------------

SaaS runs on prediction: **churn models** flag at-risk accounts for intervention,
**propensity and uplift models** target expansion, **usage analytics** drive
onboarding, and **forecasting** projects ARR. The recurring relationship makes the
data rich and the modelling continuous.
"""

CONTENT["Valuation Metric"] = r"""
What it is
----------

A **valuation metric** is a financial ratio that measures a company's value relative
to a fundamental — earnings, revenue, cash flow or assets — so investors can judge
whether it is cheap or expensive versus peers. Which metric fits depends on the
company's **stage** (startup vs mature), **industry**, and **profitability**.

Metrics for profitable, public companies
----------------------------------------

- **P/E (Price-to-Earnings)** — :math:`\text{Share Price} / \text{EPS}`; what
  investors pay per \$1 of earnings.
- **P/S (Price-to-Sales)** — :math:`\text{Market Cap} / \text{Revenue}`; useful when
  there is no profit yet.
- **P/B (Price-to-Book)** — :math:`\text{Market Cap} / \text{Book Value}`; market
  value vs net assets.
- **EV/EBITDA** — enterprise value over earnings before interest, taxes, depreciation
  and amortisation; adjusts for debt and cash, common in private equity and M&A.

Metrics for startups and high-growth companies
----------------------------------------------

- **EV/Revenue** — revenue multiples, standard for pre-profit SaaS and D2C.
- **Rule of 40** — :math:`\text{growth (\%)} + \text{EBITDA margin (\%)} \ge 40`.
- **LTV:CAC ratio** — acquisition efficiency; investors look for :math:`\ge 3`.
- **Burn multiple** — :math:`\text{Net Burn} / \text{Net New ARR}`; how much cash is
  consumed per dollar of new recurring revenue.

Worked example
--------------

A SaaS company with revenue \$50M, EBITDA \$10M and enterprise value \$350M trades at
**EV/Revenue = 7×** and **EV/EBITDA = 35×**. Investors compare those multiples to
peers to judge over- or under-valuation.

Why it matters
--------------

Valuation metrics give a **standardised way** to compare companies and anchor **M&A
negotiations, VC funding rounds and IPO pricing** — always reflecting the market's
trade-off between growth and profitability. Mature firms lean on P/E and EV/EBITDA;
growth firms on EV/Revenue, Rule of 40 and LTV:CAC.
"""

CONTENT["D2C (Direct-to-Consumer)"] = r"""
What it is
----------

**D2C (Direct-to-Consumer)** is a business model in which a brand sells **straight to
end customers**, bypassing wholesalers, distributors and retailers. It is usually
built on e-commerce but can include brand-owned physical stores. Warby Parker,
Glossier, Allbirds and Casper are classic D2C brands.

How it works
------------

The brand makes or sources its product, sells through **its own channels** (website,
app, pop-up, owned store), and therefore owns the **customer relationship, data and
experience** end to end — no middleman in between.

Advantages
----------

- **Higher margins** — no distributor or retailer taking a cut.
- **Control of the experience** — branding, packaging, service.
- **First-party data** — emails and purchase behaviour feed personalisation and LTV
  modelling.
- **Brand loyalty** — a direct relationship with customers.

Challenges
----------

- **High CAC** — paid ads, influencers and SEO are expensive, and ad costs keep
  rising.
- **Logistics and fulfillment** — the brand owns shipping, returns and support.
- **Scale limits** — reaching mass distribution is harder without retail partners.

Key metrics
-----------

- **CAC** and **LTV**, with the **payback period** — how long to recover CAC — often
  the make-or-break number.
- **Gross margin** and **repeat purchase rate**, which together determine whether the
  unit economics actually work.

D2C vs traditional retail
-------------------------

Traditional retail is *Manufacturer → Distributor → Retailer → Consumer*; D2C
collapses that to *Brand → Consumer*. The trade is **more margin and data** in
exchange for **owning marketing and logistics** yourself.

The data-science angle
----------------------

Owning first-party data is the whole point: D2C brands lean on **LTV prediction**,
**marketing-mix and attribution models** to allocate ad spend, **recommendation and
segmentation** for personalisation, and **payback / cohort analysis** to keep CAC
under control.
"""

MINDMAP.update({
    "SaaS (Software as a Service)": [
        "LTV:CAC Ratio", "Churn", "Retention", "Gross Margin", "Valuation Metric",
    ],
    "Valuation Metric": [
        "LTV:CAC Ratio", "SaaS (Software as a Service)", "Gross Margin",
        "D2C (Direct-to-Consumer)",
    ],
    "D2C (Direct-to-Consumer)": [
        "SaaS (Software as a Service)", "LTV:CAC Ratio",
        "CAC (Customer Acquisition Cost)", "LTV (Customer Lifetime Value)",
        "Gross Margin",
    ],
})


# ----------------------------------------------------------------------
# Theme: Unit Economics — LTV & CAC  (growth)
# ----------------------------------------------------------------------

CONTENT["Gross LTV (Customer Lifetime Value)"] = r"""
What it is
----------

**Gross LTV** is the **total revenue** a customer is expected to generate over their
lifetime, *before* subtracting any cost of serving them. It is the top-line view of
customer value — "how much will this customer pay us in total?", not "how much profit
do we keep?".

Formula
-------

.. math::

   \text{Gross LTV} = \text{ARPU} \times \text{Customer Lifetime},

where **ARPU** is average revenue per user per period and **customer lifetime** is the
average time a customer stays — commonly estimated as :math:`1/\text{churn rate}`.

Worked example
--------------

With ARPU = $50/month and 5% monthly churn, the expected lifetime is
:math:`1/0.05 = 20` months, so

.. math::

   \text{Gross LTV} = 50 \times 20 = 1{,}000.

Each customer pays about $1,000 in total revenue over their average lifetime.

Gross vs Net LTV
----------------

Gross LTV counts **revenue only**; **Net (Contribution) LTV** multiplies it by gross
margin to get profit, e.g. at 70% margin :math:`1{,}000 \times 0.7 = 700`. Gross LTV
is the optimistic number; Net LTV is the one to compare against CAC.

When it's used (and its limits)
-------------------------------

Gross LTV gives a **quick, high-level** read and is handy for **early-stage**
companies that lack detailed cost data. Two caveats matter:

- It ignores cost of goods sold, so it overstates true value — finance teams prefer
  Net LTV.
- The :math:`1/\text{churn}` lifetime assumes a **constant churn rate**; real
  retention curves flatten over time, so survival analysis or cohort retention curves
  give a more accurate lifetime.
"""

CONTENT["Net LTV (sometimes called Contribution LTV)"] = r"""
What it is
----------

**Net LTV** (also **Contribution LTV**) is the **profit contribution** a customer
generates over their lifetime, *after* the direct cost of serving them. Where Gross
LTV is pure revenue, Net LTV adjusts for **gross margin**, answering "how much do we
actually keep?".

Formula
-------

.. math::

   \text{Net LTV} = \text{Gross LTV} \times \text{Gross Margin \%}
   = (\text{ARPU} \times \text{Customer Lifetime}) \times \text{Gross Margin \%},

with gross margin :math:`= (\text{Revenue} - \text{COGS}) / \text{Revenue}` and
customer lifetime :math:`\approx 1/\text{churn}`.

Worked example
--------------

Continuing the running example — ARPU $50/month, 5% churn (20-month lifetime) — gives
Gross LTV $1,000. At 70% gross margin:

.. math::

   \text{Net LTV} = 1{,}000 \times 0.7 = 700.

The customer pays $1,000 in revenue, but the business keeps $700 in profit
contribution.

Why it's the number that matters
--------------------------------

Net LTV is the **realistic** measure of customer value and the correct numerator for
unit economics: the healthy benchmark is

.. math::

   \frac{\text{Net LTV}}{\text{CAC}} \ge 3.

Investors and CFOs favour Net over Gross LTV because revenue without margin is not
sustainable. As with Gross LTV, prefer **retention-curve or predictive** lifetime
estimates over a flat :math:`1/\text{churn}` when the data allows.
"""

CONTENT["LTV:CAC Ratio"] = r"""
What it is
----------

The **LTV:CAC ratio** compares the lifetime value of a customer to the cost of
acquiring them. It answers the core unit-economics question: *for every $1 spent
acquiring a customer, how many dollars of lifetime value come back?* It tells you
whether growth is **profitable and scalable**.

Formula
-------

.. math::

   \text{LTV:CAC} = \frac{\text{LTV}}{\text{CAC}}.

By convention the numerator is **Net (Contribution) LTV** — revenue alone overstates
the case — and the denominator can be **Blended**, **Paid**, or **Fully-Loaded CAC**
depending on what question you are asking.

Worked example
--------------

Chaining the standard example: ARPU $50/month and 5% churn give a 20-month lifetime,
so Gross LTV :math:`= 50 \times 20 = 1{,}000`; at 70% gross margin, Net LTV
:math:`= 700`; with blended CAC $200,

.. math::

   \text{LTV:CAC} = \frac{700}{200} = 3.5.

Every $1 of acquisition spend returns $3.50 in profit contribution.

How to read it
--------------

- **< 1** — losing money on every customer; unsustainable.
- **1–2** — barely breaking even; not yet scalable.
- **3–4** — healthy, efficient growth (the classic **3:1** benchmark).
- **> 5** — strong, but may signal **under-investment** in growth (you could afford to
  spend more to acquire).

Why it matters
--------------

It is the headline KPI for SaaS, subscription and D2C businesses and a key lens for
investors judging scalability. It is best read **alongside the payback period** — a
great ratio still hurts cash flow if CAC takes too long to recover.
"""

MINDMAP.update({
    "LTV:CAC Ratio": [
        "LTV (Customer Lifetime Value)", "CAC (Customer Acquisition Cost)",
        "Net LTV (sometimes called Contribution LTV)",
        "Gross LTV (Customer Lifetime Value)",
        "Blended CAC (Customer Acquisition Cost)",
    ],
    "Net LTV (sometimes called Contribution LTV)": [
        "Gross LTV (Customer Lifetime Value)", "LTV:CAC Ratio", "Gross Margin",
        "Predictive LTV (pLTV)", "Revenue per User (RPU / ARPU)",
    ],
    "Gross LTV (Customer Lifetime Value)": [
        "Net LTV (sometimes called Contribution LTV)",
        "Revenue per User (RPU / ARPU)", "Churn", "Retention",
        "Predictive LTV (pLTV)",
    ],
})


CONTENT["Customer Lifetime"] = r"""
What it is
----------

**Customer lifetime** is the average length of time a customer keeps buying from or
using a business before **churning**. It is measured in time units (months, years)
and is a key *input* to lifetime value (LTV) — LTV is the revenue or profit earned
across that span.

Formula
-------

If you know the churn rate (the share of customers lost per period), the expected
lifetime is its reciprocal:

.. math::

   \text{Customer Lifetime} \approx \frac{1}{\text{Churn Rate}}.

This follows from modelling churn as a constant per-period probability: the expected
number of periods until a customer leaves is :math:`1/\text{churn}`.

Worked example
--------------

At 5% monthly churn,

.. math::

   \text{Customer Lifetime} = \frac{1}{0.05} = 20 \text{ months}.

How it's used
-------------

It plugs straight into LTV and unit economics:

.. math::

   \text{LTV} = \text{ARPU} \times \text{Gross Margin} \times \text{Customer Lifetime},

and the resulting LTV is compared against CAC, with **LTV:CAC ≥ 3** as the healthy
benchmark.

Limitations
-----------

- **Constant-churn assumption** — real churn is highest early and falls for loyal
  cohorts, so :math:`1/\text{churn}` can mislead.
- **Sensitivity** — because lifetime is :math:`1/\text{churn}`, a small churn error
  swings it a lot (5% → 20 months, but 4% → 25 months).
- For retention-driven businesses, refine it with **survival analysis** or
  **cohort analysis** rather than the shortcut.
"""

CONTENT["Cohort-Based LTV (Simple Version)"] = r"""
What it is
----------

**Cohort-based LTV** estimates lifetime value from *observed* behaviour rather than a
churn assumption. Customers are grouped into **cohorts** by when they joined (the
January 2025 cohort, the February 2025 cohort, and so on), and you track how much
revenue each cohort generates month by month, then sum it — so the retention curve is
*measured*, not assumed.

The method
----------

1. **Define cohorts** — e.g. everyone acquired in January 2025.
2. **Track average revenue per customer** for each month after signup (month 0,
   month 1, and so on).
3. **Accumulate** that revenue until the cohort stabilises or churns out.
4. **Estimate the tail** — if the curve has flattened, take the total; if not, fit a
   simple decay (exponential or linear) to project the remaining months.

Formula
-------

Summing average per-customer revenue across months, optionally discounted to present
value:

.. math::

   \text{LTV} = \sum_{t=0}^{T} \frac{\text{Avg Revenue per Customer in month } t}{(1 + r)^t},

where :math:`T` is the months tracked and :math:`r` is an optional discount rate
(often dropped in a simple calculation).

Worked example
--------------

A 100-customer January cohort with average per-customer revenue of $100, 40, 38, 35,
32, 30 over months 0–5 gives

.. math::

   \text{LTV (6 months)} = 100 + 40 + 38 + 35 + 32 + 30 = 275 \text{ per customer},

with later months projected via a decay assumption if revenue is still falling.

Why it beats the churn shortcut
-------------------------------

Because it uses **real observed retention and spend**, cohort LTV captures the early
drop-off and long-tail loyalty that a flat :math:`1/\text{churn}` misses, and it
exposes differences by acquisition month, channel or segment — making it easy to see
whether **retention is improving or worsening** over time.
"""

CONTENT["Predictive LTV (pLTV)"] = r"""
What it is
----------

**Predictive LTV (pLTV)** forecasts a customer's (or cohort's) lifetime value *before*
their lifecycle is complete. Where historical LTV looks backward, pLTV is
**forward-looking**: it predicts future revenue and retention from early signals using
statistical or machine-learning models, answering "how much will this customer be
worth, given what we know now?".

Why it matters
--------------

It lets you judge **CAC vs LTV early** (no waiting years), optimise marketing in
**real time** (how much to bid for an ad impression), and spot **high-value customers
early** for targeted retention.

Methods
-------

- **Rule-based** — use early behaviour (first-week spend, first-month activity) as a
  proxy: "users who spend $20+ in week 1 are 3× more valuable at 12 months".
- **Cohort extrapolation** — fit a decay curve (exponential, Pareto, Weibull, BG/NBD)
  to historical cohorts and apply it to newer ones.
- **Probabilistic models** — **Pareto/NBD**, **BG/NBD** and **Gamma-Gamma** estimate
  purchase frequency and monetary value separately; standard in marketing analytics.
- **Machine learning** — regression/ML models (gradient-boosted trees, random
  forests, neural nets) predict revenue over a horizon from rich features.

Formula (conceptual)
--------------------

.. math::

   \text{pLTV}_i = \sum_{t=1}^{T} \mathbb{E}\!\left[\text{Revenue}_{i,t} \mid X_i\right],

summing the model's expected revenue for customer :math:`i` over horizon :math:`T`,
given features :math:`X_i` (behaviour, demographics, channel, engagement).

Worked example
--------------

Two-week-old customers: A visits 10 times and spends $50; B visits twice and spends
$5. Historical cohorts show high-activity users average a $600 twelve-month LTV and
low-activity users $60 — so pLTV(A) ≈ $600 and pLTV(B) ≈ $60, flagging A for priority
retention even this early.

Pitfalls and edge cases
-----------------------

- Needs a solid **data history** to train on.
- **Model drift** — predictions degrade as behaviour and market shift, so monitor and
  retrain.
- Always **validate** predicted vs actual LTV as cohorts mature.
"""

MINDMAP.update({
    "Customer Lifetime": [
        "Churn", "Retention", "Gross LTV (Customer Lifetime Value)",
        "LTV (Customer Lifetime Value)", "Cohort-Based LTV (Simple Version)",
    ],
    "Cohort-Based LTV (Simple Version)": [
        "Cohort", "Predictive LTV (pLTV)", "Customer Lifetime",
        "Gross LTV (Customer Lifetime Value)", "Retention",
    ],
    "Predictive LTV (pLTV)": [
        "Cohort-Based LTV (Simple Version)", "LTV (Customer Lifetime Value)",
        "Customer Lifetime", "Churn", "Gross LTV (Customer Lifetime Value)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Margins & Customer Acquisition Cost  (growth)
# ----------------------------------------------------------------------

CONTENT["Gross Margin"] = r"""
What it is
----------

**Gross margin** is the share of revenue left after subtracting the **cost of goods
sold (COGS)** — the direct cost of producing and delivering the product. It measures
the efficiency of **core operations**, before overhead, sales, marketing or admin. It
answers: for every $1 of sales, how much is kept after paying to make the product?

Formula
-------

.. math::

   \text{Gross Margin} = \frac{\text{Revenue} - \text{COGS}}{\text{Revenue}} \times 100\%,

where COGS is the direct cost of the product or service — materials, production
labour, and for SaaS, hosting and infrastructure fees.

Worked example
--------------

With revenue $1,000,000 and COGS $400,000,

.. math::

   \text{Gross Margin} = \frac{1{,}000{,}000 - 400{,}000}{1{,}000{,}000} \times 100\% = 60\%,

i.e. the company keeps 60 cents of every sales dollar after production costs.

Interpretation
--------------

- **High margin** — strong pricing power and efficient delivery; software is often
  70–90%.
- **Low margin** — heavy production or operational cost; retail and manufacturing run
  20–40%.

Gross vs net margin
-------------------

Gross margin subtracts **COGS only**; **net margin** subtracts *all* expenses (sales,
marketing, R&D, admin, interest, tax). Gross margin speaks to core operations, net
margin to overall profitability.

Why it anchors unit economics
-----------------------------

Gross margin is the bridge from revenue to profit in customer metrics: **Net
(Contribution) LTV = Gross LTV × Gross Margin**, so it is what turns a top-line LTV
into the profit number you actually compare against CAC.
"""

CONTENT["Fully Loaded CAC (Customer Acquisition Cost)"] = r"""
What it is
----------

**Fully Loaded CAC** is the **all-in cost of acquiring a customer** — every direct and
indirect sales-and-marketing expense, not just ad spend. It is the most complete CAC,
answering "what does it *really* cost to win a customer if we count everything?".

Formula
-------

.. math::

   \text{Fully Loaded CAC} = \frac{\text{All Sales \& Marketing Expenses}}{\text{New Customers Acquired}},

where the numerator includes paid media, organic/content/SEO/PR spend, sales and
marketing **salaries and commissions**, **tools** (CRM, automation, analytics),
**events and sponsorships**, and outsourced agencies.

Worked example
--------------

In a quarter: paid ads $50,000 + SEO and content $20,000 + sales salaries $40,000 +
tools $15,000 + events $10,000 = **$135,000**, against **1,500** new customers:

.. math::

   \text{Fully Loaded CAC} = \frac{135{,}000}{1{,}500} = 90 \text{ per customer}.

Why it's useful
---------------

It gives the **true cost** per customer — the number investors and CFOs want — so the
comparison against LTV is realistic, and it surfaces overhead inefficiency (too much
spend on staff or tools).

The CAC family
--------------

Each CAC answers a different question:

- **Paid CAC** — efficiency of ads only.
- **Organic CAC** — efficiency of content, SEO and brand.
- **Blended CAC** — a quick overall snapshot (paid + organic).
- **Fully Loaded CAC** — the most complete, all-in cost; best for long-term financial
  health.

Limitations
-----------

- Needs **detailed expense tracking** to compute.
- Looks **high** next to Paid CAC, especially for small teams.
- **Attribution** of multi-touch journeys is still hard.
"""

CONTENT["Organic CAC (Customer Acquisition Cost)"] = r"""
What it is
----------

**Organic CAC** is the average cost to acquire a customer from **non-paid channels** —
SEO and organic search, referrals and word-of-mouth, direct traffic, unpaid social,
and email newsletters. It **excludes** paid advertising.

Formula
-------

.. math::

   \text{Organic CAC} = \frac{\text{Organic Marketing Spend}}{\text{Customers from Organic Channels}},

where organic spend is the salaries of the content/SEO team, marketing tools, hosting
and PR, and the denominator is customers attributed to those non-paid sources.

Worked example
--------------

With a content + SEO team and tools costing $30,000 and 1,000 customers from organic
traffic,

.. math::

   \text{Organic CAC} = \frac{30{,}000}{1{,}000} = 30 \text{ per customer}.

Why it's useful
---------------

It captures the **long-term payoff** of content, brand, SEO and referrals — usually a
**lower CAC than paid**, and the other half (with Paid CAC) of the blended picture.

Limitations and the attribution lag
-----------------------------------

- **Attribution is tricky** — organic journeys are multi-touch and hard to credit.
- **Costs lag results** — content published today may bring customers months later,
  so a quarter's spend and that quarter's organic customers are mismatched in time.
- It looks **cheap**, but only after a sustained time investment to build.

Paid vs organic vs blended
--------------------------

**Paid CAC** is the short-term growth lever (more expensive); **organic CAC** is
cheaper long-run but slower and harder to attribute; **Blended CAC** mixes both into
one snapshot — which is why blended CAC drifts as the paid/organic mix shifts with
scale.
"""

MINDMAP.update({
    "Gross Margin": [
        "Net LTV (sometimes called Contribution LTV)",
        "Gross LTV (Customer Lifetime Value)", "LTV:CAC Ratio",
        "Valuation Metric", "SaaS (Software as a Service)",
    ],
    "Fully Loaded CAC (Customer Acquisition Cost)": [
        "Paid CAC (Customer Acquisition Cost)",
        "Organic CAC (Customer Acquisition Cost)",
        "Blended CAC (Customer Acquisition Cost)",
        "CAC (Customer Acquisition Cost)", "LTV:CAC Ratio",
    ],
    "Organic CAC (Customer Acquisition Cost)": [
        "Paid CAC (Customer Acquisition Cost)",
        "Blended CAC (Customer Acquisition Cost)",
        "Fully Loaded CAC (Customer Acquisition Cost)",
        "CAC (Customer Acquisition Cost)", "LTV:CAC Ratio",
    ],
})


CONTENT["Paid CAC (Customer Acquisition Cost)"] = r"""
What it is
----------

**Paid CAC** is the average cost to acquire a customer **from paid channels only** —
Google, Facebook and LinkedIn ads, sponsored content, affiliates and paid events. It
isolates the efficiency of bought growth, in contrast to **Blended CAC**, which mixes
paid and organic together.

Formula
-------

.. math::

   \text{Paid CAC} = \frac{\text{Total Paid Marketing Spend}}{\text{Customers from Paid Channels}},

with ad spend, agency fees, creative and event costs in the numerator and only the
customers attributed to paid channels in the denominator.

Worked example
--------------

With $50,000 of paid ad spend bringing 400 customers in a month,

.. math::

   \text{Paid CAC} = \frac{50{,}000}{400} = 125 \text{ per customer}.

Why it's useful
---------------

It shows the **true cost of scaling paid campaigns** — more honest than blended CAC
when growth leans on ads — and lets you compare ROI of paid vs organic acquisition.

Limitations
-----------

- Ignores cheaper organic, referral and SEO channels.
- **Attribution** is hard: customers often touch both paid and organic before
  converting.
- Can look high short-term, before retargeting and LTV payback.

Paid vs blended
---------------

**Blended CAC** (total S&M spend ÷ total customers) is the big-picture efficiency
number; **Paid CAC** (paid spend ÷ paid customers) tells you specifically how
expensive it is to *buy* growth.
"""

CONTENT["Channel-Specific CAC (Customer Acquisition Cost)"] = r"""
What it is
----------

**Channel-specific CAC** is the cost to acquire a customer through a **single
channel** — Facebook Ads, Google Search, LinkedIn, SEO, events, each measured on its
own. Where blended CAC averages everything, this isolates the efficiency of each
channel.

Formula
-------

.. math::

   \text{CAC}_{\text{channel}} = \frac{\text{Total Spend on Channel}}{\text{Customers from Channel}},

using channel-specific ad spend, agency fees, tools and creative in the numerator and
the customers directly attributable to that channel in the denominator.

Worked example
--------------

In one month, three channels can look completely different:

- Google Ads — $30,000 / 200 customers = $150
- Facebook Ads — $20,000 / 250 customers = $80
- SEO — $10,000 / 500 customers = $20

Those gaps are **invisible** in a single blended number.

Why it's useful
---------------

It tells you **which channels are most cost-effective**, so you can scale the cheap
ones and fix the expensive ones, and — paired with LTV — gives a per-channel
:math:`\text{LTV:CAC}` to rank channels by true value, not just cost.

Challenges
----------

- **Attribution** — a customer may touch ads, then email, then organic search before
  converting; deciding which channel gets credit is the central difficulty.
- **Lag** — SEO and content investment can take months to pay off, inflating
  short-term CAC.
- **Hidden overhead** — brand-building and shared salaries don't map cleanly to one
  channel.

The data-science angle
----------------------

Getting channel CAC right is fundamentally an **attribution** problem: last-touch and
first-touch are crude, so teams use **multi-touch attribution**, **Shapley-value**
credit, or **marketing-mix modelling (MMM)** — a regression of conversions on
per-channel spend with adstock/lag terms — to assign credit across the journey.

Blended vs channel-specific
---------------------------

**Blended CAC** is the snapshot for board reports and financial health;
**channel-specific CAC** is the optimisation tool for deciding where the next dollar
goes.
"""

CONTENT["Blended CAC (Customer Acquisition Cost)"] = r"""
What it is
----------

**CAC** is the cost to acquire one new customer; **Blended CAC** is the **average
across all channels at once** — paid and organic together, with no split by source.
It is "blended" because it folds ad and campaign spend together with organic effort
(SEO, referrals, inbound, brand) into one number.

Formula
-------

.. math::

   \text{Blended CAC} = \frac{\text{Total Sales \& Marketing Spend}}{\text{Total New Customers}},

counting all sales-and-marketing cost over a period against every new customer in that
period, whatever the source.

Worked example
--------------

With $100,000 of sales and marketing spend and 2,000 new customers in a month,

.. math::

   \text{Blended CAC} = \frac{100{,}000}{2{,}000} = 50 \text{ per customer}.

Why it's useful
---------------

It is a simple, **high-level snapshot** of all-in efficiency — easy to compute and
track over time, and more honest than looking at ad spend alone.

Limitations
-----------

- It **masks channel differences**: paid search at $150 and organic referral at $10
  can average to $50, hiding the inefficiency.
- It is **poor for optimisation** — it doesn't tell you which channel creates value.
- It **drifts** as you scale paid: lean harder on ads and blended CAC rises even if
  nothing else changed.

Where it fits
-------------

Use blended CAC for **big-picture health**, **Paid** and **Organic CAC** to separate
bought from earned growth, **Channel-Specific CAC** to optimise, and always read CAC
against LTV via the **LTV:CAC ratio**.
"""

MINDMAP.update({
    "Paid CAC (Customer Acquisition Cost)": [
        "Blended CAC (Customer Acquisition Cost)",
        "Organic CAC (Customer Acquisition Cost)",
        "Channel-Specific CAC (Customer Acquisition Cost)",
        "CAC (Customer Acquisition Cost)", "LTV:CAC Ratio",
    ],
    "Channel-Specific CAC (Customer Acquisition Cost)": [
        "Blended CAC (Customer Acquisition Cost)",
        "Paid CAC (Customer Acquisition Cost)",
        "Fully Loaded CAC (Customer Acquisition Cost)",
        "CAC (Customer Acquisition Cost)", "LTV:CAC Ratio",
    ],
    "Blended CAC (Customer Acquisition Cost)": [
        "Paid CAC (Customer Acquisition Cost)",
        "Organic CAC (Customer Acquisition Cost)",
        "Channel-Specific CAC (Customer Acquisition Cost)",
        "Fully Loaded CAC (Customer Acquisition Cost)",
        "CAC (Customer Acquisition Cost)", "LTV:CAC Ratio",
    ],
})


# ----------------------------------------------------------------------
# Theme: Lead-gen platforms; Bandits & Bayesian decision-making
# ----------------------------------------------------------------------

CONTENT["Lead-Gen Software"] = r"""
What it is
----------

**Lead-generation software** helps a business **find, attract and capture potential
customers (leads)** and feed them into the sales process. A *lead* is someone who has
shown interest — filled in a form, clicked an ad, downloaded a whitepaper — and the
software automates collecting their contact details, qualifying them, and pushing them
into a CRM or sales funnel.

Core features
-------------

- **Lead capture** — forms, landing pages, pop-ups, chatbots.
- **Data enrichment** — appends company size, industry, location, social profiles.
- **Lead scoring** — ranks leads by likelihood to convert (rules-based or ML).
- **CRM integration** — syncs to Salesforce, HubSpot, Zoho, Pipedrive.
- **Email/SMS automation** — nurture sequences and follow-ups.
- **Analytics** — conversion rates, cost per lead, source effectiveness.

Two families of tooling
-----------------------

- **Inbound** (attract people to you) — landing-page builders (Unbounce, Leadpages),
  SEO/content tools (Semrush, HubSpot), forms and chatbots (Typeform, Drift,
  Intercom).
- **Outbound** (reach out to people) — prospecting databases (ZoomInfo, Apollo,
  Clearbit), cold-email tools (Reply, Mailshake, Outreach), LinkedIn automation.
- **All-in-one** platforms (HubSpot, Zoho, Marketo, ActiveCampaign) combine both.

Benefits and challenges
-----------------------

It saves time over manual prospecting, raises lead quality through enrichment and
scoring, keeps the pipeline full, and makes acquisition **measurable**. The costs:
premium data tools are expensive, contact data goes stale, and you must stay compliant
with **GDPR, CAN-SPAM and CCPA**.

The data-science angle
----------------------

The intelligent core is **lead scoring** — a binary **propensity / classification
model** (logistic regression, gradient-boosted trees) that predicts conversion from
enriched features, so sales focuses on the highest-probability leads. It connects
directly to **conversion rate** and feeds the **CAC** calculation downstream.
"""

CONTENT["Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))"] = r"""
The multi-armed bandit problem
------------------------------

Picture a row of slot machines (**arms**). Each arm :math:`i` pays out from an unknown
reward distribution with mean :math:`\mu_i`, and the goal is to **maximise total reward
over time**. That forces a trade-off between:

- **Exploration** — trying arms to learn their payoffs, and
- **Exploitation** — playing the arm that looks best so far.

Pull only the current best and you may never discover a better arm; explore too much
and you waste pulls on bad ones. **Thompson Sampling** resolves this elegantly.

The Bayesian view
-----------------

Keep a **posterior distribution** over each arm's reward parameter. For Bernoulli
(success/failure) rewards the natural choice is a Beta posterior:

.. math::

   p_i \sim \text{Beta}(\alpha_i, \beta_i),

updated as successes and failures accumulate.

The algorithm
-------------

Each round:

1. **Sample** a value :math:`\tilde{\mu}_i` from every arm's posterior.
2. **Select** the arm with the highest sampled value.
3. **Play** it and observe reward :math:`r`.
4. **Update** that arm's posterior by Bayes' rule.

The intuition is the whole trick: an *uncertain* arm has a wide posterior, so its
samples are sometimes high — it gets explored; an arm you're confident is bad rarely
produces a winning sample — it's quietly dropped.

Worked example — the Bernoulli bandit
-------------------------------------

Arm :math:`i` returns 1 with probability :math:`p_i`. Start from a uniform prior
:math:`p_i \sim \text{Beta}(1,1)`. After :math:`s` successes and :math:`f` failures,
conjugacy gives

.. math::

   p_i \mid \text{data} \sim \text{Beta}(1+s,\, 1+f).

Each round, draw :math:`\tilde{p}_i \sim \text{Beta}(1+s, 1+f)` for every arm and pull
the one with the largest draw.

.. code-block:: python

   import numpy as np
   # alpha, beta hold Beta posterior params per arm (start at 1, 1)
   samples = np.random.beta(alpha, beta)      # one draw per arm
   arm = int(np.argmax(samples))              # play the best sampled arm
   # observe reward r in {0, 1}, then update:
   alpha[arm] += r
   beta[arm]  += 1 - r

Why it works, and how well
--------------------------

Exploration and exploitation are handled **automatically** by the sampling — no manual
exploration rate to tune. As evidence accumulates the posteriors concentrate, and the
best arm comes to dominate. Its **regret** — the gap between the optimal arm's expected
reward and what you actually earned — grows only **logarithmically**,
:math:`O(\log T)`, which is near the theoretical optimum.

Extensions
----------

- **Contextual Thompson Sampling** — condition on covariates :math:`x` (user features)
  via Bayesian linear or logistic regression, so the chosen arm depends on context.
- **Other reward models** — Gaussian posteriors for continuous rewards, Dirichlet for
  categorical.
- **Parallel Thompson Sampling** — for distributed settings with many simultaneous
  users.

Where it fits
-------------

Thompson Sampling is the Bayesian member of the **bandit-algorithm** family and a
direct alternative to fixed-split **A/B testing**: instead of holding a 50/50 split for
a set horizon, it shifts traffic toward the winning variant as evidence builds, cutting
the cost of showing the worse option. Compared with :math:`\epsilon`-greedy it needs no
hand-tuned exploration rate, and compared with UCB it explores via posterior sampling
rather than confidence bounds.
"""

CONTENT["Bayesian Decision Theory (BDT)"] = r"""
Core idea
---------

Bayesian inference hands you a **posterior** :math:`p(\theta \mid D)` over unknown
parameters :math:`\theta` given data :math:`D` — but in practice you don't just want
probabilities, you need to **act**: classify the email, approve the loan, treat the
patient. **Bayesian Decision Theory (BDT)** turns the posterior into an *optimal
decision* under uncertainty.

The three ingredients
---------------------

- **Actions** :math:`a` — the choices available (label spam vs not-spam).
- **States of nature** :math:`\theta` — the unknown truth (the email really is or isn't
  spam).
- **Loss function** :math:`L(a, \theta)` — the cost of taking action :math:`a` when the
  truth is :math:`\theta` (flagging real mail as spam may cost far more than missing a
  spam).

Bayes risk and the optimal rule
-------------------------------

The **Bayes risk** of an action is its posterior expected loss,

.. math::

   R(a \mid D) = \mathbb{E}_{\theta \sim p(\theta \mid D)}\big[L(a, \theta)\big],

and the **Bayes action** is the one that minimises it:

.. math::

   a^*(D) = \arg\min_a R(a \mid D).

By construction this is the choice that does best on average under everything the
posterior knows.

The classification special case
-------------------------------

Let :math:`\theta \in \{C_1, \dots, C_k\}` with posterior class probabilities
:math:`p(C_i \mid x)`. Under **0–1 loss** (0 if correct, 1 if wrong), the Bayes-optimal
classifier reduces to

.. math::

   a^*(x) = \arg\max_i \; p(C_i \mid x),

which is exactly the **MAP (maximum a posteriori)** classifier — pick the most probable
class.

Asymmetric loss shifts the boundary
-----------------------------------

When errors cost differently, BDT moves the decision threshold rather than the 0.5
default. In a medical test where a missed disease (false negative) is worse than a
false alarm, the optimal rule classifies as positive at a **lower** posterior
probability — trading more false alarms for fewer missed cases.

Where it shows up
-----------------

BDT is the formal backbone of decision-making under uncertainty across **ML**
(classification, regression, model selection), **medicine** (treat vs not), **finance**
(portfolio choice under risk) and **engineering / reinforcement learning** — including
bandit strategies like Thompson Sampling, which is BDT applied sequentially with the
posterior updated as rewards arrive.
"""

MINDMAP.update({
    "Lead-Gen Software": [
        "CAC (Customer Acquisition Cost)", "Conversion Rate (CR)",
        "Customer Segmentation", "Blended CAC (Customer Acquisition Cost)",
        "Cross-Selling",
    ],
    "Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))": [
        "Bandit Algorithms", "Bayesian Decision Theory (BDT)", "A/B Testing",
        "Bayesian Sequential Testing", "Posterior",
        "Prior Belief (or Prior Probability)",
    ],
    "Bayesian Decision Theory (BDT)": [
        "Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))",
        "Loss Functions", "Posterior", "Bayes' Theorem", "Posterior Probability",
        "Bandit Algorithms",
    ],
})


# ----------------------------------------------------------------------
# Theme: Bayesian forecasting & Bayesian A/B decisions  (bayes)
# ----------------------------------------------------------------------

CONTENT["Bayesian Time Series"] = r"""
Why go Bayesian for time series
-------------------------------

Classical models — ARIMA, exponential smoothing — fix their parameters and estimate
them by maximum likelihood, returning a single point forecast. Real forecasting often
needs more: **uncertainty quantification** (a full distribution of future values, not
one number), **flexibility** (priors encoding trend, smoothness, seasonality, and
handling missing data), and **online updating** (re-forecasting as new data lands).
Bayesian methods deliver all three through **posterior distributions**.

The general framework
---------------------

Model a series :math:`y_{1:T}` with latent parameters :math:`\theta`:

.. math::

   p(\theta \mid y_{1:T}) \propto p(y_{1:T} \mid \theta)\, p(\theta),

where the prior :math:`p(\theta)` encodes structure (smoothness, trend, seasonality)
and the likelihood :math:`p(y_{1:T} \mid \theta)` is often Gaussian or Poisson.
Forecasts come from the **posterior predictive**, which integrates over parameter
uncertainty rather than plugging in a point estimate:

.. math::

   p(y_{T+h} \mid y_{1:T}) = \int p(y_{T+h} \mid \theta, y_{1:T})\, p(\theta \mid y_{1:T})\, d\theta.

The model families
------------------

- **Bayesian ARIMA/ARMA** — ARIMA with priors on :math:`\phi, \theta, \sigma^2`;
  inference by MCMC or variational methods, yielding posterior forecasts.
- **State-space models / dynamic linear models (DLMs)** — a hidden state evolves and
  emits observations,

  .. math::

     x_t = F x_{t-1} + w_t, \qquad y_t = H x_t + v_t,

  covering local-level and trend-plus-seasonality decompositions; inference via the
  **Kalman filter** (conjugate Gaussian) or a **particle filter** (nonlinear /
  non-Gaussian).
- **Bayesian structural time series (BSTS)** — decomposes into trend + seasonality +
  regressors + holiday effects; the engine behind Google's **CausalImpact** for
  estimating intervention effects (A/B tests, policy evaluation).
- **Gaussian-process time series** — a GP prior on the latent function,
  :math:`f \sim \mathcal{GP}(0, k(t,t'))`, with the kernel :math:`k` encoding
  smoothness and periodicity; flexible but heavy on long series.
- **Bayesian VAR** — multivariate autoregression with shrinkage (Minnesota) priors on
  the coefficient matrices to tame overfitting.

Inference toolkit
-----------------

Conjugate **Gibbs sampling** (Normal–Inverse-Gamma for AR models), **Hamiltonian
Monte Carlo** (Stan, PyMC), **variational inference** for speed, and **particle MCMC**
for nonlinear state spaces.

Trade-offs
----------

The payoff is full **posterior predictive distributions**, uncertainty over both
parameters and future values, principled priors for domain knowledge, and graceful
handling of missing data and regime shifts. The cost is **computation** (MCMC on large
data), the need for **careful prior choice**, and **scaling** limits (GPs on long
series).
"""

CONTENT["Posterior probability of uplift"] = r"""
The question it answers
-----------------------

In an experiment with a binary outcome (say conversion), the **uplift** is the
difference in success rates between treatment and control,

.. math::

   u = p_T - p_C,

and the decision-relevant quantity is not a p-value but the **posterior probability
that the uplift is positive** (or beats a business threshold :math:`\tau`):

.. math::

   \Pr(u > 0 \mid \text{data}) \quad\text{or}\quad \Pr(u > \tau \mid \text{data}).

Beta–Binomial model (binary outcomes)
-------------------------------------

With :math:`s_T` successes in :math:`n_T` trials (treatment) and :math:`s_C` in
:math:`n_C` (control), conjugate Beta priors give Beta posteriors:

.. math::

   p_T \mid \text{data} \sim \text{Beta}(\alpha_T + s_T,\; \beta_T + n_T - s_T),
   \qquad
   p_C \mid \text{data} \sim \text{Beta}(\alpha_C + s_C,\; \beta_C + n_C - s_C).

The difference :math:`u = p_T - p_C` has no closed-form CDF, so estimate it by **Monte
Carlo**: draw :math:`p_T^{(m)}` and :math:`p_C^{(m)}` from their posteriors, form
:math:`u^{(m)} = p_T^{(m)} - p_C^{(m)}`, and approximate

.. math::

   \Pr(u > \tau \mid \text{data}) \approx \frac{1}{M}\sum_{m=1}^M \mathbf{1}\{u^{(m)} > \tau\}.

The empirical quantiles of :math:`\{u^{(m)}\}` give a **credible interval** for the
uplift.

.. code-block:: python

   import numpy as np
   pT = np.random.beta(1 + sT, 1 + nT - sT, size=200_000)
   pC = np.random.beta(1 + sC, 1 + nC - sC, size=200_000)
   u  = pT - pC
   prob_uplift = np.mean(u > tau)          # Pr(u > tau | data)
   ci = np.quantile(u, [0.025, 0.975])     # 95% credible interval

Continuous outcomes (revenue)
-----------------------------

For approximately Normal outcomes with unknown means, a Normal–Inverse-Gamma posterior
makes the mean difference :math:`d = \mu_T - \mu_C` (approximately) Normal, so the
uplift probability is closed-form:

.. math::

   \Pr(d > \tau \mid \text{data}) = 1 - \Phi\!\left(\frac{\tau - \hat{d}}{\text{SE}_{\text{post}}}\right).

Bayesian logistic / hierarchical regression
-------------------------------------------

With covariates and a treatment indicator,

.. math::

   \Pr(Y=1 \mid x, T) = \text{logit}^{-1}\!\big(\beta_0 + \beta^\top x + \gamma T + \delta^\top (x \cdot T)\big),

per-draw segment uplift is computed from posterior coefficient samples, and
**hierarchical priors** borrow strength across small segments to reduce false
positives.

What to report, and how to decide
---------------------------------

Decision-friendly outputs: the posterior probability :math:`\Pr(u > \tau)`, the
**expected uplift** :math:`\mathbb{E}[u \mid \text{data}]`, a 95% credible interval,
and a **risk-aware rule** — "ship if :math:`\Pr(u > \tau) \ge q`" (e.g. :math:`q =
0.9`). Practical guidance: use weakly informative priors (Beta(1,1) or Beta(0.5,0.5))
for small samples, prefer :math:`\Pr(u > \tau)` over point estimates when a bad launch
is costly, model many arms hierarchically, and — unlike fixed-horizon tests — Bayesian
posteriors support **always-valid** monitoring without naive peeking penalties.
"""

MINDMAP.update({
    "Bayesian Time Series": [
        "Time Series", "Bayesian Inference.", "Posterior",
        "Bayesian Decision Theory (BDT)", "Bayesian Neural Networks (BNNs)",
    ],
    "Posterior probability of uplift": [
        "A/B Testing", "Conversion Rate Uplift", "Bayesian Sequential Testing",
        "Posterior", "Conversion Rate (CR)", "Incremental Conversions",
    ],
})


# ----------------------------------------------------------------------
# Theme: Bayesian ML & approximate inference  (bayes)
# ----------------------------------------------------------------------

CONTENT["Gaussian Processes (GPs)"] = r"""
The big idea
------------

A **Gaussian process (GP)** is a **distribution over functions**. Just as a Gaussian
over numbers is fixed by a mean and variance, a GP over functions is fixed by a **mean
function** and a **covariance function (kernel)**. Rather than positing parameters
(like neural-network weights) and fitting them, a GP says directly: *here is the family
of functions I believe in, with uncertainty around it.*

Formal definition
-----------------

.. math::

   f(x) \sim \mathcal{GP}\big(m(x),\, k(x, x')\big),

with mean :math:`m(x) = \mathbb{E}[f(x)]` and kernel
:math:`k(x, x') = \operatorname{Cov}(f(x), f(x'))`. The defining property: **any**
finite set of inputs has a joint multivariate Gaussian,

.. math::

   (f(x_1), \dots, f(x_n)) \sim \mathcal{N}(\mathbf{m}, K),

where :math:`K_{ij} = k(x_i, x_j)` is built from the kernel.

Kernels encode your assumptions
-------------------------------

The kernel decides how correlated nearby inputs are:

- **RBF / squared-exponential** —
  :math:`k(x, x') = \exp\!\big(-\lVert x - x' \rVert^2 / 2\ell^2\big)` — smooth
  functions, with the length-scale :math:`\ell` setting how fast correlation decays.
- **Linear** — straight-line trends.
- **Periodic** — repeating patterns.

Kernels **compose** (sum or product) to build structure like trend + seasonality.

Posterior inference
-------------------

Given training data :math:`(X, y)` and test inputs :math:`X_*`, the GP prior makes
:math:`y` and the test values :math:`f_*` jointly Gaussian; **conditioning** on
:math:`y` yields a Gaussian posterior over :math:`f_*` with both a **mean prediction**
and a **variance**. So a GP returns a smooth curve *and* a calibrated confidence band —
uncertainty for free.

Cost and trade-offs
-------------------

The catch is computation: conditioning inverts an :math:`n \times n` covariance matrix
at :math:`O(n^3)` cost, so exact GPs suit **small-to-medium** data and need sparse /
inducing-point approximations to scale. Kernel choice is critical — the wrong kernel
gives poor predictions. In scikit-learn, ``GaussianProcessRegressor`` implements this.

GPs and neural networks
-----------------------

A GP places uncertainty over **functions**; a **Bayesian neural network** places it
over **weights**. The two meet at a famous limit: an **infinitely wide** neural network
with random weights *converges to a Gaussian process*.
"""

CONTENT["Bayesian Neural Networks (BNNs)"] = r"""
From point estimates to distributions
-------------------------------------

A standard neural network fixes its weights :math:`W` after training — gradient descent
finds a single best **point estimate**. That captures no **uncertainty**: on limited or
noisy data the network can be confidently wrong. A **Bayesian neural network (BNN)**
fixes this by treating the weights as **random variables** with distributions.

The Bayesian view of weights
----------------------------

Instead of one weight vector, infer a **posterior** over weights:

.. math::

   p(W \mid D) \propto p(D \mid W)\, p(W),

with prior :math:`p(W)` (often a zero-mean Gaussian), likelihood :math:`p(D \mid W)`
(how well the weights explain the data) and posterior :math:`p(W \mid D)`. Training
therefore yields a **whole distribution** of plausible networks, not one.

Prediction integrates over weights
----------------------------------

Predictions average over every weight setting, weighted by posterior probability:

.. math::

   p(y \mid x, D) = \int p(y \mid x, W)\, p(W \mid D)\, dW.

This separates two kinds of uncertainty: **aleatoric** (noise in the labels) and
**epistemic** (not enough data to pin down the weights) — and the output is a
*distribution* over predictions, not a point.

Why it's hard, and how it's done
--------------------------------

That integral is **intractable**, so BNNs rely on approximation:

- **Variational inference** — approximate the posterior with a simpler :math:`q(W)`.
- **MCMC** — sample weights from the posterior (accurate but slow).
- **MC Dropout** (Gal & Ghahramani, 2016) — keeping dropout *on at test time*
  approximates sampling from a weight distribution, a cheap practical trick.

Why it matters
--------------

A BNN **knows when it doesn't know**: it resists overconfident errors and generalises
better, which is decisive in medicine, self-driving and finance where the *cost of an
uncertain prediction* matters. The price is heavier training and inference, sensitivity
to the approximation chosen, and less mainstream tooling than standard nets.
"""

CONTENT["Variational Inference (VI)"] = r"""
The problem it solves
---------------------

Bayesian inference wants the **posterior**

.. math::

   p(\theta \mid x) = \frac{p(x \mid \theta)\, p(\theta)}{p(x)},

but the **evidence** in the denominator,
:math:`p(x) = \int p(x \mid \theta)\, p(\theta)\, d\theta`, is an integral over all
parameters that is **intractable** in most real models. MCMC tackles this by sampling;
**variational inference (VI)** tackles it by **optimisation**.

The core idea
-------------

Replace the hard posterior with the closest member of a **simpler family**
:math:`q(\theta)` (say, Gaussians):

.. math::

   q^*(\theta) = \arg\min_q \; \operatorname{KL}\!\big(q(\theta) \,\Vert\, p(\theta \mid x)\big),

measuring closeness by **Kullback–Leibler divergence**. Inference becomes a search for
the best-fitting approximation.

The ELBO
--------

We can't minimise that KL directly (it contains the unknown posterior), so VI instead
**maximises the Evidence Lower Bound**:

.. math::

   \mathcal{L}(q) = \mathbb{E}_{q(\theta)}\big[\log p(x, \theta) - \log q(\theta)\big].

Maximising the ELBO is *equivalent* to minimising the KL, and the ELBO is a genuine
**lower bound** on :math:`\log p(x)` — so gradient-based optimisation of
:math:`\mathcal{L}` drives :math:`q` toward the posterior.

How it's done in practice
-------------------------

Pick a **variational family** :math:`q(\theta; \phi)` (e.g. a Gaussian's mean and
variance), then optimise :math:`\phi` to maximise the ELBO and use :math:`q` as the
posterior. Common machinery:

- **Mean-field** — assume independence, :math:`q(\theta) = \prod_i q_i(\theta_i)`.
- **CAVI** — coordinate ascent, updating each factor in turn.
- **Stochastic VI** — mini-batch stochastic optimisation for large data.
- **Reparameterisation trick** — write :math:`\theta = g(\phi, \epsilon)` with noise
  :math:`\epsilon` so gradients flow through samples (the engine of **variational
  autoencoders**).

Where it shows up
-----------------

Topic models (latent Dirichlet allocation), variational autoencoders, Bayesian neural
networks and probabilistic graphical models. The trade-off vs **MCMC**: VI is **faster
and scalable** but gives a *biased* approximation (only as good as the family), whereas
MCMC is asymptotically exact but slower.
"""

MINDMAP.update({
    "Gaussian Processes (GPs)": [
        "Bayesian Neural Networks (BNNs)", "Bayesian Time Series",
        "Variational Inference (VI)", "Posterior", "Bayesian Inference.",
    ],
    "Bayesian Neural Networks (BNNs)": [
        "Variational Inference (VI)", "Gaussian Processes (GPs)",
        "MCMC (Markov Chain Monte Carlo)", "Posterior", "Bayesian Inference.",
    ],
    "Variational Inference (VI)": [
        "MCMC (Markov Chain Monte Carlo)", "Bayesian Neural Networks (BNNs)",
        "Kullback–Leibler (KL) Divergence", "Posterior", "Bayesian Inference.",
    ],
})


# ----------------------------------------------------------------------
# Theme: Inference engines & experimentation foundations  (bayes / inference)
# ----------------------------------------------------------------------

CONTENT["MCMC (Markov Chain Monte Carlo)"] = r"""
What it is
----------

**MCMC (Markov Chain Monte Carlo)** is a family of algorithms for **drawing samples
from complex probability distributions** — above all the **posterior** in Bayesian
inference, which usually has no closed form. Rather than computing the distribution
analytically, MCMC generates a long **sequence of samples** that approximate it: it
lets you approximate posteriors by simulation when the math is intractable.

The two ideas in the name
-------------------------

- **Markov chain** — a sequence of states where the next depends only on the current
  one (not the full history); this structure is what lets the sampler *walk* through
  the distribution.
- **Monte Carlo** — approximating integrals and expectations by repeated random
  sampling (the classic toy is estimating :math:`\pi` from random points in a square).

Put together: build a Markov chain whose **stationary distribution is the posterior**,
run it long enough, and its samples approximate that posterior.

The estimator
-------------

Most Bayesian quantities are posterior expectations,

.. math::

   \mathbb{E}[f(\theta) \mid \text{data}] = \int f(\theta)\, p(\theta \mid \text{data})\, d\theta,

intractable in high dimensions — so MCMC approximates it by the sample average over the
chain,

.. math::

   \mathbb{E}[f(\theta) \mid \text{data}] \approx \frac{1}{N} \sum_{i=1}^N f(\theta_i),
   \qquad \theta_i \sim \text{MCMC}.

The algorithms
--------------

- **Metropolis–Hastings** — propose a move, then accept or reject it by a ratio of
  posterior densities.
- **Gibbs sampling** — a special case that updates each parameter from its full
  conditional; natural for hierarchical models.
- **Hamiltonian Monte Carlo (HMC)** — uses gradients to propose distant,
  high-acceptance moves; the engine of **Stan**.
- **NUTS (No-U-Turn Sampler)** — an adaptive HMC that tunes step size and trajectory
  length automatically; the default in **PyMC** and **Stan**.

Example
-------

Coin toss with 7 heads, 3 tails and a :math:`\text{Beta}(2,2)` prior gives a
:math:`\text{Beta}(9,5)` posterior. Here it's closed-form, but if it weren't, MCMC
would draw thousands of samples of :math:`p` to estimate the posterior mean, variance
and credible interval.

Using it well
-------------

MCMC handles very complex, high-dimensional posteriors and yields **full samples** (any
posterior summary is then just a function of them). The costs: it is **computationally
expensive**, needs **tuning** (burn-in, thinning, proposal scale), and requires
**convergence diagnostics** — discard a **burn-in**, then check the
potential-scale-reduction :math:`\hat{R}` (near 1.0) and the effective sample size to
confirm the chains have mixed. Versus **variational inference**, MCMC is asymptotically
exact but slower; VI is faster but approximate.
"""

CONTENT["Sequential Settings"] = r"""
What it is
----------

A **sequential setting** is any framework where **data arrives over time** and you can
make decisions — stop or continue, accept or reject, reallocate traffic — **as the data
accumulates**, rather than only at a single fixed end point. It is the opposite of a
**fixed-horizon** design, where you pre-commit a sample size and analyse once at the
end.

What makes it different
-----------------------

Data comes one observation (or small batch) at a time, decisions can be **adaptive**,
and — crucially — naive repeated testing **inflates the Type I error rate**: every time
you "peek" at a frequentist p-value and consider stopping, you get extra chances to hit
significance by chance. Sequential settings therefore demand **specialised methods**.

Where it shows up
-----------------

- **Clinical trials** — interim analyses with stopping rules for efficacy, harm or
  futility (continuing a harmful treatment is unethical).
- **A/B testing** — users arrive over time and teams want to peek mid-experiment;
  valid sequential methods let them.
- **Online learning / bandits** — algorithms learn as data streams, shifting traffic
  to better variants.

The three families of methods
-----------------------------

- **Frequentist** — the **SPRT (sequential probability ratio test)** checks a
  likelihood ratio continuously; **group-sequential** designs (O'Brien–Fleming,
  Pocock) analyse at pre-planned checkpoints with **α-spending** to keep the overall
  error controlled.
- **Bayesian** — naturally sequential: the **posterior updates continuously** as data
  arrive, and you stop when a posterior probability crosses a threshold (say
  :math:`\Pr(\text{uplift} > 0) > 0.95`), with no peeking penalty.
- **Adaptive / online** — multi-armed bandits (:math:`\epsilon`-greedy, Thompson
  sampling) and reinforcement learning, built for sequential decisions.

Why it matters
--------------

Stopping early saves **time and cost**, avoids **ethically** continuing a clearly
harmful trial, lets businesses **adapt quickly**, and matches how many ML algorithms
actually receive data — provided the right method preserves statistical validity.
"""

CONTENT["Frequentist"] = r"""
What it is
----------

The **frequentist** approach defines probability as the **long-run frequency** of an
event over infinitely many repeated trials. Parameters — a population mean, a
conversion rate, a treatment effect — are treated as **fixed but unknown constants**,
and all the uncertainty comes from the **randomness of the data (the sample)**, never
from the parameter itself.

Key principles
--------------

- **Probability = frequency** — "P(heads) = 0.5" means about half of many flips land
  heads.
- **Parameters fixed, data random** — the true mean :math:`\mu` is a fixed constant;
  the sample mean :math:`\bar{x}` varies from sample to sample.
- **Inference via repeated sampling** — p-values, confidence intervals and tests are
  all defined by *what would happen if the experiment were repeated many times*.

The toolkit
-----------

Point estimates (the sample mean :math:`\bar{x}` for :math:`\mu`), **confidence
intervals** (an interval procedure that captures the true parameter in, say, 95% of
repeated experiments), **hypothesis tests** (a null :math:`H_0` and a p-value — the
probability of data at least as extreme as observed, *assuming* :math:`H_0`), and
**maximum likelihood estimation**.

A subtlety worth stating
------------------------

A 95% confidence interval does **not** mean "95% probability the true value is in this
interval" — under frequentism the parameter is fixed, so it's either in or out. It means
*the procedure* covers the truth 95% of the time across repetitions. The "95%
probability the mean lies here" reading is the **Bayesian credible interval**.

Example — an A/B test
------------------------

To ask whether variant B beats A, treat the true rates :math:`p_A, p_B` as fixed,
estimate them with sample proportions :math:`\hat{p}_A, \hat{p}_B`, run a
**two-proportion z-test**, and reject :math:`H_0` if the p-value is below 0.05.

Frequentist vs Bayesian
-----------------------

The two paradigms differ along clear lines: probability as **long-run frequency** vs
**degree of belief**; parameters as **fixed constants** vs **random variables**;
uncertainty from **data only** vs **prior + data**; inference via **p-values and
confidence intervals** vs **posteriors and credible intervals**; and on sequential
data, frequentist peeking **inflates error** (needs corrections) while Bayesian updating
is **continuously valid**. Frequentist methods remain dominant in medicine, the social
sciences and classic A/B testing.
"""

MINDMAP.update({
    "MCMC (Markov Chain Monte Carlo)": [
        "Variational Inference (VI)", "Bayesian Inference.", "Posterior",
        "Beta Distribution", "Bayesian Neural Networks (BNNs)",
    ],
    "Sequential Settings": [
        "Bayesian Sequential Testing",
        "Traditional A/B Test (Fixed-Horizon A/B Test)",
        "Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))",
        "A/B Testing", "Frequentist", "Bayesian Stopping Rules",
    ],
    "Frequentist": [
        "Bayesian Inference.", "A/B Testing",
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Sequential Settings",
        "Posterior",
    ],
})


# ----------------------------------------------------------------------
# Theme: Beta–Binomial conjugacy & Bayesian updating  (bayes / probstats)
# ----------------------------------------------------------------------

CONTENT["Beta Distribution"] = r"""
What it is
----------

The **Beta distribution** is a continuous distribution on the interval :math:`[0, 1]`,
which makes it the natural way to model a **probability or proportion**. It has two
**shape parameters**, :math:`\alpha` and :math:`\beta`, and density

.. math::

   f(p \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)}\, p^{\alpha - 1} (1 - p)^{\beta - 1},
   \qquad 0 \le p \le 1,

where :math:`B(\alpha, \beta)` is the Beta function, the normalising constant.

Summary statistics
------------------

.. math::

   \mathbb{E}[p] = \frac{\alpha}{\alpha + \beta}, \qquad
   \operatorname{Var}(p) = \frac{\alpha\beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)},

and, when :math:`\alpha, \beta > 1`, the mode is
:math:`(\alpha - 1)/(\alpha + \beta - 2)`. A larger :math:`\alpha + \beta` means a more
concentrated (lower-variance) belief.

The shape gallery
-----------------

The two parameters give the family enormous flexibility:

- :math:`\alpha = \beta = 1` — **uniform** on :math:`[0, 1]`.
- :math:`\alpha > \beta` — skewed toward 1; :math:`\alpha < \beta` — skewed toward 0.
- :math:`\alpha, \beta > 1` — unimodal (a single bump).
- :math:`\alpha, \beta < 1` — U-shaped (mass piling up near 0 and 1).

Why it's central: conjugacy
---------------------------

The Beta is the **conjugate prior** for the **Binomial likelihood**. With a
:math:`\text{Beta}(\alpha, \beta)` prior and :math:`k` successes in :math:`n` trials,
the posterior is again Beta:

.. math::

   p \mid \text{data} \sim \text{Beta}(\alpha + k,\; \beta + n - k).

So Bayesian updating is just *adding successes to* :math:`\alpha` *and failures to*
:math:`\beta` — no integration required.

Examples
--------

- :math:`\text{Beta}(1,1)` — a flat prior, no preference before data.
- :math:`\text{Beta}(20, 20)` — sharply peaked at 0.5, a strong "fair coin" belief.
- Start from :math:`\text{Beta}(1,1)`, observe 7 heads and 3 tails →
  :math:`\text{Beta}(8, 4)`, mean :math:`8/12 \approx 0.67`.

Where it shows up
-----------------

Bayesian inference over probabilities, **A/B testing** (belief about conversion rates),
reliability (success/failure rates), and any proportion modelling. Its multivariate
generalisation, the **Dirichlet distribution**, plays the same conjugate role for the
categorical/multinomial likelihood. In code, ``scipy.stats.beta`` provides the density,
sampling and quantiles.
"""

CONTENT["Binomial Likelihood"] = r"""
The setup
---------

Model :math:`n` independent Bernoulli (success/failure) trials with success probability
:math:`p`, observing :math:`k` successes. The **binomial PMF** gives the probability of
exactly :math:`k` successes:

.. math::

   P(X = k \mid p) = \binom{n}{k} p^k (1 - p)^{n - k}.

Likelihood vs PMF
-----------------

The **likelihood** is the *same expression read the other way*: with the data
:math:`(k, n)` fixed, it is a function of the unknown :math:`p`,

.. math::

   L(p \mid k, n) = \binom{n}{k} p^k (1 - p)^{n - k} \;\propto\; p^k (1 - p)^{n - k},

dropping the binomial coefficient because it doesn't depend on :math:`p`. The PMF asks
"given :math:`p`, how probable is this data?"; the likelihood asks "given this data, how
well does each :math:`p` explain it?".

Maximum likelihood
------------------

Maximising the log-likelihood,

.. math::

   \ell(p) = k \ln p + (n - k)\ln(1 - p),

and setting the derivative to zero gives the intuitive estimate

.. math::

   \hat{p}_{\text{MLE}} = \frac{k}{n},

the observed success proportion. For :math:`n = 10,\ k = 7`, the likelihood
:math:`\propto p^7 (1-p)^3` peaks at :math:`\hat{p} = 0.7`.

The Bayesian half
-----------------

Pair the binomial likelihood with a **Beta prior** and conjugacy makes the posterior
Beta as well:

.. math::

   \text{Beta}(\alpha, \beta) \times \text{Binomial}(k, n) \;\Rightarrow\;
   \text{Beta}(\alpha + k,\; \beta + n - k).

So the binomial likelihood sits at the centre of **both** worlds — its MLE is the
frequentist sample proportion, and with a Beta prior it gives the classic Beta–Binomial
Bayesian model.
"""

CONTENT["Posterior belief"] = r"""
What it is
----------

**Posterior belief** is your **updated belief about a parameter after seeing data** —
represented, in Bayesian statistics, by the **posterior distribution**. It fuses two
ingredients: the **prior** (what you believed beforehand) and the **likelihood** (what
the data say). In a sentence: *posterior = prior updated by evidence*.

Bayes' theorem
--------------

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)},

with prior :math:`P(\theta)`, likelihood :math:`P(D \mid \theta)`, evidence
:math:`P(D)`, and posterior :math:`P(\theta \mid D)`.

It's a distribution, not a number
---------------------------------

The posterior is a **whole distribution** over parameter values, showing how plausible
each value is *after* the data — and from it you read off summaries (mean, mode,
intervals) or specific **posterior probabilities** of events.

Example — coin toss
-------------------

A uniform prior :math:`\text{Beta}(1,1)`, updated with 7 heads in 10 tosses, gives the
posterior :math:`\text{Beta}(8, 4)`, centred near 0.67 — the updated belief that
:math:`p` is most likely around two-thirds.

Posterior belief vs posterior probability
-----------------------------------------

A useful distinction: the **posterior belief** is the entire distribution
(:math:`p \sim \text{Beta}(8, 4)`), while a **posterior probability** is a single number
pulled from it — for instance :math:`P(p > 0.5 \mid \text{data}) = 0.9`. The latter is
one summary of the former.

Where it shows up
-----------------

A/B testing (belief about a conversion-rate difference), clinical trials (belief about
treatment effect), and any Bayesian model that updates parameter distributions as data
arrive.
"""

MINDMAP.update({
    "Beta Distribution": [
        "Binomial Likelihood", "Posterior belief",
        "Prior Belief (or Prior Probability)",
        "Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))",
        "Bayes' Theorem",
    ],
    "Binomial Likelihood": [
        "Beta Distribution", "Posterior belief", "Bayes' Theorem",
        "Bayesian Inference.", "Frequentist",
    ],
    "Posterior belief": [
        "Prior Belief (or Prior Probability)", "Bayes' Theorem",
        "Posterior Probability", "Beta Distribution", "Binomial Likelihood",
    ],
})


# ----------------------------------------------------------------------
# Theme: The anatomy of Bayes' theorem  (bayes)
# ----------------------------------------------------------------------

CONTENT["Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)"] = r"""
What it is
----------

The **marginal likelihood** — also called the **model evidence** or **integrated
likelihood** — is the probability of the observed data under a model, **averaged over
all possible parameter values**. It is the **denominator** in Bayes' theorem:

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)},
   \qquad
   P(D) = \int P(D \mid \theta)\, P(\theta)\, d\theta.

In words, :math:`P(D)` is the **prior-weighted average** of the likelihood — the
overall probability of the data, considering every parameter value the prior allows.

Two jobs
--------

**1. Normalisation.** Dividing by :math:`P(D)` is what makes the posterior integrate to
1 — without it, prior × likelihood is only *proportional* to the posterior.

**2. Model comparison.** Because it scores how well an *entire model* (not one parameter
setting) predicts the data, the marginal likelihood is the basis of the **Bayes
factor**,

.. math::

   \text{BF} = \frac{P(D \mid M_1)}{P(D \mid M_2)},

which weighs two competing models — a higher marginal likelihood means the model
explains the data better, with a built-in Occam penalty for needless complexity.

Example — coin toss
-------------------

With a uniform prior :math:`\text{Beta}(1,1)` and 7 heads in 10 tosses,

.. math::

   P(D) = \int_0^1 \binom{10}{7} p^7 (1-p)^3 \, dp,

the overall probability of "7 heads out of 10" averaged across all plausible :math:`p`.

Why it's hard
-------------

That integral is usually **intractable** in real models, which is the whole reason
approximate methods exist: the **Laplace approximation**, **variational inference**
(whose ELBO is a *lower bound* on :math:`\log P(D)`), and **MCMC**-based estimators all
exist to approximate the evidence for posterior computation and model selection.
"""

CONTENT["Posterior"] = r"""
What it is
----------

The **posterior** is the **updated probability distribution of a parameter after
observing data** — the new belief formed by combining the **prior** (what you thought
before) with the **likelihood** (what the data say). The one-line version: *posterior =
prior updated with evidence*.

Bayes' theorem
--------------

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)}
   \;\propto\; \underbrace{P(D \mid \theta)}_{\text{likelihood}} \times \underbrace{P(\theta)}_{\text{prior}},

with the marginal likelihood :math:`P(D)` as the normalising constant. The
**proportional** form is what you actually work with: the posterior *shape* is just
prior times likelihood.

What you get from it
--------------------

The posterior is a **distribution**, so it supports direct probability statements — "a
95% probability the conversion rate is between 4% and 6%" — which a frequentist
confidence interval cannot make. Point summaries (the posterior **mean** or
**mode/MAP**) and **credible intervals** are all read off it.

Example — coin toss
-------------------

Uniform prior :math:`\text{Beta}(1,1)`, a Binomial likelihood, and 7 heads in 10 tosses
give

.. math::

   P(p \mid \text{data}) \propto p^7 (1-p)^3 \cdot 1 = \text{Beta}(8, 4),

a posterior centred near 0.67.

The prior washes out
--------------------

A defining property: **the more data you collect, the more the likelihood dominates and
the less the prior matters**. With small samples the prior shapes the answer; with large
ones the posterior is driven almost entirely by the data — which is why honest priors
are cheap insurance, not permanent bias. The posterior is the object *all* Bayesian
decisions are based on.
"""

CONTENT["Prior Belief (or Prior Probability)"] = r"""
What it is
----------

A **prior belief** (or **prior probability**) is your **initial belief about a parameter
before seeing any new data**, encoded as a **prior distribution**. It captures what is
plausible from past studies, domain expertise or reasonable assumptions: *what you think
before you see the evidence*.

Where it sits
-------------

In Bayes' theorem the prior :math:`P(H)` is the term that the likelihood multiplies and
the data update:

.. math::

   P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)}.

Kinds of prior
--------------

- **Informative** — built on strong prior knowledge (a drug's effect is likely 5–10%,
  from past trials).
- **Non-informative / flat** — expresses near-ignorance (every value equally likely,
  e.g. :math:`\text{Beta}(1,1)`).
- **Weakly informative** — adds gentle, realistic bounds to rule out absurd values
  (conversion rates almost never exceed 50%), which stabilises inference without forcing
  a conclusion.

Example — coin toss
-------------------

For the probability of heads :math:`p`: a dogmatic "fair coin" prior fixes
:math:`p = 0.5`; total uncertainty is :math:`\text{Beta}(1,1)` (uniform); "probably fair
but unsure" is :math:`\text{Beta}(20, 20)`, peaked at 0.5. Each is updated by the data
into a posterior.

In A/B testing
--------------

A sensible prior on the conversion-rate difference — centred at 0 with small variance,
reflecting that new features rarely move conversion more than a few points — keeps early
results from over-reacting to noise; the experiment then updates it into a posterior on
the uplift.

The key caveat
--------------

The **choice of prior matters most with small datasets** and fades as data grows (the
likelihood takes over). A weakly informative prior is usually the safe default: enough
structure to regularise, not so much that it overrides the evidence.
"""

MINDMAP.update({
    "Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)": [
        "Posterior", "Prior Belief (or Prior Probability)", "Bayes' Theorem",
        "Variational Inference (VI)", "Binomial Likelihood",
    ],
    "Posterior": [
        "Prior Belief (or Prior Probability)", "Posterior belief", "Bayes' Theorem",
        "Posterior Probability",
        "Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)",
        "Beta Distribution",
    ],
    "Prior Belief (or Prior Probability)": [
        "Posterior", "Bayes' Theorem", "Beta Distribution", "Posterior belief",
        "Binomial Likelihood",
    ],
})


# ----------------------------------------------------------------------
# Theme: Bayes' theorem capstone; inference targets; A/B uplift  (bayes / abtest)
# ----------------------------------------------------------------------

CONTENT["Parameter(s) of Interest"] = r"""
What it is
----------

The **parameter(s) of interest** are the **population characteristics you set out to
estimate or test** — usually unknown population values like a true mean, a true
proportion, a difference in means, or regression coefficients. The whole study or
experiment is designed to learn about them. In short: *the parameter of interest is the
thing you actually care about measuring.*

Parameter vs statistic
----------------------

The crucial distinction: the **parameter** is the fixed (usually unknown) population
value; the **statistic** is the sample quantity you compute to *estimate* it.

- Population mean :math:`\mu` — estimated by the sample mean :math:`\bar{x}`.
- Population proportion :math:`p` — estimated by :math:`\hat{p}`.
- Difference in means :math:`\mu_1 - \mu_2` — estimated by the difference in sample
  means.
- Regression coefficients :math:`\beta` — estimated by :math:`\hat{\beta}`.

Examples
--------

- **A/B test** — the true conversion rates :math:`p_A` and :math:`p_B`, or their
  difference :math:`p_B - p_A`.
- **Medical trial** — the average treatment effect,

  .. math::

     \text{ATE} = P(\text{recovery} \mid \text{drug}) - P(\text{recovery} \mid \text{placebo}).

- **Regression** — the coefficients :math:`\beta_1, \beta_2, \dots` linking predictors
  to the outcome.

Why it matters
--------------

Naming the parameter of interest is the **first step** in designing any study: every
estimator, hypothesis test (:math:`H_0` vs :math:`H_1`) and confidence (or credible)
interval is built to make inferences about it. Frequentists treat it as a fixed unknown
constant; Bayesians put a posterior distribution over it — but in both frameworks it is
the *target* of the analysis.
"""

CONTENT["Bayes' Theorem"] = r"""
What it is
----------

**Bayes' theorem** is the rule for **updating a belief when new evidence arrives**. It
is the single equation that ties together the four quantities of Bayesian inference —
prior, likelihood, evidence and posterior.

The formula
-----------

For a hypothesis :math:`H` and data :math:`D`,

.. math::

   P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)},

where :math:`P(H)` is the **prior**, :math:`P(D \mid H)` the **likelihood**,
:math:`P(D)` the **marginal likelihood** (evidence / normaliser) and
:math:`P(H \mid D)` the **posterior**. Stripped to its working form,

.. math::

   \text{Posterior} \propto \text{Prior} \times \text{Likelihood}.

Worked example — the base-rate trap
-----------------------------------

A disease affects 1% of people; a test is 99% sensitive and has a 5% false-positive
rate. A patient tests positive — how likely are they to be sick? The evidence is

.. math::

   P(+) = \underbrace{0.99 \times 0.01}_{\text{true positive}} + \underbrace{0.05 \times 0.99}_{\text{false positive}} = 0.0099 + 0.0495 = 0.0594,

so

.. math::

   P(\text{disease} \mid +) = \frac{0.0099}{0.0594} \approx 0.167.

Despite a positive result on a "99% accurate" test, the chance of disease is only about
**16.7%** — because the disease is rare, false positives swamp the true ones. This
**base-rate fallacy** is exactly what Bayes' theorem corrects: the rare prior pulls the
posterior far below the test's sensitivity.

Where it shows up
-----------------

Bayesian inference and updating, the **Naive Bayes** classifier and Bayesian networks,
medical diagnostics, **A/B testing** (Bayesian sequential testing, posterior probability
of uplift), fraud and risk estimation, and everyday belief revision under new
information.
"""

CONTENT["Conversion Rate Uplift"] = r"""
What it is
----------

**Conversion-rate uplift** measures the improvement (or decline) in conversion rate of a
**treatment** (variant) relative to a **control** (baseline) — the headline success
metric of an A/B test. It answers: *by how much did the new variant move conversions
versus the baseline?*

Absolute vs relative
--------------------

With :math:`CR_A` the control rate and :math:`CR_B` the treatment rate, there are two
distinct quantities:

.. math::

   \text{Absolute uplift} = CR_B - CR_A \ \text{(percentage points)}, \qquad
   \text{Relative uplift} = \frac{CR_B - CR_A}{CR_A} \times 100\%.

Examples
--------

- **Positive** — control 5%, treatment 6%: absolute uplift **+1 point**, relative uplift
  :math:`(6-5)/5 = +20\%`.
- **Negative** — control 10%, treatment 9.5%: absolute uplift **−0.5 point**, relative
  uplift :math:`(9.5-10)/10 = -5\%`.

Why both matter
---------------

The two can tell very different stories: a **1-point** absolute uplift sounds tiny, but
on a **2%** baseline it is a **50% relative** improvement. Quoting only one can mislead —
report both.

How it drives A/B decisions
---------------------------

Uplift is the primary metric, and the **minimum detectable uplift / effect (MDE)** — the
smallest effect you want to catch given sample size, :math:`\alpha` and power — sets how
big the experiment must be. The decision rule: a **significant positive** uplift → ship;
**not significant** → inconclusive; **significant negative** → stop or rethink. Whether
the uplift is real or just noise is settled by a **two-proportion z-test** (frequentist)
or the **posterior probability of uplift** (Bayesian).
"""

MINDMAP.update({
    "Parameter(s) of Interest": [
        "Frequentist", "Posterior", "Prior Belief (or Prior Probability)",
        "A/B Testing", "True Conversion Rate",
    ],
    "Bayes' Theorem": [
        "Prior Belief (or Prior Probability)", "Posterior",
        "Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)",
        "Binomial Likelihood", "Bayesian Inference.", "Posterior Probability",
    ],
    "Conversion Rate Uplift": [
        "Posterior probability of uplift", "A/B Testing", "Conversion Rate (CR)",
        "True Conversion Rate", "Incremental Conversions",
        "Traditional A/B Test (Fixed-Horizon A/B Test)",
    ],
})


# ----------------------------------------------------------------------
# Theme: A/B testing in practice — stopping rules & platforms  (abtest / platforms)
# ----------------------------------------------------------------------

CONTENT["Bayesian Stopping Rules"] = r"""
What it is
----------

A **Bayesian stopping rule** is a **predefined criterion — based on posterior
probabilities, Bayes factors, or credible intervals — for when to stop collecting data**
in an experiment. Where a frequentist test fixes the sample size to control the Type I
error rate :math:`\alpha`, a Bayesian rule watches the **strength of evidence** and stops
once it is decisive. In a sentence: *stop when the posterior probability of a hypothesis
is high enough.*

Three kinds of rule
-------------------

- **Posterior-probability threshold** — stop and accept :math:`H_1` once
  :math:`P(H_1 \mid \text{data}) > 0.95`, or stop for **futility** if
  :math:`P(H_0 \mid \text{data}) > 0.95`.
- **Bayes-factor threshold** — stop when the Bayes factor
  :math:`\text{BF} = P(\text{data} \mid H_1) / P(\text{data} \mid H_0)` reaches a strong
  level on Jeffreys' scale (:math:`\text{BF} > 10` for :math:`H_1`,
  :math:`\text{BF} < 1/10` for :math:`H_0`; otherwise keep sampling).
- **Credible-interval rule** — stop when the posterior credible interval for the effect
  **excludes zero** (e.g. a 95% interval of :math:`[0.01, 0.03]` for a conversion-rate
  difference).

Why the peeking is allowed
--------------------------

The headline advantage: **continuous monitoring does not inflate false positives** the
way repeated frequentist peeking does. The results are also directly interpretable —
"there is a 97% probability that B beats A" — priors can be folded in, and overwhelming
evidence lets you stop early and save traffic.

Worked example
--------------

Control converts at 5%; the treatment shows 5.6% after 5,000 users, and the posterior
gives :math:`P(p_B > p_A \mid \text{data}) = 0.97`. With a "stop if posterior > 0.95"
rule, the decision is to **stop and roll out B**.

The catch
---------

Stopping rules depend on the **prior** (which can sway the result), are more
**computationally intensive** (often Monte Carlo / MCMC), and ask stakeholders to think
in posterior probabilities rather than the p-values they may know better. Versus
frequentist fixed-horizon or α-spending designs, the trade is intuitive probability
statements and peeking freedom for prior-sensitivity and compute.
"""

CONTENT["Google Experiments"] = r"""
What it is
----------

"**Google Experiments**" is an umbrella for several Google A/B-testing and
experimentation products that have come and gone:

- **Google Optimize** — a free website-testing tool tied to Google Analytics (A/B,
  multivariate and split-URL tests), **deprecated in 2023**.
- **Google Ads Experiments** — still active; lets advertisers split campaign traffic to
  test bids, keywords, creatives and audiences, comparing conversions, CPC and ROAS.
- **GA4 + third-party platforms** — Google now recommends pairing GA4 (event and
  conversion tracking) with external experimentation tools (Optimizely, VWO) that handle
  randomisation, stopping rules and statistics.
- **Vertex AI "experiments"** — a *different* meaning: tracking ML model versions,
  hyperparameters and metrics, not A/B testing.

The statistics
--------------

The reporting engine differed by product: **Google Ads Experiments** uses frequentist
methods with adjusted confidence intervals, while the legacy **Optimize** ran a
**Bayesian** engine that reported a **"probability to beat baseline"** instead of
p-values — e.g. "variant B has a 95% probability of being better than A", which
non-technical users found far easier to act on.

Examples
--------

An Ads experiment splitting traffic 50/50 to test a higher-bid strategy might show 12%
more conversions at significance after two weeks → adopt it. A legacy Optimize website
test of a red vs blue call-to-action might report "red has a 96% probability of beating
blue".

The takeaway
------------

With Optimize retired, web and product experimentation on Google's stack now means
**GA4 plus an external platform** (Optimizely, VWO, LaunchDarkly or custom infra); Ads
Experiments remain the built-in option, but only for ad-campaign settings, not
full-site UX.
"""

CONTENT["Optimizely"] = r"""
What it is
----------

**Optimizely** is a commercial **experimentation and digital-experience platform**.
Originally known for website A/B testing, it has grown into a full **experimentation +
feature-management** suite used by marketers, product managers and data scientists to
test and personalise user experiences.

Core capabilities
-----------------

- **Experimentation** — A/B, multivariate and multi-page tests across web, mobile and
  **server-side**, with random assignment and a no-code WYSIWYG editor for simple tests.
- **Feature management** — **feature flags** to toggle features without redeploying,
  gradual percentage rollouts, and targeting (e.g. only Premium users).
- **Personalisation** — audience-targeting rules and custom segments.
- **Statistics** — default **frequentist with sequential-testing adjustments** (so
  monitoring doesn't inflate error), Bayesian methods in some tiers, plus **variance
  reduction (CUPED-style)** in enterprise plans.
- **Integration** — connects to Google Analytics, Segment, Salesforce, Amplitude and
  warehouses like Snowflake and BigQuery.

A typical workflow
------------------

State a hypothesis → configure the experiment → split traffic (say 50/50) → track
conversions and events → monitor significance in real time → end when conclusive (with
**sequential-testing corrections**) → roll the winner out behind a **feature flag**.

Strengths and limits
--------------------

It is **user-friendly** (non-technical teams can run tests), spans **client- and
server-side**, has **built-in feature flags**, and brings real statistical rigour. The
costs: it is a **pricey SaaS** next to open-source options, results live or die by
**metric definition**, and the most complex experimentation needs (Google/Netflix scale)
may still want bespoke internal platforms. Among peers, VWO is cheaper but lighter for
engineering, Adobe Target is enterprise-grade but complex, and LaunchDarkly leads on
feature flagging but leans on external analytics.
"""

MINDMAP.update({
    "Bayesian Stopping Rules": [
        "Sequential Settings", "Posterior probability of uplift",
        "Bayesian Sequential Testing", "Posterior Probability", "Frequentist",
        "A/B Testing",
    ],
    "Google Experiments": [
        "Optimizely", "Online Experimentation Platforms", "A/B Testing",
        "Conversion Rate Uplift", "Bayesian Stopping Rules",
    ],
    "Optimizely": [
        "Google Experiments", "Online Experimentation Platforms", "A/B Testing",
        "Conversion Rate Uplift", "Sequential Settings",
    ],
})


# ----------------------------------------------------------------------
# Theme: Experimentation infrastructure & causal effects  (abtest / causal)
# ----------------------------------------------------------------------

CONTENT["Online Experimentation Platforms"] = r"""
What it is
----------

An **online experimentation platform** is the **infrastructure that lets a company run
controlled experiments — A/B tests, multivariate tests, bandits — on a live digital
product**, usually built internally at scale. It gives product managers, data scientists
and engineers a way to randomly assign users to variants, log outcomes automatically,
analyse them with sound statistics, and enforce guardrails so experiments don't harm
users.

Core components
---------------

- **Assignment & randomisation** — bucket users (by cookie, device or account) into
  variants, with *stable* assignment so a user always sees the same one.
- **Data collection & logging** — track clicks, conversions, retention and revenue
  consistently across pipelines.
- **Metrics framework** — predefined primary KPIs plus **guardrail metrics** (engagement
  should rise, latency must not).
- **Statistical engine** — frequentist (p-values, fixed-horizon or sequential), Bayesian
  (posteriors, credible intervals), and variance reduction (CUPED) or stratified
  sampling.
- **Dashboard** — automated significance, effect sizes and time series.
- **Governance / guardrails** — flag harmful launches and enforce GDPR/HIPAA compliance.

Modern capabilities
-------------------

The leading platforms add **sequential testing** (safe peeking), **Bayesian analysis**,
**multi-armed bandits** (adaptive traffic), **heterogeneous treatment-effect / CATE**
analysis by segment, **ML-driven personalisation**, and **scale** to hundreds of
simultaneous tests.

Who builds them
---------------

Commercial tools include **Optimizely**, **Adobe Target**, **VWO** and feature-flag
platforms like **Split.io / LaunchDarkly** (Google Optimize was retired in favour of
GA4). The big platforms run their own: **Microsoft's ExP** (Bing, Office, Windows),
**Airbnb's ERF**, **Uber's** and **Netflix's** XP infrastructure, **LinkedIn's XLNT**,
and **Meta's PlanOut**.

Hard problems
-------------

Running thousands of concurrent experiments raises real challenges: **statistical
validity** at scale (many simultaneous tests), **interference / contamination** (users
caught in several experiments), **metric definition** (does the KPI track product
health?), sheer **scale**, and **ethics** (avoiding harmful or unfair experiments). The
payoff is **evidence-based decisions** instead of intuition, lower launch risk, and
continuous optimisation.
"""

CONTENT["Stopping Rules"] = r"""
What it is
----------

A **stopping rule** is the **pre-specified condition for ending an experiment** — it says
*when* to stop collecting data and decide. Without one, you can **"peek" until the result
looks significant**, which inflates the **Type I error** (false-positive) rate. The
choice of stopping rule is what makes continuous monitoring valid or invalid.

The families
------------

- **Fixed-horizon (the traditional A/B test)** — fix the sample size up front (say
  10,000 per arm), analyse **once** at the end. Simple and protects :math:`\alpha`, but
  inflexible.
- **Group-sequential** — pre-plan **interim looks** (e.g. every 25%) and spend the error
  budget with an **α-spending** rule (O'Brien–Fleming, Pocock), allowing early stops for
  efficacy or futility. Efficient; standard in clinical trials.
- **Sequential probability ratio test (SPRT)** — check a **likelihood ratio** after each
  observation and stop once it crosses a bound for :math:`H_1` or :math:`H_0`; often uses
  far fewer samples.
- **Bayesian** — stop when a **posterior probability** crosses a threshold (e.g.
  :math:`P(H_1 \mid \text{data}) > 0.95`); intuitive and needs no α-spending.
- **Ethical / practical** — in trials, stop for **harm**, for **clear benefit** (it's
  unethical to withhold), or for **futility**.

Examples
--------

An A/B test might run to a fixed **50,000 visitors per arm**; or check every **5,000**
and stop early if :math:`p < 0.001` under an O'Brien–Fleming bound; or stop once the
posterior probability that B beats A exceeds **0.95**. A clinical trial planned for 1,000
patients might look at 250, 500 and 750 — stopping early for strong benefit, immediately
for harm, or for futility.

Why they matter
---------------

Stopping rules prevent **p-hacking** (stopping the moment things look good), keep **Type
I error** controlled, **save time and cost** by ending early when results are clear, and
**protect participants** in clinical settings. The dividing line: **fixed horizon** (one
look) versus **sequential / adaptive** (group-sequential, SPRT, Bayesian) that permit
interim looks without inflating false positives.
"""

CONTENT["Treatment Effect"] = r"""
What it is
----------

The **treatment effect** is the **causal impact of an intervention** versus a control —
*how much did the treatment change the outcome compared with what would have happened
without it?* In the **potential-outcomes** framework,

.. math::

   \text{Treatment Effect} = Y(1) - Y(0),

where :math:`Y(1)` is the outcome if the unit *is* treated and :math:`Y(0)` the outcome
if it is *not*.

The fundamental problem
-----------------------

For any single unit you only ever observe **one** of :math:`Y(1)` or :math:`Y(0)` — never
both — so the individual effect is unobservable. This *fundamental problem of causal
inference* is why we estimate **averages** instead of individual effects.

The hierarchy of effects
------------------------

- **ITE (individual)** — :math:`Y_i(1) - Y_i(0)` for one unit; unobservable.
- **ATE (average)** — :math:`\text{ATE} = \mathbb{E}[Y(1) - Y(0)]`, the
  population-average effect, the usual target of RCTs and A/B tests.
- **CATE (conditional average)** —
  :math:`\text{CATE}(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]`, the effect within a
  subgroup defined by covariates :math:`x` (age, segment) — the basis of personalised
  interventions.
- **LATE (local average)** — the effect for a specific compliant subgroup, typically via
  instrumental variables.

Examples
--------

A drug with 60% recovery vs 50% on placebo has :math:`\text{ATE} = +10` points. A website
variant at 5.5% vs 5% conversion has :math:`\text{ATE} = +0.5` points, a
:math:`(0.055 - 0.05)/0.05 = +10\%` relative lift.

How it's estimated
------------------

In a **randomised controlled trial** (or A/B test), randomisation makes the groups
comparable, so the **difference in group means** is an unbiased estimate of the ATE. In
**observational** data, confounding must be removed with causal-inference tools —
matching, regression adjustment, instrumental variables, difference-in-differences or
**propensity scores**. ML methods increasingly estimate **heterogeneous (CATE)** effects
for targeting.
"""

MINDMAP.update({
    "Online Experimentation Platforms": [
        "Optimizely", "Google Experiments", "A/B Testing", "Stopping Rules",
        "Sequential Settings", "Bandit Algorithms",
    ],
    "Stopping Rules": [
        "Bayesian Stopping Rules", "Sequential Settings",
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Frequentist",
        "A/B Testing", "Online Experimentation Platforms",
    ],
    "Treatment Effect": [
        "A/B Testing", "Conversion Rate Uplift", "Parameter(s) of Interest",
        "Causal Inference", "Posterior probability of uplift",
        "Online Experimentation Platforms",
    ],
})


# ----------------------------------------------------------------------
# Theme: Bayesian evidence & sequential testing  (bayes / abtest)
# ----------------------------------------------------------------------

CONTENT["Posterior Probability"] = r"""
What it is
----------

The **posterior probability** is the **probability of a hypothesis (or parameter value)
after observing data**, computed with Bayes' theorem by combining the **prior** (belief
before) with the **likelihood** (evidence from the data):

.. math::

   P(H \mid \text{data}) = \frac{P(\text{data} \mid H)\, P(H)}{P(\text{data})}.

What it answers
---------------

It answers a directly useful question — *"given the data, how probable is this
hypothesis?"* — which is **not** what a frequentist p-value answers (*"if* :math:`H_0`
*were true, how likely is data this extreme?"*). The posterior is the quantity people
usually *think* a p-value gives them.

Worked example
--------------

Is a coin biased toward heads? Take two point hypotheses — :math:`H_0: p = 0.5` and
:math:`H_1: p = 0.7` — with equal priors, and observe 7 heads in 10 tosses. The
likelihoods are

.. math::

   P(\text{data} \mid H_0) = \binom{10}{7}(0.5)^{10} \approx 0.117, \qquad
   P(\text{data} \mid H_1) = \binom{10}{7}(0.7)^7 (0.3)^3 \approx 0.266,

so the posterior for the biased hypothesis is

.. math::

   P(H_1 \mid \text{data}) = \frac{0.266 \times 0.5}{0.266 \times 0.5 + 0.117 \times 0.5} \approx 0.69.

After 7 heads in 10, there's about a **69% probability** the coin is biased.

Parameter posteriors
--------------------

When the unknown is a continuous parameter rather than a discrete hypothesis, the
posterior is a whole **distribution**,
:math:`P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta)\, P(\theta)` — the
object used to estimate means, regression coefficients or conversion rates. (A single
**posterior probability** is then one number read off that distribution.)

Where it shows up
-----------------

Bayesian sequential testing (stop once :math:`P(H_1 \mid \text{data}) > 0.95`), Bayesian
estimation, ML models (Naive Bayes, Bayesian nets), and medical diagnosis (probability of
disease given a test result).
"""

CONTENT["Bayesian Sequential Testing"] = r"""
What it is
----------

**Bayesian sequential testing** evaluates evidence **continuously (or at interim looks)
using Bayesian updating**, instead of frequentist p-values against a fixed
:math:`\alpha`. At every step you compute a **posterior probability** or **Bayes factor**
and decide to **stop for** :math:`H_1`, **stop for** :math:`H_0`, or **keep sampling** —
with **no pre-specified sample size**.

The two evidence metrics
------------------------

- **Posterior probability** — via Bayes' theorem,
  :math:`P(H \mid \text{data}) = P(\text{data} \mid H)\,P(H) / P(\text{data})`.
- **Bayes factor** — :math:`\text{BF} = P(\text{data} \mid H_1)/P(\text{data} \mid H_0)`;
  :math:`\text{BF} > 1` favours :math:`H_1`, with :math:`\text{BF} > 10` and
  :math:`\text{BF} < 1/10` as strong-evidence thresholds.

Decision rules
--------------

- **Stop for efficacy** — :math:`P(H_1 \mid \text{data}) > 0.95` (or
  :math:`\text{BF} > 10`).
- **Stop for futility** — :math:`P(H_0 \mid \text{data}) > 0.95` (or
  :math:`\text{BF} < 1/10`).
- **Continue** — evidence still inconclusive.

The key property: **continuous monitoring does not inflate the Type I error rate**, so
peeking is allowed.

Example
-------

An A/B test with equal priors: after 5,000 visitors the posterior that B beats A is
**0.93** → keep sampling; after 8,000 it reaches **0.97** → stop and conclude B is
better.

Vs frequentist sequential
-------------------------

The upside: no fixed horizon (stop anytime), intuitive **probability statements** ("97%
chance B is better"), prior information can be folded in, and error is controlled
**without α-spending** corrections. The downside: it needs a **prior** (a subjective
choice that can sway results) and more **computation**. Where SPRT and group-sequential
methods speak in p-values and likelihood ratios with α-spending, Bayesian sequential
testing speaks in posteriors and Bayes factors and treats multiple looks as a non-issue.
"""

CONTENT["Likelihood Ratio (LR)"] = r"""
What it is
----------

The **likelihood ratio (LR)** measures **how much more likely the observed data is under
one hypothesis than another**, by dividing their likelihoods:

.. math::

   \Lambda = \frac{L(\text{data} \mid H_1)}{L(\text{data} \mid H_0)}.

:math:`\Lambda = 1` means the data is equally likely under both; :math:`\Lambda > 1`
favours :math:`H_1`; :math:`\Lambda < 1` favours :math:`H_0`.

Two roles in testing
--------------------

- **Likelihood-ratio test (LRT)** — the generalised statistic
  :math:`\Lambda = \sup_{\theta \in \Theta_0} L(\theta) / \sup_{\theta \in \Theta} L(\theta)`
  compares the best fit under the null to the best fit overall; a small :math:`\Lambda`
  is strong evidence against :math:`H_0`. (By the Neyman–Pearson lemma, the LR is the
  *most powerful* test for simple hypotheses.)
- **Sequential probability ratio test (SPRT)** — accumulate the ratio as data arrives,
  :math:`\Lambda_n = L(\text{data}_{1:n} \mid H_1)/L(\text{data}_{1:n} \mid H_0)`, and
  stop when it crosses an upper bound :math:`A` (accept :math:`H_1`) or lower bound
  :math:`B` (accept :math:`H_0`), else continue.

Worked example
--------------

Seven heads in 10 tosses, with :math:`H_0: p = 0.5` vs :math:`H_1: p = 0.7`:

.. math::

   L(H_0) = \binom{10}{7}(0.5)^{10} \approx 0.117, \quad
   L(H_1) = \binom{10}{7}(0.7)^7(0.3)^3 \approx 0.266, \quad
   \Lambda = \frac{0.266}{0.117} \approx 2.27,

so the data is about **2.3 times more likely** under the biased hypothesis. Rough
reading: :math:`\Lambda \approx 1` no evidence, :math:`> 3` moderate, :math:`> 10`
strong.

Where it shows up
-----------------

The LR is the backbone of likelihood-based inference: the **LRT** and **SPRT**, nested
**model comparison**, and — closely related — the **Bayes factor**, which *is* a
likelihood ratio when the hypotheses are simple (and an evidence ratio of marginal
likelihoods when they are composite). In medicine, diagnostic **LR+ and LR−** update the
odds of disease from a test result.
"""

MINDMAP.update({
    "Posterior Probability": [
        "Bayes' Theorem", "Posterior", "Prior Belief (or Prior Probability)",
        "Bayesian Sequential Testing", "Posterior belief", "Frequentist",
    ],
    "Bayesian Sequential Testing": [
        "Posterior Probability", "Bayesian Stopping Rules", "Sequential Settings",
        "Likelihood Ratio (LR)", "Frequentist", "Conversion Rate Uplift",
    ],
    "Likelihood Ratio (LR)": [
        "Sequential Probability Ratio Test (SPRT)", "Bayesian Sequential Testing",
        "Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)",
        "Binomial Likelihood", "Frequentist", "Posterior Probability",
    ],
})


# ----------------------------------------------------------------------
# Theme: Frequentist sequential testing — SPRT & group-sequential  (abtest)
# ----------------------------------------------------------------------

CONTENT["Sequential Probability Ratio Test (SPRT)"] = r"""
What it is
----------

The **sequential probability ratio test (SPRT)**, due to Abraham **Wald**, is a
hypothesis test that **evaluates data as it arrives** rather than fixing the sample size
in advance. It compares two **simple** hypotheses,

.. math::

   H_0 : \theta = \theta_0 \quad \text{vs} \quad H_1 : \theta = \theta_1,

and after each observation decides to **accept** :math:`H_0`, **accept** :math:`H_1`, or
**keep sampling**.

The running likelihood ratio
----------------------------

After :math:`n` observations, accumulate the likelihood ratio

.. math::

   \Lambda_n = \frac{L(\text{data}_{1:n} \mid H_1)}{L(\text{data}_{1:n} \mid H_0)}.

Decision boundaries
-------------------

Two boundaries are set from the desired Type I (:math:`\alpha`) and Type II
(:math:`\beta`) error rates:

.. math::

   A = \frac{1 - \beta}{\alpha}, \qquad B = \frac{\beta}{1 - \alpha}.

Then stop when :math:`\Lambda_n \ge A` (accept :math:`H_1`) or :math:`\Lambda_n \le B`
(accept :math:`H_0`); while :math:`B < \Lambda_n < A`, continue.

Worked example
--------------

For conversion rates :math:`H_0 : p = 0.10` vs :math:`H_1 : p = 0.12` with
:math:`\alpha = 0.05, \beta = 0.20`,

.. math::

   A = \frac{1 - 0.20}{0.05} = 16, \qquad B = \frac{0.20}{0.95} \approx 0.21.

Update :math:`\Lambda_n` as conversions come in; cross 16 → conclude :math:`H_1`, drop
below 0.21 → conclude :math:`H_0`, otherwise keep going.

Strengths and limits
--------------------

On average the SPRT needs **fewer samples** than a fixed-horizon test, stops as soon as
evidence is decisive, and holds the chosen :math:`\alpha` and :math:`\beta`. The catch:
it's built for **simple** hypotheses (fixed :math:`\theta_0, \theta_1`), is awkward for
**composite** ones (e.g. :math:`p \le 0.10`), and needs real-time monitoring. Wald
introduced it for **WWII quality control** (accepting or rejecting munitions lots with
fewer inspections); today it appears in clinical trials, sequential A/B testing and
industrial QC. Versus group-sequential designs (which look at fixed checkpoints), the
SPRT is genuinely **continuous**.
"""

CONTENT["Pocock Method"] = r"""
What it is
----------

The **Pocock method** is a **group-sequential design** that allows several interim
analyses while keeping the overall Type I error at :math:`\alpha`. Its defining choice:
**every look — interim and final — uses the same significance threshold** (the same
critical value), in contrast to O'Brien–Fleming, which varies it.

How the threshold is set
------------------------

The constant cutoff is chosen so the *overall* error across all looks still equals
:math:`\alpha`. For :math:`\alpha = 0.05` (two-sided) with 4 looks at 25/50/75/100%, the
Pocock critical value is about :math:`z \approx \pm 2.41` (a nominal :math:`p \approx
0.017`) **at every look** — stricter than 1.96 because four chances to reject must share
the budget.

Example
-------

A checkout-flow A/B test plans 40,000 users (20,000 per arm) with interim looks every
10,000. The Pocock cutoff is :math:`z \ge 2.41` at every stage: an observed
:math:`z = 2.6` at the first look means **stop early, B wins**; :math:`z = 2.0` means
continue.

Trade-offs vs OBF
-----------------

Pocock is **easy to communicate** ("same threshold every time") and **stops early more
readily** for moderate effects, which saves resources. The cost: because it spends
:math:`\alpha` evenly, the **final test is stricter** than a plain :math:`\alpha = 0.05`
(≈ 0.017 with 4 looks), so it is **less powerful** if the trial runs to completion and
may need a slightly larger sample. Rule of thumb: **Pocock when early stopping is likely**
(business/exploratory A/B), **O'Brien–Fleming when the trial will probably run to the
end** (safety-critical medicine).
"""

CONTENT["O'Brien–Fleming (OBF) Method"] = r"""
What it is
----------

The **O'Brien–Fleming (OBF) method** is a **group-sequential design** that controls the
overall Type I error :math:`\alpha` across multiple interim analyses with a distinctive
spending shape: **very strict early, lenient late**. Early on you need overwhelming
evidence to stop; by the final look the threshold is essentially the usual
:math:`\alpha`.

The spending shape
------------------

For :math:`\alpha = 0.05` (two-sided) with 4 looks, the OBF critical z-values are roughly

- 25% — :math:`z \approx 3.47` (:math:`p \approx 0.0005`)
- 50% — :math:`z \approx 2.45` (:math:`p \approx 0.014`)
- 75% — :math:`z \approx 2.00` (:math:`p \approx 0.045`)
- 100% — :math:`z \approx 1.98` (:math:`p \approx 0.048`)

so the early bar is extreme and the final bar is almost a normal :math:`\alpha = 0.05`
test. This is the opposite philosophy to Pocock's flat :math:`z \approx 2.41`.

Example
-------

A heart-drug trial with 4 interim looks: at 25% an observed :math:`p = 0.002`
(:math:`z \approx 3.1`) is **not** below the OBF bound (:math:`p \approx 0.0005`), so
continue; at 50%, :math:`p = 0.009` (:math:`z \approx 2.6`) clears the bound
(:math:`\approx 0.014`) → **stop early for efficacy**.

Strengths and trade-offs
------------------------

OBF gives **strong early protection against false positives** (random noise can't stop
the trial prematurely) while still permitting an early stop for a genuinely large effect,
and its **final test barely loses power** versus fixed-horizon. The downside: it **rarely
stops early** unless the effect is huge, so you often collect nearly the full sample.
That conservatism is exactly why it suits **safety-critical** domains like medicine,
whereas Pocock fits exploratory or business A/B tests where stopping early saves money.
"""

MINDMAP.update({
    "Sequential Probability Ratio Test (SPRT)": [
        "Likelihood Ratio (LR)", "Bayesian Sequential Testing", "Stopping Rules",
        "Sequential Settings", "Frequentist", "Group Sequential Testing",
    ],
    "Pocock Method": [
        "O'Brien–Fleming (OBF) Method", "Group Sequential Testing", "Stopping Rules",
        "Frequentist", "Sequential Settings",
        "Sequential Probability Ratio Test (SPRT)",
    ],
    "O'Brien–Fleming (OBF) Method": [
        "Pocock Method", "Group Sequential Testing", "Stopping Rules", "Frequentist",
        "Sequential Settings", "Traditional A/B Test (Fixed-Horizon A/B Test)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Group-sequential testing, Type I error, fixed-horizon baseline  (abtest)
# ----------------------------------------------------------------------

CONTENT["Group Sequential Testing"] = r"""
What it is
----------

**Group sequential testing** lets you **analyse accumulating data at several pre-planned
interim points** during an experiment — not just once at the end — and at each look
decide to **stop for efficacy** (the effect is clearly there), **stop for futility** (it
clearly isn't), or **continue**. It is standard in clinical trials and increasingly used
in A/B testing where stopping early saves resources.

The peeking problem it solves
-----------------------------

Repeatedly checking a fixed-:math:`\alpha` test and stopping the moment :math:`p < 0.05`
**inflates the Type I error** badly — with many looks, the chance of a false positive can
climb to **20–30%**. Group sequential designs fix this with an **α-spending function**
that divides the error budget across looks so the *overall* Type I rate stays at
:math:`\alpha`.

How it works
------------

Plan the number of interim analyses up front, then use an **α-spending rule** to set a
significance cutoff at each: early looks get **stricter** thresholds, later looks more
**lenient** ones, and you stop as soon as a threshold is crossed.

The α-spending rules
--------------------

- **O'Brien–Fleming** — very strict early, lenient late (final ≈ fixed :math:`\alpha`).
- **Pocock** — the same moderate threshold at every look.
- **Lan–DeMets** — a flexible spending function that allocates :math:`\alpha`
  adaptively, without fixing the look times in advance.

Example
-------

A trial of 1,000 patients, analysed every 250 with total :math:`\alpha = 0.05` under an
O'Brien–Fleming schedule, might spend :math:`\alpha = 0.001, 0.01, 0.02, 0.04` across the
four looks. A p-value of 0.008 at 500 patients crosses the second bound → **stop early
for efficacy**.

Where it sits
-------------

It is the middle ground between extremes: more efficient than the **fixed-horizon** A/B
test (one look, may waste data) and statistically valid unlike **naive peeking** (which
inflates false positives), while **bandit / adaptive** methods go further still by
reallocating traffic continuously.
"""

CONTENT["Type I Error"] = r"""
What it is
----------

A **Type I error** is a **false positive**: you **reject the null hypothesis**
:math:`H_0` **when it is actually true** — concluding there is an effect or difference
when in reality there is none.

Its probability is α
--------------------

The probability of a Type I error is exactly the **significance level** :math:`\alpha`,
fixed *before* the test: :math:`\alpha = 0.05` accepts a 5% chance of wrongly rejecting a
true :math:`H_0`; :math:`\alpha = 0.01` a 1% chance. Choosing :math:`\alpha` *is* choosing
how often you are willing to cry wolf.

Examples
--------

- **Medicine** — :math:`H_0`: the drug has no effect. If it truly doesn't, but the data
  happen to give :math:`p < 0.05`, you reject :math:`H_0` and declare it works — a Type I
  error.
- **A/B testing** — :math:`H_0`: conversion rates are equal. If they really are, but
  random variation produces a "significant" gap, you've made a Type I error.

Geometrically, with overlapping :math:`H_0` and :math:`H_1` distributions, :math:`\alpha`
is the **rejection region** in the tail; a statistic landing there *while* :math:`H_0`
holds is the error.

Type I vs Type II vs power
--------------------------

There are two ways to be wrong and one way the test "works":

- **Type I (false positive)** — reject a true :math:`H_0`; probability :math:`\alpha`.
- **Type II (false negative)** — fail to reject a false :math:`H_0`; probability
  :math:`\beta`.
- **Power** — correctly reject a false :math:`H_0`; equals :math:`1 - \beta` (and grows
  with sample size, effect size and :math:`\alpha`).

Lowering :math:`\alpha` reduces Type I errors but, all else equal, raises :math:`\beta` —
the two trade off.

Controlling it
--------------

Use a **stricter** :math:`\alpha`; apply **Bonferroni** or other multiple-testing
corrections when running many tests; stick to **fixed-horizon** testing (no peeking), or
an α-spending design if you must look early; and **replicate** to confirm.
"""

CONTENT["Traditional A/B Test (Fixed-Horizon A/B Test)"] = r"""
What it is
----------

A **traditional A/B test** — equivalently a **fixed-horizon A/B test** — is the classical
approach: **predefine a sample size or duration, collect data until that point, and run
the hypothesis test exactly once at the end**. No interim decisions.

The procedure
-------------

1. State the hypotheses (:math:`H_0`: :math:`CR_A = CR_B`).
2. Choose a significance level :math:`\alpha` (typically 0.05).
3. Run an **a-priori power analysis** to find the required sample size :math:`n`.
4. **Fix the horizon** — e.g. "stop at 10,000 users per variant."
5. Collect data to that horizon.
6. Run the test (a **two-proportion z-test** for conversion rates).
7. Reject or fail to reject :math:`H_0`.

The defining feature is **no peeking**: because the analysis happens once, at a
pre-committed sample size, the Type I error stays at :math:`\alpha`.

Example
-------

To test a new button colour with a 5% baseline and a **minimum detectable lift** of +10%,
an a-priori power analysis might call for **~8,000 users per group**. You run until each
arm hits 8,000, then run a two-proportion z-test, and only then decide whether B beats A.

Strengths and limits
--------------------

It is **rigorous, widely understood, and easy to explain**, and it controls Type I error
cleanly when its assumptions hold. The cost is **inflexibility**: you must wait for the
full horizon, which **wastes traffic** when one variant is clearly better early, and it
can't adapt in real time. The modern alternatives relax exactly this — **sequential
testing** (interim looks via α-spending), **Bayesian A/B** (probability of superiority,
peek freely), and **multi-armed bandits** (shift traffic to the winner as you learn).
"""

MINDMAP.update({
    "Group Sequential Testing": [
        "O'Brien–Fleming (OBF) Method", "Pocock Method", "Stopping Rules",
        "Sequential Settings", "Traditional A/B Test (Fixed-Horizon A/B Test)",
        "Type I Error",
    ],
    "Type I Error": [
        "Frequentist", "Traditional A/B Test (Fixed-Horizon A/B Test)",
        "Stopping Rules", "Group Sequential Testing", "A/B Testing",
        "Bayesian Sequential Testing",
    ],
    "Traditional A/B Test (Fixed-Horizon A/B Test)": [
        "Type I Error", "Stopping Rules", "Sequential Settings",
        "Bayesian Sequential Testing", "Bandit Algorithms", "Conversion Rate Uplift",
    ],
})


# ----------------------------------------------------------------------
# Theme: Estimation & inference foundations — horizon, true rate, SE  (abtest / inference)
# ----------------------------------------------------------------------

CONTENT["Fixed-Horizon Testing"] = r"""
What it is
----------

**Fixed-horizon testing** is the general statistical principle behind the classic
experiment: **decide the sample size** :math:`n` **(or end time) in advance, collect data
to that point, and run the hypothesis test exactly once.** The "horizon" is that
pre-committed stopping point. Applied to an A/B test specifically, this *is* the
**traditional A/B test**.

Why the horizon matters
-----------------------

Fixing it up front is what keeps the statistics honest: the **Type I error stays at**
:math:`\alpha`, **peeking is ruled out** (checking early and stopping on significance
inflates false positives), and the resulting **p-values and confidence intervals remain
valid** under their assumptions.

The procedure
-------------

State :math:`H_0` and :math:`H_1`; choose :math:`\alpha` (say 0.05); run an **a-priori
power analysis** to size the sample; fix the horizon; collect to it; test once; decide.

Example
-------

To detect a **+10% lift** (5% → 5.5%) at :math:`\alpha = 0.05` and **power 0.80**, an
a-priori power analysis calls for **≈ 7,850 users per variant**. You stop at that horizon
and run a single two-proportion z-test: :math:`p \le 0.05` rejects :math:`H_0`, otherwise
you fail to reject.

Strengths and limits
--------------------

It is **simple, widely accepted, and preserves error guarantees**. The price is
**rigidity**: no early stop even when the result is already obvious, wasted samples when
an effect is large, and no continuous monitoring. **Sequential and adaptive** methods
(α-spending, Bayesian updating, bandits) trade some of that simplicity for the ability to
stop early.
"""

CONTENT["True Conversion Rate"] = r"""
What it is
----------

The **true conversion rate** :math:`p` is the **actual probability that a user in the
whole population converts** (clicks, buys, signs up). It is a **population parameter** —
fixed but unknown. What an experiment actually measures is the **sample conversion rate**
:math:`\hat{p}`, an *estimate* of :math:`p`.

Parameter vs estimate
---------------------

.. math::

   p = \frac{\text{conversions in the population}}{\text{users in the population}},
   \qquad
   \hat{p} = \frac{x}{n},

where :math:`x` is conversions in the sample and :math:`n` the sample size. We can rarely
see the whole population, so we work with :math:`\hat{p}` and quantify its uncertainty.

Example
-------

1,000 users see version A and 50 convert, so :math:`\hat{p}_A = 50/1000 = 0.05` (5%). The
true rate might be :math:`p = 0.052`, but we never observe it exactly — only estimate it.

Confidence interval for p
-------------------------

Because :math:`\hat{p}` carries sampling error, a **Wald confidence interval** brackets
the likely range of :math:`p`:

.. math::

   \text{CI} = \hat{p} \pm z_{\alpha/2}\, \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}.

With :math:`\hat{p} = 0.05, n = 1000` and 95% confidence, the standard error is
:math:`\sqrt{0.05 \times 0.95 / 1000} \approx 0.0069`, giving
:math:`0.05 \pm 1.96 \times 0.0069 \approx [0.036, 0.064]` — we're 95% confident the true
rate lies between **3.6% and 6.4%**.

Why it matters
--------------

In A/B testing we never know either group's true rate; we estimate both with
:math:`\hat{p}` and use a **two-proportion z-test** to judge whether the *observed*
difference is real evidence of a difference in the **true** conversion rates — the actual
quantity of interest.
"""

CONTENT["Standard Error (SE)"] = r"""
What it is
----------

The **standard error (SE)** measures **how much a sample statistic — a mean or a
proportion — would vary across many random samples** from the same population. It is the
**uncertainty of the estimate**: how far the sample value is likely to fall from the true
population value.

The formulas
------------

For a **mean** (population SD :math:`\sigma`, or sample SD :math:`s` when :math:`\sigma`
is unknown):

.. math::

   SE_{\bar{x}} = \frac{\sigma}{\sqrt{n}} \quad\left(\text{or } \frac{s}{\sqrt{n}}\right).

For a **proportion**:

.. math::

   SE_{\hat{p}} = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}.

Key properties
--------------

**Bigger samples shrink it** (:math:`SE \propto 1/\sqrt{n}`, so estimates get more
precise), **more spread inflates it** (larger :math:`\sigma`), and it is the bridge to the
**Central Limit Theorem**: for large :math:`n` the sample mean is approximately normal
with the SE as its spread.

Examples
--------

With :math:`\mu = 100, \sigma = 20, n = 25`, :math:`SE = 20/\sqrt{25} = 4` — sample means
vary about ±4 around the truth. For a proportion, 520 "yes" out of 1,000 gives
:math:`\hat{p} = 0.52` and :math:`SE = \sqrt{0.52 \times 0.48 / 1000} \approx 0.016`, a
±1.6% margin.

SE vs standard deviation
------------------------

They are easy to confuse: the **standard deviation** describes the spread of *individual
data points*, while the **standard error** describes the spread of the *estimate*. SD is
variability in the data; SE is variability in the statistic — and SE shrinks with
:math:`n` while SD does not.

Where it shows up
-----------------

The SE is the denominator of inference: **confidence intervals**
(:math:`\text{estimate} \pm z_{\alpha/2}\, SE`), **test statistics** (the z- and
t-tests), and **A/B comparisons** of conversion rates all run on it.
"""

MINDMAP.update({
    "Fixed-Horizon Testing": [
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Stopping Rules",
        "Sequential Settings", "Type I Error", "Bayesian Sequential Testing",
        "Standard Error (SE)",
    ],
    "True Conversion Rate": [
        "Standard Error (SE)", "Conversion Rate (CR)", "Parameter(s) of Interest",
        "Conversion Rate Uplift", "Frequentist", "A/B Testing",
    ],
    "Standard Error (SE)": [
        "True Conversion Rate", "Bootstrap Confidence Intervals (CIs)", "Frequentist",
        "Conversion Rate (CR)", "True Mean (Population Mean)", "A/B Testing",
    ],
})


# ----------------------------------------------------------------------
# Theme: CI & estimation foundations — true mean, MoE, critical value  (inference / probstats)
# ----------------------------------------------------------------------

CONTENT["True Mean (Population Mean)"] = r"""
What it is
----------

The **true mean**, or **population mean** :math:`\mu`, is the **actual arithmetic average
of an entire population**. It is a **fixed** value but usually **unknown** — we can rarely
measure everyone — so we estimate it with the **sample mean** :math:`\bar{x}`.

Parameter vs estimate
---------------------

.. math::

   \mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.

Here :math:`\mu` is a **parameter** (fixed, unknown) and :math:`\bar{x}` a **statistic**
(random, changing from sample to sample). By the **Law of Large Numbers**, as :math:`n`
grows, :math:`\bar{x} \to \mu`.

Example
-------

For the population :math:`\{2, 4, 6, 8, 10\}`, the true mean is
:math:`\mu = (2+4+6+8+10)/5 = 6`. A sample :math:`\{4, 10\}` gives :math:`\bar{x} = 7` —
an *estimate* of the true 6, off by sampling luck.

Inference about μ
-----------------

Everything in classical inference targets :math:`\mu`: a **hypothesis test** checks a
claim like :math:`H_0 : \mu = 100`, and a **confidence interval** says "we're 95%
confident the true mean :math:`\mu` lies between :math:`X` and :math:`Y`." The sample mean
is the **best unbiased estimator** of :math:`\mu`, and tests and intervals quantify how
far it might be from the truth.

The proportion analogue
-----------------------

For yes/no outcomes the same parameter-vs-estimate story holds with the **true conversion
rate** :math:`p` estimated by :math:`\hat{p}` — :math:`\mu \leftrightarrow p`,
:math:`\bar{x} \leftrightarrow \hat{p}` — the mean and proportion versions of one idea.
"""

CONTENT["Margin of Error (MoE)"] = r"""
What it is
----------

The **margin of error (MoE)** is the **maximum expected gap between a sample estimate and
the true population parameter**, at a stated confidence level. It is the **± half-width**
of a confidence interval:

.. math::

   \text{Confidence Interval} = \text{Estimate} \pm \text{MoE}.

A small MoE means a precise estimate; a large one, an imprecise estimate.

How it's built
--------------

.. math::

   \text{MoE} = (\text{critical value}) \times (\text{standard error}),

where the **critical value** comes from the confidence level (1.96 for 95% under a normal
model) and the **standard error** measures sampling variability.

Two levers: confidence and sample size
--------------------------------------

- **Confidence level ↑** → larger critical value → **larger** MoE (more confidence costs
  width).
- **Sample size ↑** → smaller SE (:math:`\text{SE} \propto 1/\sqrt{n}`) → **smaller**
  MoE. Because of the square root, **halving the MoE requires 4× the sample**.

What it is not
--------------

MoE captures **random sampling error only**. It does **not** include bias, measurement
error, bad sampling design or model misspecification — so a tight MoE means **precise,
not necessarily accurate**. Two common traps: it isn't a hard maximum (it's
probabilistic), and it depends on confidence and variability, not sample size alone.

Why it matters
--------------

MoE turns a point estimate into an honest range ("support = 52% ± 3%" → true support
roughly 49–55%). It encourages **interval thinking** over point thinking, and ties
directly to significance: if a CI **excludes** the null value, the MoE is small enough to
declare a difference; if it **includes** the null, uncertainty swamps the effect.
"""

CONTENT["Critical Value"] = r"""
What it is
----------

A **critical value** is the **cutoff on a distribution that marks the rejection boundary**
for a hypothesis test. It separates the **acceptance region** (fail to reject :math:`H_0`)
from the **rejection region** (reject :math:`H_0`), and depends on three things: the
**significance level** :math:`\alpha`, whether the test is **one- or two-tailed**, and the
**distribution** used (Z, t, :math:`\chi^2`, F).

The decision rule
-----------------

If the **test statistic exceeds the critical value in magnitude**, reject :math:`H_0`; if
it falls inside, fail to reject.

Values by distribution
----------------------

- **Z** (large :math:`n`, known :math:`\sigma`) — two-tailed :math:`\alpha = 0.05` gives
  :math:`\pm 1.96`; one-tailed, :math:`1.645`.
- **t** (small :math:`n`, unknown :math:`\sigma`) — two-tailed, :math:`\alpha = 0.05`,
  :math:`df = 10` gives :math:`\pm 2.228`; as :math:`n` grows, :math:`t \to z`.
- **χ²** (goodness-of-fit, independence) — :math:`\alpha = 0.05, df = 4` gives
  :math:`\approx 9.49`.
- **F** (ANOVA) — :math:`\alpha = 0.05, df_1 = 3, df_2 = 20` gives :math:`\approx 3.10`.

Two roles
---------

The same critical value drives both **confidence intervals** —
:math:`\text{estimate} \pm (\text{critical value}) \times \text{SE}` (a 95% z-interval for
a mean of 100 with :math:`\text{SE} = 2` is :math:`[96.08, 103.92]`) — and **hypothesis
tests**.

Worked test
-----------

Test :math:`H_0 : \mu = 50` vs :math:`H_1 : \mu \neq 50` with sample mean 53,
:math:`\sigma = 10, n = 100`:

.. math::

   z = \frac{53 - 50}{10/\sqrt{100}} = \frac{3}{1} = 3.0.

At :math:`\alpha = 0.05` two-tailed the critical value is :math:`\pm 1.96`; since
:math:`3.0 > 1.96`, **reject** :math:`H_0`.
"""

MINDMAP.update({
    "True Mean (Population Mean)": [
        "True Conversion Rate", "Standard Error (SE)", "Parameter(s) of Interest",
        "Margin of Error (MoE)", "Frequentist",
        "Bootstrap Confidence Intervals (CIs)",
    ],
    "Margin of Error (MoE)": [
        "Standard Error (SE)", "Critical Value", "True Conversion Rate", "Frequentist",
        "True Mean (Population Mean)", "A/B Testing",
    ],
    "Critical Value": [
        "Margin of Error (MoE)", "Standard Error (SE)", "Type I Error", "Frequentist",
        "True Mean (Population Mean)", "Bootstrap Confidence Intervals (CIs)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Descriptive statistics & estimators — s, x-bar, beta  (probstats / inference)
# ----------------------------------------------------------------------

CONTENT["Sample Standard Deviation"] = r"""
What it is
----------

The **sample standard deviation** :math:`s` measures **how spread out the values in a
sample are around the sample mean** :math:`\bar{x}`. It is the **square root of the sample
variance** — roughly, the average distance of a data point from the mean.

The formula and Bessel's correction
-----------------------------------

.. math::

   s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}}.

The denominator is :math:`n - 1`, not :math:`n` — **Bessel's correction**. Dividing by
:math:`n` would *underestimate* the true spread, because the deviations are taken from the
sample mean (which is itself fitted to the data); using :math:`n - 1` makes :math:`s^2` an
**unbiased estimator** of the population variance :math:`\sigma^2`.

Worked example
--------------

For :math:`\{5, 7, 9\}`: the mean is :math:`\bar{x} = 7`; deviations are
:math:`-2, 0, +2`; squared, :math:`4, 0, 4`, summing to 8; divide by :math:`n - 1 = 2` to
get 4; the square root is :math:`s = 2`.

Reading it
----------

A **small** :math:`s` means points cluster near the mean (low variability); a **large**
:math:`s` means they're spread out; :math:`s = 0` means every value is identical. The
**population** standard deviation :math:`\sigma` uses :math:`N` in the denominator (the
whole population); the **sample** version uses :math:`n - 1` (an estimate of
:math:`\sigma`).

Where it shows up
-----------------

:math:`s` is the raw material of inference: it feeds the **standard error**
(:math:`SE = s/\sqrt{n}`), the **t-test**, **ANOVA** and **regression**, and the
**confidence intervals** built around sample means.
"""

CONTENT["Sample Mean"] = r"""
What it is
----------

The **sample mean** :math:`\bar{x}` is the **arithmetic average of a sample** — the sum of
the observations divided by their count. It is the **statistic** used to estimate the
**true population mean** :math:`\mu`:

.. math::

   \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.

Examples
--------

Test scores :math:`\{80, 85, 90, 95, 100\}` give :math:`\bar{x} = 450/5 = 90`. Ten people
with total height 1,720 cm give :math:`\bar{x} = 172` cm.

Its three key properties
------------------------

- **Unbiased** — :math:`\mathbb{E}[\bar{x}] = \mu`; on average the sample mean equals the
  population mean.
- **Sampling distribution (CLT)** — for large :math:`n`, the Central Limit Theorem makes
  :math:`\bar{x}` approximately normal,

  .. math::

     \bar{x} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right),

  with **standard error of the mean** :math:`SE = \sigma/\sqrt{n}` — so its variability
  shrinks as :math:`n` grows.
- **Outlier-sensitive** — being a sum, the mean is pulled by extreme values (unlike the
  median).

Where it shows up
-----------------

The sample mean is everywhere: **descriptive** summaries, **estimating** :math:`\mu`,
**hypothesis tests** (the one-sample t-test), and **confidence intervals**. It is the
**best unbiased estimator** of the population mean.
"""

CONTENT["Regression Coefficient"] = r"""
What it is
----------

A **regression coefficient** :math:`\beta` quantifies the **relationship between a
predictor** :math:`X` **and the outcome** :math:`Y` in a regression model: how much
:math:`Y` changes when :math:`X` rises by **one unit**, holding all other predictors
fixed.

Simple and multiple regression
------------------------------

In **simple** linear regression, :math:`Y = \beta_0 + \beta_1 X + \varepsilon`, where
:math:`\beta_0` is the **intercept** (the value of :math:`Y` at :math:`X = 0`) and
:math:`\beta_1` the **slope**. In **multiple** regression,
:math:`Y = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k + \varepsilon`, each
:math:`\beta_i` is a **partial** coefficient — the effect of :math:`X_i` *controlling for*
the other predictors.

Reading a coefficient
---------------------

Its **sign** gives direction (positive: :math:`Y` rises with :math:`X`; negative: it
falls), its **magnitude** the strength of the effect, and its **p-value** whether it
differs significantly from 0 (a non-significant coefficient may not contribute). For
example, ``Salary = 30000 + 2000 × years`` says each extra year adds about 2,000 in
salary; ``Price = 50000 + 100 × sqft + 20000 × garage`` says each square foot adds 100
*holding garage fixed*, and a garage adds 20,000 *holding size fixed*.

Standardised and logistic
-------------------------

**Unstandardised** coefficients are in original units (dollars, cm); **standardised** ones
(:math:`\beta^*`) are in standard-deviation units, so effects on different scales can be
compared. In **logistic** regression the coefficients are in **log-odds**; exponentiating
gives an **odds ratio** — e.g. :math:`\beta = 0.7` gives :math:`e^{0.7} \approx 2.0`, so a
one-unit increase roughly **doubles the odds** of the event.

Why it matters
--------------

Regression coefficients are the **parameters of interest** in regression — read in
context, alongside p-values, confidence intervals and effect sizes, and (in multiple
models) always as *partial* effects.
"""

MINDMAP.update({
    "Sample Standard Deviation": [
        "Sample Mean", "Standard Error (SE)", "True Mean (Population Mean)",
        "Frequentist", "Bootstrap Confidence Intervals (CIs)", "Margin of Error (MoE)",
    ],
    "Sample Mean": [
        "Sample Standard Deviation", "True Mean (Population Mean)",
        "Standard Error (SE)", "Parameter(s) of Interest", "Frequentist",
        "Regression Coefficient",
    ],
    "Regression Coefficient": [
        "Parameter(s) of Interest", "P-Value (probability value)",
        "Logistic Regression", "Sample Mean", "Frequentist", "Standard Error (SE)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Proportions, the parameter umbrella, power balancing  (probstats / inference)
# ----------------------------------------------------------------------

CONTENT["Proportion"] = r"""
What it is
----------

A **proportion** is a **part-to-whole ratio** — the fraction of a sample or population
with a given characteristic, usually the fraction of **successes** ("yes" outcomes). The
**sample proportion** estimates the true population proportion :math:`p`:

.. math::

   \hat{p} = \frac{x}{n},

with :math:`x` the number of successes and :math:`n` the sample size.

Population vs sample
--------------------

:math:`p` is the **true** proportion in the whole population (fixed, usually unknown);
:math:`\hat{p}` is computed from a sample and **estimates** :math:`p`.

Examples
--------

In a poll, 540 of 1,000 voters back candidate A → :math:`\hat{p} = 0.54` (true :math:`p`
might be 0.55). In quality control, 10 defective bulbs out of 200 →
:math:`\hat{p} = 0.05`, a 5% defect rate.

In inference
------------

Proportions drive categorical inference: a **confidence interval**

.. math::

   \hat{p} \pm z\, \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}},

and hypothesis tests — a **one-sample** proportion test compares :math:`\hat{p}` to a
hypothesised :math:`p_0`, and a **two-proportion z-test** compares :math:`\hat{p}_1` and
:math:`\hat{p}_2` (the workhorse of A/B testing).

Proportion ≈ probability
------------------------

A sample proportion **estimates a population probability**: if 30% of surveyed users
clicked an ad, the probability a random user clicks is :math:`\approx 0.30`. This is why
proportions sit at the centre of surveys, A/B tests, medical studies and quality control —
they turn yes/no data into estimable probabilities.
"""

CONTENT["True Population Parameter"] = r"""
What it is
----------

A **true population parameter** is a **fixed (but usually unknown) number that describes a
whole population**. The word *true* stresses that the value exists even though we seldom
observe it. The familiar ones are written with Greek letters: :math:`\mu` (mean),
:math:`\sigma` (standard deviation), :math:`p` (proportion) and :math:`\rho`
(correlation).

Parameter vs statistic
----------------------

Because we rarely measure an entire population, we draw a **sample** and compute a
**statistic** to estimate each parameter:

- mean — parameter :math:`\mu`, statistic :math:`\bar{x}`
- standard deviation — :math:`\sigma`, statistic :math:`s`
- proportion — :math:`p`, statistic :math:`\hat{p}`
- correlation — :math:`\rho`, statistic :math:`r`

The **statistic is random** (it depends on which sample you draw); the **parameter is
fixed but unknown**. This single distinction underlies every "true vs sample" page in
statistics.

Example
-------

The parameter :math:`p` might be the real fraction of all U.S. voters who support
candidate A; a survey of 1,000 gives the statistic :math:`\hat{p} = 0.52`, an **estimate**
of that unknown :math:`p`.

Inference is about parameters
-----------------------------

We never know a parameter exactly without a census, so we **infer** it: a **confidence
interval** brackets it ("95% confident the true parameter lies in this range"), and a
**hypothesis test** evaluates a claim about it (:math:`H_0 : \mu = 100`). Estimators,
standard errors, intervals and tests all exist to pin down population parameters from
sample data.
"""

CONTENT["Compromise Power Analysis"] = r"""
What it is
----------

**Compromise power analysis** finds a sensible **balance between the Type I error**
:math:`\alpha` (false positives) **and the Type II error** :math:`\beta` (false
negatives) **when the sample size** :math:`n` **is fixed**. Unlike *a-priori* analysis —
which fixes :math:`\alpha` and power and solves for :math:`n` — here you already know
:math:`n` and ask what :math:`\alpha`/:math:`\beta` trade-off is reasonable.

Why it's useful
---------------

Sometimes :math:`n` simply **cannot change** — a limited participant pool, a budget cap,
or a historical dataset. A-priori analysis might say "you need 500 subjects" when you have
200; compromise analysis answers the real question: *with 200, what :math:`\alpha` and
:math:`\beta` give a balanced test?*

How it works
------------

Specify the **effect size** :math:`\delta`, the **available** :math:`n`, and a desired
**ratio of Type I to Type II error** (often :math:`\alpha = \beta`, i.e. a 1:1 ratio). The
procedure then solves for the :math:`\alpha` and :math:`\beta` that satisfy the
constraint.

Example
-------

With a medium effect (Cohen's :math:`d = 0.5`), :math:`n = 40` (20 per group), and a
requirement that :math:`\alpha = \beta`, the analysis might return
:math:`\alpha = \beta = 0.12` (power :math:`= 0.88`). You **accept a higher false-positive
rate (12%)** to keep false negatives equally low, given the small sample.

The three power analyses
------------------------

- **A-priori** — input effect size, :math:`\alpha`, power → output **required** :math:`n`
  ("how many subjects do I need?").
- **Post-hoc** — input observed :math:`n` and effect size → output **achieved power**
  ("given what I saw, what was the power?").
- **Compromise** — input effect size, available :math:`n`, error ratio → output
  **appropriate** :math:`\alpha` **and** :math:`\beta` ("with this :math:`n`, how do I
  balance the two errors?").

It is the pragmatic choice when data is scarce, though it is less conventional than the
fixed :math:`\alpha = 0.05` and depends on an assumed effect size.
"""

MINDMAP.update({
    "Proportion": [
        "True Conversion Rate", "Standard Error (SE)", "Probability",
        "True Population Parameter", "Frequentist", "A/B Testing",
    ],
    "True Population Parameter": [
        "True Mean (Population Mean)", "True Conversion Rate", "Sample Mean",
        "Sample Standard Deviation", "Parameter(s) of Interest", "Proportion",
    ],
    "Compromise Power Analysis": [
        "Type I Error", "Post Hoc Power Analysis", "A Priori Power Analysis",
        "Statistical Power", "Effect Size (δ)", "Frequentist",
    ],
})


# ----------------------------------------------------------------------
# Theme: Power analysis trio & statistical significance  (abtest / inference)
# ----------------------------------------------------------------------

CONTENT["Post Hoc Power Analysis"] = r"""
What it is
----------

**Post-hoc power analysis** computes the **statistical power of a test after the study is
finished**, plugging in the **observed** sample size :math:`n`, the **observed** effect
size :math:`\delta`, and the chosen :math:`\alpha`. It asks: *given what we actually saw,
what was the probability we could have detected an effect?*

Why people run it
-----------------

Usually to interpret a **non-significant** result ("was it real-but-missed, or genuinely
null?"), to satisfy a journal asking about sensitivity, or to judge older studies in a
meta-analysis. The calculation is the a-priori one with the *observed* effect size
substituted in:

.. math::

   \text{Power} = P(\text{reject } H_0 \mid \delta_{\text{observed}}, n, \alpha).

Example
-------

If :math:`H_0` is a 10% conversion rate, the treatment shows a tiny 10.2%, with 1,000 per
group at :math:`\alpha = 0.05`, post-hoc power might be only **12%** — the study was
**underpowered** to detect so small a lift.

The tautology problem
---------------------

The deep flaw: post-hoc power is a **deterministic function of the p-value**, so it adds
nothing. A non-significant result *always* yields low post-hoc power, and a significant
one *always* high — it merely restates the test. Worse, it invites the fallacy
"non-significant + low power ⇒ :math:`H_0` is true," when it only means "this study wasn't
sensitive enough."

Report this instead
-------------------

Rather than post-hoc power, report the **observed effect size** (Cohen's d, a difference
in proportions, an odds ratio) and a **confidence interval** for the effect — these convey
the strength and precision of the result without the circularity.
"""

CONTENT["A Priori Power Analysis"] = r"""
What it is
----------

**A-priori power analysis** computes the **sample size** :math:`n` **needed before
collecting data**, from four inputs: the significance level :math:`\alpha` (Type I risk),
the desired **power** :math:`1 - \beta` (chance of detecting a true effect), the expected
**effect size** :math:`\delta`, and the data variance. The aim: a study **large enough to
detect a meaningful effect** but no larger.

Why it matters
--------------

It guards against **underpowered** studies (false negatives — missing real effects) and
**overpowered** ones (wasted resources chasing trivial effects), and forces you to commit
to a **minimum meaningful effect size** in advance rather than rationalising after the
fact.

The sample-size formula
-----------------------

For a two-sample mean test,

.. math::

   n = \left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{\delta}\right)^2,

where :math:`Z_{1-\alpha/2}` is the critical value for :math:`\alpha` (1.96 at
:math:`\alpha = 0.05`, two-tailed), :math:`Z_{1-\beta}` the value for the target power
(0.84 for 80%), and :math:`\delta = (\mu_1 - \mu_2)/\sigma` the standardised effect size
(Cohen's d). In practice tools like **G\*Power**, R or ``statsmodels`` do the arithmetic.

Example
-------

To detect a conversion lift from 10% to 11% at :math:`\alpha = 0.05` and **80% power**
with an effect of 0.01, the analysis returns about **7,850 users per group** — testing
only 1,000 per arm would be badly underpowered.

The power-analysis family
-------------------------

**A-priori** (before) sets the sample size; **post-hoc** (after) estimates achieved power
and is controversial; **sensitivity** asks, for a given :math:`n`, :math:`\alpha` and
power, the *smallest detectable* effect. The everyday convention is :math:`\alpha = 0.05`,
power = 0.80.
"""

CONTENT["Statistical Significance"] = r"""
What it is
----------

A result is **statistically significant** when it is **unlikely to have arisen by random
chance alone, assuming the null hypothesis** :math:`H_0` **is true** — operationally, when
the **p-value** :math:`\le \alpha`, the predefined significance level. It answers one
narrow question: *is this result sufficiently inconsistent with* :math:`H_0`?

What it does not tell you
-------------------------

Significance says nothing about **how large** the effect is, **whether it matters**, or
**whether it will replicate**. And the **p-value** is widely misread: it is the
probability, *under* :math:`H_0`, of data as extreme or more extreme than observed — *not*
the probability that :math:`H_0` is true, nor the probability the result is "due to
chance."

Statistical vs practical significance
-------------------------------------

These come apart. **Statistical** significance is about *detectability* and depends
heavily on sample size; **practical** significance is about *real-world importance* and
depends on effect size and context. With a large enough :math:`n`, a **trivial** effect
becomes significant; with a small :math:`n`, a **meaningful** one may not — so a result can
be significant yet practically meaningless.

Significance vs power
---------------------

Significance is a **binary** outcome (yes/no); **power** is the *probability* of achieving
it when a real effect exists. High power makes a true effect likely to register; under low
power, a non-significant result is **ambiguous** (it may just reflect too little data).

A decision rule, not a verdict
------------------------------

Treat significance as a **decision rule for controlling false positives under repeated
use** — part of a risk-management system, not a proof of truth. The :math:`\alpha = 0.05`
line is a **convention**, not a law: "significant" is not "important," and "not
significant" is not "no effect." Good practice reports **effect sizes, confidence
intervals, and power** alongside it, never significance alone.
"""

MINDMAP.update({
    "Post Hoc Power Analysis": [
        "A Priori Power Analysis", "Compromise Power Analysis", "Statistical Power",
        "Effect Size (δ)", "P-Value (probability value)", "Frequentist",
    ],
    "A Priori Power Analysis": [
        "Post Hoc Power Analysis", "Compromise Power Analysis", "Statistical Power",
        "Effect Size (δ)", "Type I Error", "Fixed-Horizon Testing",
    ],
    "Statistical Significance": [
        "P-Value (probability value)", "Type I Error", "Statistical Power",
        "Effect Size (δ)", "Frequentist", "A/B Testing",
    ],
})


# ----------------------------------------------------------------------
# Theme: A/B-test statistics — z-score, two-proportion z-test, MDL  (abtest / probstats)
# ----------------------------------------------------------------------

CONTENT["Z-Score"] = r"""
What it is
----------

A **z-score** (or **standard score**) says **how many standard deviations a value sits
from the mean** of its distribution. It re-expresses raw numbers on a common, unit-free
scale so values from different datasets can be compared directly.

The formula
-----------

For a population,

.. math::

   z = \frac{x - \mu}{\sigma},

with raw value :math:`x`, mean :math:`\mu` and standard deviation :math:`\sigma`. From a
**sample**, use :math:`z = (x - \bar{x})/s`.

Reading it
----------

:math:`z = 0` is exactly at the mean; :math:`z = +1` is one SD above; :math:`z = -2` two
SD below. Large magnitudes (:math:`|z| > 3`) flag likely **outliers**.

Examples
--------

An exam score of 85 with mean 70 and SD 10 gives :math:`z = (85 - 70)/10 = 1.5` — 1.5 SD
above average. A height of 150 cm with mean 170 and SD 8 gives
:math:`z = (150 - 170)/8 = -2.5`.

Why it's useful
---------------

Three things at once: **standardisation** (compare maths and English scores on different
scales), **probability** (in the standard normal — mean 0, SD 1 — a z maps to a tail area,
e.g. :math:`z = 1.96` bounds the central 95%), and **test statistics** (z-, t- and
χ²-tests all compare an observed value to its expected spread in z-like units). A
**z-score** standardises one data point; a **z-test** uses that machinery to test a
hypothesis about a mean or proportion.
"""

CONTENT["Two-Proportion Z-Test"] = r"""
What it is
----------

The **two-proportion z-test** decides whether an outcome's **rate differs significantly
between two independent groups** — the standard test behind A/B experiments (is control's
5% really below treatment's 6.2%, or just noise?).

Hypotheses
----------

The null is **equality**, :math:`H_0 : p_1 = p_2`; the alternative is
:math:`H_1 : p_1 \neq p_2` (two-tailed) or a one-sided version.

The statistic
-------------

.. math::

   z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}},

where :math:`\hat{p}_i = x_i / n_i` are the group proportions and
:math:`\hat{p} = (x_1 + x_2)/(n_1 + n_2)` is the **pooled** proportion — the shared rate
*assumed under* :math:`H_0`, used to build the standard error. Compare :math:`z` to a
critical value (:math:`\pm 1.96` at :math:`\alpha = 0.05`) or convert it to a p-value.

Worked example
--------------

Control: 100 of 1,000 → :math:`\hat{p}_1 = 0.10`. Treatment: 130 of 1,000 →
:math:`\hat{p}_2 = 0.13`. Pooled :math:`\hat{p} = 230/2000 = 0.115`; standard error
:math:`\sqrt{0.115 \times 0.885 \times 0.002} \approx 0.01425`; so
:math:`z = (0.10 - 0.13)/0.01425 \approx -2.11`, a two-tailed :math:`p \approx 0.035`.
Since :math:`p < 0.05`, reject :math:`H_0` — treatment converts significantly higher.

Assumptions
-----------

Independent samples, binary (success/failure) observations, and samples large enough for
the normal approximation (:math:`np \ge 5` and :math:`n(1 - p) \ge 5`). It powers A/B
tests, clinical recovery-rate comparisons and survey yes/no contrasts alike.
"""

CONTENT["Minimum Detectable Lift (MDL)"] = r"""
What it is
----------

The **minimum detectable lift (MDL)** is the **smallest relative change in a metric** —
conversion, revenue, clicks — that an experiment can **reliably detect**, given its sample
size :math:`n`, significance level :math:`\alpha` and power :math:`1 - \beta`. It is, in
effect, the smallest effect you have decided is worth catching.

Why it matters
--------------

It stops teams **over-optimising for trivial effects** and forces the design question up
front: *what improvement is big enough to justify the test?* Crucially, the relationship
is inverse — **the smaller the MDL you want to detect, the larger the sample you need**.

The formula
-----------

For conversion rates,

.. math::

   \text{MDL} = \frac{p_{\text{treatment}} - p_{\text{control}}}{p_{\text{control}}}.

With a 5% baseline, :math:`\alpha = 0.05`, power 0.80 and 20,000 per variant, the design
can detect a **10% relative lift** (5% → 5.5%) — so the MDL is +10%. A true lift of only
+2% would likely slip past undetected.

MDL vs MDE
----------

The two are easy to confuse. The **minimum detectable effect (MDE)** is the smallest
**absolute** change (e.g. +0.5 percentage points); the **MDL** is the smallest
**relative** change (a percentage lift). For control 5% → treatment 5.5%, the MDE is +0.5
points while the MDL is +10%.

Striking the balance
--------------------

Set the MDL **too high** and you miss small-but-valuable wins; set it **too low** and the
test may need millions of users. The resolution is a negotiation: the **business** names
the smallest improvement worth acting on, and the **statistician** sizes the experiment to
detect at least that.
"""

MINDMAP.update({
    "Z-Score": [
        "Standard Error (SE)", "Critical Value", "Sample Standard Deviation",
        "True Mean (Population Mean)", "Two-Proportion Z-Test",
        "Statistical Significance",
    ],
    "Two-Proportion Z-Test": [
        "Proportion", "Z-Score", "Conversion Rate Uplift", "True Conversion Rate",
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Statistical Significance",
    ],
    "Minimum Detectable Lift (MDL)": [
        "A Priori Power Analysis", "Conversion Rate Uplift", "Statistical Power",
        "Effect Size (δ)", "Two-Proportion Z-Test",
        "Traditional A/B Test (Fixed-Horizon A/B Test)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Practical significance, sample size, power  (abtest / inference)
# ----------------------------------------------------------------------

CONTENT["Trivial Effects"] = r"""
What it is
----------

A **trivial effect** is a result that is **statistically significant**
(:math:`p < \alpha`) yet **so small in magnitude that it has little or no practical
importance**. The difference is *real* — not chance — but too tiny to matter.

Why they appear
---------------

Two causes. First, **large samples**: with enough data, even a minuscule difference clears
the significance bar — in an A/B test on millions of users, a 0.05% conversion bump can be
"significant" yet meaningless. Second, **fixating on p-values**: a p-value says whether an
effect exists, not how big it is, so reading it without an **effect size** invites
over-interpretation.

Examples
--------

A drug that lowers blood pressure by **0.5 mmHg** versus standard care, tested on 10,000
patients, can show :math:`p < 0.001` — significant, but clinically trivial. A landing page
that moves conversion from 10.00% to **10.05%** across 1,000,000 users gives
:math:`p < 0.01`, yet a 0.05% lift is not worth deploying.

How to avoid the trap
---------------------

Always report an **effect size** (Cohen's d, a difference in proportions) next to the
p-value; read the **confidence interval** (a tight band hugging zero signals triviality);
judge **practical relevance**; and fix a **minimum detectable effect** in advance — "we act
only if conversion improves by at least +1%."

The link to power and n
-----------------------

Because larger samples raise power, they make even **trivial** effects detectable. The
discipline is to size a study to catch effects **worth caring about**, not the smallest
detectable ones — significance is necessary for a finding to matter, but never sufficient.
"""

CONTENT["Sample size"] = r"""
What it is
----------

**Sample size** :math:`n` is the **number of observations** used to estimate a parameter
or test a hypothesis. It is the master dial of inference, setting the **precision** of
estimates, the **power** of tests, and the **stability** of conclusions — in a word, the
*resolution* of what the data can tell you.

The square-root law
-------------------

The central relationship is

.. math::

   \text{SE} \propto \frac{1}{\sqrt{n}}.

The square root has a sharp consequence: **doubling** :math:`n` does *not* halve the
standard error — to **halve** it you must **quadruple** :math:`n`. Precision is bought at
an accelerating price.

In estimation and testing
-------------------------

For **estimation**, larger :math:`n` means **narrower confidence intervals**. For
**testing**, the statistic is roughly :math:`\text{effect}/\text{SE}`, so as the SE shrinks
the statistic grows — the same effect becomes **easier to push past the critical value**,
and **power rises**. Small effects only become detectable once :math:`n` is large enough.

The trade-off with effect size
------------------------------

Sample size and effect size **substitute** for one another: a **large** effect shows up in
a **small** sample, while a **small** effect needs a **large** one. In short, *sample size
compensates for a weak signal* — which is exactly why it is chosen up front via **power
analysis** from :math:`\alpha`, target power, the expected effect and the variance.

What n cannot do
----------------

More data **cannot fix bias, poor measurement or a wrong model**, and it can make
**trivial effects statistically significant** — precision is not correctness or importance.
So interpretation, not just design, depends on :math:`n`: a non-significant result with
small :math:`n` is *inconclusive*, and a significant one with enormous :math:`n` should be
checked for **practical relevance**. The emphasis shifts by context — precision in
estimation, power in testing, generalisation in ML, confounding in observational work — but
the concept is one.
"""

CONTENT["Power (1 – β)"] = r"""
What it is
----------

**Power** is the probability of **correctly rejecting** the null hypothesis :math:`H_0`
when the alternative is true — of **detecting a real effect**. Formally,

.. math::

   \text{Power} = 1 - \beta,

where :math:`\beta` is the probability of a **Type II error** (missing a true effect). The
usual target is **power** :math:`\ge 0.80`: an 80% chance of catching an effect that is
really there.

The error triad
---------------

Three quantities partition the possibilities when :math:`H_0` is actually false or true:
:math:`\alpha` is the **Type I** error (a false positive — rejecting a true :math:`H_0`),
:math:`\beta` the **Type II** error (a false negative), and :math:`1 - \beta` the power (a
true positive).

What raises power
-----------------

Four levers. A larger **effect size** :math:`\delta` is easier to detect; a larger
**sample size** :math:`n` shrinks the standard error and lifts power; a more lenient
**significance level** :math:`\alpha` (say 0.10 rather than 0.05) raises power but admits
more false positives; and **lower variance** :math:`\sigma^2` sharpens detection.

Example
-------

Testing whether a drug lowers blood pressure, with a medium effect (:math:`\delta = 0.5`),
:math:`n = 30` and :math:`\alpha = 0.05`, power might be only **0.60** — a 40% chance of
missing the effect. Raising :math:`n` to **100** lifts power to about **0.90**.

Where it's used
---------------

Power is the target of **a-priori power analysis**: fixing :math:`\alpha`, a desired power
(commonly 0.80) and an expected :math:`\delta`, one solves for the **minimum sample size**
needed — so that a true effect is very likely to register rather than slip away as a false
negative.
"""

MINDMAP.update({
    "Trivial Effects": [
        "Statistical Significance", "Minimum Detectable Lift (MDL)", "Effect Size (δ)",
        "P-Value (probability value)", "Sample size", "Statistical Power",
    ],
    "Sample size": [
        "Standard Error (SE)", "Statistical Power", "A Priori Power Analysis",
        "Effect Size (δ)", "Power (1 – β)", "Trivial Effects",
    ],
    "Power (1 – β)": [
        "Type I Error", "Statistical Power", "A Priori Power Analysis",
        "Effect Size (δ)", "Sample size", "Statistical Significance",
    ],
})


# ----------------------------------------------------------------------
# Theme: Core inference triad — alpha, effect size, hypothesis testing  (inference)
# ----------------------------------------------------------------------

CONTENT["Significance Level (α)"] = r"""
What it is
----------

The **significance level** :math:`\alpha` is the **threshold probability** for rejecting
the null hypothesis :math:`H_0` — the **maximum risk of a Type I error** (rejecting a true
:math:`H_0`) you are willing to accept. Common choices are :math:`\alpha = 0.05` (the
default), :math:`0.01` (stricter, stronger evidence demanded) and :math:`0.10` (more
lenient).

The decision rule
-----------------

Compute a test statistic and its **p-value**, then compare: if :math:`p \le \alpha`,
**reject** :math:`H_0`; if :math:`p > \alpha`, **fail to reject**. So :math:`\alpha` is
simply the **decision cutoff** fixed in advance.

What α is
---------

It is the **long-run false-positive rate**: at :math:`\alpha = 0.05`, about 5 in 100 tests
of a *true* null will wrongly reject it. Choose it by stakes — **0.01** in medicine,
genetics and other high-stakes settings; **0.05** as a general balance; **0.10** in
exploratory work where missing a real effect costs more than a false alarm.

Tied to the confidence level
----------------------------

Significance and confidence are complements: the **confidence level is** :math:`1 - \alpha`.
An :math:`\alpha = 0.05` test corresponds to **95% confidence** — across repeated
experiments, 95% of the resulting intervals would contain the true parameter.

What it is not
--------------

Three cautions: :math:`\alpha` is **not** the probability that :math:`H_0` is true; it is
chosen **before** the data, not tuned after; and clearing it means **statistically**
significant, not **practically** important — for that you still need an effect size.
"""

CONTENT["Effect Size (δ)"] = r"""
What it is
----------

**Effect size** :math:`\delta` measures the **magnitude** of a difference or relationship
— *how big* an effect is, not merely *whether* it exists. Where a p-value answers the
yes/no question of detectability, effect size carries the **practical importance**, and it
is the key input to **power analysis**.

Why it matters
--------------

Statistical and practical significance diverge: with a huge sample a 0.1% difference can be
"significant" yet trivial. Effect size restores the real-world magnitude, which is why it
anchors **sample-size planning**, results reporting in medicine and psychology, and A/B
testing in business.

The common forms
----------------

- **Cohen's d** (standardised mean difference): :math:`d = (\bar{X}_1 - \bar{X}_2)/s_p`
  with pooled SD :math:`s_p`; by rule of thumb **0.2 small, 0.5 medium, 0.8 large**.
- **Noncentrality** :math:`\delta` (power analysis):
  :math:`\delta = (\mu - \mu_0)/(\sigma/\sqrt{n})`, essentially the **expected
  t-statistic** under :math:`H_1` — larger :math:`\delta`, higher power.
- **Association**: Pearson :math:`r`, variance-explained :math:`R^2`, and ANOVA's
  :math:`\eta^2`.
- **Proportions** (A/B): the raw gap :math:`\delta = p_1 - p_2`, or the standardised
  :math:`h = 2\arcsin\!\sqrt{p_1} - 2\arcsin\!\sqrt{p_2}`.

Example
-------

A one-sample t-test with :math:`H_0` mean 100, sample mean 104, :math:`\sigma = 10`,
:math:`n = 25` gives :math:`\delta = (104 - 100)/(10/\sqrt{25}) = 4/2 = 2.0` — the mean is
**2 standard errors** from :math:`H_0`, a very large effect.

Effect size vs p-value
----------------------

The crucial contrast: a **p-value** says whether an effect exists and **depends on sample
size** (large :math:`n` makes tiny effects significant), while **effect size** says how big
it is and is **independent of sample size**. Report them together — significance for
detectability, effect size for meaning.
"""

CONTENT["Hypothesis Testing"] = r"""
What it is
----------

**Hypothesis testing** is a formal procedure for **deciding about a population parameter
from sample data while controlling the risk of error**. It is best read not as proof but
as **risk-managed decision-making**.

The two hypotheses
------------------

The **null** :math:`H_0` is the status quo — no effect, zero difference, zero coefficient.
The **alternative** :math:`H_1` is a departure from it, either **two-sided** (:math:`\neq`)
or **one-sided** (:math:`>` or :math:`<`). The test never *proves* :math:`H_1`; it asks
whether the data are **inconsistent with** :math:`H_0`.

The logic
---------

A proof-by-contradiction: **assume** :math:`H_0`, ask how likely data this extreme (or
more) would be under it, and if that likelihood is tiny, **reject** :math:`H_0`. The
machinery is a **test statistic**,

.. math::

   \text{test statistic} = \frac{\text{observed effect} - \text{effect under } H_0}{\text{standard error}},

which under :math:`H_0` follows a **known sampling distribution** (normal for z, t for
t-tests, :math:`\chi^2`, F) — that is what lets us turn it into a **p-value**, the
probability under :math:`H_0` of data as extreme or more.

The two errors
--------------

Decisions can fail two ways: a **Type I error** (reject a true :math:`H_0`, rate
:math:`\alpha`) and a **Type II error** (fail to reject a false one, rate :math:`\beta`),
with **power** :math:`= 1 - \beta`. The two trade off — tightening :math:`\alpha` raises
:math:`\beta`. A **two-sided** test is more conservative; a **one-sided** test has more
power but needs strong prior justification.

What "fail to reject" means
---------------------------

Crucially, failing to reject :math:`H_0` is **not** evidence that :math:`H_0` is true or
that no effect exists — only that the data are **insufficient against** :math:`H_0` at the
chosen :math:`\alpha`. It could be a true null, a **small** effect, or simply **too little
power**. And because large samples make trivial effects significant, always read the
p-value **alongside an effect size and a confidence interval** — significance for whether,
effect size for how much.
"""

MINDMAP.update({
    "Significance Level (α)": [
        "Type I Error", "P-Value (probability value)", "Statistical Significance",
        "Power (1 – β)", "Critical Value", "Effect Size (δ)",
    ],
    "Effect Size (δ)": [
        "Statistical Significance", "P-Value (probability value)", "Statistical Power",
        "Sample size", "Minimum Detectable Lift (MDL)", "Hypothesis Testing",
    ],
    "Hypothesis Testing": [
        "Statistical Significance", "Significance Level (α)",
        "P-Value (probability value)", "Type I Error", "Power (1 – β)",
        "Effect Size (δ)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Ranking & online ranker evaluation  (ranking)
# ----------------------------------------------------------------------

CONTENT["Ranking Algorithms"] = r"""
What it is
----------

A **ranking algorithm** **orders a set of items** — documents, products, ads, songs — so
the **most relevant or useful appear at the top**. Given a **query** (a search term, a user
profile), it scores items by predicted relevance, utility, or likelihood of interaction
(click, purchase, watch) and sorts by that score. Search engines, recommenders, ad systems
and social feeds are all ranking problems, and better ranking translates directly into user
satisfaction and revenue.

Classical IR
------------

The oldest rankers are lexical. **TF-IDF** weights a term by how often it appears in a
document against how rare it is across the corpus; **BM25** refines this with
**term-frequency saturation** and **document-length** normalisation, and remains a strong
search baseline.

Learning to rank
----------------

**Learning to rank (LTR)** trains a model from labelled relevance data, in three paradigms
by what the loss looks at: **pointwise** scores each item alone (regression or
classification — predict a click probability with logistic regression or gradient-boosted
trees); **pairwise** learns from comparisons ("is A better than B?", e.g. RankNet); and
**listwise** optimises the whole ordering at once (LambdaMART, ListNet), often against a
ranking metric like **NDCG**.

Neural rankers
--------------

The newest models use **embeddings** (Word2Vec, BERT, Transformers) to capture *semantic*
match rather than exact words — **DSSM** projects queries and documents into a shared space,
and **BERT-based rankers** (monoBERT, ColBERT) read query and document in context, sharply
improving relevance at higher compute cost.

How ranking is judged
---------------------

Ranking has its own metrics: **NDCG** (rewards relevant items near the top), **MAP**
(average precision across queries), **MRR** (rank of the first relevant item), **CTR**, and
**precision@k / recall@k**. The hard parts are **personalisation** (every user's "best"
differs), **position bias** (higher slots get clicks regardless of quality), **scale**
(rank billions of items fast), and **fairness** to minority items.
"""

CONTENT["Probabilistic Interleaving"] = r"""
What it is
----------

**Probabilistic interleaving** is an **online method for comparing ranking algorithms** by
blending their results into one list shown to users. Unlike **balanced interleaving**
(strict alternation) or **team-draft interleaving** (a draft pick), it builds the combined
list **probabilistically** — each algorithm defines a distribution over its ranking (higher
positions get more weight, e.g. a softmax over rank), and items are **sampled** from those
distributions.

Why use it
----------

Full **A/B testing** of rankers needs huge traffic and spends users on the worse variant;
balanced interleaving can be biased when lists overlap; team-draft works only for two
algorithms. Probabilistic interleaving is **flexible, scalable to many algorithms, and
statistically principled**.

How it works
------------

Build a rank-based probability distribution for each algorithm, **sample** items to form the
interleaved list, show it, and then **attribute clicks probabilistically** — rather than a
click belonging deterministically to one algorithm, the credit is **shared in proportion**
to how strongly each ranked the clicked item. Comparing **expected credit** across
algorithms reveals the winner.

Example
-------

With A = [A1, A2, A3] and B = [B1, A2, B3], a sampled interleaving might be
[A1, B1, A2, B3, A3]. If the user clicks **A2** — ranked highly by *both* — team-draft
would hand the whole click to one side, but probabilistic interleaving **splits the credit**
between A and B.

Strengths and costs
-------------------

It handles **more than two** algorithms, stays **fair when rankings overlap** (shared items
share credit), reduces bias, and is often **more sensitive** (detecting differences with
fewer clicks). The price is **complexity**: probabilistic attribution is harder to explain,
and the probability function (e.g. softmax temperature) needs careful design.
"""

CONTENT["Team Draft Interleaving (TDI)"] = r"""
What it is
----------

**Team-draft interleaving (TDI)** is an **online method for comparing two ranking
algorithms** A and B. Like balanced interleaving it merges both into one list shown to
users, but it fills the list by a **draft pick** — exactly like two captains choosing
players for a team — deciding which algorithm contributes the next slot.

Why use it
----------

**A/B testing** splits traffic and needs large samples; **balanced interleaving** can favour
one ranker depending on overlap and order. TDI **randomises the draft** so each algorithm
gets an equal, unbiased chance to place its items.

How it works
------------

Take the top-k from A and B; **randomly pick who drafts first**; then **alternate**: each
algorithm in turn adds its **highest-ranked item not already in the list**, until the
interleaving is full. Show it, and **attribute each click to whichever algorithm drafted
that item** — attribution is **deterministic and unambiguous**.

Example
-------

With A = [A1, A2, A3, A4] and B = [B1, B2, B3, B4], a draft might yield
[A1, B1, A2, B2, A3, B3, A4, B4]. A click on **B2** credits **algorithm B**, because B
drafted it.

Strengths and limits
--------------------

TDI is **fair** (randomised drafting removes systematic bias), **efficient** (fewer users
than A/B testing to detect a difference), gives **clear click ownership**, and is
**sensitive** to small gaps — which made it an industry standard. Its limits: it compares
**only two** algorithms at once, still assumes **clicks equal relevance** (noisy), and needs
care with **ties and overlapping** results.
"""

MINDMAP.update({
    "Ranking Algorithms": [
        "NDCG (Normalized Discounted Cumulative Gain)", "Mean Average Precision (MAP)",
        "Probabilistic Interleaving", "Team Draft Interleaving (TDI)", "Embedding",
        "Online Experimentation Platforms",
    ],
    "Probabilistic Interleaving": [
        "Team Draft Interleaving (TDI)", "Balanced Interleaving", "Ranking Algorithms",
        "Online Experimentation Platforms", "A/B Testing",
        "NDCG (Normalized Discounted Cumulative Gain)",
    ],
    "Team Draft Interleaving (TDI)": [
        "Balanced Interleaving", "Probabilistic Interleaving", "Ranking Algorithms",
        "Online Experimentation Platforms", "A/B Testing",
        "NDCG (Normalized Discounted Cumulative Gain)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Beyond fixed A/B testing — interleaving, causal impact, bandits  (abtest / causal / bandits)
# ----------------------------------------------------------------------

CONTENT["Balanced Interleaving"] = r"""
What it is
----------

**Balanced interleaving** is the simplest **online method for comparing two ranking
algorithms**: it mixes their results into one combined list shown to users, then reads off
which algorithm's items draw more clicks. Both rankers get a **fair, equal chance** to place
items, so the comparison happens *within a single session* rather than across split traffic.

Why use it
----------

Classic **A/B testing** sends half the users to A and half to B, which needs large samples
and time. Interleaving shows **both at once**, making the comparison **faster and more
sensitive** — small quality gaps surface with far fewer interactions.

How it works
------------

Take the top-k from A and B and build the list by **strict alternation** — first from A,
then B, then A, ensuring neither dominates by position. Show the combined list, **credit
each click to the algorithm that contributed that item**, and the ranker with more credited
clicks wins.

Example
-------

With A = [A1, A2, A3, A4] and B = [B1, B2, B3, B4], the interleaving is
[A1, B1, A2, B2, A3, B3, A4, B4]. If the user clicks A1, B2 and A3, then A scores **2** and
B scores **1** — A wins this impression.

Strengths and limits
--------------------

It is **efficient, fair and sensitive**, but it compares **only two** algorithms, assumes
**clicks track relevance** (which is noisy), and needs care so that **position bias** in the
alternation doesn't quietly favour one side. **Team-draft** and **probabilistic
interleaving** were developed to address exactly these weaknesses.
"""

CONTENT["Causal Impact"] = r"""
What it is
----------

**Causal impact** is the **overall effect of an intervention** — a campaign, launch or
policy — **on an outcome**. In data science the term usually points to **Google's
CausalImpact** (Brodersen et al., 2015): a **Bayesian structural time-series** model that
estimates what *would* have happened without the intervention and compares it to what
actually did.

Why it's needed
---------------

In an RCT or A/B test the causal effect is easy — you have treatment and control groups. But
in **observational or time-series** settings there may be **no untreated control**: run a
nationwide ad campaign and there is no parallel country that didn't see it. The fix is to
model the **counterfactual** — the outcome you'd have seen without the intervention — and
measure the gap.

The method and formula
----------------------

CausalImpact fits a model on the **pre-intervention** period (using historical data and
**control covariates**), forecasts the **counterfactual** for the post period, and
subtracts:

.. math::

   \text{Causal Impact} = Y_{\text{observed, post}} - Y_{\text{predicted, counterfactual}},

reporting the difference **with Bayesian credible intervals** rather than a single number.

Example
-------

A firm launches a TV campaign in July. Trained on January–June sales, the model forecasts a
**counterfactual of 50,000 units**; actual sales come in at **60,000** — an estimated impact
of **+10,000 units**, with a credible interval of roughly [7,000, 13,000].

Strengths, limits, and effect vs impact
---------------------------------------

It handles **time series** naturally, yields **full posterior distributions**, uses
**covariates** to sharpen accuracy, and needs no RCT — but it **assumes the model captures
the dynamics**, is **sensitive to the choice of controls**, and wants a **clear start date**.
Note the level distinction: the **causal effect** is micro (the change per unit — a drug
lowers blood pressure 5 mmHg), while the **causal impact** is macro (the aggregate — a
nationwide rollout prevents 10,000 hospitalisations).
"""

CONTENT["Bandit Algorithms"] = r"""
What it is
----------

**Bandit algorithms** solve the **exploration–exploitation trade-off** in sequential
decisions. Cast as a **multi-armed bandit (MAB)** — slot machines with several arms — the
goal is to **maximise cumulative reward** by choosing which arm to pull, balancing
**exploration** (trying options to learn their payoff) against **exploitation** (playing the
best-known option now).

Why it matters
--------------

An A/B test holds a **fixed split** until significance, spending traffic on losing variants
the whole time. A bandit **reallocates traffic dynamically** toward what's winning, so better
options get more users **sooner** — cutting the cost of learning. It powers ad selection,
recommendations, pricing and adaptive trials.

The core tension
----------------

Suppose three arms pay out at 5%, 7% and 6%. Always playing the current best (7%) risks
**never discovering** the 6% arm is actually better with more data; always exploring
**wastes** reward. Good bandits trade these off automatically.

The main algorithms
-------------------

- **ε-greedy**: with probability :math:`\varepsilon` explore a random arm, otherwise exploit
  the best — simple and effective.
- **UCB** (upper confidence bound): play the arm with the highest **optimistic** estimate,
  exploring when uncertainty is high and settling as data accrues.
- **Thompson sampling** (Bayesian): keep a posterior per arm (a Beta distribution for
  click/no-click), **sample** from each, and play the highest draw — a natural, highly
  effective balance.
- **Softmax / Boltzmann**: choose arms with probability rising in estimated reward, tuned by
  a temperature.

Trade-offs vs A/B testing
-------------------------

Bandits **maximise reward during the test** (not just identify a winner after it) and
typically need **fewer samples**, at the cost of **more complexity** and less clean
significance reporting. They shine when rewards are **immediate and measurable** and the
environment is **stable**; rapidly shifting preferences or delayed rewards weaken them.
"""

MINDMAP.update({
    "Balanced Interleaving": [
        "Team Draft Interleaving (TDI)", "Probabilistic Interleaving", "Ranking Algorithms",
        "Online Experimentation Platforms", "A/B Testing",
        "NDCG (Normalized Discounted Cumulative Gain)",
    ],
    "Causal Impact": [
        "Treatment Effect", "Causal Inference", "Bayesian Time Series",
        "Online Experimentation Platforms", "A/B Testing", "Counterfactual Explanations",
    ],
    "Bandit Algorithms": [
        "Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))",
        "Bayesian Decision Theory (BDT)", "A/B/n Test", "Beta Distribution", "A/B Testing",
        "Online Experimentation Platforms",
    ],
})


# ----------------------------------------------------------------------
# Theme: Multi-variant testing & the peeking problem  (abtest)
# ----------------------------------------------------------------------

CONTENT["A/B/n Test"] = r"""
What it is
----------

An **A/B/n test** generalises the two-arm A/B test to **several variants at once** —
A vs B vs C vs … n — of a page, app or feature, splitting traffic across all of them to find
which scores best on a chosen metric (conversion, click-through, engagement).

How it works
------------

Pick a **control** (A, the current design), build variants B, C, D…; **randomly split**
users across them; track the metric; and use a statistical test (two-proportion z-test,
chi-square or a Bayesian model) to pick the winner. Testing button text — "Buy Now" vs
"Shop Now" vs "Get Yours Today" vs "Order Now" — sends each of four equal groups one variant
and compares conversions.

Why and when
------------

It tests **many ideas in one experiment** rather than a sequence of A/B tests, which is
faster when you have several candidate designs and enough traffic for a **winner-takes-all**
verdict.

The costs
---------

Two prices. **Traffic**: a 50/50 split becomes 33/33/33 and beyond, so each arm gets less
data and significance takes longer. And the **multiple-comparisons problem**: every extra
variant is another chance for a false positive, so the family-wise error rate climbs unless
corrected. It also reveals less about **element interactions** than a multivariate test.

Example
-------

Optimising newsletter sign-ups across four calls to action at 25% traffic each, after two
weeks A converts 5%, B 6%, **C 7.5%** and D 5.2%. C is the significant winner and ships.
"""

CONTENT["Multivariate Test (MVT)"] = r"""
What it is
----------

A **multivariate test (MVT)** varies **several page elements at once** and tests their
**combinations**, to learn not just which single change helps but **how elements interact**.
Where A/B compares whole versions, MVT decomposes the page into factors.

How it works
------------

Choose the elements to vary — say **headline, button colour, image** — and form the
**combinations**. With two options each, that is :math:`2 \times 2 \times 2 = 8` versions;
users are split across them and the metric measured per combination.

Full vs fractional factorial
----------------------------

A **full factorial** MVT tests **every** combination, giving the cleanest read on **main
effects** (each variable alone) and **interaction effects** (how they combine) — but the
count explodes. A **fractional factorial** tests a **chosen subset**, cutting traffic and
complexity at the cost of resolving some interactions.

Why interactions matter
-----------------------

The payoff is the interaction. A landing-page test might find a strong headline worth
**+10%**, a green button **+5%** and a lifestyle image **+2%** individually — yet the
**combination** of all three lifts conversion **+25%**, more than the parts suggest. That
synergy is invisible to separate A/B tests.

The cost
--------

MVT needs **much larger samples** (variants multiply fast), is **harder to design and
analyse**, and carries the same **multiple-comparisons** false-positive risk. Use it to
**fine-tune many elements** once you have the traffic; use A/B for big, single-design
changes.
"""

CONTENT["Risk of Peeking"] = r"""
What it is
----------

**Peeking** is looking at an experiment's results **before it officially ends** and acting
on that interim data. The **risk of peeking** is the **inflated false-positive (Type I)
rate** that results from repeatedly checking and **stopping as soon as significance
appears**.

Why it breaks the test
----------------------

A fixed-horizon test assumes a **single look at a predetermined sample size**. Each extra
peek is another **independent chance** for noise to cross :math:`p < 0.05`, so the true
error rate compounds far above the nominal :math:`\alpha`. Peek ten times at a 5% threshold
and the real false-positive rate can reach **20–30%** — this compounding is called **alpha
inflation**.

What it costs
-------------

Concretely: a button test shows A ahead on day 1 (:math:`p = 0.04`), you stop and crown
A — but over the full two weeks **B** would have won. The early stop produced a **false
conclusion**, and at scale that means wrong launches, lost revenue, and eroded trust in
experimentation.

How to avoid it
---------------

Four routes: **predefine** the sample size and duration and only check at the end; use
**sequential testing / alpha-spending** designs built for interim looks (**group
sequential**, **O'Brien–Fleming**, **Pocock**); use **Bayesian** methods designed for
continuous monitoring; or, if peeks are unavoidable, apply **multiplicity corrections**
(Bonferroni, Holm). The peeking problem is precisely why the whole machinery of
fixed-horizon and sequential testing exists.
"""

MINDMAP.update({
    "A/B/n Test": [
        "A/B Testing", "Multivariate Test (MVT)",
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Bandit Algorithms",
        "Two-Proportion Z-Test", "Type I Error",
    ],
    "Multivariate Test (MVT)": [
        "A/B/n Test", "A/B Testing", "Traditional A/B Test (Fixed-Horizon A/B Test)",
        "Conversion Rate (CR)", "Type I Error", "Online Experimentation Platforms",
    ],
    "Risk of Peeking": [
        "Stopping Rules", "Fixed-Horizon Testing", "Group Sequential Testing",
        "O'Brien–Fleming (OBF) Method", "Pocock Method", "Type I Error",
    ],
})


# ----------------------------------------------------------------------
# Theme: Causal inference, the p-value, the z-test  (causal / inference)
# ----------------------------------------------------------------------

CONTENT["Causal Inference"] = r"""
What it is
----------

**Causal inference** is the discipline of establishing whether one variable **actually
causes** a change in another, rather than merely **correlating** with it. The classic trap:
ice-cream sales and drownings rise together, but neither causes the other — hot weather
drives both. Causal inference is the toolkit for separating **causation from association**.

Why it matters
--------------

Decisions hinge on it. Did the campaign **cause** the sales lift, or would sales have risen
anyway? Did the treatment **cause** recovery, or was it natural? Did the policy **cause** the
change, or did something else? Acting on a spurious correlation wastes money and harms
people.

The gold standard and its alternatives
--------------------------------------

A **randomised controlled trial (RCT)** assigns treatment and control **at random**, which
balances confounders and makes the comparison causal. When randomisation is impossible
(ethics, cost), **observational** methods approximate it: **regression** adjusts for measured
confounders; **propensity-score matching** pairs treated and untreated units with similar
covariates; **difference-in-differences** compares before/after trends against an untreated
group; **instrumental variables** exploit a variable that moves treatment but not the
outcome directly; and **regression discontinuity** compares units just above and below a
treatment cutoff.

The counterfactual core
-----------------------

Underneath sits the **potential-outcomes (Rubin) model**: each unit has a treated outcome
:math:`Y_1` and an untreated outcome :math:`Y_0`, and the causal effect is
:math:`E[Y_1 - Y_0]`. The **fundamental problem** is that we never observe **both** for the
same unit — which is exactly why we need RCTs, DiD and the rest. **Directed acyclic graphs
(DAGs)** complement this by mapping confounders and mediators visually.

The pitfalls
------------

The enemies of causal claims are **confounding** (a third variable driving both),
**selection bias** (groups that differ systematically), **reverse causality** (the outcome
also influences the cause), and **unobserved variables** that can't be adjusted for if
they're never measured.
"""

CONTENT["P-Value (probability value)"] = r"""
What it is
----------

The **p-value** is the probability of observing data **at least as extreme** as what you
saw, **assuming the null hypothesis** :math:`H_0` **is true**. It is a measure of **evidence
against** :math:`H_0`: a **low** p-value means the data would be surprising under
:math:`H_0` (stronger evidence against it), a **high** one means the data sit comfortably
with :math:`H_0`.

Reading it
----------

By convention :math:`p < 0.01` is very strong evidence, :math:`p < 0.05` is "significant",
:math:`p < 0.10` is weak, and :math:`p > 0.10` is not significant. The decision rule pairs it
with a threshold: if :math:`p \le \alpha`, reject :math:`H_0`; if :math:`p > \alpha`, fail to
reject.

How it's computed
-----------------

Form the **test statistic** (z, t, :math:`\chi^2`…) and find the tail probability of a value
that extreme under its distribution — one tail for a one-sided test, both for two-sided. For
a two-tailed t-test,

.. math::

   p = 2 \times P\!\left(T \ge |t_{\text{observed}}|\right).

For example, 17 heads in 20 tosses gives a binomial p of about **0.003** against a fair coin
(reject); a two-sample test with :math:`t = 2.1, df = 28` gives :math:`p \approx 0.045`
(reject at 0.05).

What it is not
--------------

Three persistent misconceptions: the p-value is **not** the probability that :math:`H_0` is
true; it is **not** an effect size (a tiny p means *surprising*, not *large*); and it is
**sample-size sensitive** — huge samples make trivial effects significant, tiny samples can
miss real ones. Always read it alongside an effect size and a confidence interval.
"""

CONTENT["Z-Test"] = r"""
What it is
----------

A **z-test** is a hypothesis test for whether a **sample mean (or proportion) differs from a
known population value**, built on the **standard normal** distribution. It applies when the
**sample is large** (:math:`n > 30`, so the CLT holds) and the **population variance is
known** — or well approximated by a large sample.

The statistic
-------------

For a one-sample mean,

.. math::

   z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}},

the gap between the sample mean :math:`\bar{X}` and the hypothesised :math:`\mu`, measured in
**standard errors**. There are three common forms: **one-sample** (mean vs population),
**two-sample** (two independent means), and the **proportion** z-test.

The procedure
-------------

State :math:`H_0` (no difference) and :math:`H_1`; pick :math:`\alpha`; compute :math:`z`;
and compare to the **critical value** (:math:`\pm 1.96` at :math:`\alpha = 0.05`, two-tailed)
or read off a p-value — reject when :math:`|z|` exceeds it.

Examples
--------

With :math:`\mu = 100, \sigma = 15, n = 50` and a sample mean of 105,
:math:`z = 5 / 2.12 \approx 2.36 > 1.96`, so reject :math:`H_0`. For a proportion, 320 of 500
(0.64) against a hypothesised 0.60 gives :math:`z = 0.04 / 0.022 \approx 1.82 < 1.96` —
**fail** to reject.

Z-test vs t-test
----------------

The choice turns on what you know about the variance. Use a **z-test** when :math:`\sigma` is
**known** and the sample is **large**, working from the normal distribution; use a
**t-test** when :math:`\sigma` is **unknown** and estimated from the sample :math:`s`,
working from the heavier-tailed **Student's t** — the gap between them vanishes as :math:`n`
grows.
"""

MINDMAP.update({
    "Causal Inference": [
        "Causal Impact", "Treatment Effect", "Regression Coefficient", "A/B Testing",
        "Counterfactual Explanations", "Frequentist",
    ],
    "P-Value (probability value)": [
        "Statistical Significance", "Significance Level (α)", "Hypothesis Testing",
        "Type I Error", "Effect Size (δ)", "Z-Test",
    ],
    "Z-Test": [
        "Z-Score", "Two-Proportion Z-Test", "Critical Value", "Standard Error (SE)",
        "Hypothesis Testing", "T-Test",
    ],
})


# ----------------------------------------------------------------------
# Theme: t-test, engagement & monetization metrics  (inference / growth)
# ----------------------------------------------------------------------

CONTENT["T-Test"] = r"""
What it is
----------

A **t-test** asks whether the **means of two groups differ significantly**, using the
**Student's t-distribution**. It is the tool of choice when the **sample is small**
(:math:`n < 30`) and the **population standard deviation is unknown** — estimated instead
from the data, which the heavier-tailed t accounts for.

The three forms
---------------

A **one-sample** t-test compares a sample mean to a known value (is the class average
different from 70?); an **independent two-sample** test compares two separate groups (male
vs female scores); and a **paired** test compares the *same* units measured twice (weight
before vs after a diet).

The statistic
-------------

For the independent test,

.. math::

   t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}},

the difference in means over its standard error. Compare :math:`t` to a **critical value**
from the t-distribution at the appropriate **degrees of freedom** (for the independent test,
:math:`df = n_1 + n_2 - 2`), or read a p-value.

Example
-------

A sample of 25 with mean 72, against a hypothesised population mean of 70 with
:math:`s = 5`, gives :math:`t = (72 - 70)/(5/\sqrt{25}) = 2`. The critical value at
:math:`df = 24, \alpha = 0.05` is about **2.064**, and since :math:`2 < 2.064` we **fail to
reject** :math:`H_0` — no significant difference.

Assumptions
-----------

The data should be roughly **normal**; the independent test also assumes **independent
groups** and **equal variances** — if the variances differ, use **Welch's t-test**. As
:math:`n` grows the t-distribution approaches the normal, and the t-test converges to the
z-test.
"""

CONTENT["Session Length"] = r"""
What it is
----------

**Session length** is the **time a user spends in a single session** with an app, site or
product — from opening it to closing it or timing out from inactivity — usually in seconds
or minutes.

The formula
-----------

Per session it is simply

.. math::

   \text{Session Length} = \text{end time} - \text{start time},

and across many sessions the **average session length** is total time spent divided by the
number of sessions. Analytics platforms compute both automatically.

Why it's tracked
----------------

It is a core **engagement** signal — longer sessions often mean users find value — and it
feeds **retention** (consistently short sessions hint at usability problems),
**monetisation** (in ad models, more time means more impressions), and **UX** decisions. A
12-minute news session, a 2.5-minute site average (2,500 minutes over 1,000 sessions), or a
45-minute gaming session each read differently.

The interpretation trap
-----------------------

Longer is **not always better**. A short session can mean the user **found what they needed
instantly** (checking the weather), and the "right" length is **product-specific** — a few
minutes for a news site, 30–60 for video streaming. Beware **idle time**, too: an
open-but-unused app can inflate the number. Session length is meaningful only **read against
industry, product type and user intent**.
"""

CONTENT["Revenue per User (RPU / ARPU)"] = r"""
What it is
----------

**Revenue per user (RPU)** — almost always called **average revenue per user (ARPU)** — is
the **average income a business earns per customer** over a period (monthly, quarterly,
yearly). A **unit-economics** metric, it answers how much revenue each unit of the user base
generates, and it is a staple of telecom, streaming, SaaS and mobile-gaming reporting.

The formula
-----------

.. math::

   \text{ARPU} = \frac{\text{total revenue in period}}{\text{active users in period}},

where revenue spans subscriptions, in-app purchases and ads, and "active users" is usually
**MAU** or **DAU** depending on the business.

The key variants
----------------

**ARPU** counts *all* users (free and paying); **ARPPU** counts only **paying** users —
crucial for freemium models; and ARPU can be **blended** (across everyone) or **segmented**
(by product, geography or cohort). A freemium game with ``$50,000`` revenue across 10,000
MAU has an **ARPU of** ``$5``, but if only 2,000 pay, its **ARPPU is** ``$25`` — revealing
that a small paying base carries the economics.

How it fits the other metrics
-----------------------------

ARPU is a **short-term snapshot** of revenue per user, complementing the **lifetime view**:
**CLV** is total expected revenue over the whole relationship, while **CAC** is the cost to
acquire a user. A healthy business keeps :math:`\text{CLV} > \text{CAC}`, and ideally
:math:`\text{ARPU} \times \text{retention} \gg \text{CAC}`.

The limitations
---------------

ARPU is an **average**, so it hides the **distribution** (one user at ``$500`` and one at
``$0`` both read as ``$50``), can **mask churn** (fewer customers paying more keeps ARPU flat
while the base shrinks), and means little alone — it must be paired with **CAC, churn and
CLV** to judge whether the business is actually healthy.
"""

MINDMAP.update({
    "T-Test": [
        "Z-Test", "Sample Mean", "Sample Standard Deviation", "Critical Value",
        "Hypothesis Testing", "Effect Size (δ)",
    ],
    "Session Length": [
        "Revenue per User (RPU / ARPU)", "Churn", "Retention", "Customer Segmentation",
        "Conversion Rate (CR)", "SaaS (Software as a Service)",
    ],
    "Revenue per User (RPU / ARPU)": [
        "LTV:CAC Ratio", "Customer Lifetime", "Blended CAC (Customer Acquisition Cost)",
        "Churn", "SaaS (Software as a Service)", "Customer Segmentation",
    ],
})


# ----------------------------------------------------------------------
# Theme: churn & retention; the significance result-label  (growth / inference)
# ----------------------------------------------------------------------

CONTENT["Churn"] = r"""
What it is
----------

**Churn** (customer churn) is the **rate at which customers stop using** a product or service
over a period — the **mirror image of retention**, and a defining metric for SaaS,
subscriptions, telecom and apps.

The formula and types
---------------------

The basic customer churn rate is

.. math::

   \text{churn rate} = \frac{\text{customers lost in period}}{\text{customers at start}} \times 100.

Beyond this, **revenue churn** tracks lost *revenue* (which can diverge from customer churn
when large accounts leave), and churn splits into **voluntary** (the customer cancels) and
**involuntary** (failed payments, expired cards).

Examples
--------

A SaaS product going from 1,000 to 950 paying users in a month lost 50 — a **5% churn
rate**. A mobile app where 40% of new users uninstall within 30 days has **40% 30-day
churn**.

Why it matters
--------------

Churn is **recurring revenue walking out the door**, and it caps growth: with enough churn,
even strong acquisition nets to zero. Because keeping a customer typically costs **5–7×
less** than winning a new one, reducing churn is usually the cheapest growth lever.

Churn in ML and its mirror
--------------------------

Teams build **churn-prediction models** that flag at-risk users from behaviour (logins,
purchases, complaints), then intervene with offers, better onboarding or support. And it is
exactly complementary to retention: if monthly churn is **5%**, retention is **95%**.
"""

CONTENT["Retention"] = r"""
What it is
----------

**Retention** measures how well a business **keeps its customers or users over time** — the
**complement of churn**, and a direct signal of product value: if people keep coming back,
the product is working.

The metrics
-----------

The **customer retention rate (CRR)** strips out new sign-ups to measure how many existing
customers stayed:

.. math::

   \text{CRR} = \frac{E - N}{S} \times 100,

with :math:`S` customers at the start, :math:`E` at the end, and :math:`N` newly acquired.
**Cohort retention** tracks a starting group over time (100 sign up in week 1, 40 still
active in week 4 → **40% week-4 retention**), and **revenue retention** comes in two forms:
**gross (GRR)**, which ignores expansion, and **net (NRR)**, which adds upsells and
cross-sells.

Example
-------

Start with 100 paying customers, lose 10 and gain 20: the CRR counts only the kept ones,
:math:`(110 - 20)/100 = 90\%`. A mobile app with 300 of 1,000 installs still active at 30
days has **30% retention**.

Why it matters
--------------

Retention is **cheaper than acquisition**, drives **predictable recurring revenue**, and is
the clearest **indicator of product value** — which is why it sits at the heart of
**lifetime value** and growth. Teams raise it with onboarding, A/B tests and
recommendations, and predict who is about to drop with the same models used for churn.
"""

CONTENT["Statistically Significant"] = r"""
What it is
----------

A result is **statistically significant** when the **observed effect is unlikely to have
arisen by chance** under the null hypothesis, judged against a chosen significance level
:math:`\alpha`. Operationally it means there is **enough evidence to reject** :math:`H_0` —
and crucially, "significant" here means *statistical evidence*, **not real-world
importance**.

The decision rule
-----------------

It comes down to comparing the **p-value** to the threshold: if :math:`p \le \alpha`, the
result **is** statistically significant and you reject :math:`H_0`; if :math:`p > \alpha`, it
is **not**, and you fail to reject. The usual :math:`\alpha` is 0.05.

Examples
--------

A drug trial with :math:`p = 0.01` against :math:`\alpha = 0.05` **is** significant —
evidence the drug beats placebo. An A/B test where a new button lifts clicks 3% but returns
:math:`p = 0.2` is **not** significant — the lift could be noise.

The cautions
------------

Three matter. **Significance is not importance**: with a large enough dataset a trivial 0.5%
effect can clear the bar yet mean nothing. It is **sample-size dependent**: bigger samples
make significance easier to reach. And it is vulnerable to **p-hacking** — running many tests
or slicing data until something crosses :math:`\alpha`. A significant result is a starting
point for judgement, read alongside effect size and context, not a verdict on its own.
"""

MINDMAP.update({
    "Churn": [
        "Retention", "Revenue per User (RPU / ARPU)", "Customer Lifetime",
        "Blended CAC (Customer Acquisition Cost)", "SaaS (Software as a Service)",
        "Customer Segmentation",
    ],
    "Retention": [
        "Churn", "Revenue per User (RPU / ARPU)", "Customer Lifetime",
        "Cohort-Based LTV (Simple Version)", "SaaS (Software as a Service)", "Upselling",
    ],
    "Statistically Significant": [
        "Statistical Significance", "P-Value (probability value)", "Significance Level (α)",
        "Hypothesis Testing", "Type I Error", "Effect Size (δ)",
    ],
})


# ----------------------------------------------------------------------
# Theme: IID, its time-series violation, and time-aware validation  (probstats / validation)
# ----------------------------------------------------------------------

CONTENT["IID (Independent and Identically Distributed)"] = r"""
What it is
----------

**IID** — **independent and identically distributed** — is the bedrock assumption of most
statistics and machine learning. Two conditions: each observation is **independent** (knowing
one tells you nothing about another) and all observations are **identically distributed**
(drawn from the same :math:`P(X)`). Compactly,

.. math::

   X_1, X_2, \dots, X_n \sim \text{i.i.d. } P(X).

Ten fair-coin flips are the canonical case — each flip independent, each with the same
:math:`P(H) = 0.5`.

The two halves
--------------

**Independence** fails when one sample carries information about another. **Identical
distribution** fails when the underlying distribution shifts across the sample. Both can
break separately: data can be dependent but identically distributed, or independent but
drifting.

When it holds and when it doesn't
---------------------------------

The clean case is **random sampling** from a fixed population (survey respondents drawn at
random). It breaks in **time series** (today's stock price depends on yesterday's — not
independent), under a **changing population** (early vs late customers differ — not
identical), and with **grouped data** (several records from the same patient are correlated).

Why it matters
--------------

IID is what makes the math tractable — the **law of large numbers** and the **central limit
theorem** lean on it, and linear/logistic regression, hypothesis tests and freshly
initialised neural nets all assume it. Violating it yields **biased estimates and
overconfident predictions**. In ML the training set is usually assumed IID but often isn't
(autocorrelation, drift, leakage), which is why **time-series CV, grouped CV and domain
adaptation** exist to cope.
"""

CONTENT["Temporal autocorrelation (Serial Correlation)"] = r"""
What it is
----------

**Temporal autocorrelation** (or **serial correlation**) is when the values of a **time
series correlate with their own past** — the value at time :math:`t` depends partly on
:math:`t-1, t-2, \dots`. It is the defining property of time-series data and the most common
way the IID assumption breaks.

The measure
-----------

The **autocorrelation function (ACF)** at lag :math:`k` is

.. math::

   \rho_k = \frac{\operatorname{Cov}(X_t, X_{t-k})}{\sigma^2},

the correlation between the series and its own lag-:math:`k` copy, where :math:`\sigma^2` is
the series variance. Stock prices (today near yesterday), temperature and weekly website
traffic all show it.

Why it matters
--------------

First, it **violates IID** — past strongly influences future, so models that assume
independence are wrong. Second, it **drives forecasting**: ARIMA and SARIMA explicitly model
autocorrelation, and **ACF/PACF** plots reveal the AR and MA orders. Third, it is a
**diagnostic**: the **Durbin-Watson** test checks regression residuals, and autocorrelated
residuals signal a misspecified model.

Positive, negative, and reading the plot
----------------------------------------

**Positive** autocorrelation means high tends to follow high (**momentum**); **negative**
means high tends to follow low (**mean-reversion**). On an ACF plot, strong spikes at lags
1, 2 and 7 would suggest **weekly seasonality**.
"""

CONTENT["Blocked Splits (Single Holdout)"] = r"""
What it is
----------

A **blocked split** (or **single holdout** for time series) is the simplest time-series
validation: cut the data into **one contiguous training block** and **one later contiguous
test block**. Unlike a random split, it **respects temporal order** — train on the past, test
on the future — which is mandatory when observations are autocorrelated.

How it works
------------

Order the data chronologically, take an **early block to train** and the **later block to
test**, fit on the first and evaluate on the second. For data spanning **2018–2022**, train
on **2018–2020** and test on **2021–2022** — a single cut, one split.

Why and when
------------

It is **simple, fast**, and the natural **first baseline** for a time-series model — a good
fit when retraining is infrequent or the series is stable.

The limitation
--------------

There is **only one split**, so the estimate is **sensitive to where you cut** and gives a
less robust read on generalisation than rolling schemes. An unusual test window (a pandemic
spike, say) can misrepresent the model. The more robust alternatives keep the time order but
add splits: an **expanding window** grows the train set forward over many folds, and a
**sliding window** rolls a fixed-size train set forward — trading simplicity for a steadier
estimate.
"""

MINDMAP.update({
    "IID (Independent and Identically Distributed)": [
        "Temporal autocorrelation (Serial Correlation)", "Time Series", "Probability",
        "Signal Processing", "Blocked Splits (Single Holdout)", "Cross-Validation (CV)",
    ],
    "Temporal autocorrelation (Serial Correlation)": [
        "IID (Independent and Identically Distributed)", "Time Series",
        "Blocked Splits (Single Holdout)", "Bayesian Time Series", "Signal Processing",
        "Sliding Window (Rolling Window) Cross-Validation",
    ],
    "Blocked Splits (Single Holdout)": [
        "Sliding Window (Rolling Window) Cross-Validation", "Expanding Window Cross-Validation",
        "Cross-Validation (CV)", "Temporal autocorrelation (Serial Correlation)",
        "IID (Independent and Identically Distributed)", "Time Series",
    ],
})


# ----------------------------------------------------------------------
# Theme: time-series cross-validation & data leakage  (validation)
# ----------------------------------------------------------------------

CONTENT["Sliding Window (Rolling Window) Cross-Validation"] = r"""
What it is
----------

**Sliding-window** (or **rolling-window**) cross-validation is a time-series validation
scheme where the training set has a **fixed size** and **slides forward** through time: as
new data enters the window, the **oldest data drops out**. Like all time-series CV, it
**respects time order** — past trains, future tests.

How it works
------------

Fix a **window size**, train on that window, validate on the next step, then **slide forward
and repeat**. Each fold uses a same-length training block ending just before its test block.

Example
-------

With a 2-year window over **2020–2024**: train **2020–2021** → test **2022**; train
**2021–2022** → test **2023**; train **2022–2023** → test **2024**. Notice 2020 is
**dropped** once the window moves past it.

vs the expanding window
-----------------------

The contrast is what happens to old data. An **expanding window keeps all history**, suiting
cases where old data stays relevant (macroeconomics, cumulative learning). A **sliding window
discards it**, suiting **non-stationary** settings where recent data is more predictive —
financial markets, demand forecasting, IoT sensor streams.

Benefits and the size trade-off
-------------------------------

It keeps the model **focused on recent patterns** and **bounds compute** (the training set
never grows without limit). The cost: it can **forget useful long-run history**, and the
**window size is a critical knob** — too short is noisy, too long is unresponsive.
"""

CONTENT["Expanding Window Cross-Validation"] = r"""
What it is
----------

**Expanding-window** cross-validation is a time-series scheme where the **training set grows**
over time while the test set always takes the **next** period. The rule is the usual one —
**train on the past, predict the future** — and crucially, **once data enters training it
stays** in every later fold.

How it works
------------

Order the data chronologically, start from an initial training period, and validate on the
next. Then **expand** the training window to absorb more past data and validate on the
following period, repeating to the end of the series.

Example
-------

Over **2020–2024**: train **2020** → test **2021**; train **2020–2021** → test **2022**;
train **2020–2022** → test **2023**; train **2020–2023** → test **2024**. The training block
**keeps growing**; the test always sits just after it.

vs the rolling window
---------------------

Both respect time order; they differ on memory. **Expanding** accumulates **all** history,
which helps when **older data is still relevant** (finance, macroeconomic forecasting). The
**rolling/sliding** window holds a **fixed** size and drops the oldest data, which helps when
**recent data dominates** (stock trading, demand forecasting). The choice is really a
question of whether the process is stationary.
"""

CONTENT["Data Leakage"] = r"""
What it is
----------

**Data leakage** is when **information from outside the training data slips into training**,
giving the model unfair access to **future or hidden knowledge**. The signature is a model
that looks **excellent in validation** but **collapses on real, unseen data**.

The four types
--------------

**Target leakage**: a feature encodes the answer — predicting loan default from
``debt_collected_after_default``, which only exists *because* of default. **Train-test
contamination**: test information bleeds in via preprocessing — e.g. scaling with a mean and
standard deviation computed over the **whole** dataset instead of the training fold alone.
**Temporal leakage**: using **future** data to predict the past — forecasting January's price
with March's trading volume. **Group leakage**: the **same group** (patient, user, session)
lands in both train and test, so the model just **recognises the group**.

How to prevent it
-----------------

Five guards: fit **preprocessing on the training fold only** and apply it to the rest; **drop
features that wouldn't exist at prediction time**; use **time-aware splits** for temporal
data; use **group-aware CV** (``GroupKFold``, ``StratifiedGroupKFold``) to keep groups
intact; and **monitor after deployment** — a sharp drop from validation to production is the
classic leakage tell.

Why it's dangerous
------------------

Leakage manufactures a **false sense of performance**, masking **overfitting** and poor
generalisation, and in regulated domains like finance and healthcare it can turn into a
**compliance problem** when the model fails on the data that matters.
"""

MINDMAP.update({
    "Sliding Window (Rolling Window) Cross-Validation": [
        "Expanding Window Cross-Validation", "Blocked Splits (Single Holdout)",
        "Cross-Validation (CV)", "Temporal autocorrelation (Serial Correlation)",
        "Time Series", "Data Leakage",
    ],
    "Expanding Window Cross-Validation": [
        "Sliding Window (Rolling Window) Cross-Validation", "Blocked Splits (Single Holdout)",
        "Cross-Validation (CV)", "Temporal autocorrelation (Serial Correlation)",
        "Time Series", "Data Leakage",
    ],
    "Data Leakage": [
        "Cross-Validation (CV)", "Stratified Group K-Fold", "Blocked Splits (Single Holdout)",
        "Sliding Window (Rolling Window) Cross-Validation",
        "Temporal autocorrelation (Serial Correlation)", "Data Drift",
    ],
})


# ----------------------------------------------------------------------
# Theme: stratified cross-validation family  (validation)
# ----------------------------------------------------------------------

CONTENT["Stratified Group K-Fold"] = r"""
What it is
----------

**Stratified Group K-Fold** is a cross-validation scheme that fuses **three requirements** at
once: **k-fold** splitting, **stratification** (preserve the class balance in every fold), and
**grouping** (keep every group — same patient, user, session — entirely on one side of each
split). It is the right tool for **grouped *and* imbalanced** classification.

Why it's needed
---------------

Each simpler scheme covers only part of the problem. **Stratified k-fold** balances classes
but can let one group's rows fall into both train and validation, **leaking** information.
**Group k-fold** prevents that overlap but can wreck the class balance. **Stratified group
k-fold** does both — class proportions held *and* group boundaries respected.

How it works and an example
---------------------------

Identify the group key, then build folds that are simultaneously class-balanced and
group-clean. For 1,000 samples from **100 patients** with a 20/80 disease split and ``k = 5``,
each fold holds about 20 patients, preserves the ~20/80 ratio, and shares **no patient**
between train and validation.

In scikit-learn
---------------

.. code-block:: python

   from sklearn.model_selection import StratifiedGroupKFold

   cv = StratifiedGroupKFold(n_splits=5)
   for train_idx, test_idx in cv.split(X, y, groups):  # groups = patient IDs
       ...

The comparison is clean: plain **k-fold** is neither stratified nor group-aware, **stratified
k-fold** adds class balance, **group k-fold** adds group safety, and **stratified group
k-fold** is the only one with both.
"""

CONTENT["Stratified Shuffle Split"] = r"""
What it is
----------

**Stratified Shuffle Split** repeatedly carves a dataset into **random train/test splits
while preserving the class distribution**. Unlike k-fold, it does **not** partition into fixed
folds — it **reshuffles and resamples** as many times as you ask, each split a fresh random
draw with the original class ratios intact.

How it works
------------

Set **n_splits** (how many reshuffles) and a **train/test size**; for each split, shuffle,
partition keeping the class proportions, and evaluate — then average across splits. Because
test sets can overlap between splits (they are independent draws), it is not a partition the
way k-fold is.

Example
-------

For 1,000 samples at **80% class A, 20% class B** with ``test_size=0.2`` over 5 splits, each
split yields train = 800 (A=640, B=160) and test = 200 (A=160, B=40) — the **80/20** ratio
holds every time.

In scikit-learn
---------------

.. code-block:: python

   from sklearn.model_selection import StratifiedShuffleSplit

   sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
   for train_idx, test_idx in sss.split(X, y):
       ...

It shines on **imbalanced or small** data where you want **many randomized splits** rather
than a fixed fold structure — the stratified counterpart to a plain shuffle split, which
randomizes but does *not* preserve class balance.
"""

CONTENT["Multiclass stratified CV"] = r"""
What it is
----------

**Multiclass stratified CV** is stratified k-fold extended **beyond two classes**: every fold
keeps approximately the **same distribution across all classes** as the full dataset. It is
the natural generalisation of binary stratification to **three or more** labels.

How it works
------------

Measure the overall class mix — say **A = 60%, B = 30%, C = 10%** — and build each fold to
mirror it, so every fold carries A, B and C in roughly those proportions. Both training and
validation sets then **represent all classes**.

Example
-------

For 1,000 samples split **A = 600, B = 300, C = 100** with ``k = 5``, a **regular k-fold**
might leave some folds with almost no class-C examples. **Multiclass stratified k-fold** gives
each fold about **A = 120, B = 60, C = 20** — the original shape, fold after fold.

Why it matters
--------------

In **imbalanced multiclass** problems, plain k-fold can starve a minority class in some folds,
producing **unstable, misleading metrics** (a fold with no class-C samples cannot measure
class-C performance). Stratification makes the evaluation **fair and stable**. It applies to
**classification only** — there is nothing to stratify in a regression target.
"""

MINDMAP.update({
    "Stratified Group K-Fold": [
        "Stratified Shuffle Split", "Multiclass stratified CV", "k-fold cross-validation",
        "Cross-Validation (CV)", "Data Leakage", "Class Weighting",
    ],
    "Stratified Shuffle Split": [
        "Stratified Group K-Fold", "Multiclass stratified CV", "k-fold cross-validation",
        "Cross-Validation (CV)", "Class Weighting",
        "SMOTE (Synthetic Minority Over-sampling Technique)",
    ],
    "Multiclass stratified CV": [
        "Stratified Group K-Fold", "Stratified Shuffle Split", "k-fold cross-validation",
        "Cross-Validation (CV)", "Class Weighting", "Multiclass AUROC",
    ],
})


# ----------------------------------------------------------------------
# Theme: cross-validation foundation & post-hoc re-scoring  (validation / calibration)
# ----------------------------------------------------------------------

CONTENT["k-fold cross-validation"] = r"""
What it is
----------

**k-fold cross-validation** splits the data into **k roughly equal folds** and trains and
tests the model **k times**, each run holding out a **different fold** as the test set and
training on the other :math:`k-1`. Averaging the k scores gives a lower-variance estimate of
performance than any single train/test split — which is why it is the **default** CV method.

How it works
------------

Shuffle (if order is irrelevant), split into **k folds** (commonly 5 or 10), and for each
fold :math:`i` train on the other folds and test on fold :math:`i`. Collect the k scores and
**average** them for the final metric.

Example
-------

With 1,000 samples and ``k = 5``, each fold is 200 samples: every run trains on **800** and
validates on **200**, rotating which 200 is held out, and the result is the **mean** across
the five runs.

Variations
----------

**Stratified k-fold** preserves class balance per fold (vital for imbalanced data);
**repeated k-fold** re-runs the whole process with new splits for a steadier estimate; and
**leave-one-out (LOOCV)** is the extreme :math:`k = N`, one sample per fold — very accurate,
very expensive.

In scikit-learn, and the trade-offs
-----------------------------------

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(model, X, y, cv=5)
   print(scores.mean(), scores.std())

The gains — a **reliable estimate**, less dependence on one random split, full use of the
data, and a backbone for **hyperparameter tuning** — cost **k model fits**, and plain k-fold
is **wrong for time series**, where time-aware CV is required instead.
"""

CONTENT["Cross-Validation (CV)"] = r"""
What it is
----------

**Cross-validation (CV)** is the umbrella **model-evaluation technique**: split the data into
multiple subsets, rotate which subset is held out for testing, and **average** the results so
the performance estimate does not hinge on a single lucky or unlucky split. Its goal is to
gauge how well a model **generalises to unseen data**, and it underpins **model selection,
hyperparameter tuning and overfitting prevention**.

How it works
------------

Partition into **k folds**; for each, train on the rest and test on the held-out fold; repeat
so every fold serves as test once; average across folds for the final estimate.

The main flavours
-----------------

**k-fold** is the balanced default; **stratified k-fold** holds class proportions steady for
imbalanced classification; **leave-one-out (LOOCV)** uses one sample per fold — accurate but
costly; **time-series CV** (rolling or expanding windows) respects time order for sequential
data; and **nested CV** wraps an inner tuning loop inside an outer evaluation loop so that
**hyperparameter selection does not leak** into the performance estimate.

Why it matters, and the costs
-----------------------------

CV gives a **robust, lower-variance** read on generalisation and is the standard harness for
**grid, random and Bayesian** hyperparameter search. The price is **compute** (the model is
trained many times) and the need to **match the fold type to the data** — most importantly,
never shuffling a time series.
"""

CONTENT["Re-scoring"] = r"""
What it is
----------

**Re-scoring** is **adjusting or recomputing a model's scores** — predictions, probabilities
or rankings — *after* the initial output, by layering on extra information, calibration or
rules. It improves **relevance, fairness or calibration without retraining**, and shows up in
ranking, recommendations, search, fraud detection and NLP pipelines.

Where it's used
---------------

Five common settings. In **search and recommendation**, a base ranking is re-scored with
business rules, diversity constraints or personalisation (boost new items, demote
duplicates). In **classification**, raw probabilities are re-scored by **calibration** (Platt
scaling, isotonic regression, Bayesian correction). In **ensembles**, several models' outputs
are combined — a weighted average of fraud scores, say. For **fairness**, re-scoring enforces
guardrails so no group is systematically disadvantaged. And in **NLP reranking**, an N-best
list from an acoustic or translation model is re-scored by a second model (a language model)
to pick the most fluent candidate.

Examples
--------

A fraud model's **0.7** probability is re-scored against the customer's risk profile to
**0.85**; an e-commerce relevance score gets a **+0.2** promotion boost; a speech system's
N-best hypotheses are re-scored by a language model to choose the most fluent sentence.

Benefits and challenges
-----------------------

It buys **accuracy without full retraining**, **flexibility** (rules, fairness) and
**real-time, context-aware** adjustment. The risks: a complex re-scoring stage adds
**latency**, weighting must be tuned to **avoid introducing bias**, and leaning on it too hard
can **paper over** model weaknesses instead of fixing their root cause.
"""

MINDMAP.update({
    "k-fold cross-validation": [
        "Cross-Validation (CV)", "Stratified Group K-Fold", "Stratified Shuffle Split",
        "Multiclass stratified CV", "Blocked Splits (Single Holdout)", "Data Leakage",
    ],
    "Cross-Validation (CV)": [
        "k-fold cross-validation", "Stratified Group K-Fold", "Blocked Splits (Single Holdout)",
        "Sliding Window (Rolling Window) Cross-Validation", "Expanding Window Cross-Validation",
        "Data Leakage",
    ],
    "Re-scoring": [
        "Platt Scaling", "Isotonic Regression", "Recalibration", "Ranking Algorithms",
        "NDCG (Normalized Discounted Cumulative Gain)", "Demographic Parity (Statistical Parity)",
    ],
})


# ----------------------------------------------------------------------
# Theme: drift monitoring, distillation, early stopping  (drift / mlops / training)
# ----------------------------------------------------------------------

CONTENT["Drift Detection"] = r"""
What it is
----------

**Drift detection** is the practice of **monitoring for changes in the data distribution or
model behaviour over time** that can erode performance. Because real-world data evolves —
seasonality, new users, shifting business conditions — a model that was accurate at launch can
silently decay, and drift detection is what **triggers retraining or recalibration** before
that decay bites.

The three drifts
----------------

**Covariate (feature) drift**: the input distribution moves — 80% desktop users become 60%
mobile. **Label drift** (prior shift): the target mix changes — fraud rises from 1% to 3%.
**Concept drift**: the *relationship* between features and label changes — the same features
no longer predict fraud because the tactics evolved. Only concept drift necessarily breaks the
learned mapping; the others may or may not.

How it's detected
-----------------

Four families. **Statistical tests** compare old and new distributions (Kolmogorov–Smirnov,
chi-square, **PSI** — population stability index). **Distance measures** quantify the gap (KL
and Jensen–Shannon divergence, Wasserstein, MMD). **Classifier two-sample tests** train a
model to tell "old" from "new" — if it succeeds, the distributions differ. And **performance
monitoring** tracks loss or AUC over time, though as a *lagging* signal it only fires once
labels arrive.

In practice
-----------

Set **baselines** from training data, compare **rolling windows** (last 24h vs last 4 weeks),
fire **alerts** past a threshold (e.g. PSI > 0.2), and wire the signal into the pipeline to
**retrain or recalibrate**. A credit model trained on 2022 data might, by 2025, see income
shift toward gig workers — PSI flags the covariate drift and the monitor flags concept drift,
triggering a retrain. The payoff is avoiding **silent degradation**, protecting **fairness**
(drift can hit subgroups unevenly), and meeting **compliance** demands for monitoring.
"""

CONTENT["Model Distillation (Knowledge Distillation)"] = r"""
What it is
----------

**Model distillation** (knowledge distillation) is a **compression** technique: a large,
accurate **teacher** model transfers its knowledge to a smaller, faster **student**. The aim
is a lighter model that **keeps most of the teacher's accuracy** while being cheap enough to
deploy on phones and edge devices.

How it works
------------

Train the big teacher (a large transformer, say), then have it produce **soft targets** — full
probability distributions like ``[0.7 cat, 0.2 dog, 0.1 rabbit]`` rather than a bare hard
label. The student trains to **mimic those soft outputs**, usually under a loss that blends a
**distillation term** (student vs teacher) with a **supervised term** (student vs true labels).

The objective
-------------

.. math::

   L = \alpha \, L_{\text{hard}}(y, p_s) \; + \; (1 - \alpha)\, L_{\text{soft}}(p_t, p_s, T),

where :math:`y` are true labels, :math:`p_t` and :math:`p_s` the teacher and student
probabilities, :math:`T` a **temperature** that softens the teacher's distribution to expose
its "dark knowledge", and :math:`\alpha` the balance between the two terms.

Why, examples, and costs
------------------------

It buys **efficiency** (faster, cheaper inference), **deployability** (edge devices), and even
better generalisation from the teacher's soft probabilities. **DistilBERT** is ~40% smaller and
~60% faster than BERT at ~97% of its performance; **ResNet-50** distills into ResNet-18 at
similar accuracy. The catches: the student **cannot capture everything**, :math:`T` and
:math:`\alpha` need tuning, and you still pay to **train the teacher** first.
"""

CONTENT["Early Stopping"] = r"""
What it is
----------

**Early stopping** is a **regularisation** technique: stop training **before the model
overfits**. Rather than running a fixed number of epochs, you watch a **validation metric** and
halt once it stops improving — capturing the model at its generalisation "sweet spot" and
saving the wasted epochs beyond it.

How it works
------------

Hold out a validation set, train across epochs, and after each one **score the validation
metric** and remember the **best** so far. If it fails to improve for **N consecutive epochs** —
the **patience** — stop. The key parameters are the **monitored metric**, the **mode**
(``min`` for loss, ``max`` for accuracy or AUC), the **patience**, and **restore-best-weights**,
which rolls the model back to its best epoch.

Example
-------

Capped at 100 epochs, suppose validation loss improves until **epoch 25** and then climbs as
the model starts overfitting. With **patience = 3**, training stops at **epoch 28** and restores
the weights from epoch 25 — the genuine best.

Benefits and drawbacks
----------------------

It **prevents overfitting**, **cuts training time**, and finds a near-optimal stopping point
automatically. The costs are minor: it **needs a validation set**, and a noisy metric can trip
it too soon — which is exactly what **patience** is there to absorb.
"""

MINDMAP.update({
    "Drift Detection": [
        "Data Drift", "Concept Drift", "Covariate Drift (a.k.a. Covariate Shift)",
        "PSI (Population Stability Index)", "Recalibration", "Re-scoring",
    ],
    "Model Distillation (Knowledge Distillation)": [
        "Quantization", "Frozen Encoder", "Embedding", "Autoencoder", "Early Stopping",
        "Epochs",
    ],
    "Early Stopping": [
        "Epochs", "Cross-Validation (CV)", "Model Distillation (Knowledge Distillation)",
        "Frozen Encoder", "Autoencoder", "Quantization",
    ],
})


# ----------------------------------------------------------------------
# Theme: training mechanics & the AI umbrella  (training / concepts)
# ----------------------------------------------------------------------

CONTENT["Epochs"] = r"""
What it is
----------

An **epoch** is **one complete pass of the entire training set through the model**. In each
epoch the model sees every training sample once — in **mini-batches**, not all at once — and
across many epochs it refines its weights through repeated exposure.

Epoch vs batch vs iteration
---------------------------

Three terms that are easy to conflate. A **batch** is a subset of the data processed together;
an **iteration** is one update step (a forward and backward pass on one batch); and an
**epoch** is one full cycle through *all* batches. So with **10,000 samples** and a **batch
size of 100**, it takes **100 iterations to complete 1 epoch**.

Too few, too many
-----------------

Epoch count trades **underfitting against overfitting**. Too few and the model **hasn't
learned enough**; too many and it begins to **memorise** the training data and generalises
worse. Loss falls with more epochs only **up to a point**.

The training loop and an example
--------------------------------

The loop is: initialise weights; for each epoch, run forward/backward passes over the
mini-batches and update, then evaluate on a validation set; stop when validation stops
improving (**early stopping**). Training an image classifier on **CIFAR-10** for 20 epochs,
the model sees all 50,000 images each epoch and, by epoch 20, the loss has stabilised. Since
the right count is data-dependent — roughly **50–200** for small sets, **5–30** with early
stopping for large ones — the **epoch count is itself a hyperparameter**.
"""

CONTENT["Hyperparameter"] = r"""
What it is
----------

A **hyperparameter** is a setting **chosen before training** that controls **how a model
learns or how it is structured** — as opposed to **parameters** (weights and biases), which the
optimiser **learns from the data**. Hyperparameters are the model's *learning strategy*;
parameters are its *learned knowledge*.

The main kinds
--------------

They span four areas. **Model structure**: number of layers, neurons per layer, a tree's max
depth. **Training process**: learning rate, batch size, number of epochs, dropout rate.
**Regularisation**: L1/L2 penalty strength, weight decay. **Optimisation**: SGD momentum, the
Adam beta values.

Why they matter
---------------

The same model with different hyperparameters can perform **very differently** — poor choices
cause underfitting, overfitting or painfully slow training, so **tuning is critical** for
accuracy and generalisation.

Tuning, and an example
----------------------

Common strategies escalate in sophistication: **manual** trial, **grid search** (every
combination), **random search** (sample combinations — often more efficient), **Bayesian
optimisation** (a probabilistic model guides the search), and **Hyperband / population-based**
methods at scale. Training a net on MNIST one might fix learning rate ``0.001``, batch size
``64``, dropout ``0.5`` and ``20`` epochs; the weights are learned automatically, but
performance hinges on those chosen hyperparameters.
"""

CONTENT["AI (Artificial Intelligence)"] = r"""
What it is
----------

**Artificial intelligence (AI)** is the broad field of computer science aimed at building
systems that **perform tasks that normally require human intelligence** — reasoning, learning,
problem-solving, perception, language and decision-making. It is the **umbrella term** beneath
which Machine Learning, Deep Learning and other methods sit.

Core capabilities
-----------------

AI systems aim at a handful of capabilities: **learning** from data or experience (ML, DL);
**reasoning** to logical decisions (symbolic AI, expert systems); **perception** of the world
(vision, speech); **language understanding** (NLP, large language models); and
**action/autonomy** (robotics, self-driving cars, agents).

Narrow, general, super
----------------------

Today's systems are **narrow (weak) AI** — focused on one task (spam filters, recommenders,
voice assistants, chatbots). **General AI (AGI)** — human-level competence across arbitrary
tasks — remains **hypothetical and unachieved**, and **superintelligence**, surpassing humans
in every domain, is a theoretical further step.

AI vs ML vs DL, and the trade-offs
----------------------------------

The nesting is the key relationship: **AI** is the broad goal, **machine learning** is the
subset that learns from data, and **deep learning** is the subset of ML built on neural
networks for complex perception and language. The field automates tasks, augments
decision-making and unlocks insight from large data — across healthcare, finance, transport
and generative tools — but carries real challenges: **bias and fairness**, the
**interpretability** of black-box models, the **energy and cost** of training, and **safety,
regulation and ethics**.
"""

MINDMAP.update({
    "Epochs": [
        "Hyperparameter", "Early Stopping", "Cross-Validation (CV)",
        "Model Distillation (Knowledge Distillation)", "Neural Networks", "Quantization",
    ],
    "Hyperparameter": [
        "Epochs", "Cross-Validation (CV)", "Early Stopping", "Neural Networks",
        "Model Distillation (Knowledge Distillation)", "Quantization",
    ],
    "AI (Artificial Intelligence)": [
        "Machine Learning (ML)", "Neural Networks", "Bayesian Neural Networks (BNNs)",
        "Variational Inference (VI)", "Model Distillation (Knowledge Distillation)",
        "Quantization",
    ],
})


# ----------------------------------------------------------------------
# Theme: machine learning & two regulated-domain applications  (concepts / applied)
# ----------------------------------------------------------------------

CONTENT["Machine Learning (ML)"] = r"""
What it is
----------

**Machine learning (ML)** is the branch of AI in which computers **learn patterns from data**
rather than following hand-written rules. You supply **examples**, the model **learns the
relationship between inputs (features) and outputs (labels)**, and once trained it
**predicts** on new, unseen data.

The core idea
-------------

Formally, ML fits a function

.. math::

   y = f(X) + \varepsilon,

where :math:`X` are the input features, :math:`y` the output, :math:`f` the function learned
from data, and :math:`\varepsilon` irreducible noise. Learning means estimating :math:`f`.

The kinds of learning
---------------------

**Supervised** learning uses labelled data — **regression** for continuous targets (house
price), **classification** for categories (spam or not). **Unsupervised** learning works on
unlabelled data — **clustering** (customer segmentation) and **dimensionality reduction** (PCA,
embeddings). **Semi-supervised** mixes a little labelled with much unlabelled data (costly
medical labels). **Reinforcement learning** has an agent learn from rewards by acting in an
environment. And **self-supervised** learning predicts part of the input from the rest (masked
words) — the engine behind modern LLMs.

Workflow and an example
-----------------------

The lifecycle is collect → clean → choose a model → train → evaluate → deploy → **monitor and
retrain**. Train a model on thousands of houses — 1,000 sqft and 3 rooms sold for ``$250,000`` —
and it learns that price rises with size and rooms, so a new 1,200 sqft, 4-room house is
predicted at roughly ``$300,000``. ML matters because it **automates pattern discovery** at a
scale and complexity beyond hand-coded rules.
"""

CONTENT["Medical AI"] = r"""
What it is
----------

**Medical AI** is the application of AI techniques — machine learning, deep learning, NLP,
computer vision — to **healthcare and medicine**: diagnosis, treatment, drug discovery, patient
monitoring and operational efficiency.

Where it's applied
------------------

The reach is broad. **Imaging and diagnostics** detect disease in X-rays, CT, MRI and
ultrasound (lung nodules, diabetic retinopathy, fractures). **Clinical decision support**
suggests treatments and dosages. **Predictive analytics** flag readmission or sepsis risk in
the ICU. **Drug discovery** tackles protein folding and molecule generation — AlphaFold being
the landmark. **NLP** mines electronic health records and powers triage chatbots.
**Personalised medicine** fuses genomics with patient data, and **wearables** catch irregular
heart rhythms or track glucose continuously.

The benefits
------------

Done well, it brings **faster, more accurate diagnosis**, **less repetitive workload** for
clinicians, **earlier detection** and better outcomes, and **lower cost** through efficiency.

The hard parts
--------------

The obstacles are as real as the promise: **data privacy** (HIPAA, GDPR), **bias and fairness**
when training data underrepresents groups, **interpretability** of black-box models that
clinicians must trust, **regulatory approval** (FDA, EMA), and **integration** into clinical
workflows without disrupting care. Deployed systems like FDA-cleared diabetic-retinopathy
screening show the path through these constraints.
"""

CONTENT["KYC"] = r"""
What it is
----------

**KYC — Know Your Customer** (or Client) — is the **regulatory process** by which banks,
financial institutions and fintechs **verify a customer's identity**. Its purpose is to prevent
**fraud, money laundering (AML), terrorism financing and identity theft**.

The three steps
---------------

**Customer identification**: collect official documents (passport, licence, ID, utility bills)
and verify name, address and date of birth. **Customer due diligence (CDD)**: assess the
customer's **risk level** — basic checks for most, **enhanced** due diligence for high-risk
cases like large transactions or politically exposed persons. **Ongoing monitoring**: watch
transactions for suspicious activity and periodically **re-verify** (re-KYC).

Where it shows up
-----------------

Opening a **bank account** needs ID plus proof of address; a **crypto exchange** asks for a
photo ID and a selfie before trading; a **loan application** verifies income documents and ID
to cut fraud risk. KYC is **required by AML law**, shields institutions from fraud and fines,
and builds trust with customers and regulators.

The technology
--------------

Modern KYC leans on automation: **OCR** to read documents, **facial recognition** to match a
selfie to an ID photo, **database checks** against sanction and politically-exposed-person
lists, and **AI/ML models** to flag forged IDs and suspicious transaction patterns — the same
fraud-detection machinery used across finance.
"""

MINDMAP.update({
    "Machine Learning (ML)": [
        "AI (Artificial Intelligence)", "Neural Networks", "Customer Segmentation",
        "Embedding", "Regression Coefficient", "Medical AI",
    ],
    "Medical AI": [
        "AI (Artificial Intelligence)", "Machine Learning (ML)",
        "Demographic Parity (Statistical Parity)", "Counterfactual Explanations",
        "Drift Detection", "Neural Networks",
    ],
    "KYC": [
        "AI (Artificial Intelligence)", "Machine Learning (ML)", "Drift Detection",
        "Demographic Parity (Statistical Parity)", "SaaS (Software as a Service)",
        "Counterfactual Explanations",
    ],
})


# ----------------------------------------------------------------------
# Theme: workforce metric & managed ML platforms  (growth / platforms)
# ----------------------------------------------------------------------

CONTENT["FTEs"] = r"""
What it is
----------

An **FTE — full-time equivalent** — standardises workforce size by converting **total hours
worked into a number of full-time employees**. **1.0 FTE** is one person working full-time
(commonly 40 hours/week in the US, though it varies by company and country), so the measure
lets part-timers and contractors be compared on one scale.

Why it's used
-------------

Three jobs. **Headcount planning** measures real staffing even when many work part-time;
**budgeting** computes labour cost consistently; and **project management** estimates the
resources available to deliver.

Examples
--------

The arithmetic is hours-based: one full-timer at 40h/week is **1.0 FTE**, and **two
part-timers** at 20h each also sum to **1.0 FTE**. A project needing **5 FTEs** for six months
could be five full-time staff, ten half-time staff, or any mix totalling the same hours. On
cost, if one FTE earns ``$100,000``/year, then 3.5 FTEs run about ``$350,000``/year.

FTE vs headcount
----------------

The two answer different questions. **Headcount** is the raw number of people; **FTE** is that
number **weighted by hours**. Ten employees each working half-time are a **headcount of 10**
but an **FTE of 5** — which is why FTE, not headcount, is the fair basis for comparing
workloads and costs.
"""

CONTENT["AWS SageMaker"] = r"""
What it is
----------

**Amazon SageMaker** is AWS's **end-to-end, fully managed ML platform** for **building,
training, deploying and monitoring** models at scale. It is the AWS answer to Google's Vertex
AI and Microsoft's Azure ML.

The pipeline it covers
----------------------

SageMaker spans the whole lifecycle. **Data prep** with Data Wrangler and a Feature Store
(over S3, Redshift, Athena). **Training** for any major framework (TensorFlow, PyTorch,
scikit-learn, XGBoost) on managed CPU/GPU instances, with distributed training and **automatic
hyperparameter tuning**. **Deployment** as real-time, asynchronous or batch endpoints, with
autoscaling. **MLOps** via Pipelines (CI/CD), Model Monitor (drift, bias, quality), Clarify
(bias and explainability) and Debugger. And **generative AI** through JumpStart's pre-trained
models.

Workflow, benefits, costs
-------------------------

The flow is: prepare data in S3 → train on managed infrastructure → auto-tune → deploy to an
endpoint → monitor and retrain. The upside is **scale, framework flexibility** (any Docker
container), deep **AWS integration** and a full **MLOps** toolkit. The downside is
**complexity** — many services and a steep curve — and **cost** that climbs fast if endpoints
run 24/7.

Versus the alternatives
-----------------------

Against **Vertex AI**, the two are close peers — both full end-to-end platforms, differing
mainly by cloud ecosystem (AWS vs GCP). Against the **OpenAI API**, the trade is control for
simplicity: SageMaker trains and serves *any* custom model, while an inference-only API is
faster to start but limited to the provider's models.
"""

CONTENT["Vertex AI"] = r"""
What it is
----------

**Vertex AI** is **Google Cloud's managed ML platform** for building, training, deploying and
monitoring models end-to-end, unifying **data, training, inference and monitoring** in one
workflow. It is the GCP counterpart to AWS SageMaker and Azure ML.

The pipeline it covers
----------------------

**Training** for custom models (TensorFlow, PyTorch, scikit-learn, XGBoost) with distributed
runs across CPUs, GPUs and **TPUs**, plus built-in tuning. **Deployment** as real-time or batch
endpoints over Google Cloud Storage, with autoscaling. **Foundation models** — Google's PaLM,
Gemini, Imagen and Chirp — accessible through **Vertex AI Studio** without training from
scratch. **MLOps** with monitoring (drift, bias, feature skew), pipelines, **explainable AI**
and metadata versioning. And a **Feature Store** integrated with BigQuery and Dataflow.

Workflow, benefits, costs
-------------------------

The flow runs ingest (BigQuery, GCS) → prepare features → train (custom or **AutoML**) →
deploy → monitor for drift, performance and fairness. Its strengths are being **fully
managed**, **first-class access to Google's foundation models**, and tight **GCP integration**.
The costs mirror SageMaker's: it can be **expensive at scale** and brings **vendor lock-in** to
GCP.

Versus SageMaker
----------------

The two are near-mirror images — both end-to-end, both supporting any framework and full
MLOps. The real differentiators are **ecosystem** (BigQuery/GCS vs S3/Redshift) and
**foundation-model access** (Gemini and friends native to Vertex). The choice usually follows
whichever cloud an organisation already lives in.
"""

MINDMAP.update({
    "FTEs": [
        "SaaS (Software as a Service)", "Gross Margin",
        "Blended CAC (Customer Acquisition Cost)", "Customer Segmentation", "KYC",
        "D2C (Direct-to-Consumer)",
    ],
    "AWS SageMaker": [
        "Vertex AI", "OpenAI API (ML API)", "Online Experimentation Platforms",
        "Drift Detection", "Hyperparameter", "ONNX (Open Neural Network Exchange)",
    ],
    "Vertex AI": [
        "AWS SageMaker", "OpenAI API (ML API)", "Online Experimentation Platforms",
        "Drift Detection", "Hyperparameter", "Quantization",
    ],
})


# ----------------------------------------------------------------------
# Theme: inference APIs, managed endpoints, big-payload serving  (platforms / ops)
# ----------------------------------------------------------------------

CONTENT["OpenAI API (ML API)"] = r"""
What it is
----------

The **OpenAI API** is a cloud **machine-learning API**: it exposes OpenAI's models — LLMs,
embeddings, fine-tuned variants — through simple **HTTP endpoints**. Rather than training and
hosting a model yourself, you **send a request, the cloud runs inference, and JSON comes
back** — inference-as-a-service.

What it offers
--------------

The surface is broad. **Chat and text generation** (GPT-family models) for Q&A, summarisation
and reasoning. An **embeddings** endpoint turning text into vectors for semantic search,
clustering and **RAG**. A **fine-tuning** endpoint to specialise a model on your data.
**Moderation** for unsafe content, **vision/multimodal** for images alongside text, and
**speech** both ways (text-to-speech and Whisper transcription). Plus **batch/async**
processing for volume.

Calling it
----------

You POST to a REST endpoint (or use an SDK) and read the response:

.. code-block:: python

   from openai import OpenAI
   client = OpenAI()

   response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": "Explain Bayesian inference simply."}],
   )

Benefits and limits
-------------------

The appeal is **no training or hosting**, **automatic scale**, and **simple integration** that
improves as OpenAI ships new models. The constraints: it is **cloud-only** (needs internet),
adds **network latency**, **costs per token** of input and output, and raises **privacy**
questions — sensitive data must clear HIPAA/GDPR review before it leaves your systems.
"""

CONTENT["AWS SageMaker Endpoints"] = r"""
What it is
----------

An **AWS SageMaker endpoint** is a **fully managed API** for serving a trained model: you
deploy the model and SageMaker handles **infrastructure, scaling and serving** so there are no
servers to run. Input goes in, the endpoint runs inference, predictions come out.

The three types
---------------

**Real-time** endpoints are always-on HTTPS with low latency and autoscaling — for fraud
checks or chatbots. **Asynchronous** endpoints handle **large payloads or long jobs**: upload
to S3, the endpoint processes, results land back in S3 — good for video or big documents.
**Batch transform** runs **offline** over a whole S3 dataset with no live endpoint — ideal for
scheduled jobs like scoring every customer monthly.

Deploying one
-------------

The path is: train or **import** a model (any framework, or ONNX) and save artifacts to S3;
**register** it as a Model with its inference script; create an **endpoint configuration**
(instance type, scaling); deploy; then **invoke** via the SDK or REST:

.. code-block:: python

   import boto3
   sm = boto3.client("sagemaker-runtime")
   response = sm.invoke_endpoint(
       EndpointName="my-endpoint",
       ContentType="application/json",
       Body='{"data":[1.0, 2.0, 3.0]}',
   )

Why use them
------------

They are **fully managed** (no EC2 to babysit), **autoscaling** for traffic spikes,
**framework-flexible** (custom Docker too), **secure** (IAM, VPC, encryption), and **cost
controllable** — pay for instance time, or shift to async/batch to save when latency is not
critical.
"""

CONTENT["Cloud Inference with Big Payloads"] = r"""
What it is
----------

In **cloud inference**, the **payload** is the data you send the model. **Small payloads** are
tabular rows or short prompts; **big payloads** are large images, long videos, audio, sprawling
JSON feature sets or massive documents — and their size changes everything about how you serve
the request.

The challenges
--------------

Five bite. **Network latency and bandwidth**: uploading a 500MB video dwarfs a 1KB JSON call.
**Serialisation and transfer costs**: JSON/Protobuf/gRPC overhead grows with size, and many
APIs **cap request size**. **Compute cost**: more data means more FLOPs and higher OpEx — a 4K
frame sequence costs far more than one image. **Timeouts and reliability**: big uploads time
out more and are **harder to retry**. And **privacy**: sensitive payloads (medical scans) need
encryption and compliance checks.

The mitigations
---------------

Five answers. **Compress client-side** — resize images, sample video frames, extract audio
features (MFCCs) before sending. **Stream in chunks** (gRPC, WebSocket) for live video or
speech. **Go asynchronous** — upload to S3/GCS and send only a **file reference**, then poll or
get a callback. **Split edge and cloud** — run a feature extractor on-device and send small
**embeddings** rather than raw input. And **optimise the format** — binary serialisation
(Protobuf, Avro, Arrow) over raw JSON, batching small requests together.

Examples
--------

A 200MB **CT scan** goes to cloud storage, with only its reference passed to an async pipeline.
**Video analytics** extracts one frame per second locally instead of streaming full HD. And a
200-page document for an **LLM** is chunked into sections, processed sequentially or through
**retrieval (RAG)** rather than sent whole.
"""

MINDMAP.update({
    "OpenAI API (ML API)": [
        "AWS SageMaker", "Vertex AI", "AWS SageMaker Endpoints", "Embedding",
        "Cloud Inference with Big Payloads", "Online Experimentation Platforms",
    ],
    "AWS SageMaker Endpoints": [
        "AWS SageMaker", "Cloud Inference with Big Payloads", "OpenAI API (ML API)",
        "ONNX (Open Neural Network Exchange)", "Drift Detection", "Vertex AI",
    ],
    "Cloud Inference with Big Payloads": [
        "AWS SageMaker Endpoints", "OpenAI API (ML API)", "Embedding", "Quantization",
        "Model Distillation (Knowledge Distillation)", "Cloud Inference",
    ],
})


# ----------------------------------------------------------------------
# Theme: cloud inference, ensembles, model weights  (ops / training)
# ----------------------------------------------------------------------

CONTENT["Cloud Inference"] = r"""
What it is
----------

**Cloud inference** is running a **trained model on cloud infrastructure** to make predictions
on new data. "Inference" means *using* a trained model rather than training one, and "cloud"
means it runs on AWS, GCP or Azure rather than on local or on-device hardware.

How it works
------------

Three stages. **Train** a model (locally or in the cloud) and save its weights. **Deploy** it to
a cloud service (SageMaker, Vertex AI, Azure ML) behind an **API endpoint** (REST or gRPC).
Then **infer**: the client sends input, the service loads the model, runs a forward pass and
returns the prediction.

Benefits and challenges
-----------------------

The upsides are **scalability** (autoscale to millions of requests), **low maintenance** (the
provider runs servers and GPUs), **flexibility** (serve and version many models), and
**accessibility** (any device can call the API). The costs are **latency** (a network round
trip), **OpEx** per prediction at scale, **privacy** obligations when sensitive data leaves the
device, and the need for **monitoring** to keep uptime and fairness in check.

Cloud vs edge
-------------

The contrast is with **edge / on-device** inference. Cloud runs on powerful servers (GPUs,
TPUs) and is easy to update but pays a network-latency cost; edge runs locally on a phone or
IoT device with **low latency** but limited compute and harder remote updates. Image-tagging
APIs, LLM chat services and fraud-scoring endpoints are all cloud inference.
"""

CONTENT["Ensemble"] = r"""
What it is
----------

An **ensemble** combines **multiple models into one stronger predictor**. The intuition is the
**wisdom of the crowd** — a group of diverse or individually "weak" models, averaged together,
is usually more accurate and more stable than any one of them.

Why use one
-----------

Ensembles **reduce variance** (steadier, less noise-sensitive predictions), can **reduce bias**
(capturing more complex patterns), and **improve robustness** (one model's errors offset by
others). They routinely win competitions like Kaggle for exactly these reasons.

The five strategies
-------------------

**Bagging** (bootstrap aggregating) trains models on different random samples and averages or
votes — **random forest** is bagged decision trees. **Boosting** trains models
**sequentially**, each fixing the last one's errors (AdaBoost, gradient boosting, XGBoost,
LightGBM, CatBoost). **Stacking** trains base models and a **meta-learner** to combine them.
**Voting** takes a majority (hard) or averages probabilities (soft). **Blending** is stacking
with the meta-learner trained on a **holdout set** rather than CV folds.

Trade-offs
----------

The wins — **higher accuracy**, **resistance to overfitting**, complementary patterns from
different algorithms — come at a price: **more compute**, **harder interpretability** than a
single model, and **deployment complexity** (several models mean more latency and OpEx).
"""

CONTENT["Model Weights"] = r"""
What it is
----------

**Model weights** are a model's **trainable parameters** — the numbers that determine how input
features are turned into predictions. Each weight encodes the **importance** of a feature, and
training **adjusts** them to minimise loss.

By model type
-------------

In **linear regression**, :math:`y = w_1 x_1 + w_2 x_2 + b`, the weights :math:`w_1, w_2` say
how much each feature contributes — if :math:`w_1 = 200`, every extra square foot adds ``$200``
to the predicted price. In a **neural network**, every connection between neurons has a weight;
the forward pass multiplies inputs by weights and applies an activation, and CNN filter weights
learn patterns like edges. In **logistic regression**, weights are the **log-odds**
contribution — positive pushes toward the positive class, negative away.

How they're learned
-------------------

Weights start **random** (or from a heuristic), then a loop refines them: **forward pass** to
predict, a **loss** against the truth, **backpropagation** for the gradients of loss with
respect to each weight, and an **optimiser** (SGD, Adam) to update them — repeated until the
loss settles.

Why they matter
---------------

Weights **are** the model's learned knowledge: saving or loading a model *is* saving or loading
its weights. In transfer learning we often **freeze** the encoder's weights and fine-tune only
the last layers. Concretely, a spam classifier might learn a weight of **+2.5** for "free"
(strongly spammy) and **-1.0** for "invoice" (less so).
"""

MINDMAP.update({
    "Cloud Inference": [
        "Cloud Inference with Big Payloads", "AWS SageMaker Endpoints", "OpenAI API (ML API)",
        "AWS SageMaker", "Vertex AI", "Quantization",
    ],
    "Ensemble": [
        "Re-scoring", "Cross-Validation (CV)", "Model Distillation (Knowledge Distillation)",
        "Cloud Inference", "Bayesian Neural Networks (BNNs)", "Model Weights",
    ],
    "Model Weights": [
        "Hyperparameter", "Frozen Encoder", "Regression Coefficient", "Epochs",
        "Neural Networks", "FLOPs",
    ],
})


# ----------------------------------------------------------------------
# Theme: compute, cost, and large language models  (compute / cost / concepts)
# ----------------------------------------------------------------------

CONTENT["FLOPs"] = r"""
What it is
----------

**FLOPs — floating-point operations** — count the **mathematical operations** (adds, multiplies,
divides) a model or algorithm performs. In ML, FLOPs are a standard **proxy for computational
complexity** and for training/inference **cost** — they measure *workload*, not memory access or
I/O.

The units
---------

One operation is a **FLOP**, and the scale climbs fast: **MFLOPs** (millions), **GFLOPs**
(billions), **TFLOPs** (trillions), **PFLOPs** (quadrillions). A single large matrix
multiplication already involves enormous counts.

Counting them
-------------

Multiplying an :math:`m \times n` matrix by an :math:`n \times p` matrix costs about
:math:`2 \, m \, n \, p` FLOPs — so two :math:`1000 \times 1000` matrices run to roughly **2
billion**. In deep learning, **training** FLOPs scale with dataset size × model size × epochs,
while **inference** FLOPs per forward pass drive deployment latency. For comparison, ResNet-50 is
~4 GFLOPs per image, BERT-base ~22 GFLOPs per sequence, and GPT-3 took an estimated
:math:`3 \times 10^{23}` FLOPs to train.

FLOPs vs FLOPS
--------------

Two letters, two meanings. **FLOPs** is a **count** of operations (the workload); **FLOPS** is
**floating-point operations per second**, the **speed** of hardware — an NVIDIA A100 delivers
~312 TFLOPS on its tensor cores. FLOPs is the job; FLOPS is how fast the worker does it. Because
training cost scales almost linearly with FLOPs, the count is a key lever in the
efficiency-versus-accuracy trade-off.
"""

CONTENT["OpEx"] = r"""
What it is
----------

**OpEx — operating expenses** — are the **ongoing, day-to-day costs** of running a business or
system. They contrast with **CapEx (capital expenditures)**, the one-time, long-term investments
like buying servers or buildings; OpEx is the **recurring** spend — salaries, rent, cloud usage.

What counts as OpEx
-------------------

In a business: salaries, rent, utilities, marketing, maintenance. In a **tech/ML** setting:
**cloud compute** (AWS, GCP, Azure), **data storage** (S3, BigQuery, Snowflake), **inference
costs** (GPU time per prediction), **third-party API** fees, and **monitoring/logging**
subscriptions.

CapEx vs OpEx
-------------

The split is timing and accounting. **CapEx** is a **one-time** investment in a long-lived asset,
**capitalised and depreciated** over years. **OpEx** is a **recurring** operating cost,
**expensed immediately** on the income statement. Buying ``$500k`` of servers is CapEx; paying
``$50k`` a month for cloud, salaries and electricity is OpEx.

Why it matters
--------------

OpEx is **subtracted from revenue** to get operating profit, so businesses track it closely to
**control recurring costs**. In ML specifically, **reducing OpEx** — more efficient models that
burn fewer GPU hours — is what makes a deployment **sustainable** at scale, tying it directly to
FLOPs and inference cost.
"""

CONTENT["LLMs (Large Language Models)"] = r"""
What it is
----------

**LLMs — large language models** — are ML models trained on **massive text corpora** to
understand and generate human language. Built on **transformer** architectures, they handle a
wide range of NLP tasks **without task-specific training**, via zero-shot and few-shot prompting.

How they work
-------------

The architecture is the **transformer** with **self-attention**, which lets the model weigh every
word in a sequence for context. Training runs over **billions to trillions of tokens** (books,
articles, code, the web) with a deceptively simple objective: **predict the next token**. From
that single objective emerge text generation, summarisation, translation, reasoning and code
generation.

The landscape and uses
----------------------

The major families include OpenAI's **GPT**, Google's **Gemini**, Meta's **LLaMA**, Anthropic's
**Claude** and Mistral's **Mixtral**. They power conversational assistants, content creation,
**code copilots**, semantic search and **retrieval-augmented generation (RAG)**, and
data-science tasks like query-to-SQL.

The limitations
---------------

The caveats are serious: **hallucinations** (fluent but wrong answers), **bias** inherited from
training data, **cost and energy** (very high OpEx to train and serve), **stale knowledge** that
lags real-time events, and limited **explainability**. A useful mental model is a **very powerful
autocomplete** — predict the most likely continuation — that, at sufficient scale, begins to show
reasoning-like behaviour.
"""

MINDMAP.update({
    "FLOPs": [
        "Model Weights", "Quantization", "Model Distillation (Knowledge Distillation)",
        "Cloud Inference", "OpEx", "Neural Networks",
    ],
    "OpEx": [
        "FLOPs", "Gross Margin", "Cloud Inference", "SaaS (Software as a Service)",
        "LLMs (Large Language Models)", "Blended CAC (Customer Acquisition Cost)",
    ],
    "LLMs (Large Language Models)": [
        "AI (Artificial Intelligence)", "Machine Learning (ML)", "Neural Networks", "Embedding",
        "OpenAI API (ML API)", "FLOPs",
    ],
})


# ----------------------------------------------------------------------
# Theme: probability recalibration, reweighting, continuous retraining  (calibration / drift / mlops)
# ----------------------------------------------------------------------

CONTENT["Recalibration"] = r"""
What it is
----------

**Recalibration** adjusts a model's **predicted probabilities** so they match the **true
likelihood** of outcomes. A model may output a 0.9 "probability of fraud", but if that score is
**over- or under-confident**, recalibration corrects the mismatch — without touching the model's
ranking.

Why it's needed
---------------

Many models (SVMs, neural nets, boosted trees) are **poorly calibrated**: their raw outputs are
useful for *ordering* cases but are not true probabilities. In high-stakes domains — medicine,
finance, fraud — decisions need **trustworthy probabilities**, not just a ranking. The target is
simple: of all cases scored **0.7**, about **70%** should actually be positive.

How it's done
-------------

Fit a small **calibration model on a validation set**, comparing predicted scores to true
outcomes. The common methods are **Platt scaling** (a logistic fit on the raw scores),
**isotonic regression** (a non-parametric monotonic mapping), **temperature scaling** (one
parameter on the logits, popular in deep learning), and **Bayesian** recalibration. A
**calibration curve** — predicted probability versus observed frequency — diagnoses the need:
deviation from the diagonal means miscalibration.

Examples and the contrast
-------------------------

A spam filter scores an email **0.9**, but only 70% of such emails are truly spam — Platt scaling
corrects it toward **0.72**. A mortality model predicts **0.2** where the real rate is **30%** —
recalibration nudges it to **0.3**. This is distinct from **threshold tuning**, which moves the
decision cutoff (say 0.5 → 0.3) to trade precision against recall; recalibration changes the
**probabilities themselves**. It works as a **post-processing** step (no retraining), though it
needs a reliable calibration set and may leave ranking metrics like AUC unchanged.
"""

CONTENT["Reweighting"] = r"""
What it is
----------

**Reweighting** assigns **different weights to samples, features or loss terms** so that training
or evaluation reflects the **true importance, fairness or distribution** of the data. It leaves
the raw data untouched and instead changes **how much influence** each part has.

Where it's used
---------------

Several settings. **Class imbalance**: upweight rare positives (fraud, disease) so the model
can't ignore them. **Covariate shift / domain adaptation**: reweight training samples to match
the production distribution (an 80%-desktop training set toward 50%-mobile production).
**Fairness**: upweight underrepresented protected groups for equal contribution. Plus **loss
reweighting** (balancing terms in multi-task or adversarial training) and **importance sampling**
(correcting biased draws for unbiased estimates).

The math
--------

The ordinary average loss

.. math::

   L = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), y_i)

becomes, with weights,

.. math::

   L = \frac{1}{N} \sum_{i=1}^{N} w_i \, \ell(f(x_i), y_i),

where :math:`w_i` is the weight on sample :math:`i` — a larger :math:`w_i` gives that example
more influence.

Examples and trade-offs
-----------------------

On a **1%-fraud** dataset, an unweighted model predicts "not fraud" always; weighting fraud at 99
against 1 makes those cases count and **lifts recall**. A loan model that is 80% male can upweight
female applicants toward fairness. The benefits — handling imbalance and drift, improving
fairness, **keeping data intact** (no over/undersampling) — come with risks: **extreme weights**
can overfit the minority, and choosing weights well takes validation.
"""

CONTENT["Continuous Retraining"] = r"""
What it is
----------

**Continuous retraining** is the practice of **regularly updating a model with new data** to keep
it accurate in production — also called online retraining or model refresh. It exists because
real-world data **changes over time**: distribution shift, new categories, seasonal patterns.

Why it's needed
---------------

Three pressures. **Drift** — feature distributions move (customer behaviour, fraud tactics) and
feature-target relationships evolve. **Business change** — new products, regulations or customer
segments. And **operational resilience** — keeping KPIs (AUC, calibration, accuracy) stable
rather than letting the model **decay** on stale data.

How it works
------------

A monitoring pipeline watches for **drift and KPI degradation**, then a **trigger** fires —
**scheduled** (weekly/monthly refresh) or **event-driven** (drift past a threshold, KPI below
target). The retraining pipeline pulls **new labelled data**, retrains or fine-tunes, **validates
on a fresh holdout**, compares against the current model (A/B or shadow deployment), and
**deploys only if it improves**.

Approaches and trade-offs
-------------------------

**Batch** retraining rebuilds from scratch periodically — simple but resource-heavy.
**Incremental / online** learning updates weights as data streams in. **Hybrid** keeps a frozen
base and fine-tunes on recent data. The payoff is **stability under drift** with less manual work;
the costs are needing **robust MLOps** (validation, reproducibility), **label availability** (no
labels, no retraining), and guarding against **catastrophic forgetting** when old data is dropped.
"""

MINDMAP.update({
    "Recalibration": [
        "Re-scoring", "Platt Scaling", "Isotonic Regression", "Temperature Scaling",
        "Recalibrate Thresholds", "Drift Detection",
    ],
    "Reweighting": [
        "Class Weighting", "Demographic Parity (Statistical Parity)",
        "SMOTE (Synthetic Minority Over-sampling Technique)", "Recalibration",
        "Continuous Retraining", "Covariate Drift (a.k.a. Covariate Shift)",
    ],
    "Continuous Retraining": [
        "Drift Detection", "Monitoring Pipelines", "Data Drift", "Concept Drift",
        "Recalibration", "Reweighting",
    ],
})


# ----------------------------------------------------------------------
# Theme: monitoring, active learning, Bayesian correction  (mlops / training / bayes)
# ----------------------------------------------------------------------

CONTENT["Monitoring Pipelines"] = r"""
What it is
----------

A **monitoring pipeline** is the system of checks and data flows that **continuously tracks the
health and performance** of an ML model in production — a "control tower" whose job is to catch
**drift, degradation, anomalies and failures early**, before they cause silent harm.

What it watches
---------------

Four layers. **Data monitoring**: schema validation, missing values and outliers, **feature
drift** (PSI, KS test, MMD) and representation drift in embeddings. **Model performance**: AUC,
precision, recall, F1 and calibration for classifiers; MSE/RMSE/MAE/R² for regressors; business
metrics like CTR and fraud savings. **Operational**: latency, throughput, uptime, cost per
prediction. And **guardrails**: alerts when thresholds break (drift > 0.2, latency > 200ms),
triggering auto-retrain or rollback.

How it flows
------------

The cycle is **collect** (log predictions, inputs, metadata, eventual outcomes) → **aggregate**
(metrics over daily/weekly windows) → **compare** (against training baselines and SLAs) →
**alert** (flag anomalies and degraded KPIs) → **action** (retrain, adjust thresholds, or
investigate the data). Dashboards slice these signals by geo, device or cohort to separate
leading from lagging indicators.

Why it matters
--------------

A fraud model whose AUC quietly slips from 0.9 to 0.75, with latency spiking past 300ms, fails
**silently** without monitoring. Pipelines prevent that — and underwrite **fairness and
compliance** (no group disproportionately harmed), **accountability** to stakeholders, and the
feedback loop that drives continuous retraining.
"""

CONTENT["Active Learning"] = r"""
What it is
----------

**Active learning** trains a model **iteratively** and lets it **choose the most informative
examples to label**, instead of labelling everything. The goal is **high accuracy from far fewer
labels** — invaluable when annotation is expensive or slow, as with medical images or legal
documents.

The loop
--------

Start with a **small labelled set** and a large **unlabelled pool**. Train an initial model, use
a **query strategy** to pick the most valuable unlabelled samples, send them to an **oracle** (a
human expert) for labels, add them to the training set and retrain — repeating until the model is
good enough or the labelling budget runs out.

Query strategies
----------------

How to choose what to label. **Uncertainty sampling** picks the least confident cases (for binary
classification, predicted probability near 0.5). **Query by committee** trains several models and
picks where they **disagree** most. **Expected model change** chooses points that would most move
the model. **Diversity sampling** picks examples **unlike** the existing training data to cover
the input space.

Why it works
------------

Given 100,000 unlabelled emails at ``$2`` each to label, training on 1,000 and then querying the
500 most uncertain improves accuracy **faster than random labelling**. Active learning cuts
**annotation cost**, accelerates learning, and naturally **prioritises rare or uncertain cases** —
helping with imbalance for free.
"""

CONTENT["Bayesian Correction"] = r"""
What it is
----------

**Bayesian correction** uses **Bayes' theorem** to **adjust raw probabilities, predictions or test
results** when the observed data is biased, noisy or incomplete. It is, in essence, "correcting"
outputs by **Bayesian updating** — folding in what we already know.

Where it appears
----------------

Several places. **Base-rate adjustment**: a model trained as if classes were balanced can be
corrected toward the **true prior** — a 90% rare-disease score shrinks sharply once low prevalence
is accounted for, via :math:`P(y=1 \mid x) \propto P(x \mid y=1)\, P(y=1)`. **Diagnostic tests**:
combining sensitivity and specificity with prevalence gives the true posterior,
:math:`P(\text{disease} \mid +) = \frac{P(+ \mid \text{disease})\, P(\text{disease})}{P(+)}`.
**Label-noise correction** estimates a noise transition matrix (e.g. for crowdsourced labels), and
**Bayesian calibration** updates scores much like Platt scaling but through Bayesian inference.

An example
----------

A spam classifier scores an email **0.8**, but the true spam base rate is only **10%**. After
Bayesian correction with that prior, the calibrated probability might fall to **0.4** — preventing
**overconfidence** when the prior is low.

Why it's useful
---------------

The method handles **class imbalance**, corrects for **measurement noise**, and yields
**better-calibrated probabilities** with **principled uncertainty** — exactly what risk-sensitive
decisions need. It is the Bayesian sibling of recalibration: where Platt scaling fits a curve,
Bayesian correction reasons from priors.
"""

MINDMAP.update({
    "Monitoring Pipelines": [
        "Drift Detection", "Continuous Retraining", "PSI (Population Stability Index)",
        "Data Drift", "Concept Drift", "Re-scoring",
    ],
    "Active Learning": [
        "Reweighting", "Continuous Retraining", "Ensemble", "Medical AI",
        "Monitoring Pipelines", "Bayesian Correction",
    ],
    "Bayesian Correction": [
        "Bayes' Theorem", "Recalibration", "Platt Scaling", "Posterior",
        "Prior Belief (or Prior Probability)", "Recalibrate Thresholds",
    ],
})


# ----------------------------------------------------------------------
# Theme: thresholds, guardrails, model KPIs  (mlops monitoring cluster)
# ----------------------------------------------------------------------

CONTENT["Recalibrate Thresholds"] = r"""
What it is
----------

To **recalibrate thresholds** is to **adjust the decision cutoffs or alert limits** of a model or
monitoring system so they stay valid as data or business needs change. In a **classifier**, that
means moving the probability threshold (the default 0.5) used to split positive from negative; in
**monitoring**, it means tuning alert limits on drift, anomaly rate or latency.

Why it's needed
---------------

Thresholds go stale for four reasons. **Data drift** shifts the distribution so the old cutoff no
longer fits. **Business shift** changes the relative cost of false positives versus false
negatives. **Model updates** alter the calibration curve and thus the optimal cutoff. And
**operational noise** makes metrics fluctuate more or less than before.

Examples
--------

A fraud model defaults to flagging when :math:`p > 0.5`; once the business decides missed fraud is
too costly, the threshold drops to **0.3** — more flags, higher recall, lower precision. A PSI
drift alert at **0.1** proves too jumpy against seasonality, so it is relaxed to **0.2**. And after
recalibrating an overconfident model (Platt scaling, isotonic regression), the decision cutoffs
must move with it.

How to do it
------------

Four approaches. **Empirical evaluation** — test candidate thresholds on a validation set.
**Cost-sensitive analysis** — pick the threshold that maximises expected business value.
**Periodic review** — revisit thresholds when retraining or after drift. And **dynamic thresholds**
that adapt automatically to recent performance.
"""

CONTENT["Guardrails (in ML & Data Systems)"] = r"""
What it is
----------

**Guardrails** are **secondary checks, rules or constraints** that keep an ML system **safe** (no
crashes or invalid inputs), **fair** (no harmful bias) and **reliable** (stable over time). They
are not the primary objective — minimising loss, maximising AUC — but they **prevent unacceptable
outcomes** once a system is deployed.

The five kinds
--------------

**Data guardrails** clamp outliers (cap ages at 120), handle unseen categories and validate schema.
**Performance guardrails** set a minimum acceptable accuracy or AUC and trigger retraining below it.
**Fairness guardrails** enforce parity across groups (e.g. loan-approval gaps within a few points)
and block non-compliant deployments. **Operational guardrails** cap latency (< 200ms), require
throughput and hold cost per prediction under budget. And **monitoring guardrails** alert when
drift (PSI, KL, MMD) or anomaly rates exceed limits, triggering rollback.

Why they matter
---------------

A model can look excellent on paper — AUC 0.9 — and still fail in practice. Guardrails **catch the
hidden risks** that headline metrics miss, which is what makes a system **trustworthy in
production**.

An analogy
----------

The primary metric is the **speedometer** — how fast the car is going. Guardrails are the **safety
rails on the road**: they don't make you faster, but they stop you driving off a cliff, however
fast you go.
"""

CONTENT["Model KPIs (Key Performance Indicators)"] = r"""
What it is
----------

**Model KPIs** are the **key metrics that track a model's performance, reliability and impact**.
They span **technical performance** (loss, accuracy, AUC, calibration) and **business impact** (ROI,
churn reduction, revenue uplift) — answering both "does it predict well?" and "does it help the
business?".

The four families
-----------------

**Prediction quality**: for classifiers, accuracy, precision/recall/F1, ROC-AUC and PR-AUC, log
loss and calibration; for regressors, MSE/RMSE, MAE and R². **Drift and stability**: feature drift
(PSI, KS test), data-quality checks and representation drift. **Operational**: latency, throughput,
uptime and cost per prediction. **Business impact**: revenue uplift, churn reduction, fraud savings
and ROI.

Examples
--------

A **fraud model** might report a technical KPI of **AUC 0.92**, an operational KPI of **50ms**
latency, and a business KPI of ``$1.2M`` of fraud prevented last quarter. A **recommender** might
report **NDCG@10 of 0.65**, sub-100ms response time, and an **8% lift in click-through rate**.

Leading vs lagging
------------------

KPIs divide into two roles. **Leading indicators** like drift **warn of future trouble** before it
hits performance. **Lagging indicators** like AUC, calibration and loss **confirm actual impact**
after the fact. A healthy dashboard watches both.
"""

MINDMAP.update({
    "Recalibrate Thresholds": [
        "Recalibration", "Bayesian Correction", "Monitoring Pipelines",
        "Guardrails (in ML & Data Systems)", "Platt Scaling", "Critical Value",
    ],
    "Guardrails (in ML & Data Systems)": [
        "Monitoring Pipelines", "Recalibrate Thresholds",
        "Model KPIs (Key Performance Indicators)", "Drift Detection",
        "Demographic Parity (Statistical Parity)", "Continuous Retraining",
    ],
    "Model KPIs (Key Performance Indicators)": [
        "Monitoring Pipelines", "Guardrails (in ML & Data Systems)", "Drift Detection",
        "PSI (Population Stability Index)", "Lagging Indicators", "Recalibrate Thresholds",
    ],
})


# ----------------------------------------------------------------------
# Theme: leading/lagging indicators, time-series windows  (mlops metrics / signal)
# ----------------------------------------------------------------------

CONTENT["Lagging Indicators"] = r"""
What it is
----------

A **lagging indicator** is a metric that **confirms the impact of a change after it has already
happened**. Lagging indicators measure **outcomes**, not early signals — in ML monitoring, they are
the **model-performance metrics**: loss, AUC, accuracy, calibration.

Characteristics
---------------

They are **reactive**, surfacing a problem only once it has occurred. They are a **direct measure**
of end results — model performance and business KPIs — which makes them the natural tools for
**validation and confirmation** rather than early warning.

Examples
--------

Three groups. **Performance metrics**: accuracy, precision/recall/F1, AUC, log loss, calibration
error. **Business KPIs after the fact**: CTR falling, fraud losses rising, churn climbing. And in a
**monitoring** context: an AUC drop that means drift *already* hurt predictions, or a loss spike
that follows a distribution change.

Why they matter
---------------

Lagging indicators **confirm whether the leading indicators actually mattered** — whether drift or a
data-quality issue translated into real damage. They are what go/no-go **retraining decisions** rest
on. In a fraud model, a leading indicator might be drift in ``transaction_type``; the **lagging**
indicator is AUC sliding from 0.87 to 0.72 — proof the model is now underperforming.
"""

CONTENT["Leading Indicators"] = r"""
What it is
----------

A **leading indicator** is an **early signal** that gives **advance warning** of a possible future
problem. Leading indicators **predict** what might happen rather than confirming what already did,
and in ML they usually concern **input data quality and distribution**.

Characteristics
---------------

They are **proactive** — you can act before performance drops. They are **indirect**, measuring not
the end result but the *conditions* that affect it. And they have **short-term sensitivity**,
catching changes quickly.

Examples
--------

Four kinds. **Data drift**: feature distributions shift (incomes skew higher) or category
frequencies change (new device types). **Input-data quality**: a sudden rise in missing values or
unexpected schema. **Operational**: latency spikes in feature pipelines, errors in upstream sources.
And **representation shift**: embeddings of user behaviour drifting from historical patterns.

Why they matter
---------------

Leading indicators are an **early-warning system** that fires *before* lagging metrics (AUC, loss,
accuracy) degrade, enabling proactive retraining, pipeline fixes or alerts. In a fraud model, a
**leading** signal — a surge in transactions from new countries — can precede the **lagging** AUC
drop by a week, buying time to respond.
"""

CONTENT["Windows (in Time-Series)"] = r"""
What it is
----------

In time-series analysis and monitoring, a **window** is a **slice of time** over which you compute a
statistic — a mean, sum, baseline or anomaly score. The window defines **how much past data** counts,
and it can **stay fixed**, **slide forward**, or **expand**.

The three types
---------------

A **fixed window** always covers a set period — "weekly sales, Monday to Sunday". A **rolling (or
sliding) window** moves forward step by step, recomputing as it goes — "a 7-day rolling average of
temperature" — and is the workhorse for smoothing, baselines and anomaly detection. An **expanding
window** starts at a point and **grows** as data arrives — "cumulative average sales since launch".

Baseline vs current
-------------------

Monitoring usually compares two windows. A **baseline window** — a rolling average over the last N
weeks — defines what "normal" looks like. A short **current window** — the last 24 hours to 7 days —
shows whether recent behaviour has **deviated** from that baseline.

An example
----------

For API monitoring, a rolling **8-week baseline** gives an average latency of **200ms**. The
**current 24-hour window** shows **350ms**. Because 350ms far exceeds the baseline, the comparison
**flags performance degradation** — the same windowed-comparison logic that underlies drift detection
and leading indicators.
"""

MINDMAP.update({
    "Lagging Indicators": [
        "Leading Indicators", "Model KPIs (Key Performance Indicators)", "Monitoring Pipelines",
        "Drift Detection", "Concept Drift", "Re-scoring",
    ],
    "Leading Indicators": [
        "Lagging Indicators", "Model KPIs (Key Performance Indicators)", "Monitoring Pipelines",
        "Drift Detection", "Data Drift", "Windows (in Time-Series)",
    ],
    "Windows (in Time-Series)": [
        "Sliding Window (Rolling Window) Cross-Validation", "Expanding Window Cross-Validation",
        "Time Series", "Monitoring Pipelines", "Drift Detection", "Leading Indicators",
    ],
})


# ----------------------------------------------------------------------
# Theme: representation shift & distribution-distance drift tests  (drift / signal)
# ----------------------------------------------------------------------

CONTENT["Representation Shift"] = r"""
What it is
----------

**Representation shift** occurs when the **internal representation** of data — learned embeddings,
feature vectors — **changes over time** between training and deployment, *even if the raw input
distribution looks similar*. It is a special case of distribution shift, but focused on the
**feature/embedding space** rather than the raw input.

Where it appears
----------------

Three places. In **neural networks and embeddings**, the learned mapping can change (through
retraining or new data), so downstream tasks built on the old space fail. In **preprocessing
pipelines**, steps like TF-IDF, PCA or scaling drift as the data changes — TF-IDF weights move as
new vocabulary dominates. And in **domain shift**, inputs that look similar can still drift in
embedding space — a face model trained on frontal faces, deployed on side profiles.

Why it matters
--------------

Downstream classifiers and regressors that **assume a stable representation degrade**;
**similarity search** (nearest-neighbour in embedding space) returns wrong results; and **fairness**
suffers if some groups' embeddings drift more than others.

Detecting it, with an example
------------------------------

Detection works on the embeddings themselves: **distance metrics** (MMD, energy distance, KL),
tracking cosine or Euclidean shifts; **visualisation** with t-SNE or UMAP to watch clusters move;
and **classifier two-sample tests**. The classic example: the word "mask" embedded mostly as
*cosmetic* in 2019 shifts toward *face covering* in 2020 — breaking any downstream sentiment or
topic model that relied on the old representation.
"""

CONTENT["Classifier Two-Sample Tests (C2STs)"] = r"""
What it is
----------

A **classifier two-sample test (C2ST)** checks whether two datasets come from the **same
distribution** by **training a classifier to tell them apart**. If the classifier **can't** beat
chance, the distributions are likely the same; if it **can** separate them well, they differ — a
distribution shift. It is a modern, high-dimensional-friendly alternative to classical tests like
the KS test, MMD or energy distance.

How it works
------------

Three steps. **Label** dataset A as 0 (say training data) and dataset B as 1 (production). **Train**
a classifier — logistic regression, random forest, neural net — to predict which set a sample came
from. **Evaluate**: accuracy near **50%** means the two are indistinguishable (same distribution),
while accuracy well above 50% (or a high AUC) signals they differ.

The hypothesis test
-------------------

Formally it tests :math:`H_0: P = Q` against :math:`H_1: P \neq Q`, using **classifier accuracy or
AUC** as the test statistic and **permutation testing or bootstrapping** to get a p-value. Its
strengths are working in **very high dimensions** (where KS or chi-square fail) and using
off-the-shelf classifiers; its costs are the **compute** of training, **sensitivity to classifier
choice**, and the need for enough data.

Where it's used
---------------

C2STs power **data-drift detection** (train vs production), **generative-model evaluation** (real vs
generated), and **bias detection** (comparing subgroups). Concretely: with 10,000 samples from 2024
and 10,000 from 2025, an XGBoost separator reaching **AUC 0.90** means the two are easily
distinguishable — strong drift.
"""

CONTENT["Energy Distance"] = r"""
What it is
----------

**Energy distance** is a **statistical distance** between two probability distributions :math:`P`
and :math:`Q`, built from **expected pairwise distances** between samples. It is **0 exactly when
the distributions are identical**, and grows as they differ — making it a natural **two-sample
test**, in the same family as MMD.

The formula
-----------

For :math:`X \sim P` and :math:`Y \sim Q`,

.. math::

   D_E^2(P, Q) = 2\, \mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\|,

where :math:`X, X'` are independent draws from :math:`P`, :math:`Y, Y'` from :math:`Q`, and
:math:`\|\cdot\|` is the Euclidean norm. The empirical version replaces these expectations with
averages over the two samples — cross-distances minus within-distances.

Properties
----------

It is a true **metric**: non-negative, symmetric, and zero iff :math:`P = Q`. Unlike MMD it is
**kernel-free**, working directly with Euclidean distances, and like MMD it stays sensitive in
**high dimensions** where the KS test fails. Both energy distance and MMD are **integral
probability metrics**.

Where it's used, with an example
----------------------------------

It serves **two-sample testing**, **drift detection** (training vs production), **GAN evaluation**
and **clustering validation**. Comparing two customer-age distributions, an energy distance of
**0.15** says they are fairly similar, while **1.2** signals a real difference — perhaps a much
younger incoming sample.
"""

MINDMAP.update({
    "Representation Shift": [
        "Embedding", "Drift Detection", "Covariate Drift (a.k.a. Covariate Shift)",
        "Leading Indicators", "Classifier Two-Sample Tests (C2STs)", "Energy Distance",
    ],
    "Classifier Two-Sample Tests (C2STs)": [
        "Representation Shift", "Drift Detection", "Energy Distance", "Data Drift",
        "PSI (Population Stability Index)", "Maximum Mean Discrepancy (MMD)",
    ],
    "Energy Distance": [
        "Classifier Two-Sample Tests (C2STs)", "Representation Shift", "Drift Detection",
        "Maximum Mean Discrepancy (MMD)", "PSI (Population Stability Index)", "Data Drift",
    ],
})


# ----------------------------------------------------------------------
# Theme: MMD distance, categorical cardinality & drift  (drift / features)
# ----------------------------------------------------------------------

CONTENT["Maximum Mean Discrepancy (MMD)"] = r"""
What it is
----------

**Maximum mean discrepancy (MMD)** is a statistical test for the **difference between two
distributions** :math:`P` and :math:`Q` from samples. Under a chosen kernel, **MMD is 0 exactly
when the distributions match**, and grows as they diverge. It is widely used to detect **drift**,
compare **train versus test** data, and evaluate **generative models** (GANs, VAEs, diffusion).

The idea
--------

MMD compares the **mean embeddings** of the two distributions in a **reproducing kernel Hilbert
space (RKHS)**:

.. math::

   \text{MMD}^2(P, Q) = \left\| \mu_P - \mu_Q \right\|_{\mathcal{H}}^2,

where :math:`\mu_P = \mathbb{E}_{x \sim P}[\phi(x)]` is the mean embedding of :math:`P` under the
feature map :math:`\phi` of a kernel :math:`k(x, y)`.

Estimating it
-------------

The **kernel trick** gives an estimate from samples without ever touching the infinite-dimensional
space — averaging within-:math:`P` kernels plus within-:math:`Q` kernels minus twice the cross
kernels:

.. math::

   \widehat{\text{MMD}}^2 = \frac{1}{m^2}\sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{n^2}\sum_{j,j'} k(y_j, y_{j'}) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j),

usually with a Gaussian RBF kernel.

Where it's used
---------------

A **small** MMD means similar samples, a **large** one means different. It powers **drift detection**
(train vs production, covariate drift), **GAN evaluation** (generated vs real), and **domain
adaptation** (aligning source and target features, as in MMD-regularised networks). Train a fraud
model on last year's data :math:`P`; if this year's :math:`Q` gives a high MMD, you have covariate
drift and likely need to retrain.
"""

CONTENT["Cardinality in Categorical Data"] = r"""
What it is
----------

**Cardinality** is the number of **unique categories** in a categorical feature. ``Gender`` has just
two values — **low cardinality**; a ``Zip Code`` field has thousands — **high cardinality**. The
distinction matters because it changes how features should be measured and encoded.

Why it matters for association
--------------------------------

Cardinality directly affects measures like **Cramér's V**, which divides by the smaller table
dimension :math:`k = \min(\text{rows}, \text{cols})`. With **low cardinality** (Gender, Yes/No) the
statistic is easy to read — Gender versus product preference giving V = 0.3 is a clear moderate
association. With **high cardinality** (zip codes, product IDs) it grows unreliable: many categories
have tiny counts, the chi-square statistic inflates, and an association can look strong when it is
really just **sparsity**.

Handling high cardinality
---------------------------

Three remedies. **Group** rare categories into an "Other" bucket. Use **target encoding** or
**frequency encoding** rather than raw category comparison. And if using Cramér's V, ensure the
**sample is large enough** that expected cell counts are not tiny.

An example
----------

Comparing ``City`` (100 categories) against ``Purchase`` (Yes/No) might yield Cramér's V of **0.6**,
suggesting a strong link — but that can simply reflect **too few samples per city**, not a real
effect of city on purchasing.
"""

CONTENT["Categorical Drift"] = r"""
What it is
----------

**Categorical drift** is a change over time in the **distribution of categories** between training
and production data. It is a form of data drift that specifically affects **categorical features**
rather than continuous ones.

What happens
------------

The **frequency of categories** shifts — if 80% of customers came from *Region A* in training but
only 40% do in production, the feature has drifted. This **hurts models** trained on the old mix:
predictions skew, once-rare categories become common, and entirely **unseen categories** can appear
in production that the model never learned.

Detecting it
------------

Standard tools compare category frequencies. A **chi-square test** weighs observed against expected
counts; **Cramér's V** measures the strength of the shift; and the **Population Stability Index
(PSI)** quantifies how much a categorical distribution has moved.

Where it bites
--------------

The effects are concrete. In **e-commerce**, a recommender fails when new products dominate. In
**healthcare**, a diagnosis model degrades as disease-code frequencies change. In **finance**, fraud
detection weakens as transaction types (online, POS, crypto) shift — each a categorical drift the
monitoring must catch.
"""

MINDMAP.update({
    "Maximum Mean Discrepancy (MMD)": [
        "Energy Distance", "Classifier Two-Sample Tests (C2STs)", "Drift Detection",
        "Covariate Drift (a.k.a. Covariate Shift)", "Representation Shift",
        "PSI (Population Stability Index)",
    ],
    "Cardinality in Categorical Data": [
        "Categorical Drift", "Cramér's V", "Embedding", "Data Drift", "Drift Detection",
        "PSI (Population Stability Index)",
    ],
    "Categorical Drift": [
        "Cardinality in Categorical Data", "Cramér's V", "Data Drift",
        "Covariate Drift (a.k.a. Covariate Shift)", "PSI (Population Stability Index)",
        "Drift Detection",
    ],
})


# ----------------------------------------------------------------------
# Theme: Cramer's V, macro shifts, categorical explosions  (features / drift)
# ----------------------------------------------------------------------

CONTENT["Cramér's V"] = r"""
What it is
----------

**Cramér's V** measures the **strength of association between two categorical variables**. Built on
the **chi-square statistic**, it **normalises** the result to lie between **0 and 1**: 0 means the
variables are completely independent, 1 means one fully determines the other.

The formula
-----------

.. math::

   V = \sqrt{\frac{\chi^2}{n \,(k - 1)}},

where :math:`\chi^2` is the chi-square statistic, :math:`n` the total sample size, and :math:`k` the
smaller of the number of rows and columns in the contingency table. Dividing by :math:`k - 1` is
what keeps :math:`V` bounded regardless of table size.

Reading the number
------------------

A rough guide: **0.0-0.1** very weak, **0.1-0.3** weak, **0.3-0.5** moderate, and **above 0.5**
strong — though exact thresholds vary by field. Surveying 1,000 people on gender (male/female) and
drink preference (coffee/tea), a computed **V = 0.25** signals a weak-to-moderate link between the
two.

Where it's used
---------------

In data science it does three jobs: detecting **categorical drift** by comparing distributions over
time, flagging **redundant features** (two categoricals so strongly associated that one can be
dropped), and testing **feature-target association** in classification.
"""

CONTENT["Macro Shifts"] = r"""
What it is
----------

**Macro shifts** are **large-scale, external changes** in the broader environment — economic, social,
political or technological — big enough to move markets and break models. In ML terms they are
**system-wide distribution changes**, structural shifts well beyond ordinary small drift and usually
outside the business's control.

Examples
--------

The pattern recurs across domains. A **global recession** reshapes consumer spending; a **pandemic**
collapses travel and surges e-commerce overnight; **inflation** rewrites buying habits. Each breaks
models trained on the old world — a pre-pandemic **credit-risk** model misreads new borrower
behaviour, **demand forecasts** built on old habits miss, and **supply-chain** lead times jump after
a geopolitical disruption.

Why they matter, and detecting them
-------------------------------------

Models assume **stationarity** — that the future resembles the past — and macro shifts shatter that
assumption, causing prediction failure, strategic risk, and new **fairness** problems. They are
caught with **drift measures** (PSI, KL or Jensen-Shannon divergence, KS tests), **performance
monitoring** (sudden AUC or lift drops), and **external signals** (economic indicators, policy
changes).

Responding to them
------------------

The playbook: **retrain** on post-shift data, prefer **adaptive** models (online or fast time-series
learners), run **scenario planning and stress tests**, keep **humans in the loop** under drastic
change, and **diversify data sources**. A macro shift is the broad external force that often drives
**concept drift** and **data drift** at the same time.
"""

CONTENT["Categorical Explosions"] = r"""
What it is
----------

A **categorical explosion** happens when a categorical feature has a **very large number of unique
levels**, so that naive encoding — one-hot in particular — produces a **feature explosion**: the
dataset becomes enormously **wide and sparse**, straining storage, computation and model quality.

The problem in numbers
----------------------

A ``Zip Code`` field with **50,000** values becomes **50,000 binary columns** after one-hot encoding;
a ``Product ID`` with a million values becomes a **million columns**. The damage is fourfold: **high
dimensionality** (overfitting), **sparsity** (mostly zeros), **compute cost** (slow, memory-hungry
training), and **poor generalisation** to unseen categories.

Handling it
-----------

Six strategies replace naive one-hot. **Group** rare categories into "Other" or bucket by region.
**Frequency or target encoding** replaces a category with its count or mean target. The **hashing
trick** maps categories into a fixed number of buckets. **Entity embeddings** learn dense vectors for
each category during training. **Dimension reduction** (PCA, autoencoders) compresses the encoding.
And **domain knowledge** lowers granularity — "Product Category" instead of "Product ID".

Where it appears
----------------

The usual sources are **retail** (product and user IDs), **geography** (zip codes, GPS), **web data**
(URLs, session and device IDs) and **healthcare** (ICD-10 codes, tens of thousands of them).
"""

MINDMAP.update({
    "Cramér's V": [
        "Categorical Drift", "Cardinality in Categorical Data", "Data Drift",
        "PSI (Population Stability Index)", "Drift Detection", "Categorical Explosions",
    ],
    "Macro Shifts": [
        "Concept Drift", "Data Drift", "Covariate Drift (a.k.a. Covariate Shift)",
        "Drift Detection", "PSI (Population Stability Index)", "Continuous Retraining",
    ],
    "Categorical Explosions": [
        "Cardinality in Categorical Data", "Embedding", "Categorical Drift", "Cramér's V",
        "Autoencoder", "Drift Detection",
    ],
})


# ----------------------------------------------------------------------
# Theme: cohorts, off-distribution data, discriminatory power  (analytics / drift / metrics)
# ----------------------------------------------------------------------

CONTENT["Cohort"] = r"""
What it is
----------

A **cohort** is a **group of individuals who share a common characteristic within a defined time
period**. Across statistics, business, healthcare and ML, cohorts let you study **patterns within one
group over time** rather than blurring everyone together — the idea is to track the *journey* of a
group that has something in common.

Kinds of cohort
---------------

The shared trait varies. A **signup cohort** is everyone who joined in January 2024; an **acquisition
cohort** is everyone won through a Q1 ad campaign; a **behavioural cohort** is those who purchased
within their first week. The same idea drives a **medical** cohort (patients diagnosed in 2022) and
an **education** cohort (the Fall 2020 entering class).

Cohort analysis
---------------

**Cohort analysis** tracks a metric for each cohort over time. A retention table, for instance,
follows monthly signup cohorts down the rows and elapsed months across the columns — a January cohort
might retain 70% at month 1, 50% at month 2, 35% at month 3 — revealing whether **later cohorts
behave differently** from earlier ones. It **removes noise** by comparing groups with shared starting
points, and it controls for time-based effects like seasonality.

Cohort vs segment
-----------------

The distinction is time. A **cohort** is defined by a **shared event in a time window** ("everyone who
joined in March"); a **segment** is defined by **attributes regardless of when** ("all users aged
18-24"). Cohorts answer "how does this group evolve?"; segments answer "who are these people?".
"""

CONTENT["Off-Distribution"] = r"""
What it is
----------

**Off-distribution** data are points that **differ significantly from the distribution the model was
trained on** — inputs outside its "familiar range". Models assume production data is drawn from the
**same distribution** as training (the i.i.d. assumption); when that breaks, the input is
off-distribution, also called **out-of-distribution (OOD)**.

Examples
--------

A cats-versus-dogs classifier shown a **giraffe** is off-distribution. A credit model trained on
2015-2020 applications meets **post-COVID** borrower behaviour it never saw. A diagnostic model
trained on adult MRIs is handed a **child's** scan. In each case the input falls outside the learned
scope.

Why it's a problem, and detecting it
--------------------------------------

Models optimised for in-distribution data make **unreliable or overconfident** predictions on OOD
inputs, with fairness risks for unseen subgroups. Detection draws on **distance metrics** (KL,
Jensen-Shannon, KS), **embedding methods** (Mahalanobis or cosine distance in latent space),
**uncertainty estimation** (Bayesian neural nets, deep ensembles, MC dropout), and dedicated **OOD
classifiers**.

Handling it
-----------

Five responses: **augment** the training data to broaden coverage; **adapt** the model to the new
domain; build **robust** models with regularisation or adversarial training; add a **reject option**
so the model can abstain ("I don't know"); and run **monitoring pipelines** to flag drift in
production. Off-distribution is the abrupt cousin of gradual **data** and **concept drift**.
"""

CONTENT["Discriminatory Power"] = r"""
What it is
----------

**Discriminatory power** is a model's (or test's) ability to **distinguish correctly between two
groups** — usually positives versus negatives: good versus bad credit, disease versus healthy,
responder versus not. The plain question it answers: *how well can the model separate those who will
do X from those who won't?*

Where it applies
----------------

A classifier with high discriminatory power assigns **higher scores to positives** than negatives. A
**credit scorecard** is judged on how cleanly it separates defaulters from non-defaulters; a
**diagnostic test** on how well it separates the sick from the healthy.

How it's measured
-----------------

Several metrics. **AUC** is the probability a random positive outscores a random negative — 0.5 is
random, 1.0 is perfect. The **KS statistic** is the maximum gap between the cumulative score
distributions of positives and negatives (0.4-0.6 is strong in credit risk). The **Gini coefficient**
rescales AUC, :math:`\text{Gini} = 2 \times \text{AUC} - 1`. Lift and CAP curves give a visual read.

An example, and why it matters
--------------------------------

If good borrowers average a score of 0.8 and bad ones 0.3, with **AUC 0.85 and KS 0.45**, the model
has strong discriminatory power; **AUC 0.55, KS 0.08** is near-random. It drives better **targeting**
and **lending** decisions, is a **regulatory** reporting requirement in finance, and reduces false
positives and negatives for fairer outcomes.
"""

MINDMAP.update({
    "Cohort": [
        "Customer Segmentation", "Retention", "Churn", "Causal Inference",
        "Revenue per User (RPU / ARPU)",
    ],
    "Off-Distribution": [
        "IID (Independent and Identically Distributed)", "Data Drift", "Concept Drift",
        "Covariate Drift (a.k.a. Covariate Shift)", "Drift Detection", "Representation Shift",
    ],
    "Discriminatory Power": [
        "Gini Coefficient", "Multiclass AUROC", "KS Statistic (Kolmogorov–Smirnov Statistic)",
        "Demographic Parity (Statistical Parity)", "Recalibration",
    ],
})
