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
**MAP**, **sMAPE** and **WAPE**. Choose carefully: MAP explodes when actual
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
