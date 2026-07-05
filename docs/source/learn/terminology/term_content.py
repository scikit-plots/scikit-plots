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


# ----------------------------------------------------------------------
# Theme: KS statistic, model stability, feature values  (metrics / mlops / features)
# ----------------------------------------------------------------------

CONTENT["KS Statistic (Kolmogorov–Smirnov Statistic)"] = r"""
What it is
----------

The **KS statistic** measures the **maximum difference between two cumulative distribution functions
(CDFs)**. In statistics it underpins the **KS test** — do two samples come from the same
distribution? In ML, especially credit scoring and binary classification, it measures how well a
model **separates positives from negatives**: far-apart distributions give a large KS, heavily
overlapping ones a small KS.

The formula
-----------

For a binary problem, build the CDF of positives and the CDF of negatives across the score, then take
the largest vertical gap between them:

.. math::

   KS = \max_x \left| F_{\text{pos}}(x) - F_{\text{neg}}(x) \right|,

where :math:`F` is a cumulative distribution. Because it scans every threshold for the single point
of greatest separation, KS is **threshold-independent**.

A worked example
----------------

Score 1,000 customers — 500 good, 500 bad — and sort by predicted score. At each threshold track the
share of bads captured against the share of goods captured; KS is the largest gap between those
curves. If at score 0.65 the CDF of bads is 0.70 and the CDF of goods is 0.30, the gap is 0.40, so
**KS = 40%**.

Reading it, and its cousin AUC
--------------------------------

KS runs from **0 to 1**: 0 means no separating power (identical distributions), **0.4-0.6** is strong
(typical in credit risk), below 0.2 is weak. It is closely related to **AUC** — both measure class
separability and both are threshold-independent — but where AUC integrates the whole ROC curve, KS
reports only the **single maximum point of separation**.
"""

CONTENT["Model Stability"] = r"""
What it is
----------

**Model stability** is how **consistently** a model performs when faced with different data samples
(train versus validation), new production data, and **time-evolving** data subject to drift. A stable
model is reliable and predictable and does not swing wildly; an unstable one is sensitive to small
changes in data or training splits and gives inconsistent results.

Four dimensions
---------------

Stability has several faces: **performance** stability (accuracy, AUC, RMSE hold across datasets and
time), **prediction** stability (similar inputs give similar outputs after retraining), **feature**
stability (importances and coefficients stay consistent), and **temporal** stability (the model
resists drift as new data arrives).

Measuring it
------------

The tools are familiar: low **cross-validation variance** across folds, **retraining consistency**
across random seeds, the **Population Stability Index (PSI)** for train-versus-production distribution
shift, **feature-importance** consistency across runs, and ongoing **drift monitoring**. A credit
model with validation AUC **0.85** that falls to **0.70** on next year's production data is unstable
over time — concept drift has set in, and it needs retraining.

Improving it, and why it matters
----------------------------------

Stability improves with **robust feature engineering**, **regularisation**, **ensembles** (which cut
variance), **data-quality monitoring**, **drift-detection systems** (PSI, KS test), and a
**retraining schedule**. It matters because stakeholders need **trust**, regulated industries demand
it for **compliance**, and unstable models make **inconsistent business decisions** — approving and
denying similar loans at random.
"""

CONTENT["Feature Values"] = r"""
What it is
----------

**Feature values** are the actual **numerical, categorical or textual values** describing an
observation. The distinction in terms: a **feature** is a variable — a column in the dataset — while
a **feature value** is the concrete entry for one observation, a single cell in a given row.

An example
----------

In a table predicting whether a customer buys, the **features** are Age, Gender and Income; the
**feature values** for one customer might be Age = 25, Gender = Male, Income = ``40,000``, and for
another Age = 32, Gender = Female, Income = ``55,000``. Each row supplies one set of feature values.

Types of value
--------------

They come in several kinds: **numerical** (Age = 25), **categorical** (Gender = Male/Female),
**binary** (Yes/No, 0/1), **textual** (reviews, turned into embeddings or bag-of-words), and
**derived** features engineered from raw data ("income per household member").

Their role, and why they matter
---------------------------------

Feature values are the **inputs** a model learns from to predict a target. A linear model writes the
prediction as

.. math::

   \hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b,

with the :math:`x_i` the feature values and the :math:`w_i` the learned weights. Because they drive
everything downstream, **data quality**, **scaling** (so no variable dominates), **engineering**, and
**interpretability** all hinge on getting feature values right.
"""

MINDMAP.update({
    "KS Statistic (Kolmogorov–Smirnov Statistic)": [
        "Discriminatory Power", "Gini Coefficient", "Multiclass AUROC", "Energy Distance",
        "PSI (Population Stability Index)",
    ],
    "Model Stability": [
        "PSI (Population Stability Index)", "Drift Detection",
        "KS Statistic (Kolmogorov–Smirnov Statistic)", "Concept Drift",
        "Continuous Retraining", "Monitoring Pipelines",
    ],
    "Feature Values": [
        "Hyperparameter", "Model Weights", "Embedding", "Cardinality in Categorical Data",
        "Machine Learning (ML)", "Model Stability",
    ],
})


# ----------------------------------------------------------------------
# Theme: four-fifths / 80% rule — adverse impact  (fairness)
# ----------------------------------------------------------------------

CONTENT["Four-Fifths (80%) Rule"] = r"""
What it is
----------

The **four-fifths rule** — also called the **80% rule** — is the primary rule-of-thumb the U.S.
**EEOC** uses to flag **adverse (disparate) impact** in a selection process. It comes from the
**Uniform Guidelines on Employee Selection Procedures (UGESP, 1978)** and operates under **Title VII**
of the Civil Rights Act. The rule: the **selection rate** for any protected group (by race, sex or
ethnicity) that is **less than four-fifths (80%) of the rate of the highest-selected group** is
generally treated as evidence of adverse impact.

How the ratio is computed
---------------------------

Four steps. Compute each group's **selection rate** — the number selected divided by the number of
applicants. Identify the group with the **highest** rate; that becomes the **baseline**. Divide every
other group's rate by that baseline to get an **impact ratio**. If any ratio falls **below 0.80**,
the procedure is flagged for that group.

A worked example
----------------

An employer hires for a warehouse role: of 100 white applicants, 40 are hired (a **40%** rate); of 80
Black applicants, 24 are hired (a **30%** rate). The impact ratio is 30 / 40 = **0.75**. Because 75%
is below 80%, adverse impact is indicated against Black applicants and the procedure would be flagged
for review. Note the test compares **rates, not head-counts** — a smaller group can still clear the
bar if its rate is high enough.

A threshold, not a verdict
----------------------------

Falling below 80% does **not** prove discrimination — it triggers **investigation**. Because the raw
ratio ignores whether the gap is beyond chance, practitioners pair it with **significance tests**
(Fisher's exact, chi-square, or a standard-deviation analysis). Once adverse impact is indicated, the
employer must show the procedure is **job-related and consistent with business necessity**, offer a
**less-discriminatory alternative**, or **modify** it. Courts treat the rule as guidance, not a
definitive legal standard.

Why it matters for ML
---------------------

The rule governs **automated selection** exactly as it governs human decisions: an AI resume screener
or scoring model that selects one protected group at **under 80%** of the top group's rate carries
indicated adverse impact — **even with no discriminatory intent**. This is why the 80% ratio is the
legal touchstone behind fairness criteria like **demographic parity**, which compares
positive-outcome rates across groups.
"""

MINDMAP.update({
    "Four-Fifths (80%) Rule": [
        "Demographic Parity (Statistical Parity)", "Equal Opportunity (Fairness)",
        "Equalized Odds (Fairness)", "Predictive Parity (Calibration)",
        "Discriminatory Power", "Statistical Significance",
    ],
})


# ----------------------------------------------------------------------
# Theme: SLI, ROI, treatment cost  (ops / business value / uplift)
# ----------------------------------------------------------------------

CONTENT["SLI (Service Level Indicator)"] = r"""
What it is
----------

A **Service Level Indicator (SLI)** is a **specific, measurable metric** that reflects a service's
actual performance against what was agreed or expected — the **"thermometer"** of how well a service
is running. It is the **raw measurement** used to judge whether a service is meeting its objective.

SLI vs SLO vs SLA
-----------------

Three layers stack up. The **SLI** is the metric measured — "percentage of requests completed within
300 ms". The **SLO** is the target for that metric — "99% of requests must complete within 300 ms".
The **SLA** is the formal contract, often with penalties — "if availability drops below 99%, the
provider credits the customer". SLI is the most granular layer; SLO and SLA build on it.

Common SLIs
-----------

They fall into families: **reliability** (uptime %, mean time between failures), **performance**
(latency, throughput in requests per second), and **quality** (error rate, data accuracy). The same
idea appears in operations as **on-time delivery %**, **fill rate**, or **stockout frequency**.

An example, and why it matters
--------------------------------

An e-commerce site might measure that **95%** of checkout requests finish in under two seconds (the
SLI) against an SLO target of **99%**, under an SLA that compensates customers if uptime falls below
**98%**. SLIs matter because they give **objective evidence** of performance, guide **where to
improve**, and form the measurable **foundation** of every SLO and SLA.
"""

CONTENT["ROI (Return on Investment)"] = r"""
What it is
----------

**ROI (Return on Investment)** is a **profitability metric**: how much return — gain or loss — an
investment produces relative to its cost. It answers, for every ``$1`` spent, how much profit came
back. It appears everywhere — marketing campaigns, product launches, financial portfolios, and
operational change.

The formula
-----------

.. math::

   \text{ROI} = \frac{\text{Gain from Investment} - \text{Cost of Investment}}{\text{Cost of Investment}} \times 100\%,

where the **gain** is the incremental revenue or benefit and the **cost** is the treatment cost,
campaign spend, or project cost.

A worked example
----------------

A promotion generates ``$15,000`` of incremental revenue for a ``$10,000`` cost. Then ROI =
(15,000 − 10,000) / 10,000 × 100% = **50%** — every ``$1`` spent returned ``$1.50``, a 50-cent profit
on the dollar.

Variations, and the uplift view
---------------------------------

Common variants include **marketing ROI** (on incremental campaign revenue), **ROAS** (revenue over
ad spend), **risk-adjusted** ROI (accounting for variance), and **time-adjusted** ROI (NPV, IRR). In
**uplift modeling**, ROI is computed from incremental revenue against treatment cost, which steers
spend toward **persuadables** — those where incremental benefit exceeds treatment cost — and away
from sure things and lost causes.
"""

CONTENT["Treatment Cost"] = r"""
What it is
----------

In **causal ML and uplift modeling**, **treatment cost** is the expense of applying an intervention
to one individual, customer or unit — the **per-unit cost** of sending a coupon, serving an ad,
granting a discount, or administering a drug. Summed over everyone treated, it becomes the **total
cost of the campaign**.

The formula
-----------

Per unit it is simply the direct cost of the intervention per customer; in total,

.. math::

   \text{Total Treatment Cost} = \text{Cost per unit} \times \text{Number of Treated Customers}.

A worked example
----------------

A retailer sends a ``$5`` coupon to 2,000 customers, so the per-unit cost is ``$5`` and the total
treatment cost is 2,000 × 5 = ``$10,000``. If that campaign produces ``$15,000`` in incremental
revenue, the **net incremental profit** is 15,000 − 10,000 = ``$5,000``.

Why it matters for uplift and ROI
-----------------------------------

Treatment cost is what turns raw uplift into *profit*. Incremental revenue is the extra money earned;
treatment cost is what was spent to earn it; the difference is **incremental profit**, and the ratio
is the **ROI of treatment**. Tracking it prevents targeting customers where **cost exceeds benefit**,
enables **profit-based** (not conversion-only) uplift modeling, and forces attention on **net value**
rather than gross outcomes.
"""

MINDMAP.update({
    "SLI (Service Level Indicator)": [
        "SLOs (Service Level Objectives)", "Model KPIs (Key Performance Indicators)",
        "Monitoring Pipelines", "Guardrails (in ML & Data Systems)", "Model Stability",
        "ROI (Return on Investment)",
    ],
    "ROI (Return on Investment)": [
        "Treatment Cost", "Incremental Revenue", "Treatment Effect", "Valuation Metric",
        "Gross Margin", "SLI (Service Level Indicator)",
    ],
    "Treatment Cost": [
        "Incremental Revenue", "ROI (Return on Investment)", "Treatment Effect",
        "Conversion Rate Uplift", "Causal Inference", "Posterior probability of uplift",
    ],
})


# ----------------------------------------------------------------------
# Theme: incremental revenue — causal revenue lift  (uplift / business value)
# ----------------------------------------------------------------------

CONTENT["Incremental Revenue"] = r"""
What it is
----------

**Incremental revenue** is the additional revenue a business earns **specifically because of an
action** — a campaign, promotion or treatment — **beyond the baseline it would have earned anyway**.
It is a **causal** measure: it captures revenue the action *created*, not revenue it merely took
credit for.

The formula
-----------

.. math::

   \text{Incremental Revenue} = \text{Actual Revenue} - \text{Baseline Revenue}.

If an e-commerce store averaging ``$50,000`` a month launches an affiliate program and revenue rises
to ``$60,000``, the incremental revenue attributable to the program is ``$10,000``.

Getting the baseline right
----------------------------

The **baseline is the whole game**. Because it represents what would have happened *without* the
action, a careless baseline lets ordinary fluctuations masquerade as lift. Four forces routinely
inflate a naive estimate: **seasonality**, pre-existing **organic growth trends**, **competitive
dynamics**, and broader **shifts in customer behaviour**. Failing to adjust for them credits normal
business change to the campaign.

Measuring it credibly
-----------------------

The most reliable method is a **holdout (control-group) test** — a randomised experiment comparing an
exposed group against an unexposed one, so the difference is genuinely causal. **Marketing mix
modelling (MMM)** and **attribution modelling** offer alternative lenses, each isolating true lift
from sales that would have occurred regardless.

Link to profit and ROI
------------------------

Incremental revenue is the **foundation of profit-based evaluation**. Subtract the treatment cost and
it becomes **incremental profit**; divide that profit by the cost and it becomes **ROI**. It is also
the objective that **revenue uplift models** maximise, steering spend toward *persuadable* customers
whose spending the action actually changes.
"""

MINDMAP.update({
    "Incremental Revenue": [
        "Treatment Cost", "ROI (Return on Investment)", "Conversion Rate Uplift",
        "Treatment Effect", "Causal Inference", "Revenue per User (RPU / ARPU)",
    ],
})


# ----------------------------------------------------------------------
# Theme: Qini-curve construction — incremental gain, CIG, TIB  (uplift / ranking)
# ----------------------------------------------------------------------

CONTENT["Incremental Gain"] = r"""
What it is
----------

**Incremental gain** is the **extra positive outcome** — conversions, sales, sign-ups — produced by
applying a treatment (a campaign, discount or intervention) **versus not applying it**, measured for a
**specific slice of the population**. It is the **building block** of the cumulative incremental gain
and the Qini curve: the step-by-step value added as you target more people.

The formula
-----------

For one segment,

.. math::

   \text{Incremental Gain} = \text{Responses}_{\text{treatment}} - \text{Responses}_{\text{control}},

or, in probability terms, :math:`\left(P(\text{Outcome} \mid \text{Treatment}) - P(\text{Outcome} \mid \text{Control})\right) \times n`, where :math:`n` is the number of individuals targeted in that
segment and the control response is what the same group would have done untreated.

A worked example
----------------

Rank 2,000 customers by uplift score and split into deciles of 200. In the **top decile**, the
treatment group converts at 18% (36 purchases) against a control of 10% (20) — an incremental gain of
**16**. The **second decile** converts at 14% (28) versus 12% (24) — a gain of **4**. Their running
total, the cumulative incremental gain after the top 20%, is **20 extra purchases**.

Where it fits
-------------

Incremental gain is the **local** effect in each segment; summing it as you move down the ranking
gives the **cumulative incremental gain**, and plotting that against the fraction targeted traces the
**Qini curve**. Reading it segment by segment shows **where uplift is strongest** and where to stop
targeting — once the gain flattens or turns negative, additional targeting destroys value.
"""

CONTENT["Total Incremental Benefit (TIB)"] = r"""
What it is
----------

**Total incremental benefit (TIB)** is the **overall net gain** from applying a treatment compared to
**not applying it at all** — the absolute improvement an uplift strategy delivers across the whole
population. In uplift modelling it is the **final value of the cumulative incremental gain curve**, at
100% of the targeted population.

The formula
-----------

.. math::

   \text{TIB} = \sum_{i=1}^{N} \left( y_i^{\text{treat}} - y_i^{\text{control}} \right),

summing the treated-minus-untreated outcome over the whole population :math:`N`. Expressed in money,
it becomes **incremental conversions × profit per conversion**.

A worked example
----------------

A promotion runs on 10,000 customers: the treatment group (5,000) makes 700 purchases (14%), the
control group (5,000) makes 500 (10%). That is **200 incremental conversions**; at ``$50`` profit
each, the total incremental benefit is **``$10,000``** — 200 extra purchases the campaign genuinely
caused.

Why it matters
--------------

TIB puts a single number on a campaign's **business value**. It feeds ROI directly —

.. math::

   \text{ROI} = \frac{\text{TIB} - \text{Campaign Cost}}{\text{Campaign Cost}},

and serves as a **benchmark** for comparing uplift models and a basis for **budget allocation**, since
it reflects the true added impact rather than gross outcomes.
"""

CONTENT["Cumulative Incremental Gain (CIG)"] = r"""
What it is
----------

**Cumulative incremental gain (CIG)** is an uplift-modelling measure of the **total additional
outcomes** — purchases, sign-ups, conversions — won by targeting customers **ranked by uplift score**,
relative to random or no targeting. It is the quantity plotted on the **Y-axis of a Qini curve**.

How it's built
---------------

Four steps: **rank** customers by predicted uplift score, highest to lowest; **split** them into
deciles or percentiles; for each segment compute the **incremental gain** as treatment responses minus
control responses; then take the **running sum** of those gains down to the segment of interest.

The formula
-----------

.. math::

   \text{CIG}(p) = \sum_{i=1}^{pN} \left( y_i^{\text{treat}} - y_i^{\text{control}} \right),

the treated-minus-untreated outcome accumulated over the top proportion :math:`p` of a population of
size :math:`N`.

A worked example
----------------

Across 10,000 customers, targeting the **top 20%** yields 400 treatment versus 300 control
conversions — a gain of **100**; extending to the **top 40%** adds 750 versus 550 — a further **200**.
The cumulative incremental gain at 40% is therefore **300 conversions**.

Reading the Qini curve
------------------------

Plotting CIG against the percentage of the population targeted gives the **Qini curve**. A **steep
early slope** means the model finds *persuadables* first and targeting pays off quickly; a **flat**
curve means the model is no better than random. The shape reveals the **optimal targeting fraction** —
where gain stops rising — and lets you **compare models** by how much incremental value each delivers.
"""

MINDMAP.update({
    "Incremental Gain": [
        "Cumulative Incremental Gain (CIG)", "Total Incremental Benefit (TIB)", "Qini Curve",
        "Conversion Rate Uplift", "Treatment Effect", "Incremental Revenue",
    ],
    "Total Incremental Benefit (TIB)": [
        "Cumulative Incremental Gain (CIG)", "Incremental Gain", "Qini Curve",
        "ROI (Return on Investment)", "Incremental Revenue", "Treatment Cost",
    ],
    "Cumulative Incremental Gain (CIG)": [
        "Incremental Gain", "Total Incremental Benefit (TIB)", "Qini Curve",
        "Treatment Effect", "Posterior probability of uplift", "Conversion Rate Uplift",
    ],
})


# ----------------------------------------------------------------------
# Theme: uplift-model evaluation — uplift score, Qini curve, Qini coefficient  (uplift / ranking)
# ----------------------------------------------------------------------

CONTENT["Uplift Score"] = r"""
What it is
----------

An **uplift score** is the number an **uplift model** assigns to an individual, estimating the
**incremental impact** — positive or negative — of a treatment (a campaign, discount or medical
intervention) on that person's likelihood of acting. It answers: *how much more (or less) likely is
this person to act if we intervene than if we do not?*

The formula
-----------

At the individual level,

.. math::

   \text{Uplift Score}_i = P(\text{Outcome} \mid \text{Treatment}, i) - P(\text{Outcome} \mid \text{Control}, i).

A **positive** score means the treatment raises the chance of the desired outcome, a **negative**
score means it backfires, and a score **near zero** means the treatment barely matters.

A worked example
----------------

For a subscription campaign: **Customer A** has a treated probability of 0.40 versus 0.25 untreated —
an uplift of **+0.15**, a strong target. **Customer B** sits at 0.70 versus 0.68 — just **+0.02**,
barely worth the spend. **Customer C** is 0.10 versus 0.20 — an uplift of **-0.10**, meaning the
campaign actively *hurts* (a "Do-Not-Disturb" case).

How it's used
-------------

Uplift scores let you **rank and target** the highest-scoring customers — the *persuadables* — while
avoiding *sure things* and *lost causes*, and **excluding negative-uplift** customers who might churn
or unsubscribe if contacted. Aggregated across the population, the scores build the **uplift and Qini
curves** whose quality the **Qini coefficient** summarises.
"""

CONTENT["Qini Curve"] = r"""
What it is
----------

The **Qini curve** is the **visual evaluation tool** for uplift models (causal or
incremental-response models). It shows the **incremental benefit** gained by targeting customers
**ranked by predicted uplift score** — the uplift-modelling analogue of an ROC curve, measuring how
well the model finds **persuadables**. The higher and more "bowed upward" it is, the better.

How it's built
---------------

Rank customers by uplift score from highest to lowest, **incrementally target** the top *x*%, and for
each portion compute the **incremental response** — treatment responses minus control responses. The
**cumulative** incremental response is what gets plotted.

Axes and baselines
------------------

The **X-axis** is the proportion of the population targeted (0% to 100%); the **Y-axis** is the
cumulative incremental response. A **random-targeting diagonal** marks the uplift you would expect from
targeting at random, and the **model curve** ideally sits well above it.

Reading it, with an example
-----------------------------

Targeting the top 20% might show a treatment purchase rate of 18% against a control of 12% — a **+6%**
incremental lift on those customers — and so on at 40%, 60%. A **steep early rise** means the model
concentrates persuadables at the top; a **flat, near-diagonal** line means it is no better than
random. The area between the model curve and the diagonal is the **Qini coefficient**.
"""

CONTENT["Qini Coefficient"] = r"""
What it is
----------

The **Qini coefficient** is a single-number **performance metric for uplift models** — their
equivalent of AUC. Where AUC asks how well a classifier predicts *who will buy*, the Qini coefficient
asks how well an uplift model ranks people by *who will buy because of the treatment*.

How it's computed
-----------------

Sort customers by predicted uplift (descending), split them into deciles or percentiles, and plot the
**Qini curve** — cumulative incremental response (treatment minus control) against the proportion
targeted. The coefficient is the **normalised area between the model's curve and the random-targeting
diagonal**.

The formula
-----------

.. math::

   \text{Qini} = \frac{\int_0^1 \left( G_{\text{model}}(x) - G_{\text{random}}(x) \right) dx}{\int_0^1 \left( G_{\text{perfect}}(x) - G_{\text{random}}(x) \right) dx}.

It ranges from **0 to 1** — 0.5 and up is decent, and close to 1 is excellent separation.

A worked example, and cousins
-------------------------------

Targeting the top 20%, a random selection might yield +100 purchases while the model yields **+300** —
the model curve sits above random, and the coefficient quantifies that gap. It is the **normalised**
form of the **AUUC** (raw area under the uplift curve), which makes it comparable across datasets:
ROC-AUC is to classification what the Qini coefficient is to uplift.
"""

MINDMAP.update({
    "Uplift Score": [
        "Uplift Models", "Qini Curve", "Qini Coefficient", "Conversion Rate Uplift",
        "Treatment Effect", "Posterior probability of uplift",
    ],
    "Qini Curve": [
        "Uplift Score", "Qini Coefficient", "Cumulative Incremental Gain (CIG)",
        "Incremental Gain", "Treatment Effect", "Uplift Models",
    ],
    "Qini Coefficient": [
        "Qini Curve", "Uplift Score", "AUUC (Area Under the Uplift Curve)", "Uplift Models",
        "Cumulative Incremental Gain (CIG)", "Total Incremental Benefit (TIB)",
    ],
})


# ----------------------------------------------------------------------
# Theme: uplift foundations — uplift, uplift models, AUUC  (uplift / causal)
# ----------------------------------------------------------------------

CONTENT["Uplift"] = r"""
What it is
----------

**Uplift** measures the **incremental impact** of an action, treatment or intervention compared to
**not taking it**. In machine learning, **uplift modelling** — also called *incremental response* or
*true lift* modelling — predicts the **change in outcome probability** caused by applying a treatment
(say, sending a marketing offer). It is about **causal effect**, not mere correlation.

The formula
-----------

.. math::

   \text{Uplift}(x) = P(Y = 1 \mid T = 1, X = x) - P(Y = 1 \mid T = 0, X = x),

where :math:`Y` is the outcome (purchase, churn, click), :math:`T` is the treatment (1 = received,
0 = not), and :math:`X` are the individual's features.

A worked example
----------------

For a promotional email, compare each customer's treated and untreated purchase probability.
**Customer A**: 30% with the email versus 25% without — an uplift of **+5%**, worth targeting.
**Customer B**: 60% versus 60% — **0%**, the email is irrelevant. **Customer C**: 20% versus 30% — a
**-10%** uplift, meaning the email actively *reduces* the chance (a spam-sensitive recipient).

Why it matters
--------------

Uplift buys **targeting efficiency** — spending only on customers who change behaviour *because of*
the treatment — and **cost savings**, by skipping "sure things" who would act anyway and "sleeping
dogs" who react badly. Used across **marketing, healthcare, finance and policy**, it is measured with
the **uplift (Qini) curve**, its area **AUUC**, and the **Qini coefficient**.
"""

CONTENT["Uplift Models"] = r"""
What it is
----------

**Uplift models** — also called incremental, net-lift or true-lift models — are predictive models
that estimate the **causal impact** of an action (a campaign, discount or treatment) on an
individual's behaviour, rather than the behaviour itself. They shift the question from *"who will
buy?"* to *"who will buy* **because of** *the campaign?"* — predicting incremental change, not raw
outcome.

Four kinds of customer
------------------------

Uplift modelling divides people by how they respond to intervention: **persuadables** act *only*
because of it (the target), **sure things** would act anyway, **lost causes** never act, and
**do-not-disturbs** react *negatively* if contacted. The goal is to reach persuadables, stop wasting
budget on sure things and lost causes, and avoid provoking do-not-disturbs.

Techniques
----------

Four families. The **two-model approach** trains separate treated- and control-group models and
subtracts their probabilities. **Class transformation** relabels the target to encode both outcome
and treatment, so a single classifier predicts uplift directly. **Uplift trees and forests** choose
splits that maximise the treatment-control difference. And **meta-learners** (T-, S- and X-learners)
build uplift on top of standard models within causal-ML frameworks.

Evaluation and trade-offs
---------------------------

Because uplift concerns *causal* effect, accuracy and AUC are not enough; models are judged by the
**Qini curve and coefficient**, the **uplift curve**, and **AUUC**. In a subscription campaign where
the treatment group buys at 20% against a control's 15%, the model tries to pinpoint which
individuals make up that **+5%**. The payoff is better spend, ROI and causal insight; the costs are a
need for **experimental (A/B) data**, greater model complexity, and harder interpretation.
"""

CONTENT["AUUC (Area Under the Uplift Curve)"] = r"""
What it is
----------

**AUUC (Area Under the Uplift Curve)** is the **cumulative incremental gain** of an uplift model,
integrated over the **entire population** — the total area under the uplift curve (incremental gain
against population proportion). It is an **overall** measure of how well the model ranks individuals
by uplift.

How it's computed
------------------

Sort customers by predicted uplift score (descending) and partition them into bins such as deciles.
For each bin, compute the uplift as the treatment response rate minus the control rate,

.. math::

   \text{Uplift}_k = \frac{y^{T}_k}{n^{T}_k} - \frac{y^{C}_k}{n^{C}_k},

then plot cumulative uplift against the fraction of the population targeted; **AUUC is the area under
that curve**.

The formula
-----------

.. math::

   \text{AUUC} = \int_0^1 U(x) \, dx,

where :math:`U(x)` is the cumulative uplift at population fraction :math:`x`. A strong model traces a
steep curve — targeting the top 20% might capture 80% of all achievable incremental responses — while
random targeting hugs the baseline near zero.

AUUC versus the Qini coefficient
---------------------------------

The two are close cousins. **AUUC** is a **raw** area, so it depends on dataset size and response
rate — much like raw accuracy. The **Qini coefficient** is a **normalised** AUUC, scaled between
random and perfect targeting, which makes it **comparable across datasets** — much like AUC. Both are
core metrics for evaluating uplift models.
"""

MINDMAP.update({
    "Uplift": [
        "Uplift Models", "Uplift Score", "Treatment Effect", "Causal Inference",
        "Conversion Rate Uplift", "Qini Coefficient",
    ],
    "Uplift Models": [
        "Uplift", "Uplift Score", "Qini Curve", "AUUC (Area Under the Uplift Curve)",
        "Causal ML (Causal Machine Learning)", "Treatment Effect",
    ],
    "AUUC (Area Under the Uplift Curve)": [
        "Qini Coefficient", "Uplift Curve", "Cumulative Incremental Gain (CIG)",
        "Total Incremental Benefit (TIB)", "Uplift Score", "Uplift",
    ],
})


# ----------------------------------------------------------------------
# Theme: uplift@k, uplift random forests, uplift curve  (uplift / algorithms / eval)
# ----------------------------------------------------------------------

CONTENT["Uplift@k"] = r"""
What it is
----------

**Uplift@k** is a performance metric in uplift modelling and causal ML. It measures the **incremental
effect** achieved if you target only the **top k%** of customers, ranked by the model's predicted
uplift score. In plain terms: *if I contact only the top k%, how much extra impact do I get compared
to not contacting them?*

The formula
-----------

It is the **difference in average outcome** between the treatment and control groups **within the
top-k% segment**,

.. math::

   \text{Uplift@}k = \bar{y}_{\text{treatment}}^{(k)} - \bar{y}_{\text{control}}^{(k)},

where the averages are taken over the top k% by predicted uplift. Random targeting yields a small or
zero value (treatment and control behave alike); a good model makes treatment clearly outperform
control in that segment.

A worked example
----------------

With 10,000 customers, targeting the **top 20%** (k = 20%) selects 2,000. If, within that segment, the
treatment group's purchase probability exceeds the control group's by **5 percentage points**, then
uplift@k = **+5pp** — the extra impact the model captures by choosing those 2,000.

Uses, and versus uplift
-------------------------

In **marketing** it estimates incremental sales from promoting only the top k%; in **healthcare**, the
incremental recovery from treating the top-k patients; in **recommendation**, the extra engagement
from targeting the top-k users. The distinction from plain uplift is subtle but important: **uplift**
asks *how effective is the treatment overall?*, while **uplift@k** asks *how good is my model at
selecting the best subset to treat?*
"""

CONTENT["Uplift Random Forests"] = r"""
What it is
----------

An **uplift random forest** is a modified random forest built to **estimate the causal effect** of a
treatment directly, not merely to predict an outcome. Instead of modelling :math:`P(Y \mid X)`, it
models the **treatment-versus-control difference**,

.. math::

   \Delta(X) = P(Y = 1 \mid T = 1, X) - P(Y = 1 \mid T = 0, X).

How it works
------------

The trees are **uplift trees**: each split is chosen to **maximise the difference in treatment effect**
between its branches, rather than to maximise class purity. Many such trees are then **averaged in an
ensemble**, exactly as in an ordinary random forest, for stability — and each individual receives an
estimated uplift, the incremental probability change caused by the treatment.

Why use it
-----------

It **handles nonlinear relationships and feature interactions** automatically, **reduces variance**
compared with a single uplift tree, and delivers **individual-level treatment-effect** predictions —
learning how features *modify* the treatment effect, not just how they drive the outcome.

Applications
------------

It powers **marketing** (targeting customers who respond *because of* a campaign), **personalised
medicine** (patients who benefit most from a drug), and **policy** (subgroups most positively
affected). It requires **both treated and control data** — an A/B setup. In an email sign-up campaign,
a standard forest predicts the probability of signing up, while the uplift forest predicts the *extra*
probability caused by the email — separating loyal always-signers (low uplift) from persuadables
(high) and negative reactors (negative uplift).
"""

CONTENT["Uplift Curve"] = r"""
What it is
----------

An **uplift curve** is a tool for **evaluating uplift models** (incremental-response models). It plots
the **incremental gain** — the extra benefit caused by the treatment — against the **proportion of the
population targeted**, showing how much improvement you gain by targeting the top fraction of customers
ranked by uplift score. It is closely related to, and often drawn as, the **Qini curve**.

Why not target everyone
-------------------------

Traditional models (like logistic regression) predict response probability, but in marketing you
**don't want to target everyone** likely to buy — some would buy anyway, and some are even negatively
influenced. The uplift curve reflects the model's job of isolating **persuadables**, those who change
behaviour because of the treatment.

How it's built
---------------

Rank customers by predicted uplift score from high to low, split them into buckets (top 10%, next
10%, …), and for each bucket compute the **treated-minus-control** difference in outcome rate. Plotting
the **cumulative** incremental gains against the percentage targeted gives the curve: the **X-axis** is
the fraction targeted, the **Y-axis** the incremental gain, with a **random baseline** line and the
**model curve** ideally above it.

Reading it, with an example
-----------------------------

A **steeper** curve means the model is better at finding the most-influenced customers; a curve **close
to the random line** adds little value; and the **area between the model curve and the random line**
serves as a performance metric, much like AUC. For an email renewal campaign, the curve might show
that targeting the **top 20%** by uplift score generates most of the incremental renewals, while
targeting everyone simply wastes resources.
"""

MINDMAP.update({
    "Uplift@k": [
        "Uplift Models", "Uplift Score", "AUUC (Area Under the Uplift Curve)", "Qini Curve",
        "Conversion Rate Uplift", "Uplift",
    ],
    "Uplift Random Forests": [
        "Uplift Models", "Uplift", "Treatment Effect", "Causal Inference", "Uplift Score",
        "Causal ML (Causal Machine Learning)",
    ],
    "Uplift Curve": [
        "Qini Curve", "AUUC (Area Under the Uplift Curve)", "Uplift Score", "Incremental Gain",
        "Uplift Models", "Uplift",
    ],
})


# ----------------------------------------------------------------------
# Theme: causal ML, random-targeting baseline, cumulative uplift  (causal / uplift)
# ----------------------------------------------------------------------

CONTENT["Causal ML (Causal Machine Learning)"] = r"""
What it is
----------

**Causal ML (causal machine learning)** is the branch of machine learning that estimates
**cause-and-effect relationships**, not merely correlations or predictions. Where traditional ML asks
*"what is likely to happen?"*, causal ML asks *"what will happen* **because of** *this
intervention?"* — making it essential for policy evaluation, medical treatment, marketing and any
decision where an action *changes* the outcome.

The counterfactual problem
----------------------------

Everything rests on **treatment versus control** — the intervention against no intervention. The
difficulty is the **counterfactual problem**: for any individual we observe only *one* outcome
(treated or not), never the other, so the effect must be **estimated**. That effect comes in three
grains: the **ATE** (average treatment effect across the population,
:math:`\text{ATE} = \mathbb{E}[Y(1) - Y(0)]`), the **CATE** (conditional on a subgroup), and the
**ITE** (for a single individual).

Methods
-------

The gold standard is **experimental** — a randomised controlled trial (RCT) assigns treatment at
random, eliminating confounding. When experiments are impossible, **observational** methods step in:
propensity-score matching, inverse-propensity weighting, and doubly-robust estimators that combine
regression with weighting. **ML extensions** add **meta-learners** (S-, T-, X- and R-learners),
**causal trees and forests** for heterogeneous effects, and deep models (Dragonnet, TARNet) — with
libraries such as ``CausalML``, ``EconML`` and ``DoWhy``.

An example
----------

In a marketing email campaign, traditional ML predicts **who will buy**; causal ML predicts **who
will buy because of the email**. If the treatment group buys at 15% and the control at 10%, the
**ATE is a 5% uplift** — and causal ML then estimates **CATE/ITE** to reveal which segments or
individuals respond most.

Benefits and challenges
-----------------------

The payoff is real: it captures **true causal effect** rather than correlation, targets only those
who benefit, and supports **counterfactual reasoning** ("what if?"). The costs are structural — it
needs a **treatment-control design** (experiments or strong assumptions), is **sensitive to
confounding** in observational data, and is harder to explain and validate than standard ML.
"""

CONTENT["Random Targeting Strategy"] = r"""
What it is
----------

A **random targeting strategy** selects customers (or units) **at random** for an intervention — a
campaign, treatment or policy — instead of using a predictive or uplift model. In uplift modelling it
is the **baseline** against which a model is judged: if a model cannot beat random targeting, it is
not useful.

The diagonal baseline
-----------------------

Treat a random proportion of the whole population — 20%, 50%, 100% — and, because the choice is
random, the **treatment effect spreads evenly**. On a Qini or uplift curve (proportion targeted on
the X-axis, cumulative incremental gain on the Y-axis) this traces a **straight diagonal line**:
uplift accumulates linearly with the fraction treated.

A worked example
----------------

With 10,000 customers and an average treatment effect of **+5%**, random targeting of 20% (2,000)
yields about **100** incremental conversions; 40% yields about **200**; and 100% yields about **500**
— the campaign's total incremental benefit. The line from (0, 0) to (100%, 500) is the random
baseline.

Role, pros and cons
---------------------

That line is the **benchmark**: a good model's curve starts steep — finding persuadables first — and
stays **above** it throughout; a curve that hugs the diagonal adds no value. Random targeting is
**simple** and **fair** (hence its use in A/B testing), but it **wastes budget** on sure things, lost
causes and do-not-disturbs, delivers lower ROI, and cannot adapt to customer heterogeneity.
"""

CONTENT["Cumulative Uplift"] = r"""
What it is
----------

**Cumulative uplift** is the **running total of the incremental effect** — extra responses,
conversions or purchases — gained by targeting customers ranked by uplift score. It says how much
total benefit has accrued up to a given proportion of the population, and it is the quantity on the
**Y-axis of the Qini curve**.

The formula
-----------

For the top :math:`k\%` of customers ranked by uplift score,

.. math::

   \text{Cumulative Uplift}(k) = \sum_{i=1}^{kN} \left( y_i^{\text{treatment}} - y_i^{\text{control}} \right),

summing the treated-minus-untreated outcome over those customers, where :math:`N` is the total count.

A worked example
----------------

Rank 1,000 customers (split treatment/control) by uplift score and walk down the deciles. The **top
10%** gives 18 treatment versus 12 control conversions — **+6**, so cumulative uplift is **6**. The
**top 20%** adds 32 versus 24 — **+8**, bringing it to **14**. The **top 30%** adds **+10**, reaching
**24**. At 100% the final cumulative uplift equals the **total incremental benefit**.

Why it matters, and its cousins
---------------------------------

The curve shows **where uplift concentrates** (the top 20% may hold most of the benefit), fixes the
**optimal targeting cutoff** (stop where it flattens), and **evaluates models** — steep early is
good, flat like the random baseline is bad. Its family: **incremental gain** is the per-group step,
**cumulative uplift** (the cumulative incremental gain) the running sum, **total incremental benefit**
the final value, and the **Qini curve** their plot.
"""

MINDMAP.update({
    "Causal ML (Causal Machine Learning)": [
        "Treatment Effect", "Causal Inference", "Uplift Models", "Uplift",
        "Random Targeting Strategy", "Conversion Rate Uplift",
    ],
    "Random Targeting Strategy": [
        "Qini Curve", "Cumulative Uplift", "Uplift Models", "Uplift Score",
        "Total Incremental Benefit (TIB)", "Causal ML (Causal Machine Learning)",
    ],
    "Cumulative Uplift": [
        "Cumulative Incremental Gain (CIG)", "Incremental Gain", "Total Incremental Benefit (TIB)",
        "Qini Curve", "Uplift Score", "Uplift Models",
    ],
})


# ----------------------------------------------------------------------
# Theme: incremental conversions, net revenue, causal effect  (uplift econ / causal)
# ----------------------------------------------------------------------

CONTENT["Incremental Conversions"] = r"""
What it is
----------

**Incremental conversions** measure the **additional number of conversions caused by a treatment** —
a campaign, promotion or intervention — compared with what would have happened without it (the
control). They are the extra conversions **directly attributable** to the treatment, not the raw
total.

The formula
-----------

The basic form is treatment conversions minus control conversions. To scale the effect to a full
population, use the rate-based version,

.. math::

   \text{Incremental Conversions} = \left( \frac{y^T}{n^T} - \frac{y^C}{n^C} \right) \times N,

where :math:`y^T/n^T` and :math:`y^C/n^C` are the treatment and control conversion rates and
:math:`N` is the total population you would target. This adjusts for the baseline rate and scales the
per-user uplift up to everyone.

A worked example
----------------

Run a campaign on 10,000 users: the treatment group (5,000) converts at 12% (600), the control group
(5,000) at 8% (400). The rate difference is **4%**, so across the full 10,000 the incremental
conversions are :math:`0.04 \times 10{,}000 = 400` — the extra conversions the campaign genuinely
caused.

Why it matters
--------------

**Total** conversions mislead, because some users would have converted anyway. **Incremental**
conversions isolate the **causal effect** of the treatment, which is why they are the foundation of
**uplift modelling** and incrementality testing across marketing, ads and A/B testing.
"""

CONTENT["Revenue net of treatment cost"] = r"""
What it is
----------

**Revenue net of treatment cost** is the extra revenue a treatment earns **after subtracting the cost
of running it**. It answers the blunt question: *after paying for the campaign, how much incremental
revenue did we really keep?*

The formula
-----------

.. math::

   \text{Net Revenue} = \text{Incremental Revenue} - \text{Treatment Cost},

where **incremental revenue** is treatment-group revenue minus control-group revenue, and **treatment
cost** covers marketing spend, incentives, delivery fees — any direct cost of the treatment.

A worked example
----------------

An email upsell campaign brings the treatment group to ``$120,000`` in revenue against the control's
``$100,000`` — an incremental revenue of ``$20,000``. If the campaign cost ``$5,000`` to run, then net
revenue is ``$15,000`` — the ``$20,000`` of incremental revenue minus the ``$5,000`` cost. The
headline ``$20K`` uplift is really **``$15K``** once costs are counted.

Why it matters
--------------

Uplift measured on revenue alone **overstates** the benefit. Net revenue sits closer to **ROI**
(though ROI further accounts for profit margin) and is the right lens for **comparing campaigns**: two
treatments with equal uplift are not equal if one costs far less to run.
"""

CONTENT["Causal Effect"] = r"""
What it is
----------

A **causal effect** is the change in an outcome **directly caused** by a change in a treatment,
**holding everything else constant** — *if X changes, how much does Y change because of X, and not
because of other factors?* This is strictly stronger than **correlation**, which shows only
association; causation means X **produces** the change in Y.

The potential-outcomes view
-----------------------------

The **Rubin causal model** frames it with potential outcomes: :math:`Y(1)` is a unit's outcome if
treated and :math:`Y(0)` its outcome if not, so the causal effect for that unit is
:math:`Y(1) - Y(0)`. Because we can only ever observe **one** of the two for any individual — the
**fundamental problem of causal inference** — we estimate the **average causal effect** instead,
:math:`\text{ACE} = \mathbb{E}[Y(1) - Y(0)]`.

Identifying it
--------------

Every method exists to **control for confounders** — factors that influence both treatment and
outcome. The toolkit runs from **randomised controlled trials** and **matching / stratification**
through **regression adjustment**, **instrumental variables** and **difference-in-differences**, up to
**causal graphs (DAGs)** for reasoning about confounding formally.

Why it matters
--------------

A causal effect is the **true impact** of one variable on another — what would change *if we
intervened*. It tests mechanisms in **science**, tells **policy** whether an intervention actually
works, and measures the real business impact of **marketing, pricing and product** changes.
"""

MINDMAP.update({
    "Incremental Conversions": [
        "Conversion Rate Uplift", "Treatment Effect", "Causal Effect", "Incremental Revenue",
        "Uplift", "Causal Inference",
    ],
    "Revenue net of treatment cost": [
        "Incremental Revenue", "Treatment Cost", "ROI (Return on Investment)",
        "Incremental Conversions", "Conversion Rate Uplift", "Gross Margin",
    ],
    "Causal Effect": [
        "Causal Inference", "Causal ML (Causal Machine Learning)", "Treatment Effect", "Uplift",
        "Incremental Conversions", "Conversion Rate Uplift",
    ],
})


# ----------------------------------------------------------------------
# Theme: cannibalization + service-level SLO/SLA  (uplift econ / ops)
# ----------------------------------------------------------------------

CONTENT["Cannibalization"] = r"""
What it is
----------

**Cannibalization** is the reduction in sales or revenue of an **existing** product, service or
channel caused by introducing or promoting a **new** one **from the same company**. Put plainly: your
new sales come at the **expense of your own old sales**, not from genuine market growth.

Two examples
------------

A **discount campaign** cuts 20% off a product; many loyal customers who would have paid full price
simply buy at the discount, so the apparent uplift is really **full-price sales shifted into
discounted ones**. A **product launch** does the same across a portfolio: when a company releases its
newest phone, part of that model's strong sales comes from customers who would have bought the
previous model anyway — the new revenue **cannibalises** the old.

The formula
-----------

In uplift and ROI analysis, cannibalization enters as a **negative adjustment**:

.. math::

   \text{Net Revenue} = \text{Incremental Revenue} - \text{Treatment Cost} - \text{Cannibalization Loss},

where the **cannibalization loss** is the revenue lost from existing products because of the
treatment.

Why it matters
--------------

Ignore it and campaigns look **more successful than they are** — the overestimation risk. Accounting
for it forces **whole-portfolio** thinking rather than single-product metrics, so true impact is
**incremental gain minus treatment cost minus cannibalization**. In short, cannibalization is
*stealing from yourself*: the hidden downside of a promotion or launch that must be subtracted from
uplift to find the real net benefit.
"""

CONTENT["SLOs (Service Level Objectives)"] = r"""
What it is
----------

A **Service Level Objective (SLO)** is a **measurable target** for how a service should perform —
usually a percentage or number. It is the **goal** for reliability, performance or availability that a
team commits to.

The SLA, SLO, SLI hierarchy
-----------------------------

The three fit together. The **SLA** is the external **contract** with customers; the **SLO** is the
internal **target** that supports it; and the **SLI** is the actual **measurement** used to check
performance. The SLO sits in the middle — stricter than the SLA, expressed in terms the SLI can
measure.

Examples and error budgets
----------------------------

Typical SLOs cover **availability** ("99.9% uptime per month", SLI = uptime %), **latency** ("95% of
requests under 300 ms"), **error rate** ("under 0.1% of requests return 5xx in 30 days") and
**throughput** ("at least 5,000 requests per second at peak"). An SLO of 99.9% implies an **error
budget** of 0.1% allowed downtime — the slack that decides when to prioritise reliability over new
features, and that stops teams from the costly, unrealistic chase for 100%.

A worked example
----------------

Suppose an SLA requires **99.5%** uptime. The internal SLO is set at **99.9%** for margin, and the SLI
is measured uptime over the last 30 days. If uptime slips to **99.6%**, the SLA is still safe but the
**SLO is breached** — an early warning of reliability risk *before* any SLA penalty is triggered.
"""

CONTENT["SLA (Service Level Agreement)"] = r"""
What it is
----------

A **Service Level Agreement (SLA)** is a **formal contract** between a service provider and a client
(or between two internal teams) that defines **what** will be delivered, the **quality and performance
standards** expected, **how** performance is measured, and the **consequences** of missing targets. It
governs supplier performance in **supply chains** (delivery, quality, responsiveness) and uptime and
support speed in **IT**.

Key components
--------------

A complete SLA names the **scope of services**; the **performance metrics (KPIs)** — on-time delivery,
fill rate, defect rate, uptime, response and resolution time; the **service-level targets**;
**monitoring and reporting**; **penalties and remedies**; each party's **responsibilities**; and the
agreement's **duration and review** cycle.

Example clauses
---------------

In a supply-chain context: **98%** of deliveries must arrive within three days of the promised date;
at least **95%** of order lines must be filled completely on first shipment; defective items must stay
**at or below 0.5%**; and the supplier must confirm receipt of a purchase order **within 24 hours**.

Benefits and challenges
-----------------------

An SLA brings **clarity**, **accountability**, measurable **tracking**, **risk reduction** and
**stronger relationships**. The pitfalls are equally concrete: **overly rigid** or **unrealistic**
targets strain the relationship, **monitoring** every metric has real cost, and an SLA must **adapt**
as business realities change.
"""

MINDMAP.update({
    "Cannibalization": [
        "Revenue net of treatment cost", "Incremental Revenue", "Treatment Cost",
        "ROI (Return on Investment)", "Incremental Gain", "Conversion Rate Uplift",
    ],
    "SLOs (Service Level Objectives)": [
        "SLI (Service Level Indicator)", "SLA (Service Level Agreement)",
        "Model KPIs (Key Performance Indicators)", "Monitoring Pipelines",
        "Guardrails (in ML & Data Systems)", "Model Stability",
    ],
    "SLA (Service Level Agreement)": [
        "SLOs (Service Level Objectives)", "SLI (Service Level Indicator)",
        "Model KPIs (Key Performance Indicators)", "SLA Breach Rate", "Monitoring Pipelines",
        "Guardrails (in ML & Data Systems)",
    ],
})


# ----------------------------------------------------------------------
# Theme: SLA breach rate, SLA breaches, ops health dashboard  (ops / service)
# ----------------------------------------------------------------------

CONTENT["SLA Breach Rate"] = r"""
What it is
----------

**SLA Breach Rate** is a **performance metric** measuring how often a service provider — a supplier,
IT team or internal group — **fails to meet the service levels agreed in the SLA**. Each **breach** is
a missed target (a late delivery, missed uptime, poor quality), and the rate reports the **percentage
of obligations not met** over a period. It is the exact **opposite of the SLA compliance rate**.

The formula
-----------

.. math::

   \text{SLA Breach Rate} = \frac{\text{Number of SLA breaches}}{\text{Total SLA obligations}} \times 100\%,

where **breaches** count the times performance fell below target and **obligations** count every
chance there was to meet it.

Worked examples
---------------

In a **supply chain**, 1,000 orders with 50 delivered late gives a breach rate of 50 / 1,000 =
**5%**. In an **IT support desk**, 2,000 tickets with 200 missing their resolution deadline gives
200 / 2,000 = **10%**.

Why it matters, and reducing it
---------------------------------

The rate drives **accountability**, signals **customer satisfaction**, triggers **contractual
penalties or credits**, and pinpoints weak areas. It falls with **better forecasting**, **real-time
monitoring and alerts**, **supplier collaboration**, **automated escalation**, and **realistic SLA
reviews**. Its mirror image is the compliance rate (100% minus the breach rate), and the common KPIs
it is measured against are on-time delivery, fill rate and uptime.
"""

CONTENT["SLA Breaches"] = r"""
What it is
----------

An **SLA breach** occurs when a service falls **below the standard promised** in the Service Level
Agreement — a missed uptime, a late delivery, a slow response. Breaches are tracked with the **SLA
breach rate**, the percentage of commitments missed over a period.

Where breaches happen
-----------------------

They span industries. In **IT and cloud** services, uptime dips below 99.9% or a response exceeds its
threshold. In **customer support**, a ticket goes unanswered past four hours or unresolved past 24. In
**logistics and supply chain**, deliveries run late or order accuracy falls short. In
**manufacturing**, the defect rate climbs above the agreed limit.

The consequences
----------------

The fallout is concrete: **financial penalties** (refunds, service credits), **customer
dissatisfaction** and lost trust, **reputational damage** through negative reviews and churn, and
**operational strain** as escalations and firefighting multiply.

A worked example
----------------

An SLA promises that **95% of orders ship within 48 hours**. Of 1,000 orders, 920 arrive on time, so
80 fall short — a breach rate of 80 / 1,000 = **8%**. A high rate translates directly into penalties,
churn and inefficiency.
"""

CONTENT["Ops Health Dashboard"] = r"""
What it is
----------

An **Ops (Operations) Health Dashboard** is a **visual, real-time monitoring tool** that gives an
at-a-glance overview of key operational metrics — the **"control panel"** that tells managers whether
supply chain, IT, production or customer service is running smoothly, by consolidating **KPIs, trends
and alerts** in one place.

Core features
-------------

Five capabilities define it: **real-time data** integration (with ERP, WMS, CRM and monitoring tools),
**KPI visualisation** (charts, gauges, traffic-light indicators), **drill-down** from overall health
to a specific issue, an **alerting system** that flags SLA breaches and anomalies, and **comparisons**
against history and targets.

What it tracks
--------------

The KPIs depend on context. **Supply-chain** ops watch stockout rate, fill rate, backorder rate,
inventory turnover, lead time and supplier SLA breach rate; **IT/service** ops watch uptime, SLA
breach rate, incident response time, MTTR and open-versus-resolved tickets; **business** ops watch
order-processing time, OTIF, CSAT/NPS and revenue versus target. A typical layout leads with a
composite score (say **92/100**) and green/yellow/red columns — for example a fill rate of 97%, uptime
of 99.7%, and an order-processing cost of ``$2.50`` per unit.

Benefits, and tools
-------------------

The dashboard becomes a **single source of truth**, speeds **issue detection**, improves
**accountability**, and supports **data-driven decisions**. It is built with BI tools (Tableau,
Power BI, Looker, Qlik), ops platforms (ServiceNow, Splunk, Datadog), built-in ERP/WMS dashboards
(SAP, Oracle NetSuite), or custom stacks (Python Dash or Streamlit, R Shiny, Grafana).
"""

MINDMAP.update({
    "SLA Breach Rate": [
        "SLA (Service Level Agreement)", "SLOs (Service Level Objectives)",
        "SLI (Service Level Indicator)", "SLA Breaches", "Ops Health Dashboard",
        "Model KPIs (Key Performance Indicators)",
    ],
    "SLA Breaches": [
        "SLA Breach Rate", "SLA (Service Level Agreement)", "SLOs (Service Level Objectives)",
        "Ops Health Dashboard", "Supplier Management", "Model KPIs (Key Performance Indicators)",
    ],
    "Ops Health Dashboard": [
        "SLA Breach Rate", "Model KPIs (Key Performance Indicators)", "Monitoring Pipelines",
        "SLA (Service Level Agreement)", "Supplier Management", "Long Lead Times",
    ],
})


# ----------------------------------------------------------------------
# Theme: inventory service metrics — fill rate, backorder rate, lost sales value  (ops / supply chain)
# ----------------------------------------------------------------------

CONTENT["Fill Rate"] = r"""
What it is
----------

**Fill Rate** is a supply-chain and inventory metric measuring the **percentage of customer demand
met immediately from available stock, without delay**. It answers: *of everything customers wanted,
how much did we deliver on time and in full, from stock?* A high fill rate means availability; a low
one means frequent shortages, backorders and delays.

The formulas
------------

It has three common forms. **Order fill rate** divides orders completely fulfilled by total orders;
**line fill rate** divides order lines completely fulfilled by total order lines; and **unit fill
rate** divides units delivered on the first shipment by total units ordered — each expressed as a
percentage,

.. math::

   \text{Fill Rate (unit)} = \frac{\text{Units delivered on first shipment}}{\text{Total Units Ordered}} \times 100\%.

A worked example
----------------

A week's demand is 1,000 units; the warehouse ships 970 immediately and backorders the remaining 30.
The fill rate is 970 / 1,000 = **97%** — 97% of demand satisfied instantly from stock.

Why it matters, and its mirror
--------------------------------

Fill rate drives **customer satisfaction** and **revenue**, signals forecasting and replenishment
health, and is benchmarked at **95-98%** across many industries. It is the exact **complement of the
stockout rate**: :math:`\text{Fill Rate} = 100\% - \text{Stockout Rate}`.
"""

CONTENT["Backorder Rate"] = r"""
What it is
----------

**Backorder Rate** is a supply-chain KPI measuring the **proportion of customer demand that cannot be
filled immediately** and must be placed on backorder — **delayed** fulfillment, not necessarily lost.
A high rate points to frequent shortages and weak planning; a low one to efficient inventory and
satisfied customers.

The formula
-----------

It is computed by units or by orders,

.. math::

   \text{Backorder Rate} = \frac{\text{Backordered Units}}{\text{Total Units Ordered}} \times 100\%,

or, order-based, orders containing backordered items divided by total orders.

A worked example
----------------

Of 1,000 units ordered in a month, 920 ship immediately and 80 are backordered, giving 80 / 1,000 =
**8%** of demand delayed.

Where it sits
-------------

Backorder rate shapes **customer experience**, **revenue** (some backorders convert later, some
cancel), **forecasting accuracy** and **supply-chain health**, and it falls with better forecasting
(ARIMA, Prophet, LSTM), proper safety stock, reliable suppliers and ABC segmentation. In one full
snapshot — 920 shipped, 50 backordered then filled, 30 cancelled — the fill rate is 92%, the backorder
rate 5%, the stockout rate 3%, and those 30 cancellations become **lost sales value**.
"""

CONTENT["Lost Sales Value"] = r"""
What it is
----------

**Lost Sales Value** is the **monetary value of sales that could not be realised** because products
were out of stock or unavailable when customers wanted them. It puts the **dollar impact** on
stockouts, going beyond the percentages of stockout rate or fill rate. High lost-sales value means
serious revenue leakage; low means sound inventory and demand planning.

The formula
-----------

.. math::

   \text{Lost Sales Value} = \text{Unfulfilled Units} \times \text{Unit Selling Price},

or equivalently the total demand value (demand units times price) minus the actual sales value
(fulfilled units times price).

A worked example
----------------

A customer wants 500 units but only 450 are in stock, so 50 go unfilled. At ``$20`` each, the lost
sales value is 50 × 20 = ``$1,000`` — a thousand dollars of revenue forgone to the stockout.

Why it matters
--------------

It shows the money lost **directly**, drives **customer-loyalty** concerns, informs the **inventory
cost versus service level** trade-off, and prioritises the SKUs whose stockouts hurt most (typically
high-margin ones). Alongside its percentage cousins — stockout rate (how *often*) and fill rate (how
*much* is filled) — lost sales value answers how much revenue is actually lost: a snapshot might read
5% stockout, 95% fill, and ``$50,000`` lost.
"""

MINDMAP.update({
    "Fill Rate": [
        "Backorder Rate", "Stockout Rate", "Lost Sales Value", "Safety Stock",
        "Ops Health Dashboard", "SLA (Service Level Agreement)",
    ],
    "Backorder Rate": [
        "Fill Rate", "Stockout Rate", "Lost Sales Value", "Safety Stock",
        "Ops Health Dashboard", "SLA (Service Level Agreement)",
    ],
    "Lost Sales Value": [
        "Stockout Rate", "Fill Rate", "Backorder Rate", "Safety Stock",
        "Ops Health Dashboard", "SLA Breach Rate",
    ],
})


# ----------------------------------------------------------------------
# Theme: stockout metric + forecasting methods (Prophet, LSTM)  (ops / repr / drift)
# ----------------------------------------------------------------------

CONTENT["Stockout Rate"] = r"""
What it is
----------

**Stockout Rate** is a supply-chain and inventory metric measuring **how often items are out of stock
when there is demand** — the percentage of demand that could not be filled because the product was
unavailable. It is a direct reading of **service-level** performance: a low rate means shelves stay
stocked, a high rate means customers keep hitting empty ones.

The formula
-----------

It is computed by demand or by orders,

.. math::

   \text{Stockout Rate} = \frac{\text{Unfulfilled Units}}{\text{Total Demand}} \times 100\%,

or, order-based, the share of orders that hit at least one stockout.

A worked example
----------------

A month's demand is 1,000 units; inventory covers 950 and 50 go unfilled. The stockout rate is
50 / 1,000 = **5%** — one order in twenty meets an empty shelf.

Why it matters, and reducing it
---------------------------------

Frequent stockouts **push customers to competitors**, forfeit sales, and flag weak forecasting, so the
metric captures the tension between **service level** and **holding cost**. It falls with better
**demand forecasting** (ARIMA, Prophet, LSTM), adequate **safety stock**, shorter vendor **lead
times**, **multi-location** inventory and **real-time tracking**. It is the exact **complement of the
fill rate**.
"""

CONTENT["Prophet — Time Series Forecasting by Facebook (Meta)"] = r"""
What it is
----------

**Prophet** is an **open-source forecasting library** from Facebook (now Meta) that makes time-series
forecasting **simple, scalable and interpretable** — designed for business data with **trends,
seasonality and holidays**, and usable without deep statistical expertise.

The decomposable model
------------------------

Prophet models a series as a sum of interpretable components,

.. math::

   y(t) = g(t) + s(t) + h(t) + \varepsilon_t,

where :math:`g(t)` is the **trend** (linear, or logistic with saturation
:math:`g(t) = \frac{C}{1 + \exp(-k(t - m))}`, plus automatic **changepoints**), :math:`s(t)` is
**seasonality** (a Fourier series for weekly, yearly or custom cycles), :math:`h(t)` captures
**holidays and events** from a supplied list, and :math:`\varepsilon_t` is noise.

Strengths and limits
--------------------

Prophet is **user-friendly** (just a ``ds``/``y`` DataFrame), **interpretable** (each component is
separable), **robust** to missing data and outliers, detects changepoints automatically, and
**scales** across many series. Its limits follow from its **additive** design: it does not model
autoregressive correlations, it suits **daily/weekly/monthly** business data rather than
high-frequency signals, and it is **less powerful than LSTMs or Transformers** on complex patterns.

In practice
-----------

.. code-block:: python

   from prophet import Prophet

   # df has two columns: ds (datestamp) and y (value)
   model = Prophet()
   model.fit(df)

   future = model.make_future_dataframe(periods=90)
   forecast = model.predict(future)
   model.plot(forecast)
   model.plot_components(forecast)
"""

CONTENT["LSTM — Long Short-Term Memory Networks"] = r"""
What it is
----------

An **LSTM (Long Short-Term Memory network)** is a special **recurrent neural network** for sequential
data that overcomes the **vanishing- and exploding-gradient** problem of vanilla RNNs. It adds a
**gated memory** structure that learns what to **keep, update and forget** across long sequences.

The gated cell
--------------

A **cell state** runs through the sequence like a conveyor belt, and three **gates** — sigmoids in
:math:`[0, 1]` that decide *how much* to let through — govern it: a **forget gate**
:math:`f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)`, an **input gate** :math:`i_t` with candidate
values :math:`\tilde{C}_t = \tanh(\cdot)`, and an **output gate** :math:`o_t`. The updates are

.. math::

   C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \qquad h_t = o_t \odot \tanh(C_t),

with :math:`C_t` the long-term cell state and :math:`h_t` the short-term hidden state.

Variants
--------

Common variants include the **bidirectional LSTM** (reads a sequence forwards and backwards), the
**stacked LSTM** (several layers for depth), the **peephole LSTM** (gates can see the cell state), and
the **GRU**, a simpler cousin that merges the forget and input gates.

Uses, strengths, weaknesses
---------------------------

LSTMs power **NLP** (text generation, translation, sentiment), **speech recognition**, **time-series
forecasting** (demand, stock prices, anomalies) and **control systems**. They capture dependencies
across **50-100+ steps**, but they are **computationally heavy**, slower than GRUs, and now often
**outperformed by Transformers** on very long sequences.
"""

MINDMAP.update({
    "Stockout Rate": [
        "Fill Rate", "Backorder Rate", "Lost Sales Value", "Safety Stock",
        "Ops Health Dashboard", "Prophet — Time Series Forecasting by Facebook (Meta)",
    ],
    "Prophet — Time Series Forecasting by Facebook (Meta)": [
        "LSTM — Long Short-Term Memory Networks",
        "ARIMA (AutoRegressive Integrated Moving Average)", "Time Series",
        "Bayesian Time Series", "Signal Processing", "Stockout Rate",
    ],
    "LSTM — Long Short-Term Memory Networks": [
        "Prophet — Time Series Forecasting by Facebook (Meta)",
        "ARIMA (AutoRegressive Integrated Moving Average)", "Autoencoder", "Embedding",
        "Bayesian Neural Networks (BNNs)", "Time Series",
    ],
})


# ----------------------------------------------------------------------
# Theme: classical time-series — ARIMA, seasonality  (concepts / forecasting)
# ----------------------------------------------------------------------

CONTENT["ARIMA (AutoRegressive Integrated Moving Average)"] = r"""
What it is
----------

**ARIMA (AutoRegressive Integrated Moving Average)** is a classical statistical model for
**time-series forecasting** that predicts future values from a series' **own past**. It fuses three
ideas — **autoregression (AR)**, **integration / differencing (I)** and **moving average (MA)** — and
describes a series by its **autocorrelations** rather than by explicit trend or seasonality. A model
is written **ARIMA(p, d, q)**.

The three parameters
--------------------

Each letter is one parameter. **p** is the **AR order** — how many lagged past *values* the current
value is regressed on. **d** is the **differencing order** — how many times the series is differenced
to make it **stationary** (removing trend). **q** is the **MA order** — how many past *error* terms
feed the forecast. In backshift form,

.. math::

   \phi_p(B)\,(1 - B)^d\, y_t = \theta_q(B)\,\varepsilon_t,

where :math:`B` is the backshift operator (:math:`B y_t = y_{t-1}`), :math:`\phi_p` and
:math:`\theta_q` are the AR and MA polynomials, and :math:`(1 - B)^d` applies the differencing.
Setting parameters to zero recovers the simpler **AR**, **MA** and **ARMA** models.

Building one (Box-Jenkins)
----------------------------

The Box-Jenkins recipe has three stages. First, make the series **stationary** by differencing,
checked with a unit-root test such as the **Dickey-Fuller** test. Next, pick **p** and **q**: the
**ACF** (autocorrelation function) guides **q**, the **PACF** (partial autocorrelation function)
guides **p**, and among candidates you choose the one with the lowest **AIC** (or BIC). Finally,
**validate the residuals** — they should be uncorrelated white noise; if not, revisit the orders.

Strengths, limits, and SARIMA
-------------------------------

ARIMA is **flexible** and **interpretable** for **linear, univariate** series — finance, demand,
sales — and gives stable longer-term forecasts. But it **assumes a linear autocorrelation structure**,
**requires stationarity**, and struggles with non-linear patterns where **LSTMs or Transformers**
do better. For periodic data, the **SARIMA** extension adds seasonal terms, written
:math:`\text{ARIMA}(p, d, q)(P, D, Q)_m` with season length :math:`m`.
"""

CONTENT["Seasonality"] = r"""
What it is
----------

**Seasonality** is a pattern in a time series that **repeats at fixed, regular intervals** — driven by
seasonal factors such as the time of year, the month, the day of the week or the hour of the day.
Unlike a **trend** (a long-term rise or fall), seasonality is **periodic**, recurring with a fixed
frequency. In a decomposition, a series splits into trend, seasonal and residual parts,
:math:`y_t = T_t + S_t + \varepsilon_t`, with :math:`S_t` the seasonal component.

Examples
--------

Retail sales spike every December, ice-cream demand rises each summer, website traffic dips every
weekend, and electricity load peaks on hot afternoons. In each case the same shape returns on a
predictable cycle.

Detecting it
------------

Seasonality shows up as a **repeating shape** in a line plot, and is confirmed with **seasonal
subseries** or **decomposition** plots and with the **autocorrelation function (ACF)**, which spikes at
the **seasonal lag** — for example lag 12 for monthly data with yearly seasonality.

Handling it in models
---------------------

Several tools absorb it: **seasonal differencing** removes it, **SARIMA** adds seasonal
:math:`(P, D, Q)_m` terms, **Prophet** fits it with a Fourier series, and simpler models use
**seasonal dummy variables**. Getting seasonality right is essential for accurate forecasting —
ignoring it leaves systematic, repeating errors in the residuals.
"""

MINDMAP.update({
    "ARIMA (AutoRegressive Integrated Moving Average)": [
        "Prophet — Time Series Forecasting by Facebook (Meta)",
        "LSTM — Long Short-Term Memory Networks", "Time Series", "Bayesian Time Series",
        "Seasonality", "Temporal autocorrelation (Serial Correlation)",
    ],
    "Seasonality": [
        "ARIMA (AutoRegressive Integrated Moving Average)",
        "Prophet — Time Series Forecasting by Facebook (Meta)", "Time Series", "Seasonal Lag",
        "Signal Processing", "Temporal autocorrelation (Serial Correlation)",
    ],
})


# ----------------------------------------------------------------------
# Theme: explainability trio — counterfactuals, LIME, SHAP  (xai)
# ----------------------------------------------------------------------

CONTENT["Counterfactual Explanations"] = r"""
What it is
----------

A **counterfactual explanation** shows how a prediction would change **if the input were different**,
answering: *what minimal change to this input would flip the model's decision?* It borrows the causal
idea of a counterfactual — *what would have happened if ...?*

The formal definition
---------------------

Given an instance :math:`x` with prediction :math:`f(x)`, a counterfactual is an alternative
:math:`x'` that changes the outcome yet stays as close as possible to :math:`x`:

.. math::

   \min_{x'} \; d(x, x') \quad \text{s.t.} \quad f(x') = y_{\text{desired}},

with :math:`f(x') \neq f(x)` and :math:`d(\cdot)` a distance metric (L1, L2 or a feature-specific
cost).

A worked example
----------------

For a **denied** loan, a counterfactual might say: *if income rose by* ``$5,000``\ *, or the credit
score were 650 instead of 600, the loan would be approved* — actionable **recourse**. Good
counterfactuals are **valid** (they flip the prediction), **proximal** and **sparse** (few, small
changes), **actionable** (realistic and controllable — "get older" is not), and **diverse** (several
options).

Methods and uses
----------------

They are generated by **optimisation**, **gradient search** (for differentiable models) or
**generative models** (VAE, GAN) for realism, via libraries like ``alibi`` and ``dice-ml``. They power
**user recourse** ("what must I change?"), **fairness audits** (do minority groups need unfairly large
changes for the same decision?) and **model debugging** — but they can be **unrealistic** (e.g. "if
gender changed"), are **non-unique**, and need **domain constraints** to stay valid and fair.
"""

CONTENT["LIME (Local Interpretable Model-agnostic Explanations)"] = r"""
What it is
----------

**LIME (Local Interpretable Model-agnostic Explanations)** explains an **individual prediction** of
any ML model by approximating the black box **locally** — around the instance of interest — with a
simpler, interpretable model such as linear regression. It answers: *why did the model predict this
for this example?*

How it works
------------

Six steps. Take the instance to explain; create **perturbed samples** by slightly varying its
features; collect the black-box **predictions** for those samples; **weight** each sample by proximity
to the original; fit a **simple interpretable model** (linear or tree) on that local neighbourhood;
and read its **coefficients** as the feature contributions.

A worked example
----------------

For a **denied** loan, perturbing income, age and debt and fitting a local linear model might yield
Income **-0.4**, high debt **+0.3** and employment length **+0.1** — low income plus high debt pushed
the decision toward denial.

Strengths, limits, and SHAP
-----------------------------

LIME is **model-agnostic**, sharply **local** (one prediction at a time) and **human-friendly**, but
it is **unstable** (different perturbations give different explanations), only **locally faithful**,
**computationally expensive**, and shaky under **correlated features**. Against SHAP: LIME fits
**local surrogate** models (faster, less stable) while SHAP uses **game theory** (local *and* global,
more stable, an exact decomposition).

In practice
-----------

.. code-block:: python

   import lime.lime_tabular
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier

   X, y = load_iris(return_X_y=True)
   model = RandomForestClassifier().fit(X, y)

   explainer = lime.lime_tabular.LimeTabularExplainer(
       X,
       feature_names=["f1", "f2", "f3", "f4"],
       class_names=["setosa", "versicolor", "virginica"],
       discretize_continuous=True,
   )
   exp = explainer.explain_instance(X[0], model.predict_proba, num_features=2)
   exp.show_in_notebook()
"""

CONTENT["SHAP (SHapley Additive exPlanations)"] = r"""
What it is
----------

**SHAP (SHapley Additive exPlanations)** is a unified framework for explaining the predictions of
**any** machine-learning model, built on **Shapley values** from cooperative game theory (Lloyd
Shapley, 1953). Treat each feature as a **player** in a game and the prediction as the **payout**;
SHAP assigns each feature a **fair share** of the contribution — how much it pushed the prediction up
or down from a baseline.

The additive decomposition
----------------------------

A prediction is decomposed **exactly** into per-feature contributions,

.. math::

   \hat{y} = \phi_0 + \sum_{i=1}^{M} \phi_i,

where :math:`\phi_0` is the **baseline** (the average prediction when no features are known) and
:math:`\phi_i` is the **SHAP value** of feature :math:`i`.

A worked example
----------------

A loan-approval probability of **0.8** against a baseline of **0.5** might break down as Income
**+0.2**, Employment history **+0.1**, Debt ratio **0** and Age **0**, so
:math:`0.8 = 0.5 + 0.2 + 0.1`. Income and employment history raised the approval; the other features
were neutral.

Strengths, visuals, limits
----------------------------

SHAP is **consistent** (a fair allocation), works both **locally and globally**, and is
**model-agnostic or model-specific** (tree models, deep nets, linear), with **force**, **summary** and
**dependence** plots. Its costs: exact Shapley values are **exponential** in the number of features
(SHAP uses approximations), explanations can be **misused** out of context, and **correlated
features** are hard to attribute fairly.

In practice
-----------

.. code-block:: python

   import shap
   import xgboost as xgb
   from sklearn.datasets import fetch_california_housing

   X, y = fetch_california_housing(return_X_y=True, as_frame=True)
   model = xgb.XGBRegressor().fit(X, y)

   explainer = shap.Explainer(model, X)
   shap_values = explainer(X)

   shap.summary_plot(shap_values, X)   # global feature importance
   shap.plots.force(shap_values[0])    # local explanation for one row
"""

MINDMAP.update({
    "Counterfactual Explanations": [
        "SHAP (SHapley Additive exPlanations)",
        "LIME (Local Interpretable Model-agnostic Explanations)", "Causal Effect",
        "Causal Inference", "Post-hoc Explainability", "Equalized Odds (Fairness)",
    ],
    "LIME (Local Interpretable Model-agnostic Explanations)": [
        "SHAP (SHapley Additive exPlanations)", "Counterfactual Explanations",
        "Post-hoc Explainability", "Feature Values", "Discriminatory Power", "Deep Ensembles",
    ],
    "SHAP (SHapley Additive exPlanations)": [
        "LIME (Local Interpretable Model-agnostic Explanations)", "Counterfactual Explanations",
        "Post-hoc Explainability", "Feature Values", "Discriminatory Power", "Deep Ensembles",
    ],
})


# ----------------------------------------------------------------------
# Theme: interpretability spectrum — decision trees, deep ensembles, post-hoc XAI  (xai / training)
# ----------------------------------------------------------------------

CONTENT["Decision Trees"] = r"""
What it is
----------

A **decision tree** is a supervised-learning algorithm that splits data into branches by **feature
values**, forming a tree. Each **internal node** is a decision (a feature and a threshold) and each
**leaf** a prediction (a class label or a number) — think of it as a flowchart: ask questions, follow
branches, reach a prediction. It comes in two flavours: **classification trees** (discrete labels) and
**regression trees** (continuous values).

How it learns, and splitting
------------------------------

Training is recursive: start with all data at the **root**, evaluate candidate **splits** for each
feature, keep the split that best separates the data (minimises impurity), and repeat until a
**stopping rule** (max depth, minimum samples per leaf). Classification trees split by **Gini
impurity** or **entropy** (information gain); regression trees by **MSE reduction**. The Gini impurity
at a node is

.. math::

   G = 1 - \sum_{k} p_k^2,

where :math:`p_k` is the proportion of class :math:`k` at that node.

Strengths and weaknesses
--------------------------

Trees are **easy to interpret and visualise**, handle **mixed numeric and categorical** features,
capture **nonlinear boundaries and interactions**, and need no feature scaling. But a deep tree
**overfits**, is **unstable** (a small data change reshapes it), splits **greedily** (it can miss the
global optimum), and is **weaker alone** than an ensemble.

From one tree to many
-----------------------

**Pruning** cuts back branches to curb overfitting; **random forests** bag many trees; and
**gradient-boosted trees** (XGBoost, LightGBM, CatBoost) build trees sequentially to correct earlier
errors.

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.tree import DecisionTreeClassifier, export_text

   X, y = load_iris(return_X_y=True)
   tree = DecisionTreeClassifier(max_depth=3).fit(X, y)

   print(export_text(tree, feature_names=["sepal_length", "sepal_width",
                                          "petal_length", "petal_width"]))
"""

CONTENT["Deep Ensembles"] = r"""
What it is
----------

A **deep ensemble** trains several neural networks **independently** and **aggregates** their
predictions, improving both **accuracy** and **uncertainty estimation**. Each member shares the
architecture but starts from a **different random initialisation** (and data order), so the networks
settle into different modes of the loss landscape; averaging their outputs cancels errors and reduces
variance:

.. math::

   \bar{p}(y \mid x) = \frac{1}{M} \sum_{m=1}^{M} p_{\theta_m}(y \mid x).

Why it works, and uncertainty
-------------------------------

Because independently-initialised networks explore **different functions**, their **disagreement** is
informative. Deep ensembles decompose predictive uncertainty into **aleatoric** (data noise) and
**epistemic** (model) components and produce well-calibrated **predictive intervals** — rivalling
**Bayesian neural networks** while being far simpler to implement.

Calibration and cost
--------------------

They usually still need **calibration** (for example temperature scaling), especially under
**distribution shift**. The main drawback is that cost grows **linearly** with the number of members
:math:`M`, which motivates efficient variants such as **BatchEnsemble**, **snapshot ensembles**, and
spreading members over time.

Where it's used
---------------

Deep ensembles shine wherever **reliable confidence** matters as much as the point prediction —
climate downscaling, **robotic perception** and safe human-robot interaction, and **low-data transfer
learning**.
"""

CONTENT["Post-hoc Explainability"] = r"""
What it is
----------

**Post-hoc explainability** means explaining a model's behaviour **after training**, without changing
its internal structure — making **black-box models** (deep nets, ensembles, gradient boosting)
interpretable. *Post-hoc* means *after the fact*: you do not train the model to be interpretable, you
**analyse its outputs afterward**. It contrasts with **intrinsically interpretable** models (linear
regression, small decision trees) that are transparent by design.

Why it matters
--------------

Many high-performing models are **opaque**, yet users, regulators and businesses need to know **why** a
prediction was made — for **debugging**, **trust and transparency**, and **compliance** (finance,
healthcare, GDPR / AI Act).

The techniques
--------------

The toolkit spans **feature importance** (global permutation or gain-based, and local),
**surrogate models** (a simple tree approximating the black box), the local methods **LIME** and
**SHAP**, **visualisations** — partial-dependence plots, ICE plots, and saliency maps / Grad-CAM for
images — and **counterfactual explanations**. For a black-box loan model, SHAP might flag low income
and short employment, while a counterfactual says "two more years of employment would flip the
decision."

Limitations
-----------

Post-hoc explanations are **approximations** of the true model logic, risk being **misleading** (the
faithfulness problem), can be **computationally expensive**, and are **diagnostic only** — not a
substitute for fair training practices.
"""

MINDMAP.update({
    "Deep Ensembles": [
        "Decision Trees", "Post-hoc Explainability", "Bayesian Neural Networks (BNNs)",
        "Model Stability", "Uplift Random Forests", "SHAP (SHapley Additive exPlanations)",
    ],
    "Post-hoc Explainability": [
        "SHAP (SHapley Additive exPlanations)",
        "LIME (Local Interpretable Model-agnostic Explanations)", "Counterfactual Explanations",
        "Decision Trees", "Feature Values", "Discriminatory Power",
    ],
    "Decision Trees": [
        "Post-hoc Explainability", "Deep Ensembles", "Uplift Random Forests",
        "SHAP (SHapley Additive exPlanations)", "Discriminatory Power",
        "Bayesian Neural Networks (BNNs)",
    ],
})


# ----------------------------------------------------------------------
# Theme: model deployment format — ONNX  (platforms / mlops)
# ----------------------------------------------------------------------

CONTENT["ONNX (Open Neural Network Exchange)"] = r"""
What it is
----------

**ONNX (Open Neural Network Exchange)** is an **open-source, standardised format** for representing
machine-learning models — both deep-learning and traditional ML. It acts as a **universal
translator**: build a model in one framework (PyTorch, TensorFlow, scikit-learn) and **deploy it in
another** environment optimised for inference, without rewriting or retraining. Originally called
*Toffee* and built by the PyTorch team at Facebook, it was renamed ONNX in September 2017 and is now
backed by Microsoft, IBM, Intel, AMD, Arm, Qualcomm and others.

How it represents a model
---------------------------

An ONNX model is an **extensible computation graph** — a directed acyclic graph whose **nodes** are
**operators** (convolution, pooling, activation) with typed inputs and outputs. It defines a set of
**built-in operators** and **standard data types**, and serialises the network structure (layers,
connections) and parameters (weights, biases) in a **framework-agnostic** way using **Protocol
Buffers** as the container format. The focus is on **inference** (evaluation), not training.

The workflow
------------

The lifecycle is **train, export, run**. Train in any framework, **export** to a single ``.onnx`` file
(for example with ``torch.onnx.export``), then execute it with a runtime such as **ONNX Runtime**,
whose **execution providers** target CPUs, GPUs and specialised accelerators. The runtime applies
**graph optimisations** — node fusion, constant folding — that cut inference latency.

Why it matters
--------------

ONNX delivers **framework interoperability** (no ecosystem lock-in), **hardware optimisation** (one
``.onnx`` runs on NVIDIA GPUs, Intel CPUs or mobile NPUs via tools like OpenVINO and CoreML),
**faster inference**, and **simplified deployment** — one delivery format across cloud, edge, mobile
and even the browser (ONNX Runtime Web on WebGL/WebAssembly). Its main limitation is that some
**proprietary or very new operators** are not yet supported.

.. code-block:: python

   import torch
   import onnxruntime as ort

   # Export a trained PyTorch model to ONNX
   dummy = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy, "model.onnx",
                     input_names=["input"], output_names=["output"],
                     dynamic_axes={"input": {0: "batch_size"}})

   # Run inference with ONNX Runtime
   session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
   outputs = session.run(None, {"input": input_array})
"""

MINDMAP.update({
    "ONNX (Open Neural Network Exchange)": [
        "Quantization", "Caching", "Neural Networks", "TPU Clusters", "Latency Guardrails",
        "Monitoring Pipelines",
    ],
})


# ----------------------------------------------------------------------
# Theme: trustworthy evaluation & data quality — DeLong, dataset shift, label noise  (evaluation / drift)
# ----------------------------------------------------------------------

CONTENT["DeLong's Test"] = r"""
What it is
----------

**DeLong's test** is a **non-parametric statistical test** for comparing the **ROC-AUCs of two
correlated models** — answering *is model A's AUC significantly better than model B's?* Proposed by
DeLong, DeLong & Clarke-Pearson (1988), it matters because two models are usually evaluated on the
**same test set**, so their AUCs are **correlated** and their difference must be judged with a p-value,
not by eye.

Why you need it
---------------

AUCs are **random variables** — they depend on the sample — so a raw ``AUC_A - AUC_B`` gap could be
noise. DeLong's test supplies the **standard error** of each AUC, the SE of their **difference**, and a
**z-statistic and p-value**.

How it works
------------

It treats AUC as the probability that a random positive outranks a random negative, and uses
**U-statistics** to estimate the variances and covariances of those pairwise comparisons, giving

.. math::

   z = \frac{\text{AUC}_1 - \text{AUC}_2}{\text{SE}(\text{AUC}_1 - \text{AUC}_2)},

which is referred to the standard normal for a p-value.

A worked example
----------------

Model A scores **AUC = 0.88** and Model B **0.84**. The difference of **0.04** with an SE of **0.015**
gives :math:`z \approx 2.67` and :math:`p \approx 0.0076` — so A is **significantly better**
(p < 0.01).

.. code-block:: python

   # delong_roc_test is a small helper (from a gist/package)
   from delong import delong_roc_test

   # y_true = true labels; y_pred1, y_pred2 = the two models' scores
   p_value = delong_roc_test(y_true, y_pred1, y_pred2)
   print("p-value:", p_value)

Alternatives
------------

Other options include **bootstrap confidence intervals** (resample and recompute the AUC-difference
distribution), the older, less accurate **Hanley-McNeil** approximation, and **permutation tests**
(shuffle labels to build the null).
"""

CONTENT["Dataset Shift"] = r"""
What it is
----------

**Dataset shift** is when the **training** data distribution differs from the **test / production**
distribution — formally :math:`P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)`. Because a model
learns its patterns from training data, a shift makes it **perform worse in the real world**. It is the
formal **umbrella** over the whole drift family.

The three types
---------------

**Covariate shift**: :math:`P(X)` changes but :math:`P(Y \mid X)` stays — a spam filter trained on old
emails, tested on new ones. **Prior (label) shift**: :math:`P(Y)` changes but :math:`P(X \mid Y)`
stays — fraud is 1% in training but 5% in production. **Concept shift**: :math:`P(Y \mid X)` itself
changes — the meaning of a label evolves, the hardest case to handle.

Detecting it
------------

Use **statistical tests** (KS, chi-square, PSI, KL-divergence), a **train-versus-test discriminator**
(if a classifier can tell the two sets apart, they differ), and **production monitoring** of accuracy,
AUC and calibration.

Coping
------

**Reweight** samples by importance, :math:`w(x) = \frac{P_{\text{test}}(x)}{P_{\text{train}}(x)}`, for
covariate shift; **resample** to match real prevalence; apply **domain adaptation**; **retrain
continually**; or build **robust, invariant** models. A model trained on one hospital's older, less
diverse patients loses accuracy on another's younger, more diverse population.
"""

CONTENT["Label Noise"] = r"""
What it is
----------

**Label noise** is **incorrect or unreliable labels** in a dataset — training examples assigned the
wrong class or target, arising from human labellers, weak heuristics or imperfect sensors (a tweet "I
love this" marked **negative**, or a misrecorded diagnosis).

The three types
---------------

**Random (uniform) noise**: labels flipped independently of the features (say 10% at random).
**Class-conditional noise**: mislabeling depends on the class — "cat" is confused for "dog" more often
than for "car". **Feature-dependent (systematic) noise**: ambiguous or low-quality inputs are
mislabeled more — blurry dog photos read as cats.

Why it hurts
------------

It **degrades training** (models learn wrong patterns), **miscalibrates** probabilities, and
**distorts evaluation** (a noisy test set makes metrics meaningless). Tell-tale signs: training
accuracy never reaching 100%, a **stalling loss**, memorisation of noise with low validation
performance, and **high disagreement among annotators**.

Coping
------

**Clean** the data (multiple annotators, keep high-agreement labels); use **noise-robust losses** (MAE,
generalised cross-entropy) with regularisation and early stopping; **model the noise** with a
transition matrix; combine a small clean set with a large noisy one (**weak / semi-supervised**); and
always keep a **clean gold-standard evaluation set** with robust metrics like AUC.
"""

MINDMAP.update({
    "DeLong's Test": [
        "Multiclass AUROC", "Discriminatory Power", "Statistical Power", "Dataset Shift",
        "Evaluation Set", "Model KPIs (Key Performance Indicators)",
    ],
    "Dataset Shift": [
        "Covariate Drift (a.k.a. Covariate Shift)", "Concept Drift", "Data Drift",
        "PSI (Population Stability Index)", "KS Statistic (Kolmogorov–Smirnov Statistic)",
        "Label Noise",
    ],
    "Label Noise": [
        "Dataset Shift", "Evaluation Set", "Weak Supervision", "Full Annotation",
        "Discriminatory Power", "Multiclass AUROC",
    ],
})


# ----------------------------------------------------------------------
# Theme: classical supervised-learning models — SVM, logistic regression, neural networks  (concepts / training)
# ----------------------------------------------------------------------

CONTENT["Support Vector Machines (SVMs)"] = r"""
What it is
----------

A **support vector machine** is a **supervised max-margin** algorithm for classification (and, as SVR,
regression). It finds the **optimal separating hyperplane** — the decision boundary that **maximizes the
margin**, the distance to the nearest points of each class. Developed from the work of Vapnik and
Chervonenkis, with the soft-margin form due to Cortes & Vapnik (1995).

.. math::

   \mathbf{w}^\top \mathbf{x} + b = 0 \quad\text{(the separating hyperplane)}

Margin and support vectors
--------------------------

The **support vectors** are the training points closest to the boundary — and they **alone** define it
(remove any other point and nothing changes). A **hard margin** separates the classes perfectly; a
**soft margin** tolerates some violations through **slack variables** :math:`\xi_i`, with the penalty
**C** trading margin width against misclassification (large ``C`` → stricter, narrower margin; small
``C`` → wider, more tolerant). The objective minimizes

.. math::

   \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i,

where the per-point cost is the **hinge loss** :math:`\max(0,\, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))`.

The kernel trick
----------------

When data is not linearly separable, a **kernel** maps it into a higher-dimensional space where it is —
**without ever computing the coordinates**, using only pairwise dot products. Common kernels are
**linear**, **polynomial**, **RBF** (the most popular) and **sigmoid**; the choice trades accuracy
against complexity and compute.

When to use it
--------------

SVMs are strong on **high-dimensional** data (text, images, bioinformatics), **resilient to noise**, and
guard against overfitting, but are **expensive on very large datasets** and sensitive to kernel choice.

.. code-block:: python

   from sklearn.svm import SVC

   clf = SVC(kernel="rbf", C=1.0)   # RBF kernel, soft margin
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
"""

CONTENT["Logistic Regression"] = r"""
What it is
----------

**Logistic regression** is a **linear model for binary classification** that predicts the **probability**
of the positive class. Despite *regression* in the name it **classifies**: it outputs a probability, then
a **threshold** (usually 0.5) assigns the label.

The sigmoid and log-odds
------------------------

It passes a linear combination through the **sigmoid**, squashing any real number into :math:`(0, 1)`:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \theta_0 + \boldsymbol{\theta}^\top \mathbf{x}.

Equivalently, the **log-odds** (logit) — the natural log of the odds :math:`p/(1-p)` — is **linear in the
features**, which is what makes the coefficients interpretable:

.. math::

   \log \frac{p}{1 - p} = \theta_0 + \boldsymbol{\theta}^\top \mathbf{x}.

Fitting by maximum likelihood
-----------------------------

Coefficients are chosen to **maximize the likelihood** of the observed labels — equivalently, to minimize
the **log loss** (negative log-likelihood), a **convex** objective solved by gradient-based methods:

.. math::

   \mathcal{L} = -\sum_i \big[\, y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i) \,\big].

Assumptions and multiclass
--------------------------

It assumes a **linear log-odds** relationship, **few extreme outliers**, and enough data; it extends to
several classes via **one-vs-rest** or **multinomial (softmax)**. Simple, fast and interpretable, it is a
workhorse for spam, fraud and medical diagnosis.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression

   clf = LogisticRegression(max_iter=1000)
   clf.fit(X_train, y_train)
   proba = clf.predict_proba(X_test)[:, 1]   # P(class 1)
"""

CONTENT["Neural Networks"] = r"""
What it is
----------

A **neural network** is a model of interconnected **neurons** arranged in **layers** — an **input**
layer, one or more **hidden** layers, and an **output** layer. Each neuron computes a **weighted sum** of
its inputs, adds a **bias**, and passes the result through a **non-linear activation function**, letting
the network model complex relationships.

Weights, biases, activations
----------------------------

**Weights** set the strength of each connection and **biases** shift the activation; the **non-linearity**
is what lets stacked layers represent functions a linear model cannot. Common activations are **ReLU**,
**sigmoid** and **tanh**. A single unit computes

.. math::

   a = \phi\!\left( \sum_i w_i x_i + b \right),

where :math:`\phi` is the activation function.

Forward pass and backpropagation
--------------------------------

In the **forward pass**, inputs flow layer by layer to an output, and a **loss function** (MSE for
regression, cross-entropy for classification) measures the error. **Backpropagation** then applies the
**chain rule** to compute the gradient of that loss with respect to every weight, propagating the error
**backward** from output to input; **gradient descent** updates the weights, with the **learning rate**
:math:`\eta` setting the step size:

.. math::

   w \leftarrow w - \eta \, \frac{\partial \mathcal{L}}{\partial w}.

In practice
-----------

Depth and width are chosen for the task, **validation** data guards against overfitting, and frameworks
like **PyTorch**, **TensorFlow** and **Keras** implement backpropagation automatically.

.. code-block:: python

   from sklearn.neural_network import MLPClassifier

   clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500)
   clf.fit(X_train, y_train)
"""

MINDMAP.update({
    "Support Vector Machines (SVMs)": [
        "Logistic Regression", "Neural Networks", "Decision Trees", "Linear Models",
        "Multiclass AUROC", "Discriminatory Power",
    ],
    "Logistic Regression": [
        "Linear Models", "Support Vector Machines (SVMs)", "Neural Networks", "Decision Trees",
        "Multiclass AUROC", "Discriminatory Power",
    ],
    "Neural Networks": [
        "Support Vector Machines (SVMs)", "Logistic Regression", "Decision Trees",
        "LSTM — Long Short-Term Memory Networks", "Deep Ensembles", "Autoencoder",
    ],
})


# ----------------------------------------------------------------------
# Theme: probabilistic forecasting backbone — point vs probabilistic forecasts, proper scoring  (evaluation / inference)
# ----------------------------------------------------------------------

CONTENT["Point Forecasts"] = r"""
What it is
----------

A **point forecast** gives a **single predicted value** for each future period — the **deterministic**
forecast, with all the outcome probability mass placed on one number. Historically the dominant approach
because it is **easy to interpret and act on**: a demand at 5 p.m., a price in the day-ahead market.

What it hides
-------------

A point forecast says nothing about **uncertainty** — two forecasts can share the *same* point estimate
yet imply very different **risk**. The value reported is usually a **summary** of an underlying
predictive distribution (typically its **mean** or **median**). As the saying goes, *it is better to be
vaguely right than exactly wrong*.

Scoring and context
-------------------

Point forecasts are scored against the realized value with **error metrics** — MAE, MSE / RMSE — which
reward closeness but ignore calibration. Probabilistic forecasting does not *eliminate* point forecasts;
it **places them in context** as one functional (mean, median, a quantile) of the full distribution.
"""

CONTENT["Probabilistic Forecasts"] = r"""
What it is
----------

A **probabilistic forecast** produces a **full predictive distribution** — a range of possible outcomes
together with their probabilities — rather than a single value. By **quantifying uncertainty**, it lets
decision-makers weigh risk instead of betting on one number.

Forms and quality
-----------------

It can be expressed as **prediction intervals**, **quantiles**, a **density**, or **samples**. Quality is
judged on two axes: **calibration** — do the stated probabilities match observed frequencies? — and
**sharpness** — are the intervals as **tight** as possible *subject to* being calibrated? Sharp but
miscalibrated is misleading; calibrated but diffuse is uninformative.

Why it matters
--------------

Probabilistic forecasts support **decisions under uncertainty** — hedging, safety stock, capacity
planning — especially where the **cost of outcomes is asymmetric**. Narrow bands signal agreement; wide
spreads flag where more flexibility or hedging is needed. A point estimate alone cannot convey this.
"""

CONTENT["Strictly Proper Scoring Rules"] = r"""
What it is
----------

A **scoring rule** assigns a numerical score to a **probabilistic forecast** given the outcome that
materializes. It is **proper** if the forecaster's *expected* score is maximized by reporting the **true**
distribution, and **strictly proper** if that maximum is **unique** — attained *only* at the truth:

.. math::

   S(p, q) \;\le\; S(q, q) \quad \text{for all } p, q, \qquad \text{with equality} \iff p = q.

Why it matters
--------------

Strict propriety makes **honesty optimal**: a forecaster cannot improve the expected score by hedging or
shading probabilities away from their true beliefs. That single property is why these rules serve **both**
as **training objectives** (to calibrate probabilistic models) and as **evaluation metrics** (to rank
forecasts fairly). Foundational reference: Gneiting & Raftery (2007).

Common examples
---------------

The **logarithmic score**, which is also **local** (it depends only on the density assigned to what
actually happened):

.. math::

   S(q, x) = \log q(x) \quad\text{(its negative, } -\log q(x)\text{, is the log loss / NLL).}

Others include the **Brier score** for probabilities and the **CRPS** and **pinball (quantile) loss** for
full distributions and quantiles.
"""

MINDMAP.update({
    "Point Forecasts": [
        "Probabilistic Forecasts", "Deterministic forecasts", "Prediction Intervals (PI)",
        "Forecast Error", "Quantile Forecasts", "Time Series Forecasting",
    ],
    "Probabilistic Forecasts": [
        "Point Forecasts", "Deterministic forecasts", "Continuous Probabilistic Forecasts",
        "Prediction Intervals (PI)", "Quantile Forecasts", "Strictly Proper Scoring Rules",
    ],
    "Strictly Proper Scoring Rules": [
        "Probabilistic Forecasts", "Probability Forecasts", "Continuous Probabilistic Forecasts",
        "Point Forecasts", "Quantile Regression", "Forecast Error",
    ],
})


# ----------------------------------------------------------------------
# Theme: distribution representation — quantile forecasts, prediction intervals, quantile regression  (evaluation / inference)
# ----------------------------------------------------------------------

CONTENT["Quantile Forecasts"] = r"""
What it is
----------

A **quantile forecast** expresses the prediction not as one number but as one or more **conditional
quantiles** of the future value's distribution — for example the 10th, 50th and 90th percentiles. The
0.5 level is the **median**; a pair like 0.1 / 0.9 brackets the outcome, saying it is unlikely (:math:`\le`
10%) to fall below the lower or above the upper. Formally the :math:`\tau`-quantile forecast satisfies

.. math::

   \Pr\!\left(Y \le \hat{y}_\tau \mid X\right) = \tau.

Why use them
------------

They convey **uncertainty and asymmetry** directly: you can act differently when under- and over-shooting
carry different costs, **without** assuming a parametric (e.g. Gaussian) shape for the distribution.

How they're scored
------------------

Quantile forecasts are evaluated with the **pinball (quantile) loss**, matched to each level; the
**CRPS** (Continuous Ranked Probability Score) generalizes it by integrating the pinball loss across
**all** quantiles, giving a single distributional score.
"""

CONTENT["Prediction Intervals (PI)"] = r"""
What it is
----------

A **prediction interval** is a range :math:`[\,\text{lower},\ \text{upper}\,]` expected to contain the
**future observation** with a stated probability — its **nominal coverage**, e.g. 90%. It is typically
built from a **pair of quantiles**, the :math:`(1-\alpha)/2` and :math:`(1+\alpha)/2` levels (the 5th and
95th percentiles for 90% coverage):

.. math::

   \big[\, \hat{Q}_{(1-\alpha)/2},\ \ \hat{Q}_{(1+\alpha)/2} \,\big]
   \quad\Rightarrow\quad \text{nominal coverage } 1-\alpha.

Coverage vs width
-----------------

Quality trades **coverage** — does the empirical fraction of actuals landing inside match the nominal
level? — against **width / sharpness** — narrower is more useful, but only if coverage holds. (A PI is
about a *future value*, distinct from a **confidence interval**, which is about a *parameter*.)

Calibrating them
----------------

**Conformal prediction** is a model-agnostic wrapper that adjusts interval width on a held-out
**calibration** set to guarantee the target coverage in **finite samples**, under mild assumptions.
"""

CONTENT["Quantile Regression"] = r"""
What it is
----------

**Quantile regression** estimates a **conditional quantile** of the target instead of its mean: for a
chosen level :math:`\tau \in (0, 1)` it predicts the :math:`\tau`-th quantile given the features. Fit
every level and you recover the **inverse CDF** — the whole conditional distribution.

The pinball loss and τ
----------------------

It minimizes the **pinball loss**, which weights over- and under-prediction **asymmetrically** by
:math:`\tau`:

.. math::

   \ell_\tau(y, \hat{y}) =
   \begin{cases}
     \tau\,(y - \hat{y}) & y \ge \hat{y}, \\[2pt]
     (1 - \tau)(\hat{y} - y) & y < \hat{y}.
   \end{cases}

At :math:`\tau = 0.5` this is symmetric and recovers the **median** (equivalent to minimizing MAE);
:math:`\tau < 0.5` pushes the model to **under-predict**, :math:`\tau > 0.5` to **over-predict**, and the
further :math:`\tau` is from 0.5 the stronger the asymmetry.

In practice
-----------

It is **distribution-free** and robust (built on absolute differences), but fits each quantile
**separately**, which can cause **quantile crossing** (a lower quantile predicted above a higher one)
unless constrained. Common estimators: the linear ``QuantileRegressor``, gradient-boosted quantile
models, and quantile random forests.

.. code-block:: python

   from sklearn.linear_model import QuantileRegressor

   lower = QuantileRegressor(quantile=0.05, alpha=0.0).fit(X_train, y_train)
   upper = QuantileRegressor(quantile=0.95, alpha=0.0).fit(X_train, y_train)
   # [lower.predict(X), upper.predict(X)] is a 90% prediction interval
"""

MINDMAP.update({
    "Quantile Forecasts": [
        "Quantile Regression", "Prediction Intervals (PI)", "Probabilistic Forecasts",
        "Quantile Level", "Predicting Percentiles", "Point Forecasts",
    ],
    "Prediction Intervals (PI)": [
        "Quantile Forecasts", "Quantile Regression", "Probabilistic Forecasts",
        "Continuous Probabilistic Forecasts", "Quantile Level", "Strictly Proper Scoring Rules",
    ],
    "Quantile Regression": [
        "Quantile Forecasts", "Prediction Intervals (PI)", "Quantile Level",
        "Predicting Percentiles", "Probabilistic Forecasts", "R² (R-squared)",
    ],
})


# ----------------------------------------------------------------------
# Theme: forecast error & benchmarking — error metrics, naive baseline, M-competitions  (evaluation)
# ----------------------------------------------------------------------

CONTENT["Forecast Error"] = r"""
What it is
----------

**Forecast error** is the gap between what happened and what was predicted — the residual

.. math::

   e_t = y_t - \hat{y}_t.

A single error means little; forecast quality is summarized by **aggregating** errors into metrics.

Common metrics
--------------

**MAE** :math:`= \frac{1}{N}\sum_t |y_t - \hat{y}_t|` is robust and interpretable, and the forecast that
minimizes it is the **median**. **RMSE** :math:`= \sqrt{\frac{1}{N}\sum_t (y_t - \hat{y}_t)^2}` penalizes
large misses more and is minimized by the **mean**, but is harder to read. **MAPE** (mean absolute
*percentage* error) is scale-free but **explodes** when actuals are near zero; **sMAPE** is a bounded
symmetric variant, still shaky near zero. **MASE** scales MAE by a naive forecast's error, making it
**scale-free** and interpretable (:math:`<1` beats naive).

Use several
-----------

No single metric tells the whole story — **MAPE** can look great while **bias** quietly builds, and
**MAE** can hide a few enormous misses — so report several: an absolute metric (MAE / RMSE), a scaled one
(MASE), and a **bias** measure.
"""

CONTENT["Naïve Baseline Forecast"] = r"""
What it is
----------

A **naïve baseline forecast** is the simplest possible forecast — use the **last observed value** as the
prediction for the next period (the random-walk forecast). Its seasonal cousin, the **seasonal naïve**,
uses the value from one season ago:

.. math::

   \hat{y}_{t+1} = y_t \qquad\text{(naïve)}, \qquad\qquad \hat{y}_{t+h} = y_{t+h-m} \quad\text{(seasonal naïve, period } m).

Why it matters
--------------

It is the **benchmark every model must beat**. If a complex model cannot outperform *"just repeat the
last value"*, the complexity is not paying off. It is also the reference in the **denominator of MASE**,
computed on the **in-sample (training)** series so the benchmark does not leak future information.

Which variant
-------------

The plain naïve suits **non-seasonal** data; the **seasonal naïve** is the right reference when there is a
clear period (set :math:`m = 12` for monthly data with yearly seasonality, not :math:`m = 1`). Cheap,
robust, and — for noisy or short series — surprisingly hard to beat.
"""

CONTENT["M-Competitions (Makridakis Competitions)"] = r"""
What they are
-------------

The **M-competitions** are a series of large-scale **forecasting competitions** (M1 through M6) organized
by **Spyros Makridakis** and colleagues to gather **empirical evidence** about which forecasting methods
actually work best in practice — not just in theory.

Key findings
------------

Across the early competitions **no single method dominated**, and **simple methods** (naïve, exponential
smoothing, ARIMA) proved **tough baselines** that often matched or beat more complex statistical models;
**combining** forecasts reliably improved accuracy. **M4** (2018; 100,000 series, 61 methods) found the
best results came from **hybrid statistical + ML** approaches and combinations, while pure-ML methods
fared poorly. **M5** (2020; Walmart hierarchical retail data on Kaggle, roughly ``$100,000`` in prizes)
was the first in which **ML methods dominated** the leaderboard, and it put **probabilistic / uncertainty**
forecasting center stage.

Why they matter
---------------

For four decades the M-competitions have shaped forecasting — establishing that **combinations and
hybrids** win, that **simple baselines** must always be checked, and that **probabilistic forecasting** is
now the standard. They directly inspired modern ML forecasting competitions (such as those on Kaggle).
"""

MINDMAP.update({
    "Forecast Error": [
        "Naïve Baseline Forecast", "Point Forecasts", "Forecasting Benchmarks",
        "Average Absolute Error (AAE)", "Time Series Forecasting", "Relative accuracy",
    ],
    "Naïve Baseline Forecast": [
        "Forecast Error", "Simple Baseline Methods", "Seasonal Lag", "Forecasting Benchmarks",
        "M-Competitions (Makridakis Competitions)", "Point Forecasts",
    ],
    "M-Competitions (Makridakis Competitions)": [
        "Forecasting Competitions", "Forecasting Benchmarks", "Naïve Baseline Forecast",
        "Forecast Error", "Probabilistic Forecasts", "Time Series Forecasting",
    ],
})


# ----------------------------------------------------------------------
# Theme: probability fundamentals — PMF, PDF, CDF  (probstats)
# ----------------------------------------------------------------------

CONTENT["Probability Mass"] = r"""
What it is
----------

The **probability mass function (PMF)** of a **discrete** random variable gives the probability that it
takes **exactly** a given value:

.. math::

   p(x) = \Pr[X = x].

It maps each possible value to a probability — for a fair die, :math:`p(k) = 1/6` for :math:`k = 1,\dots,6`.

Properties
----------

Every mass lies in :math:`[0, 1]`, and all masses **sum to one**:

.. math::

   \sum_x p(x) = 1.

Discrete only
-------------

Mass applies to **discrete** outcomes, where a single value can carry **positive** probability — unlike a
continuous variable, where any *exact* point has probability **zero** (there, density takes its place).
"""

CONTENT["Probability Density"] = r"""
What it is
----------

The **probability density function (PDF)** :math:`f(x)` describes a **continuous** random variable. It is
**not** a probability itself — its **area** is: the probability of landing in an interval is the integral

.. math::

   P(a \le X \le b) = \int_a^b f(x)\,dx.

Properties
----------

The density satisfies :math:`f(x) \ge 0`, may **exceed 1** (it is a density, not a probability),
integrates to **one** over the whole line, and assigns probability **zero** to any *exact* value. The bell
curve of the **normal distribution** is the classic PDF.

Link to the CDF
---------------

The density is the **derivative** of the cumulative distribution function,

.. math::

   f(x) = F'(x),

so equivalently :math:`F` is the running integral of :math:`f`.
"""

CONTENT["Cumulative Distribution Function (CDF)"] = r"""
What it is
----------

The **cumulative distribution function (CDF)** gives the probability that a random variable is **at
most** :math:`x`:

.. math::

   F(x) = \Pr[X \le x].

Unlike the PMF or PDF, it works for **both** discrete and continuous variables.

Properties
----------

The CDF is **non-decreasing**, runs from **0 to 1**, and is right-continuous. For a **discrete** variable
it is a **step function**; for a **continuous** one it is smooth:

.. math::

   F(x) = \sum_{k \le x} p(k) \quad\text{(discrete)}, \qquad F(x) = \int_{-\infty}^{x} f(t)\,dt \quad\text{(continuous)}.

Why it's useful
---------------

It directly answers "**at most**" and interval questions, and it is the **bridge** between
representations: **quantiles** are read off its inverse and the **density** is its derivative. For a fair
die, :math:`F(2) = 1/3`.
"""

MINDMAP.update({
    "Probability Density": [
        "Probability Mass", "Cumulative Distribution Function (CDF)", "Probability Distribution",
        "Normal Distribution", "Probabilistic Forecasts", "Quantile Regression",
    ],
    "Probability Mass": [
        "Probability Density", "Cumulative Distribution Function (CDF)", "Probability Distribution",
        "Classification Probability", "Probability", "Normal Distribution",
    ],
    "Cumulative Distribution Function (CDF)": [
        "Probability Density", "Probability Mass", "Probability Distribution",
        "Quantile Level", "Quantile Regression", "Normal Distribution",
    ],
})


# ----------------------------------------------------------------------
# Theme: distribution & supervised-learning foundations — distribution, Gaussian, target  (probstats / concepts)
# ----------------------------------------------------------------------

CONTENT["Probability Distribution"] = r"""
What it is
----------

A **probability distribution** describes how probability is **spread over the possible values** of a
random variable — the full accounting of what can happen and how likely each outcome is.

How it's described
------------------

For a **discrete** variable it is given by a **probability mass function**, for a **continuous** one by a
**probability density function**, and for either by a **cumulative distribution function**. Whatever the
form, the total probability is **one**:

.. math::

   \sum_x p(x) = 1 \quad\text{(discrete)}, \qquad \int_{-\infty}^{\infty} f(x)\,dx = 1 \quad\text{(continuous)}.

Characterizing it
-----------------

Distributions are summarized by **parameters** (a **mean** for location, a **variance** for spread) and by
**moments**; you can **fit** a distribution to data or **sample** synthetic data from one. Common families
include the **normal**, **Bernoulli / binomial**, **Poisson** and **exponential**.
"""

CONTENT["Normal Distribution"] = r"""
What it is
----------

The **normal (Gaussian) distribution** is the continuous, **bell-shaped**, symmetric distribution defined
by two parameters — the **mean** :math:`\mu` (its center) and the **standard deviation** :math:`\sigma`
(its spread; variance :math:`\sigma^2`):

.. math::

   X \sim \mathcal{N}(\mu, \sigma^2).

Its **mean, median and mode coincide**, and it extends from :math:`-\infty` to :math:`+\infty`.

The density
-----------

.. math::

   f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).

About **68%** of values lie within :math:`1\sigma` of the mean, **95%** within :math:`2\sigma`, and
**99.7%** within :math:`3\sigma` (the *68–95–99.7 rule*). Standardizing with :math:`z = (x-\mu)/\sigma`
maps any normal onto the **standard normal** :math:`\mathcal{N}(0, 1)`, so a single table serves all.

Why it's everywhere
-------------------

The **Central Limit Theorem** — averages of many independent, finite-variance quantities tend toward a
normal — makes it the default model for **measurement errors** and **aggregates**. But it has **light
tails**: with heavy-tailed data or frequent **outliers** (e.g. Cauchy, Pareto) it fits poorly and
least-squares methods grow unreliable.
"""

CONTENT["Target Variable"] = r"""
What it is
----------

The **target variable** (:math:`y`) — also called the **dependent**, **response**, or **outcome**
variable, or the **label** — is the quantity a **supervised** model is trained to **predict** from the
input **features** (the independent variables). It is the "correct answer" that must be **observed** in
the training data.

Types
-----

It can be **continuous** (a regression target, e.g. a price) or **categorical** (a classification target,
e.g. spam / not-spam), and also **ordinal** or **multi-label**. Its type **determines the problem** and
which models fit.

Why it matters
--------------

The algorithm only ever learns a **function mapping features to target**, so a **well-defined** target is
decisive: without a labeled target, supervised learning cannot proceed, and a poorly chosen or **biased**
target propagates straight into the model's behavior.
"""

MINDMAP.update({
    "Probability Distribution": [
        "Probability Density", "Probability Mass", "Cumulative Distribution Function (CDF)",
        "Normal Distribution", "Probability", "Probabilistic Forecasts",
    ],
    "Normal Distribution": [
        "Probability Distribution", "Probability Density", "Cumulative Distribution Function (CDF)",
        "Z-Score", "Standard Error (SE)", "Bootstrap Confidence Intervals (CIs)",
    ],
    "Target Variable": [
        "Feature Values", "Classification Probability", "Regression Coefficient",
        "Point Forecasts", "Probabilistic Forecasts", "Label Noise",
    ],
})


# ----------------------------------------------------------------------
# Theme: risk forecasting — return distribution, Value-at-Risk, risk forecast  (risk)
# ----------------------------------------------------------------------

CONTENT["Return Distribution"] = r"""
What it is
----------

A **return distribution** is the probability distribution of an asset's (or portfolio's) **returns** over
a period — the range of possible returns and how likely each is. From a price series
:math:`(S_0, \dots, S_n)`, the simple return is

.. math::

   R_t = \frac{S_t - S_{t-1}}{S_{t-1}}

(or, equivalently for many purposes, the **log return**).

Why it matters
--------------

Every downside-risk measure is read **off this distribution** — for instance, **Value-at-Risk** is a
tail **quantile** of it. Model the return distribution and you can **price** risk.

The reality: fat tails
----------------------

Empirical returns are **not** normal — they have **heavy tails** and **skewness**, so extreme moves
happen far more often than a Gaussian predicts. Assuming normality **understates** tail risk;
heavy-tailed (**Student-t**) or **location-scale** models fit better.
"""

CONTENT["Value-at-Risk (VaR)"] = r"""
What it is
----------

**Value-at-Risk** summarizes downside risk in a single number: the **maximum loss** over a holding period
:math:`h` that will **not be exceeded** with confidence :math:`\alpha` (typically 95% or 99%) — anything
worse occurs only with probability :math:`1 - \alpha`. Formally it is the :math:`\alpha`-**quantile** of
the loss distribution (the negative :math:`\alpha`-quantile of returns):

.. math::

   \mathrm{VaR}_\alpha = -F_r^{-1}(\alpha), \qquad \Pr\!\left(L > \mathrm{VaR}_\alpha\right) = 1 - \alpha,

with :math:`F_r` the return CDF and :math:`L` the loss.

Where it comes from
-------------------

VaR was introduced by J. P. Morgan's **RiskMetrics** (1994) and enshrined by the **Basel** framework for
bank regulatory capital. It is estimated by **historical simulation** (the empirical quantile over a
rolling window), **parametric** methods (assume a normal / t distribution and scale by volatility, often
via **GARCH**), or **Monte Carlo**.

Its blind spot
--------------

VaR says **nothing about how bad** losses beyond the threshold are, and it is **not coherent** — it can
violate **subadditivity**, so a diversified portfolio's VaR may exceed the sum of its parts. **Expected
Shortfall** (CVaR) — the *average* loss **given** VaR is breached — repairs both and is coherent.
"""

CONTENT["Risk Forecast"] = r"""
What it is
----------

A **risk forecast** predicts a risk measure — most often **VaR** or **Expected Shortfall** — for a
**future** period. Because VaR is a quantile, forecasting it means forecasting the :math:`\tau`-**quantile
of future returns** given today's information; the quantity is **unobserved** and estimated ahead of time.

How it's done
-------------

Methods forecast the future **return distribution** (or just its **scale**): **GARCH**-family volatility
models (forecast the variance, then scale a distributional quantile), **historical simulation**, **Extreme
Value Theory** for the far tail, **quantile regression**, and hybrids of these.

How it's judged
---------------

By **backtesting**: over a long out-of-sample run, the fraction of days the loss **breaches** the forecast
VaR should match the stated level (about 1% of days for 99% VaR). Too many breaches means risk was
**under-forecast**. This discipline is vital for banks, risk managers and regulators.
"""

MINDMAP.update({
    "Return Distribution": [
        "Value-at-Risk (VaR)", "Risk Forecast", "Probability Distribution",
        "Normal Distribution", "Quantile Level", "Probabilistic Forecasts",
    ],
    "Value-at-Risk (VaR)": [
        "Return Distribution", "Risk Forecast", "Quantile Level", "Quantile Regression",
        "Cumulative Distribution Function (CDF)", "Probabilistic Scoring",
    ],
    "Risk Forecast": [
        "Value-at-Risk (VaR)", "Return Distribution", "Probabilistic Forecasts",
        "Quantile Regression", "Probabilistic Scoring", "Forecast Error",
    ],
})


# ----------------------------------------------------------------------
# Theme: full-distribution forecasting — continuous prob. forecasts, full distribution, probabilistic scoring  (evaluation)
# ----------------------------------------------------------------------

CONTENT["Continuous Probabilistic Forecasts"] = r"""
What it is
----------

A **continuous probabilistic forecast** is a probabilistic forecast for a **continuous (real-valued)**
outcome — a full **predictive distribution** over the variable (a density or CDF), rather than a single
value or a class probability. It answers *what is the whole distribution of tomorrow's demand, price, or
temperature?*

How it's represented
--------------------

It can be given as a **parametric** distribution (e.g. a normal :math:`\mathcal{N}(\mu, \sigma^2)` with a
forecast mean and variance), a set of **quantiles**, or an **ensemble** of sampled trajectories — each a
way to describe the continuous outcome's uncertainty.

Why it's useful
---------------

From one object it exposes **every** downstream quantity — the mean, any **quantile**, a **prediction
interval**, or a **tail probability**. Because it lives on a continuum (unlike a discrete / categorical
probabilistic forecast), it is scored with the **CRPS**.
"""

CONTENT["Full Distribution"] = r"""
What it is
----------

Forecasting the **full distribution** means predicting the **entire** predictive distribution — the
complete CDF / PDF over all possible outcomes — rather than a **summary** of it. A point forecast
collapses it to one number; a quantile forecast reports a few points; the full distribution keeps
**everything**.

The richest target
------------------

From the full distribution you can **derive any summary** after the fact — the mean, the median, any
**quantile**, a **prediction interval**, the probability of exceeding a threshold, or a risk measure such
as **VaR**. Nothing about the uncertainty is discarded.

How it's judged
---------------

Because it is a whole distribution, it is scored by a rule that reads the **entire shape** against the
outcome — the **CRPS**, which compares the forecast CDF to the observation's step CDF — not a point-error
metric like MAE.
"""

CONTENT["Probabilistic Scoring"] = r"""
What it is
----------

**Probabilistic scoring** evaluates a **probabilistic forecast** — a whole predictive distribution —
against the outcome that occurs, using a **scoring rule** (a loss function for distributions). To be
trustworthy the rule should be **strictly proper**, so a forecaster **minimizes** the expected score
*only* by reporting their true distribution.

The workhorse: CRPS
-------------------

The **Continuous Ranked Probability Score** is the most-used score for real-valued forecasts. It
integrates the **squared gap** between the forecast CDF :math:`F` and the step CDF of the observation
:math:`y`, and is **negatively oriented** (lower is better):

.. math::

   \mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(x) - \mathbb{1}\{x \ge y\}\big)^2 \, dx.

It **generalizes the MAE** (for point forecasts) and the **Brier score** (for binary ones), reducing to
them in those cases.

What it captures
----------------

A proper score rewards both **calibration** (probabilities match reality) and **sharpness** (tight
distributions) — the CRPS in fact **decomposes** into calibration, discrimination and uncertainty parts.
The **log score** is a **local** alternative that looks only at the density assigned to the outcome.
"""

MINDMAP.update({
    "Probabilistic Scoring": [
        "Strictly Proper Scoring Rules", "Continuous Probabilistic Forecasts", "Probabilistic Forecasts",
        "Full Distribution", "Cumulative Distribution Function (CDF)", "Quantile Forecasts",
    ],
    "Full Distribution": [
        "Continuous Probabilistic Forecasts", "Probabilistic Forecasts", "Point Forecasts",
        "Quantile Forecasts", "Probability Distribution", "Probabilistic Scoring",
    ],
    "Continuous Probabilistic Forecasts": [
        "Probabilistic Forecasts", "Full Distribution", "Probabilistic Scoring",
        "Strictly Proper Scoring Rules", "Quantile Forecasts", "Prediction Intervals (PI)",
    ],
})


# ----------------------------------------------------------------------
# Theme: forecast-accuracy metrics — AAE, relative accuracy, R-squared  (metrics)
# ----------------------------------------------------------------------

CONTENT["Average Absolute Error (AAE)"] = r"""
What it is
----------

The **average absolute error** is the mean of the **absolute** errors — the average of
:math:`|\text{forecast} - \text{actual}|` across all points. It answers *how big is the typical error?*
in the **same units** as the data, and is identical to the **Mean Absolute Error (MAE)** (often also
called the Mean Absolute Deviation):

.. math::

   \mathrm{AAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|.

Properties
----------

Because it uses **absolute** (not squared) errors, AAE is **robust to outliers** — a few large misses do
not dominate — and it treats over- and under-prediction **symmetrically**, ignoring the *direction* of
error. The forecast that minimizes it is the **median**.

Limitation
----------

AAE is **scale-dependent**: you cannot compare it across series on different scales (an AAE of 10 is tiny
for house prices, huge for temperatures). For that, switch to a **percentage** or **relative** metric.
"""

CONTENT["Relative accuracy"] = r"""
What it is
----------

**Relative accuracy** measures forecast accuracy **relative to a benchmark** rather than in absolute
units. You **normalize** the error by a reference method's error — usually the **naïve** (or
seasonal-naïve) forecast — so results land on a **common, scale-free** scale comparable across series:

.. math::

   \text{relative error} = \frac{\mathrm{MAE}_{\text{model}}}{\mathrm{MAE}_{\text{benchmark}}}.

How to read it
--------------

A value **< 1** means the model **beats** the benchmark, **= 1** means it **matches** it, and **> 1**
means it is **worse** — a relative error of 0.6 is roughly **40% better** than the benchmark. This family
includes **MASE**, **Theil's U** (:math:`<1` beats a naïve guess), and relative / bounded relative
absolute errors.

Why it matters
--------------

Absolute errors like MAE and RMSE are **meaningless without a reference** — *is an MAE of 10 good?*
depends entirely on the problem — whereas relative accuracy is **interpretable** and puts easy and
hard-to-forecast series on equal footing.
"""

CONTENT["R² (R-squared)"] = r"""
What it is
----------

**R²**, the **coefficient of determination**, is the **proportion of variance** in the target that the
model explains. It compares the model's **residual** error to the variance around the **mean**:

.. math::

   R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}, \qquad
   SS_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2, \quad SS_{\text{tot}} = \sum_i (y_i - \bar{y})^2.

How to read it
--------------

It usually runs **0 to 1** — **1** is a perfect fit, **0** means the model does no better than predicting
the **mean** (and it can go **negative** for a model worse than that). Being **dimensionless**, it
complements MAE / RMSE, which report error in the target's units; in simple regression it equals
:math:`r^2`, the squared **Pearson correlation**.

Caveats
-------

R² **never decreases** when predictors are added (even noise), so use **adjusted R²** to compare models
of different size; it is sensitive to **outliers**, assumes the modeled relationship, and a high value
implies **neither causation nor good out-of-sample** performance.
"""

MINDMAP.update({
    "Average Absolute Error (AAE)": [
        "Forecast Error", "Naïve Baseline Forecast", "Relative accuracy", "R² (R-squared)",
        "Point Forecasts", "Time Series Forecasting",
    ],
    "Relative accuracy": [
        "Forecast Error", "Naïve Baseline Forecast", "Average Absolute Error (AAE)",
        "M-Competitions (Makridakis Competitions)", "R² (R-squared)", "Forecasting Benchmarks",
    ],
    "R² (R-squared)": [
        "Forecast Error", "Average Absolute Error (AAE)", "Relative accuracy",
        "Regression Coefficient", "Target Variable", "Point Forecasts",
    ],
})


# ----------------------------------------------------------------------
# Theme: forecast benchmarking — simple baselines, benchmarks, competitions  (evaluation)
# ----------------------------------------------------------------------

CONTENT["Simple Baseline Methods"] = r"""
What they are
-------------

**Simple baseline methods** are a small family of trivially simple forecasts used to set a **statistical
baseline** before anything complex. The staples: the **mean** method (forecast the historical **average**,
which smooths away seasonality), the **naïve** method (carry the **last value** forward — a random walk),
the **seasonal naïve** (carry the value from **one season ago**), and the **drift** method (naïve plus a
**trend** equal to the average historical change).

The drift formula
-----------------

.. math::

   \hat{y}_{T+h} = y_T + h \cdot \frac{y_T - y_1}{T-1},

equivalent to drawing a line through the **first and last** observations and extrapolating it forward.

Why use them
------------

They are **cheap and transparent**, and any sophisticated model must **beat** them to earn its
complexity. Choose by structure — the **mean** for a flat series, the **naïve** for a random walk, the
**seasonal naïve** for strong seasonality, **drift** for a trend — and note that **averaging** several
often improves accuracy. *KISS: keep it sophisticatedly simple.*
"""

CONTENT["Forecasting Benchmarks"] = r"""
What it is
----------

A **forecasting benchmark** is the **reference forecast** a proposed model must **outperform** to be worth
using. In practice the **simple baseline methods** (naïve, seasonal naïve, mean, drift) fill this role —
the **naïve** forecast is the standard reference, and the basis of **MASE**.

The discipline
--------------

Always establish a benchmark **before** reaching for complex models — a step that is often skipped. *Any
complex model must be better than the baseline to be considered.* A model that only **marginally** beats
the naïve forecast probably is not worth its added complexity and maintenance.

Beyond one series
-----------------

Shared **datasets** (such as the M-competition series) act as **community benchmarks**, letting different
methods be compared on **common ground** rather than on each author's private data.
"""

CONTENT["Forecasting Competitions"] = r"""
What they are
-------------

**Forecasting competitions** are organized contests in which many teams forecast the **same datasets** and
are **ranked** by accuracy on a held-out period — turning *which method is best?* into an **empirical**,
reproducible question. The **M-competitions** are the archetype; **Kaggle** hosts many modern ones.

How they work
-------------

They use standardized **data**, a hidden **test** horizon, and common **metrics** (often scale-free ones
like MASE) so entries are **directly comparable**. Prizes and public **leaderboards** draw large fields of
participants.

Why they matter
---------------

They produce **durable evidence** — that **combinations and hybrids** tend to win, that **simple
baselines** are hard to beat, and increasingly that **ML** is competitive — and they leave behind
**reusable benchmark datasets** that shape later research.
"""

MINDMAP.update({
    "Forecasting Benchmarks": [
        "Simple Baseline Methods", "Naïve Baseline Forecast", "Forecast Error",
        "M-Competitions (Makridakis Competitions)", "Relative accuracy", "Forecasting Competitions",
    ],
    "Simple Baseline Methods": [
        "Naïve Baseline Forecast", "Forecasting Benchmarks", "Seasonal Lag", "Forecast Error",
        "Time Series Forecasting", "Forecasting Competitions",
    ],
    "Forecasting Competitions": [
        "M-Competitions (Makridakis Competitions)", "Forecasting Benchmarks", "Naïve Baseline Forecast",
        "Simple Baseline Methods", "Forecast Error", "Probabilistic Forecasts",
    ],
})


# ----------------------------------------------------------------------
# Theme: time-series concepts — TS forecasting, seasonal lag, log-space  (concepts / signal)
# ----------------------------------------------------------------------

CONTENT["Time Series Forecasting"] = r"""
What it is
----------

**Time series forecasting** predicts **future values** of a **time-ordered** series from its past. Unlike
ordinary supervised learning — where rows are independent and can be shuffled — a time series has
**memory**: order matters, and each observation is **dependent** on its neighbors.

The components
--------------

A series decomposes into a **trend** (long-term drift up or down), **seasonality** (regular patterns at a
**fixed** period), **cycles** (longer, **aperiodic** swings with no fixed length), and **noise** (the
irregular residual). Modeling means **separating** these.

Stationarity and diagnostics
----------------------------

Many methods need **stationarity** (constant mean and variance); non-stationary series are made modelable
by **differencing** (removes trend / seasonality from the mean) and a **log / Box-Cox** transform
(stabilizes variance). Choose the model **after** diagnostics — time, seasonal and **lag** plots, and the
**ACF / PACF** — and evaluate with **rolling-origin backtesting**, reporting error by **horizon** and
publishing **intervals**, not just points.
"""

CONTENT["Seasonal Lag"] = r"""
What it is
----------

A **lag** is a past value of the series, :math:`y_{t-k}`; the **seasonal lag** is the lag equal to the
**seasonal period** :math:`m` — the value from the **same point one cycle ago**:

.. math::

   \text{seasonal lag: } y_{t-m}, \qquad \text{seasonal difference: } y_t - y_{t-m}.

Common periods
--------------

The period is set by the calendar of the data: :math:`m = 12` for **monthly** data with yearly
seasonality, :math:`m = 7` for **daily** data with weekly seasonality, :math:`m = 24` for **hourly** data
with daily cycles.

Where it's used
---------------

The seasonal lag underlies the **seasonal naïve** forecast (:math:`\hat{y}_t = y_{t-m}`), **seasonal
differencing** (which strips out seasonality), and **lag features** in ML forecasting. A large
**autocorrelation** at the seasonal lag is the signature of seasonality.
"""

CONTENT["Log-Space"] = r"""
What it is
----------

Working in **log-space** means transforming a series to its **logarithm**, :math:`w_t = \log(y_t)`,
instead of the raw values — a **variance-stabilizing** transform for when fluctuations **grow with the
level** of the series (multiplicative or heteroscedastic behavior).

What it does
------------

It **compresses large values** while leaving small ones nearly untouched, turns **multiplicative**
structure into **additive**, and makes **relative (percentage) change** the natural unit — a difference in
log-space is approximately a proportional change:

.. math::

   \log(y_t) - \log(y_{t-1}) \approx \frac{y_t - y_{t-1}}{y_{t-1}}.

Caveats
-------

The logarithm is **undefined for zero or negative** values — use :math:`\log(y + 1)` or a **Box-Cox**
transform instead — and any forecast made in log-space must be **back-transformed** (exponentiated) to the
original scale.
"""

MINDMAP.update({
    "Time Series Forecasting": [
        "Seasonal Lag", "Log-Space", "Seasonality", "Naïve Baseline Forecast",
        "Forecast Error", "Point Forecasts",
    ],
    "Seasonal Lag": [
        "Time Series Forecasting", "Seasonality", "Naïve Baseline Forecast", "Log-Space",
        "Temporal autocorrelation (Serial Correlation)", "Simple Baseline Methods",
    ],
    "Log-Space": [
        "Time Series Forecasting", "Seasonal Lag", "Normal Distribution", "Forecast Error",
        "Relative accuracy", "Seasonality",
    ],
})


# ----------------------------------------------------------------------
# Theme: quantile levels & forecast types — quantile level, percentiles, deterministic  (evaluation / inference)
# ----------------------------------------------------------------------

CONTENT["Quantile Level"] = r"""
What it is
----------

The **quantile level** :math:`\tau \in (0,1)` (sometimes written :math:`\alpha`) is the probability that
**names which quantile** a forecast targets — the value :math:`q_\tau` below which the outcome is expected
to fall a fraction :math:`\tau` of the time:

.. math::

   \Pr\!\left(Y \le q_\tau\right) = \tau, \qquad \tau \in (0, 1).

Reading levels
--------------

:math:`\tau = 0.5` is the **median**; :math:`\tau = 0.1` is the 10th percentile (a pessimistic lower value
the outcome undercuts 10% of the time); :math:`\tau = 0.9` the 90th (an optimistic upper value). A **set**
of levels :math:`\{0.1, 0.5, 0.9, \dots\}` traces out the whole predictive distribution.

Monotonicity
------------

In **quantile regression** the level is the **tilting parameter** of the pinball loss. Estimated quantiles
should be **monotonically increasing** in :math:`\tau` — when a lower level's forecast exceeds a higher
one's, that is **quantile crossing**, an error to constrain away.
"""

CONTENT["Predicting Percentiles"] = r"""
What it is
----------

**Predicting percentiles** means forecasting specific **percentiles** (quantiles) of the outcome
distribution — the value below which a given **percentage** of outcomes fall — instead of only a single
mean. A percentile is a quantile stated as a percent: the 0.9 quantile is the **90th percentile**.

Why percentiles
---------------

A handful of percentiles (say the **10th, 50th and 90th**) sketch the **range** of outcomes and their
**best- and worst-case** scenarios, exposing **uncertainty** and enabling **asymmetric** decisions —
without committing to a parametric distribution.

How it's done
-------------

Percentiles are produced by **quantile regression** (and its tree / boosting variants), each trained on
the **pinball loss** for its level; stacking many percentiles approximates the **full distribution**.
"""

CONTENT["Deterministic forecasts"] = r"""
What it is
----------

A **deterministic forecast** outputs a **single value** for each future point — a **point estimate** with
**no uncertainty** attached. It is the counterpart of a **probabilistic** forecast, which predicts a whole
distribution; here all the probability sits on **one number**.

What it hides
-------------

Two deterministic forecasts can agree on the number yet face very different **risk**, and the value is
usually a **summary** of an implicit distribution — the **mean** (if fit by minimizing RMSE) or the
**median** (if fit by MAE).

When it's enough
----------------

It is simple to produce, communicate and act on, and fine when uncertainty is small or irrelevant — but
where the **cost of being wrong is asymmetric**, a **probabilistic** or **quantile** forecast conveys far
more.
"""

MINDMAP.update({
    "Quantile Level": [
        "Quantile Forecasts", "Quantile Regression", "Predicting Percentiles",
        "Prediction Intervals (PI)", "Probabilistic Forecasts", "Cumulative Distribution Function (CDF)",
    ],
    "Predicting Percentiles": [
        "Quantile Forecasts", "Quantile Level", "Quantile Regression",
        "Prediction Intervals (PI)", "Probabilistic Forecasts", "Point Forecasts",
    ],
    "Deterministic forecasts": [
        "Point Forecasts", "Probabilistic Forecasts", "Forecast Error", "Quantile Forecasts",
        "Time Series Forecasting", "Full Distribution",
    ],
})


# ----------------------------------------------------------------------
# Theme: recommender coverage — catalog, item, user coverage  (recsys)
# ----------------------------------------------------------------------

CONTENT["Catalog Coverage"] = r"""
What it is
----------

**Catalog coverage** is the proportion of the entire item catalog that a recommender actually **surfaces**
to users — the share of products that ever appear in someone's recommendations. It measures the
**breadth** of the system's reach, not the quality of any one list.

The formula
-----------

.. math::

   \text{Catalog Coverage} = \frac{|\text{distinct items recommended}|}{|\text{total items in catalog}|},

computed as an **aggregate** over all users and a time window (often expressed as a percentage).

Why it matters
--------------

**Low** coverage signals **popularity bias** — the system funnels everyone toward the same few hits,
starving the **long tail** and neglecting niche tastes; **high** coverage means diverse, inclusive
recommendations. Because recommending popular items is often **accurate but narrow**, coverage is reported
**alongside** accuracy — a good system scores well on both.
"""

CONTENT["Item Coverage"] = r"""
What it is
----------

**Item coverage** is the fraction of individual items the recommender is **able to recommend at all** —
for which it can produce a prediction or place in a list. It asks *how many products can the system
reach?*, item by item:

.. math::

   \text{Item Coverage} = \frac{|\text{items the system can recommend}|}{|\text{total items}|}.

What limits it
--------------

Items with **no or few interactions** — new or niche products (the **cold-start** problem, data
**sparsity**) — may be impossible to recommend, dragging item coverage down. **Content-based** signals or
**hybrid** models raise it by letting the system reason about **unseen** items from their features.

Recommendability
----------------

Item coverage is about **recommendability** (can this item ever surface?), whereas **catalog coverage** is
about how much of the catalog **actually** surfaces in practice.
"""

CONTENT["User Coverage"] = r"""
What it is
----------

**User coverage** is the fraction of the user base for whom the recommender can produce **useful
recommendations** — *how many users can the system serve?* A system with excellent recommendations for
only a subset of users has limited reach:

.. math::

   \text{User Coverage} = \frac{|\text{users served}|}{|\text{total users}|}.

What limits it
--------------

**New users** with little or no history (the user-side **cold-start** problem) may get no personalized
recommendations; **sparse** or atypical users are also hard to serve. Fallbacks — **popularity** lists,
onboarding questionnaires, **content-based** profiles — extend coverage to these users.

Why it matters
--------------

Coverage and accuracy trade off — it is easy to look accurate by only serving **well-understood** users —
so user coverage keeps the evaluation **honest** about the whole population.
"""

MINDMAP.update({
    "Catalog Coverage": [
        "Item Coverage", "User Coverage", "Long-Tail Items", "Intra-List Diversity (ILD)",
        "Diminishing Utility", "Relevance in Recommender Systems",
    ],
    "Item Coverage": [
        "Catalog Coverage", "User Coverage", "Long-Tail Items", "Relevance in Recommender Systems",
        "Intra-List Diversity (ILD)", "Diminishing Utility",
    ],
    "User Coverage": [
        "Catalog Coverage", "Item Coverage", "Relevance in Recommender Systems",
        "Customer Segmentation", "Long-Tail Items", "Intra-List Diversity (ILD)",
    ],
})


# ----------------------------------------------------------------------
# Theme: recommender similarity & diversity — cosine, Jaccard, ILD  (recsys)
# ----------------------------------------------------------------------

CONTENT["Cosine Similarity of Item Features"] = r"""
What it is
----------

**Cosine similarity** measures how alike two items are by the **cosine of the angle** between their
**feature vectors** — representations built from metadata (one-hot genres, tags, text) or learned
**embeddings**. It captures **orientation**, not magnitude, so it is invariant to vector length.

The formula
-----------

.. math::

   \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\,\|\mathbf{B}\|},

ranging from :math:`-1` to :math:`1` (0 to 1 for non-negative features) — **1** means identical direction
(very similar), **0** means unrelated (orthogonal).

Where it's used
---------------

It powers **content-based filtering** and **item-item** similarity (recommend items close to those a user
liked), and it is the usual kernel for computing **intra-list similarity / diversity**.
"""

CONTENT["Jaccard index"] = r"""
What it is
----------

The **Jaccard index** (Jaccard similarity coefficient) measures the overlap between two **sets** — the size
of their **intersection** over the size of their **union**. For items, the sets are typically the **users
who liked** each item, or their **tags / features**.

The formula
-----------

.. math::

   J(A, B) = \frac{|A \cap B|}{|A \cup B|},

ranging 0 to 1 — **0** for disjoint sets, **1** for identical ones; the complement :math:`1 - J` is the
**Jaccard distance**.

When to use it
--------------

It is the natural choice for **binary** (like / dislike, present / absent) data, where magnitudes don't
matter — only which elements are shared. Contrast with **cosine**, which works on real-valued vectors.
"""

CONTENT["Intra-List Diversity (ILD)"] = r"""
What it is
----------

**Intra-list diversity** measures how **varied** the items within a **single** recommendation list are —
the antidote to lists of near-identical products. It is defined from the **intra-list similarity (ILS)**,
the **average pairwise similarity** of all items in the list; diversity is its complement.

The formula
-----------

.. math::

   \mathrm{ILD} = \frac{2}{|L|\,(|L|-1)} \sum_{i < j} \big(1 - \mathrm{sim}(i, j)\big),

with :math:`\mathrm{sim}` a **cosine** (over embeddings / features) or **Jaccard** (over tags / genres)
similarity. High ILS → similar items → **low** diversity; low ILS → varied items → **high** diversity.

Why it matters
--------------

Accuracy alone rewards recommending ten versions of the same hit; **diversity** captures whether a list
actually **broadens** what the user sees. It trades off against relevance — the art is a **diverse yet
relevant** list.
"""

MINDMAP.update({
    "Cosine Similarity of Item Features": [
        "Jaccard index", "Intra-List Diversity (ILD)", "Embedding Similarity", "Genre Overlap",
        "Embedding", "Relevance in Recommender Systems",
    ],
    "Jaccard index": [
        "Cosine Similarity of Item Features", "Intra-List Diversity (ILD)", "Genre Overlap",
        "Cramér's V", "Relevance in Recommender Systems", "Catalog Coverage",
    ],
    "Intra-List Diversity (ILD)": [
        "Cosine Similarity of Item Features", "Jaccard index", "Genre Overlap", "Catalog Coverage",
        "Long-Tail Items", "Relevance in Recommender Systems",
    ],
})


# ----------------------------------------------------------------------
# Theme: recommender long-tail, novelty & relevance  (recsys)
# ----------------------------------------------------------------------

CONTENT["Long-Tail Items"] = r"""
What it is
----------

In user-item interaction data, a **small number of "head" items** draw **most** of the interactions, while
a **large number of "long-tail" items** each attract **very few** — a **power-law** (Pareto) popularity
distribution. The long tail is where niche, specialized products live.

Why they matter
---------------

Recommending only head items reinforces **popularity bias** and gives every user the same obvious hits;
surfacing the long tail improves **coverage**, **diversity** and **novelty**, drives **discovery**, and can
expand **sales diversity**. Long-tail items are inherently more **novel** because users are unlikely to
already know them.

The challenge
-------------

Long-tail items have **sparse** interaction data, so they suffer the **cold-start** problem and are hard to
model — the reason accuracy-only systems ignore them. Long-tail recommendation therefore adds **coverage**
and **diversity** metrics on top of accuracy.
"""

CONTENT["Self-Information of Popularity"] = r"""
What it is
----------

A **novelty** measure borrowed from **information theory**. The **self-information** (surprisal) of
recommending an item is the **negative base-2 logarithm** of its **popularity** — the probability that a
random user has interacted with it. Rare events carry more information, so **rare items score high**.

The formula
-----------

.. math::

   \text{self-information}(i) = -\log_2\!\left(\frac{\text{count}(i)}{|U|}\right),

where :math:`\text{count}(i)` is the number of users who consumed item :math:`i` and :math:`|U|` is the
total number of users. A metric averages this over a top-:math:`N` list and across users.

What it captures
----------------

**Popular** items (high probability) have **low** self-information — they are unsurprising; **long-tail**
items have **high** self-information — they are novel. It quantifies how much a recommendation tells the
user something **new**.
"""

CONTENT["Relevance in Recommender Systems"] = r"""
What it is
----------

**Relevance** is whether a recommended item actually **matches the user's tastes and needs** — an item they
would find useful and want to engage with. It is the property that **accuracy** metrics (precision, recall,
NDCG, MAP) are built to measure.

The traditional goal
--------------------

Recommend **as many relevant items as possible**, maximizing accuracy. For a long time this was the sole
objective of recommender systems.

Not enough alone
----------------

A perfectly relevant list can still be **boring** — ten near-identical hits the user already knows. So
relevance is balanced against **novelty**, **diversity** and **coverage**, and modern novelty / diversity
metrics are made **relevance-aware** (rewarding items that are novel **and** relevant) so a system is not
credited for surfacing surprising-but-useless items. The aim is **relevant *and* diverse**.
"""

MINDMAP.update({
    "Long-Tail Items": [
        "Self-Information of Popularity", "Catalog Coverage", "Item Coverage",
        "Intra-List Diversity (ILD)", "Relevance in Recommender Systems", "Diminishing Utility",
    ],
    "Self-Information of Popularity": [
        "Long-Tail Items", "Relevance in Recommender Systems", "Intra-List Diversity (ILD)",
        "Catalog Coverage", "Diminishing Utility", "Dominating in Recommender Systems",
    ],
    "Relevance in Recommender Systems": [
        "Long-Tail Items", "Self-Information of Popularity", "Intra-List Diversity (ILD)",
        "Catalog Coverage", "Cosine Similarity of Item Features", "Dominating in Recommender Systems",
    ],
})


# ----------------------------------------------------------------------
# Theme: recommender genre similarity, dominance & diminishing utility  (recsys)
# ----------------------------------------------------------------------

CONTENT["Genre Overlap"] = r"""
What it is
----------

**Genre overlap** measures how similar two items are by **how many genres (or categories) they share** — a
**metadata-based** similarity for items that carry categorical labels, like movies, music or books. It is
the domain-specific counterpart to **cosine** or **Jaccard** similarity when the features are **genres**.

How it's computed
-----------------

Often as the **Jaccard** of the two items' genre sets — the shared genres over the total distinct genres:

.. math::

   \text{overlap}(i, j) = \frac{|G_i \cap G_j|}{|G_i \cup G_j|}.

Two movies both tagged :math:`\{\text{action}, \text{thriller}\}` overlap fully; an action film and a
documentary don't overlap at all.

Where it's used
---------------

It serves as the **similarity kernel** for **intra-list similarity / diversity** (a list of same-genre
items has high overlap → low diversity), and it underpins **calibrated** recommendation, where the **genre
mix** of a list is kept aligned with the user's historical tastes.
"""

CONTENT["Dominating in Recommender Systems"] = r"""
What it is
----------

A **multi-criteria** notion. One item **dominates** another (**Pareto dominance**) when it is **at least as
good on every criterion** and **strictly better on at least one**:

.. math::

   a \succ b \iff \big(\forall k:\ a_k \ge b_k\big) \ \wedge\ \big(\exists k:\ a_k > b_k\big).

If a hotel is cheaper, closer **and** higher-rated than another, it dominates it.

The skyline
-----------

The items **not dominated** by any other form the **skyline** (the **Pareto frontier**) — the only
candidates worth recommending under multiple objectives, since every dominated item is beaten outright by
something in the skyline.

Why it matters
--------------

Multi-criteria recommenders (price, distance, rating, recency) use dominance to **prune** clearly-inferior
options before ranking. Dominance also names a **failure mode** — when a few **popular** items dominate
everyone's lists, crowding out the long tail and harming exposure fairness.
"""

CONTENT["Diminishing Utility"] = r"""
What it is
----------

The principle of **diminishing marginal utility** applied to recommendation: **each additional similar item
adds less value** than the one before. The third action movie in a row is not three times as useful as the
first — its **marginal** contribution shrinks.

Why it matters
--------------

This is the economic reason to **diversify**. Because utility from redundant items is **submodular**
(diminishing returns), a **varied** list delivers more total value than a list of near-duplicates — so
**marginal relevance** matters more than absolute relevance.

How it's used
-------------

Diversification methods like **Maximal Marginal Relevance (MMR)** and submodular utility-maximization
objectives encode diminishing utility directly — each pick is scored by its relevance **minus** its
redundancy with what's already chosen, so once a genre or topic is covered, further items of the same kind
are penalized.
"""

MINDMAP.update({
    "Genre Overlap": [
        "Cosine Similarity of Item Features", "Jaccard index", "Intra-List Diversity (ILD)",
        "Relevance in Recommender Systems", "Embedding Similarity", "Catalog Coverage",
    ],
    "Dominating in Recommender Systems": [
        "Relevance in Recommender Systems", "Long-Tail Items", "Intra-List Diversity (ILD)",
        "Diminishing Utility", "Catalog Coverage", "Self-Information of Popularity",
    ],
    "Diminishing Utility": [
        "Intra-List Diversity (ILD)", "Relevance in Recommender Systems", "Long-Tail Items",
        "Dominating in Recommender Systems", "Genre Overlap", "Self-Information of Popularity",
    ],
})


# ----------------------------------------------------------------------
# Theme: inventory replenishment — demand forecasting, ROP, safety stock  (ops / supply-chain)
# ----------------------------------------------------------------------

CONTENT["Demand Forecasting"] = r"""
What it is
----------

**Demand forecasting** predicts **future customer demand** for a product from **historical sales**, market
signals, seasonality and known upcoming events. It is the foundation of inventory planning — nearly every
replenishment decision depends on it.

How it's done
-------------

Methods range from **simple baselines** (moving averages) to **time-series** models (ARIMA, exponential
smoothing, Prophet) and **ML**; what matters is capturing **trend**, **seasonality** and demand
**variability** (its standard deviation), not just the average.

Why accuracy matters
--------------------

Forecast **error** propagates downstream — the **reorder point** and **safety stock** are both sized from
the forecast, so a biased or noisy forecast either **starves** shelves (stockouts) or **bloats** inventory
(holding cost). Better forecasts shrink the safety buffer needed for a given service level.
"""

CONTENT["Reorder Point (ROP) Optimization"] = r"""
What it is
----------

The **reorder point** is the inventory level that **triggers a new order** — set so stock arrives just
before you run out. Under a **continuous-review** policy, when on-hand inventory falls to the ROP, a
replenishment order is placed.

The formula
-----------

.. math::

   \text{ROP} = \underbrace{\mu_D \times L}_{\text{demand during lead time}} \;+\; \underbrace{\text{SS}}_{\text{safety stock}},

where :math:`\mu_D` is average demand per period and :math:`L` is the lead time. The first term covers
**expected** usage while the order is in transit; the second buffers **variability**.

Optimizing it
-------------

A good ROP balances **stockout risk** against **holding cost**. It is tuned with **accurate demand
forecasts**, measured **demand and lead-time variability**, and a chosen **service level**; modern systems
recompute it in **real time** as those inputs drift. Drop the safety-stock term only when demand is very
stable and suppliers are reliable.
"""

CONTENT["Safety Stock"] = r"""
What it is
----------

**Safety stock** is the **buffer inventory** held to absorb **uncertainty** in demand and supply — the
cushion that keeps you selling when demand spikes or a shipment is late. It is the difference between a
naive "average" reorder level and a robust one.

The formula
-----------

The statistical form sizes it from the **service level** and demand **variability**:

.. math::

   \text{SS} = Z \times \sigma_D \times \sqrt{L},

where :math:`Z` is the service-level z-score (1.28 for 90\%, 1.65 for 95\%, 2.33 for 99\%), :math:`\sigma_D`
is the standard deviation of demand per period, and :math:`L` is the lead time.

The trade-off
-------------

More safety stock **raises** the service level (fewer stockouts) but **ties up capital** in holding cost —
so the **service level** is chosen by weighing stockout cost against carrying cost, often set higher for
critical or perishable items (via ABC / XYZ classing).
"""

MINDMAP.update({
    "Demand Forecasting": [
        "Reorder Point (ROP) Optimization", "Safety Stock", "Time Series Forecasting",
        "Forecast Error", "Stockout Rate", "Long Lead Times",
    ],
    "Reorder Point (ROP) Optimization": [
        "Safety Stock", "Demand Forecasting", "Long Lead Times", "Fill Rate",
        "Stockout Rate", "Supplier Management",
    ],
    "Safety Stock": [
        "Reorder Point (ROP) Optimization", "Demand Forecasting", "Fill Rate",
        "Stockout Rate", "Backorder Rate", "Long Lead Times",
    ],
})


# ----------------------------------------------------------------------
# Theme: supply side — supplier management, constraints, long lead times  (ops / supply-chain)
# ----------------------------------------------------------------------

CONTENT["Supplier Management"] = r"""
What it is
----------

**Supplier management** is the discipline of selecting, monitoring and collaborating with the **vendors**
who supply materials and goods — turning a loose set of purchase orders into managed, accountable
**relationships**. It spans sourcing, performance tracking and communication.

What it involves
----------------

**Supplier selection** and **diversification** (spreading dependence across several vendors so one failure
doesn't halt production), **performance monitoring** via scorecards and audit trails, and **structured
collaboration** that keeps purchase orders confirmed and current as dates, prices and quantities change.

Why it matters
--------------

Strong supplier relationships **reduce bottlenecks**, delays and quality failures, and give planners early
**visibility** into inbound risk. Over-reliance on a **single supplier** is a key vulnerability —
diversification is the standard hedge against disruption.
"""

CONTENT["Supplier Constraints"] = r"""
What it is
----------

**Supplier constraints** are the limits on what a supplier can actually deliver — **capacity** ceilings,
**minimum order quantities**, **long or changing lead times**, **quality holds**, and transportation
delays. They are the gap between the plan in your system and **supplier reality**.

Why they bite
-------------

A constraint means the original schedule or lead time **no longer reflects** current conditions — parts are
produced but can't ship, a promised date slips, an order can't be filled in full. Left invisible, these
desynchronize the plan from the actual inbound supply.

Managing them
-------------

Constraints aren't always avoidable, but **visibility** — capturing supplier confirmations and changes,
flagging open-order risk — buys **time to respond**: re-plan, expedite, or shift to a **backup supplier**.
Diversification and buffers absorb the rest.
"""

CONTENT["Long Lead Times"] = r"""
What it is
----------

**Lead time** is the total span from **placing an order to having the goods available**; **long lead
times** are extended delays — weeks or months for overseas sourcing, slow procurement, or constrained
production.

The inventory cost
------------------

Longer lead times force businesses to hold **more safety stock** to cover demand over the wait, directly
**raising holding costs** — safety stock scales with the **square root of lead time**:

.. math::

   \text{SS} = Z \times \sigma_D \times \sqrt{L}.

They also cut **agility**: you can't respond quickly to demand shifts, and the risk of **stockouts** and
missed deliveries climbs.

Variability is worse
--------------------

An unpredictable lead time is more damaging than a long-but-stable one, because its **variance** widens the
buffer needed. Causes include geographic distance, production constraints, poor visibility and transport
disruptions.
"""

MINDMAP.update({
    "Supplier Management": [
        "Supplier Constraints", "Long Lead Times", "Safety Stock",
        "Reorder Point (ROP) Optimization", "Demand Forecasting", "Real-Time Inventory Tracking",
    ],
    "Supplier Constraints": [
        "Supplier Management", "Long Lead Times", "Safety Stock", "Slow-Moving SKUs",
        "Demand Forecasting", "Reorder Point (ROP) Optimization",
    ],
    "Long Lead Times": [
        "Safety Stock", "Reorder Point (ROP) Optimization", "Supplier Constraints",
        "Supplier Management", "Demand Forecasting", "Stockout Rate",
    ],
})


# ----------------------------------------------------------------------
# Theme: SKU tracking — SKU, slow-moving SKUs, real-time tracking  (ops / supply-chain)
# ----------------------------------------------------------------------

CONTENT["SKU"] = r"""
What it is
----------

A **stock keeping unit (SKU)** is a unique **alphanumeric code** a business assigns **internally** to each
distinct product or variant, so it can be tracked through inventory. It is a **fingerprint** for an item —
no two distinct products share one — and it encodes key **attributes** (category, brand, color, size), as
in ``WREN-101-RED-SM`` (brand WREN, category 101, red, small).

SKU vs UPC
----------

A SKU is **internal and custom** to each company — you design the format — whereas a **UPC** is a
**universal**, standardized barcode shared across all retailers. The same physical product can carry
different SKUs at different companies.

Why it matters
--------------

The SKU is the **atom** of inventory operations — every movement (sold, received, returned, written off)
references it. It drives **stock monitoring**, **reorder** triggers, warehouse **location** and picking,
**ABC classification**, and per-product **sales analytics** that feed **demand forecasting**.
"""

CONTENT["Slow-Moving SKUs"] = r"""
What it is
----------

**Slow-moving SKUs** are products with **low sales velocity** — they turn over slowly, sitting in the
warehouse and **tying up capital** and space. Items that barely move at all become **dead stock**. They are
the sluggish tail of the sales distribution, the opposite of fast-moving bestsellers.

Why they cost
-------------

Unsold inventory accrues **holding costs** and risks **obsolescence** and write-offs — insufficient demand
for a SKU can inflate its annual cost substantially. Every dollar in a slow-mover is a dollar not in a
product that sells.

What to do
----------

**Identify them early** with SKU-level velocity analytics, before they go obsolete, then act — **discount**,
**bundle** a slow-mover with a fast one, **liquidate**, or **discontinue** — and **reduce or delay**
purchase orders for them. In warehouse layout, they move to **less-accessible** zones (ABC classification).
"""

CONTENT["Real-Time Inventory Tracking"] = r"""
What it is
----------

**Real-time inventory tracking** monitors stock levels **as they change** — continuously and instantly —
instead of relying on periodic manual counts. Every scan at sale, receipt or transfer updates the count
automatically, keyed by **SKU**.

How it works
------------

Modern **POS**, **ERP** and warehouse systems update inventory the moment a product is scanned;
**cloud-based** platforms expose that data from anywhere, keeping **multiple locations and sales channels**
synchronized.

Why it matters
--------------

It prevents **overselling** and **stockouts**, keeps multi-channel stock consistent, and lets the system
**trigger reorders** and **flag anomalies** (a sudden drop in availability) without human polling. It is the
**visibility layer** that makes automated **reorder-point** and **safety-stock** policies actually work.
"""

MINDMAP.update({
    "SKU": [
        "Slow-Moving SKUs", "Real-Time Inventory Tracking", "Demand Forecasting",
        "Reorder Point (ROP) Optimization", "Safety Stock", "Stockout Rate",
    ],
    "Slow-Moving SKUs": [
        "SKU", "Real-Time Inventory Tracking", "Long-Tail Items", "Demand Forecasting",
        "Stockout Rate", "Reorder Point (ROP) Optimization",
    ],
    "Real-Time Inventory Tracking": [
        "SKU", "Slow-Moving SKUs", "Reorder Point (ROP) Optimization", "Safety Stock",
        "Demand Forecasting", "Stockout Rate",
    ],
})


# ----------------------------------------------------------------------
# Theme: classification foundations — binary classification, class probability, log-odds  (concepts / metrics)
# ----------------------------------------------------------------------

CONTENT["Binary Classification"] = r"""
What it is
----------

**Binary classification** predicts one of **two classes** — positive/negative, 1/0, spam/not-spam. The
model doesn't output a bare label directly; it estimates the **probability** that an instance belongs to
the **positive** class, then a **decision threshold** turns that probability into a hard label.

The threshold
-------------

By default the cutoff is **0.5** — probability :math:`\ge 0.5` → class 1, else class 0 — but 0.5 is **not**
always right. On **imbalanced** data (e.g. fraud at 1%), 0.5 may label everything negative; the threshold
is tuned against **precision / recall** or an **ROC** curve to match the cost of each error.

How it's judged
---------------

Predictions map to the **confusion matrix** — true and false positives and negatives — from which
precision, recall, F1 and AUROC follow. The threshold choice moves directly along that trade-off.
"""

CONTENT["Classification Probability"] = r"""
What it is
----------

The **classification probability** is the **score** a classifier assigns that an instance belongs to a
class — an estimate of :math:`P(\text{class} \mid \text{features})` between **0 and 1**. It is the model's
**confidence** *before* any hard decision is made.

From probability to label
-------------------------

A **threshold** converts it to a class (in scikit-learn, ``predict_proba`` gives the probability,
``predict`` applies the cutoff). Two instances scored 0.51 and 0.99 both become "positive," but they are
**not** equally certain — which is why the probability carries more information than the label.

Why calibration matters
-----------------------

The probability is only trustworthy if it is **calibrated** — if events predicted at 0.7 actually happen
about 70% of the time. **Over-** or **under-confident** scores mislead any downstream **risk-based
decision**, so probabilities are validated with calibration curves, not just accuracy.
"""

CONTENT["Log-Odds"] = r"""
What it is
----------

The **log-odds** (or **logit**) is the natural logarithm of the **odds** of an event — the ratio of its
probability to its complement:

.. math::

   \text{logit}(p) = \log\!\left(\frac{p}{1 - p}\right), \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}.

Odds run from 0 (at :math:`p=0`) through 1 (at :math:`p=0.5`) to :math:`\infty` (at :math:`p=1`); taking
the log spreads them onto the full line, from :math:`-\infty` to :math:`+\infty`.

Why models use it
-----------------

A probability is trapped in :math:`[0,1]`, awkward to model with a **linear** function; the log-odds is
**unbounded**, so **logistic regression** (and the final layer of many classifiers) models the log-odds as
a **linear** combination of features — the raw "score" before conversion.

Back to probability
-------------------

The **sigmoid** :math:`\sigma` is the **inverse** of the logit — it maps a log-odds score :math:`z` back to
a probability. So the pipeline runs linear score → log-odds → sigmoid → **classification probability** →
threshold → class.
"""

MINDMAP.update({
    "Classification Probability": [
        "Binary Classification", "Log-Odds", "Sigmoid Function", "Logistic Regression",
        "Probabilistic Forecasts", "Multiclass AUROC",
    ],
    "Binary Classification": [
        "Classification Probability", "Log-Odds", "Sigmoid Function", "Logistic Regression",
        "Classification Models", "Neural Networks",
    ],
    "Log-Odds": [
        "Classification Probability", "Binary Classification", "Sigmoid Function", "Softmax Function",
        "Logistic Regression", "Neural Networks",
    ],
})


# ----------------------------------------------------------------------
# Theme: activations — sigmoid, softmax, squashing functions  (concepts / repr)
# ----------------------------------------------------------------------

CONTENT["Sigmoid Function"] = r"""
What it is
----------

The **sigmoid** function :math:`\sigma` maps **any** real number to the open interval **(0, 1)**, tracing an
**S-shaped** curve:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}.

At :math:`z=0` it returns **0.5**; large positive :math:`z` → near **1**, large negative :math:`z` → near
**0**.

Its role
--------

It is the **inverse of the logit** — it turns a **log-odds** score back into a **probability** — which makes
it the output activation of **logistic regression** and of **binary**-classification output layers, and the
basis of **binary cross-entropy** loss. Each output is an **independent** probability, so sigmoid also
serves **multi-label** problems.

Watch out
---------

In the **hidden** layers of deep networks the sigmoid causes **vanishing gradients** (its slope flattens for
large :math:`|z|`), so ReLU-family activations are preferred there; sigmoid is kept for the **output**.
"""

CONTENT["Softmax Function"] = r"""
What it is
----------

The **softmax** function generalizes the sigmoid to **many** classes. It takes a vector of **K** raw scores
(**logits**), exponentiates each and normalizes by their sum, producing **K** probabilities that each lie
in (0,1) and **sum to 1** — a full **distribution** over mutually exclusive classes:

.. math::

   \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.

Its role
--------

It is the standard **output layer** for **multi-class** classification, trained with **categorical
cross-entropy**. It **amplifies** the largest score toward 1 while **dampening** the rest — a soft "winner."
When :math:`K=2` it **reduces to the sigmoid**.

Watch out
---------

Because the outputs are coupled (they must sum to 1), softmax assumes classes are **mutually exclusive** —
for **multi-label** problems (independent classes) use per-class sigmoids instead. Its probabilities can
also be **poorly calibrated**.
"""

CONTENT["Squashing Function"] = r"""
What it is
----------

A **squashing function** is any function that **compresses** an unbounded input — any real number in
:math:`(-\infty, \infty)` — into a **bounded** range, giving the characteristic **S-shape**. It "squashes"
an infinite domain into a finite interval.

Examples
--------

The **sigmoid** squashes to **(0, 1)** (a probability), **tanh** to **(−1, 1)**, and **softmax** squashes a
vector of logits into probabilities on (0,1). They are the classic **non-linear activations** that let a
network turn raw scores into interpretable, constrained outputs.

Why it matters
--------------

Squashing is what converts an **unbounded** linear score (like the **log-odds**) into something usable — a
probability, or a normalized signal — and the non-linearity is what lets stacked layers model **complex**
patterns. Its flat tails are also the source of **saturation** and vanishing gradients.
"""

MINDMAP.update({
    "Sigmoid Function": [
        "Log-Odds", "Softmax Function", "Squashing Function", "Logistic Regression",
        "Binary Classification", "Classification Probability",
    ],
    "Softmax Function": [
        "Sigmoid Function", "Squashing Function", "Log-Odds", "Neural Networks",
        "Classification Models", "Binary Classification",
    ],
    "Squashing Function": [
        "Sigmoid Function", "Softmax Function", "Neural Networks", "Log-Odds",
        "Binary Classification", "Classification Probability",
    ],
})


# ----------------------------------------------------------------------
# Theme: losses & stability — loss functions, BCE, logit space  (training / concepts)
# ----------------------------------------------------------------------

CONTENT["Loss Functions"] = r"""
What it is
----------

A **loss function** (cost or objective) measures **how wrong** a model's predictions are on the training
data — a single number the model **minimizes**. Lower loss means predictions closer to targets; it is the
signal that **gradient descent** follows to update parameters.

Matching loss to task
---------------------

The loss encodes what "wrong" means — **regression** uses **mean squared error** or MAE; **binary
classification** uses **binary cross-entropy**; **multi-class** uses **categorical cross-entropy**. Squared
error can't tell a bad classification from a disastrous one, which is why classification uses cross-entropy.

Why it matters
--------------

The loss defines what the model actually **optimizes**, so a mismatched loss silently optimizes the wrong
thing (MSE on a sigmoid gives a **non-convex** surface). A good loss is **differentiable**, fits the task,
and aligns with the real objective.
"""

CONTENT["Binary Cross-Entropy (BCE)"] = r"""
What it is
----------

**Binary cross-entropy** (log loss) is the standard **loss** for **binary classification**, for a true
label :math:`y \in \{0,1\}` and predicted probability :math:`p = \sigma(z)`:

.. math::

   \text{BCE} = -\big[\,y \log(p) + (1 - y)\log(1 - p)\,\big], \qquad p = \sigma(z),

averaged over the data.

How it behaves
--------------

The **logarithm** gives an **asymmetric** penalty — a confident-correct prediction costs almost nothing
(:math:`-\log 0.99 \approx 0.01`), a confident-wrong one costs a lot (:math:`-\log 0.01 \approx 4.6`). This
pressures the model to be **confident when right and hesitant when wrong**, pushing toward **calibrated**
probabilities.

Why it fits
-----------

BCE is the **negative log-likelihood** of the **Bernoulli** distribution, so minimizing it is **maximum
likelihood** — the natural partner of the **sigmoid**, with a clean gradient :math:`(p - y)`. Its
multi-class analogue is **categorical cross-entropy** with **softmax**.
"""

CONTENT["Logit Space"] = r"""
What it is
----------

**Logit space** means working with the **raw log-odds** scores :math:`z` (the **logits**) instead of the
probabilities :math:`p = \sigma(z)` — the "pre-sigmoid" world, where values span the **whole real line**
rather than being squeezed into (0,1).

Why it's used
-------------

Computing the loss **directly from logits** is far more **numerically stable**. Converting a logit to a
probability and then taking its log can **underflow** (a tiny :math:`p` rounds to 0, and :math:`\log 0 =
-\infty`) or **overflow** (:math:`e^{z}` for a large logit exceeds the float range); staying in logit space
with the **log-sum-exp** trick avoids both. This is why frameworks fuse sigmoid + BCE
(``BCEWithLogitsLoss``) and softmax + cross-entropy into a single ``from_logits`` op.

The payoff
----------

Better stability and cleaner gradients — the same reason **log-space** helps elsewhere. Logit space ties
the **log-odds**, the **sigmoid**, and the **cross-entropy** loss into one numerically safe computation.
"""

MINDMAP.update({
    "Loss Functions": [
        "Binary Cross-Entropy (BCE)", "Logit Space", "Mean Squared Error (MSE)",
        "Sigmoid Function", "Softmax Function", "Neural Networks",
    ],
    "Binary Cross-Entropy (BCE)": [
        "Loss Functions", "Logit Space", "Sigmoid Function", "Log-Odds",
        "Binary Classification", "Classification Probability",
    ],
    "Logit Space": [
        "Log-Odds", "Binary Cross-Entropy (BCE)", "Sigmoid Function", "Loss Functions",
        "Underflow", "Log-Space",
    ],
})


# ----------------------------------------------------------------------
# Theme: calibration — confidence level, overconfident, underconfident  (calibration)
# ----------------------------------------------------------------------

CONTENT["Confidence Level"] = r"""
What it is
----------

A model's **confidence level** is the **probability it attaches** to its prediction — how sure it is that an
instance belongs to the predicted class. It is only meaningful if it is **calibrated**: a model is
**well-calibrated** when predictions made with confidence :math:`p` are correct about **100p%** of the time
(predict 0.9 → right 90% of the time).

How it's checked
----------------

A **reliability diagram** bins predictions by confidence level and plots confidence against actual
**accuracy**; perfect calibration lies on the **diagonal**. The gap is summarized by the **Expected
Calibration Error (ECE)** — the average distance between confidence and accuracy across bins.

Why it matters
--------------

Raw **accuracy** says nothing about whether the confidence is honest, yet downstream **risk-based
decisions** (which cases to escalate, when to defer) depend on trusting the number. This is the *model*
sense of confidence, distinct from a statistical **confidence interval**.
"""

CONTENT["Overconfident"] = r"""
What it is
----------

A model is **overconfident** when its predicted probabilities are **too high** for its actual accuracy — it
claims more certainty than it earns. A model that is 99% confident but only 90% accurate is overconfident.

How to spot it
--------------

On a **reliability diagram**, overconfident points fall **below** the diagonal (accuracy < confidence), and
the histogram piles predictions near **1.0**. Modern deep networks are frequently overconfident, having
"memorized" training data and carried that certainty to new inputs; **log loss** flags it by punishing
confident-wrong predictions heavily.

Why it's dangerous
------------------

Overconfidence is a **safety** hazard in high-stakes settings — an overconfident medical or fraud model
triggers costly actions on cases it has wrong. It is corrected **post-hoc** with methods like temperature
scaling, Platt scaling or isotonic regression.
"""

CONTENT["Underconfident"] = r"""
What it is
----------

A model is **underconfident** when its predicted probabilities are **too low** for its actual accuracy — it
hedges, claiming less certainty than it deserves. A model that is only 80% confident but 90% accurate is
underconfident.

How to spot it
--------------

On a **reliability diagram**, underconfident points fall **above** the diagonal (accuracy > confidence), and
predictions **cluster near 0.5** rather than committing. It is the mirror image of overconfidence, and a
single model can be **overconfident in some ranges and underconfident in others**.

Why it matters
--------------

Though it feels "safe," underconfidence **wastes** the model's discriminative signal — useful, correct
predictions get muted probabilities, so **thresholds** and **risk-based decisions** under-trigger. Like
overconfidence, it is fixed by **recalibration**.
"""

MINDMAP.update({
    "Confidence Level": [
        "Overconfident", "Underconfident", "Classification Probability",
        "Binary Cross-Entropy (BCE)", "Risk-Based Decisions", "Multiclass AUROC",
    ],
    "Overconfident": [
        "Underconfident", "Confidence Level", "Classification Probability",
        "Binary Cross-Entropy (BCE)", "Risk-Based Decisions", "Softmax Function",
    ],
    "Underconfident": [
        "Overconfident", "Confidence Level", "Classification Probability",
        "Risk-Based Decisions", "Binary Cross-Entropy (BCE)", "Sigmoid Function",
    ],
})


# ----------------------------------------------------------------------
# Theme: probabilistic reasoning & decisions — probability forecasts, risk-based decisions, likelihood  (inference / risk)
# ----------------------------------------------------------------------

CONTENT["Probability Forecasts"] = r"""
What it is
----------

A **probability forecast** states the **probability of a specific outcome or event** rather than a single
deterministic value — "a 70% chance of rain," "a 4% probability of default." It is the event-focused face of
**probabilistic forecasting**: instead of committing to one number, it quantifies **how likely** each
outcome is.

How it's expressed
------------------

As a probability per event, or as **certainty levels** on a distribution — a **P80** forecast is 80% certain
(a 20% chance of being exceeded), a **P50** is the median. A full probability forecast carries the **whole
distribution** of outcomes.

Why it matters
--------------

Probabilities are what **risk-based decisions** consume — a threshold on the forecast (escalate if
:math:`P(\text{default}) > 5\%`) turns uncertainty into action. They must be **calibrated** to be trusted,
and are scored with **strictly proper scoring rules** (e.g. the log score), which reward honest
probabilities.
"""

CONTENT["Risk-Based Decisions"] = r"""
What it is
----------

A **risk-based decision** chooses the action that best balances **predicted probabilities against costs** —
it acts on **expected risk**, not on the raw label. The basic definition of risk is the **expected cost**: a
loss weighted by its probability,

.. math::

   R = \mathbb{E}[L] = \sum_i L_i \, p_i.

How it works
------------

Under **Bayesian decision theory**, you pick the action that **minimizes total expected risk** given the
**cost** of each error. Because false positives and false negatives usually cost **differently**, the
optimal **decision threshold** on a probability is generally **not 0.5** — a cheap-to-check,
expensive-to-miss event (fraud, disease) warrants a **lower** threshold.

Why calibration matters
-----------------------

The whole scheme assumes the probabilities are **honest** — an **overconfident** model makes the
expected-cost arithmetic wrong and triggers bad actions. So risk-based decisions rest on **calibrated**
probability forecasts.
"""

CONTENT["Likelihood"] = r"""
What it is
----------

The **likelihood** is the probability of the **observed data given a model's parameters** —
:math:`\mathcal{L}(\theta) = P(\text{data} \mid \theta)`. The twist is perspective: it is read as a function
of the **parameters** :math:`\theta` (with the data fixed), asking *which parameters make what we saw most
probable?*

Maximum likelihood
------------------

**MLE** picks the parameters that **maximize** the likelihood (in practice the **log**-likelihood, since
sums are easier and more stable than products):

.. math::

   \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; \mathcal{L}(\theta).

It is the dominant engine of statistical **inference** — logistic regression, and most classifiers, are fit
this way.

The connection
--------------

Minimizing **cross-entropy** (like **binary cross-entropy**) is exactly **maximizing likelihood** — BCE is
the **negative log-likelihood** of the Bernoulli model. Likelihood **ratios** also underlie optimal
**decision** rules, tying estimation and decision-making together.
"""

MINDMAP.update({
    "Probability Forecasts": [
        "Probabilistic Forecasts", "Risk-Based Decisions", "Classification Probability",
        "Quantile Forecasts", "Prediction Intervals (PI)", "Strictly Proper Scoring Rules",
    ],
    "Risk-Based Decisions": [
        "Probability Forecasts", "Confidence Level", "Classification Probability",
        "Overconfident", "Likelihood", "Binary Classification",
    ],
    "Likelihood": [
        "Binary Cross-Entropy (BCE)", "Logistic Regression", "Probability Forecasts",
        "Risk-Based Decisions", "Loss Functions", "Correlation",
    ],
})


# ----------------------------------------------------------------------
# Theme: regression error metrics — MSE, RMSE, MAPE  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Mean Squared Error (MSE)"] = r"""
What it is
----------

**Mean squared error** is the average of the **squared** differences between predictions and truth — the
workhorse **loss** and metric for **regression** (also called **L2** or quadratic loss):

.. math::

   \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2.

Squaring makes every error positive and **weights large errors far more** than small ones.

How it behaves
--------------

Because errors are squared, MSE is dominated by **big misses** and is **sensitive to outliers** — one large
error can swamp many small ones. Its units are the **square** of the target's, so it doesn't read directly;
minimizing MSE yields the **mean** (conditional expectation) as the optimal prediction.

Why it's used
-------------

It is **smooth and differentiable**, ideal for **gradient descent** (it is the loss regression networks
minimize), and it is the **maximum-likelihood** loss under **Gaussian** noise. When outliers should count
less, **MAE** is preferred.
"""

CONTENT["Root Mean Squared Error (RMSE)"] = r"""
What it is
----------

**Root mean squared error** is the **square root** of the MSE:

.. math::

   \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2}.

The root returns the error to the target's **original units**, so it reads as a **typical error magnitude**.

How it behaves
--------------

RMSE keeps MSE's heavy **penalty on large errors** and its **outlier sensitivity**, but is far more
**interpretable** — an RMSE of 5 means predictions are off by about **5 units** on average. It ranks models
**identically** to MSE, is always **≥ the MAE**, and the RMSE–MAE gap widens as the **error variance** grows.

When to use it
--------------

Report RMSE for regression when **large errors are costly** and you want a number in the data's units; pair
it with **R²** for a scale-free complement. Like R², it also **falls** as you add variables, so watch
**overfitting**.
"""

CONTENT["Mean Absolute Percentage Error (MAPE)"] = r"""
What it is
----------

**Mean absolute percentage error** expresses each error as a **percentage of the actual** value, averaged:

.. math::

   \text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|.

This makes it **scale-free** — comparable across series of wildly different magnitudes.

Why it's popular
----------------

"8% off" is instantly meaningful to non-specialists and lets you compare accuracy across products or regions
on different scales — hence its ubiquity in **demand forecasting** and business reporting.

The pitfalls
------------

MAPE **explodes when actuals are zero or near-zero** (the denominator → 0), and it is **asymmetric** —
over-forecasts can incur unbounded percentage error while under-forecasts are capped at 100%, biasing it
toward models that **under-predict**. For intermittent or zero-heavy data, **scaled** errors like **MASE**
are safer.
"""

MINDMAP.update({
    "Mean Squared Error (MSE)": [
        "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)", "Loss Functions",
        "R² (R-squared)", "Regression Models", "Forecast Error",
    ],
    "Root Mean Squared Error (RMSE)": [
        "Mean Squared Error (MSE)", "Mean Absolute Percentage Error (MAPE)", "R² (R-squared)",
        "Loss Functions", "Regression Models", "Average Absolute Error (AAE)",
    ],
    "Mean Absolute Percentage Error (MAPE)": [
        "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Relative accuracy",
        "Forecast Error", "R² (R-squared)", "Regression Models",
    ],
})


# ----------------------------------------------------------------------
# Theme: classification evaluation — precision, ROC-AUC, PR-AUC  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Precision (a.k.a. Positive Predictive Value, PPV)"] = r"""
What it is
----------

**Precision** (also **positive predictive value**, **PPV**) is the fraction of **correct** positive
predictions among **all** positive predictions:

.. math::

   \text{Precision} = \frac{TP}{TP + FP}.

It answers: *of everything the model flagged positive, how much really was?*

The trade-off
-------------

Precision counts **false positives** against you but ignores **false negatives** — so a model can reach high
precision by only flagging the cases it is surest about. It is therefore read **together with recall**
(:math:`TP/(TP+FN)`), which counts the positives **missed**; the **F1** score is their harmonic mean.

When it matters
---------------

Precision is the priority when a **false positive is costly** — a spam filter deleting real mail, a system
flagging innocent transactions as fraud — where you would rather miss some positives than raise false
alarms.
"""

CONTENT["ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)"] = r"""
What it is
----------

**ROC-AUC** is the **area under the ROC curve**, which plots the **true positive rate** (recall) against the
**false positive rate** as the decision **threshold** sweeps from 0 to 1. It condenses that whole curve into
**one number** in :math:`[0, 1]`.

How to read it
--------------

**1.0** is a perfect classifier, **0.5** is random guessing. It has a clean probabilistic meaning — the
chance that a randomly chosen **positive** is scored **higher** than a randomly chosen **negative** — so it
measures **ranking** quality, independent of any single threshold and invariant to the score **scale**.

The caveat
----------

Because the **false positive rate** has all the **true negatives** in its denominator, ROC-AUC can look
**optimistically high** on **imbalanced** data where negatives dominate — a model can score well while still
flooding a rare positive class with false alarms. There, **PR-AUC** is more honest.
"""

CONTENT["Precision–Recall AUC (PR-AUC)"] = r"""
What it is
----------

**PR-AUC** is the **area under the precision–recall curve**, which plots **precision** against **recall**
across thresholds — also called **Average Precision (AP)**, the mean precision over all recall levels. It
ranges :math:`[0, 1]`, higher is better.

Why it's imbalance-friendly
---------------------------

Unlike ROC-AUC, PR-AUC ignores **true negatives** entirely and focuses on the **positive** class, so it
stays informative when positives are **rare**. Its **baseline** also shifts with prevalence — random
guessing scores the **positive-class ratio** (0.5 when balanced, 0.01 at 1% positive), not a fixed 0.5.

When to use it
--------------

Reach for PR-AUC on **highly imbalanced** problems where finding the **minority** positive is the goal —
fraud, rare-disease, anomaly detection — where a high ROC-AUC can be misleading. Report it **alongside**
ROC-AUC for the full picture.
"""

MINDMAP.update({
    "Precision (a.k.a. Positive Predictive Value, PPV)": [
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Precision–Recall AUC (PR-AUC)", "Binary Classification", "Multiclass AUROC",
        "Classification Probability", "Multiclass Classification",
    ],
    "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)": [
        "Precision–Recall AUC (PR-AUC)", "Precision (a.k.a. Positive Predictive Value, PPV)",
        "Multiclass AUROC", "Binary Classification", "Classification Probability",
        "Multiclass Classification",
    ],
    "Precision–Recall AUC (PR-AUC)": [
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Precision (a.k.a. Positive Predictive Value, PPV)", "Binary Classification",
        "Multiclass AUROC", "Classification Probability", "Macro AUC",
    ],
})


# ----------------------------------------------------------------------
# Theme: multiclass AUROC — multiclass classification, OvR, macro AUC  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Multiclass Classification"] = r"""
What it is
----------

**Multiclass classification** predicts one of **more than two** mutually exclusive classes — a handwritten
digit (0–9), a species, a product category. It generalizes **binary** classification, and its models usually
end in a **softmax** layer that outputs a probability over the **K** classes.

The evaluation twist
--------------------

Metrics built for two classes — **ROC-AUC**, **precision**, recall — have no direct multiclass definition,
because "**positive** vs negative" is ambiguous with many classes. To use them, the problem is **binarized**
(one class vs the others) and the per-class scores are **averaged**.

The two decompositions
----------------------

**One-vs-Rest** turns K classes into K binary problems (each class against the rest); **One-vs-One** compares
every **pair**. Either produces a set of per-class or per-pair scores that a **micro** or **macro** average
then collapses into a single number.
"""

CONTENT["One-vs-Rest (OvR)"] = r"""
What it is
----------

**One-vs-Rest** (also **one-vs-all**) reduces a **K-class** problem to **K binary** ones — in each, a single
class is the **positive** and all the others are lumped together as the **negative**. It's the simplest way
to let binary tools handle many classes.

How it's used
-------------

For a K-class model you get **K** ROC curves and AUCs, one per class, each answering *how well does the model
separate this class from everything else?* scikit-learn exposes it as ``multi_class='ovr'``; it also matches
the **multilabel** setting, where classes aren't exclusive.

The catch
---------

Each binary split is **imbalanced** — the positive class is only about **1/K** of the data, and the "rest"
group's makeup shifts with the class distribution, so OvR scores are **sensitive to class imbalance**. The
alternative, **One-vs-One**, compares class **pairs** and is less imbalance-prone but trains
:math:`O(K^2)` classifiers.
"""

CONTENT["Macro AUC"] = r"""
What it is
----------

**Macro AUC** averages the per-class AUCs (from **One-vs-Rest**) with **equal weight** — every class counts
the **same**, no matter how rare or common. It answers *how well does the model do on the average class?*

Macro vs micro
--------------

The contrast is **micro AUC**, which pools every class's true/false-positive contributions into **one global**
curve, effectively **weighting by prevalence** so frequent classes dominate. Macro treats a class with 10
samples exactly like one with 10,000.

When to use which
-----------------

**Macro** is the choice when the **rare** classes matter as much as the common ones (you want minority
performance to show), while **micro** (or a **weighted** macro) better reflects **overall** accuracy on
**imbalanced** data. Reporting both reveals whether a good score is carried by the majority classes.
"""

MINDMAP.update({
    "Multiclass Classification": [
        "One-vs-Rest (OvR)", "Macro AUC", "Micro AUC", "Binary Classification",
        "Softmax Function", "Multiclass AUROC",
    ],
    "One-vs-Rest (OvR)": [
        "Multiclass Classification", "Macro AUC", "Micro AUC", "Binary Classification",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)", "Multiclass AUROC",
    ],
    "Macro AUC": [
        "Micro AUC", "One-vs-Rest (OvR)", "Multiclass Classification",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Multiclass AUROC", "Partial AUC (pAUC)",
    ],
})


# ----------------------------------------------------------------------
# Theme: AUC variants & baseline — micro AUC, partial AUC, accuracy  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Micro AUC"] = r"""
What it is
----------

**Micro AUC** is the multiclass / multilabel AUROC that **pools** every class's predictions into **one
global** ROC curve — aggregating all true-positive and false-positive counts across classes, then computing
a single AUC. Because it counts **every prediction** equally, frequent classes contribute more.

Micro vs macro
--------------

Where **macro AUC** averages per-class AUCs with equal weight, micro AUC is effectively **weighted by
prevalence** — a rare class with few samples barely moves it. Micro answers *how well does the model do on
the average prediction?*, macro *on the average class?*

When to use it
--------------

Micro AUC suits **imbalanced** multiclass problems when you care about **overall** performance dominated by
common classes, and it matches how a **multilabel** system is scored (over the flattened label matrix).
Report it beside **macro** to expose class-size effects.
"""

CONTENT["Partial AUC (pAUC)"] = r"""
What it is
----------

**Partial AUC** is the area under **only a slice** of the ROC curve — typically a confined range of **false
positive rate** (say :math:`\text{FPR} \le 0.1`), sometimes of TPR, or both. It focuses the metric on the
**operating region** that actually matters.

Why restrict
------------

Full AUC weights **all** FPR regions equally, but many are operationally irrelevant — a radiologist doesn't
care about performance at 80% FPR, a bank won't run a fraud model that flags half of transactions. pAUC
scores the model **where it will be used** (low FPR / high TPR for screening), and is especially apt for
**low-prevalence** data needing high specificity.

The trade-off
-------------

pAUC is more **decision-relevant** than full AUC and distinguishes curves that **cross** yet share the same
total AUC, but it **ignores** the ROC outside the band and its raw value depends on the **interval width**
(so it's often **standardized**, e.g. McClish, back to :math:`[0,1]`). The bounds must be justified by the
use case.
"""

CONTENT["Accuracy"] = r"""
What it is
----------

**Accuracy** is the simplest classification metric — the **fraction of predictions that are correct**:

.. math::

   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}.

It answers *what share did the model get right?*

The imbalance trap
------------------

Accuracy is **misleading on imbalanced** data — if 99% of cases are negative, a model that predicts
"negative" for **everything** scores **99%** while catching **zero** positives. That false sense of security
is why fraud, disease and anomaly tasks report **precision / recall**, **AUC** or **F1** instead.

When it's fine
--------------

Accuracy is a reasonable headline when classes are **roughly balanced** and every error costs about the
**same**. Otherwise it hides **which** errors happen — a **confusion matrix** and threshold-aware metrics
tell the real story.
"""

MINDMAP.update({
    "Micro AUC": [
        "Macro AUC", "One-vs-Rest (OvR)", "Multiclass Classification",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Multiclass AUROC", "Partial AUC (pAUC)",
    ],
    "Partial AUC (pAUC)": [
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)", "Macro AUC",
        "Micro AUC", "Multiclass AUROC", "Binary Classification", "Precision–Recall AUC (PR-AUC)",
    ],
    "Accuracy": [
        "Precision (a.k.a. Positive Predictive Value, PPV)",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Binary Classification", "Multiclass Classification", "Macro AUC",
        "Precision–Recall AUC (PR-AUC)",
    ],
})


# ----------------------------------------------------------------------
# Theme: distribution divergences — KL, JS, KS test  (drift / inference)
# ----------------------------------------------------------------------

CONTENT["Kullback–Leibler (KL) Divergence"] = r"""
What it is
----------

**KL divergence** measures how much one probability distribution **P** differs from a reference **Q** — the
**relative entropy**, the expected extra "surprise" from using Q when the truth is P:

.. math::

   D_{\text{KL}}(P \,\|\, Q) = \sum_{x} P(x)\,\log\!\frac{P(x)}{Q(x)} \;\ge\; 0,

(an integral for continuous distributions). It is **0** only when P = Q.

Its quirks
----------

KL is **asymmetric** — :math:`D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)` — so it is **not a true
distance**, and it **blows up** when Q assigns zero probability where P doesn't (it needs **absolute
continuity**). It is a **directed** measure, not a symmetric metric.

Where it's used
---------------

It is the core of **variational** methods (the VAE loss), of **feature selection**, and of **drift
detection**, where it behaves much like **PSI** — a solid default on large datasets, though its asymmetry
means you read it as a **degree** of drift, not a comparable distance.
"""

CONTENT["Jensen–Shannon (JS) Divergence"] = r"""
What it is
----------

**JS divergence** is the **symmetric, bounded** repair of KL. It averages the KL of each distribution to
their **mixture** :math:`M = \tfrac{1}{2}(P+Q)`:

.. math::

   D_{\text{JS}}(P \,\|\, Q) = \tfrac{1}{2}D_{\text{KL}}(P \,\|\, M) + \tfrac{1}{2}D_{\text{KL}}(Q \,\|\, M).

Unlike KL it is always **finite** and **symmetric**.

Its properties
--------------

JS ranges from **0** (identical) to a bounded maximum (**1** in bits, :math:`\log 2` in nats, when the
distributions are **disjoint**). Its **square root** is the **Jensen–Shannon distance**, which *is* a proper
metric — so JS gives a well-behaved, comparable measure of distributional difference.

Where it's used
---------------

It works on **numerical and categorical** features alike and is a popular **drift** signal — stable, less
noisy, and slightly **more sensitive** than KL or PSI — which is why monitoring systems favour it when a
symmetric, bounded score is wanted.
"""

CONTENT["Kolmogorov–Smirnov (KS) Test"] = r"""
What it is
----------

The **KS test** is a **non-parametric** test of whether two samples come from the **same distribution**
(two-sample), or whether a sample matches a **reference** distribution (goodness-of-fit). It compares their
**cumulative distribution functions (CDFs)**.

The statistic
-------------

Its **D-statistic** is the **largest vertical gap** between the two CDFs:

.. math::

   D = \sup_{x} \,\big|F_1(x) - F_2(x)\big|.

A bigger D means the distributions are further apart. Because it uses the CDF directly, it makes **no
assumptions** about the distribution's shape — its great strength.

Where it's used
---------------

With a null of "same distribution," a small p-value flags a **significant** difference — making the KS test
a standard tool for **drift detection** on continuous features and for **goodness-of-fit** checks. It
underlies the **KS statistic** used as a drift metric.
"""

MINDMAP.update({
    "Kullback–Leibler (KL) Divergence": [
        "Jensen–Shannon (JS) Divergence", "Kolmogorov–Smirnov (KS) Test", "Data Drift",
        "Statistical Tests", "Chi-square (χ²) Test", "Cramér's V",
    ],
    "Jensen–Shannon (JS) Divergence": [
        "Kullback–Leibler (KL) Divergence", "Kolmogorov–Smirnov (KS) Test", "Data Drift",
        "Concept Drift", "Statistical Tests", "Cramér's V",
    ],
    "Kolmogorov–Smirnov (KS) Test": [
        "KS Statistic (Kolmogorov–Smirnov Statistic)", "Kullback–Leibler (KL) Divergence",
        "Jensen–Shannon (JS) Divergence", "Data Drift", "Cumulative Distribution Function (CDF)",
        "Statistical Tests",
    ],
})


# ----------------------------------------------------------------------
# Theme: descriptive statistics — mean, median, outlier  (probstats)
# ----------------------------------------------------------------------

CONTENT["Mean"] = r"""
What it is
----------

The **(arithmetic) mean** is the **average** — the sum of all values divided by their count:

.. math::

   \mu = \frac{1}{n}\sum_{i=1}^{n} x_i.

It uses **every** data point, which makes it the natural summary of a **symmetric** distribution and the
value that **minimizes squared error**.

Its weakness
------------

Because it incorporates every value, the mean is **sensitive to outliers** and **skew** — a single extreme
value drags it toward the tail. Statisticians say its **breakdown point is 0%**: one bad point can move it
arbitrarily. On **skewed** data like income, the mean **overstates** where most values sit.

When to use it
--------------

Prefer the mean for **roughly symmetric**, outlier-free numeric data, where it is efficient and
mathematically convenient (it feeds variance, standard error, regression). For skewed or outlier-heavy
data, reach for the **median**. It is undefined for **categorical** data.
"""

CONTENT["Median"] = r"""
What it is
----------

The **median** is the **middle value** of the data once sorted — for an odd count the single center value,
for an even count the **average of the two** middle values. It splits the data into two **equal halves** and
is exactly the **0.5 quantile**.

Its strength
------------

The median is **robust to outliers** — extreme values change **which** point sits in the middle only
slightly, and never by their magnitude, so its **breakdown point is 50%**. That resilience makes it the
right summary for **skewed** data (income, house prices) and the value that **minimizes absolute error**.

The trade-off
-------------

The median ignores the **magnitudes** of all but the central values, so it discards information the mean
uses, and it is **harder to manipulate** algebraically. It needs **orderable** data — it works on ordinal,
not nominal, values.
"""

CONTENT["Outlier"] = r"""
What it is
----------

An **outlier** is an **extreme, atypical** value that sits far from the bulk of the data. Outliers arise
from genuine rare events, measurement or data-entry **errors**, or a mixture of populations — and spotting
them matters because they can **distort** an analysis.

What they break
---------------

Outliers hit statistics that use **every** value hardest — the **mean**, the **variance / standard
deviation**, and **squared-error** losses like **MSE**, which square the large residual — while **robust**
measures like the **median** barely move. This gap between mean and median is itself a **signal** of outliers
or skew.

How they're handled
-------------------

Outliers are **detected** (z-scores, the **IQR** rule, distance- or model-based methods), then
**investigated** — a true error is corrected or removed, but a genuine extreme is often **kept** and handled
with **robust** methods or transforms rather than silently discarded.
"""

MINDMAP.update({
    "Mean": [
        "Median", "Outlier", "Mean Squared Error (MSE)", "Normal Distribution",
        "Standard Error (SE)", "Regression Models",
    ],
    "Median": [
        "Mean", "Outlier", "Quantile Level", "Normal Distribution",
        "Regression Models", "Mean Squared Error (MSE)",
    ],
    "Outlier": [
        "Mean", "Median", "Mean Squared Error (MSE)", "Normal Distribution",
        "Z-Score", "Standard Error (SE)",
    ],
})


# ----------------------------------------------------------------------
# Theme: feature engineering — sensitivity, encode, normalize  (features)
# ----------------------------------------------------------------------

CONTENT["Sensitivity in Feature Engineering"] = r"""
What it is
----------

**Sensitivity** in feature engineering is the degree to which a model's predictions **depend on how features
are represented** — their **scale**, distribution, and encoding. Some algorithms are highly **sensitive** to
these choices; others are nearly **invariant** — and that difference decides how much preprocessing you must
do.

Who's sensitive
---------------

**Distance-based** models (KNN, SVM), **gradient-descent** learners (linear / logistic regression, neural
nets), and **regularized** models are **scale-sensitive** — a feature with a large range will **dominate**
distances or gradients unless it's **normalized**. **Tree-based** models (decision trees, random forests,
gradient boosting) split one feature at a time and are essentially **scale-invariant**.

Why it matters
--------------

Knowing a model's sensitivity tells you what preprocessing is **required** versus **wasted** — you **must**
scale for KNN or a neural net, but scaling for a random forest changes little. The same lens underlies
**feature-sensitivity analysis**: measuring how much the output moves when a feature changes reveals which
features the model actually **relies on**.
"""

CONTENT["Encode (in Feature Engineering)"] = r"""
What it is
----------

**Encoding** converts **categorical** (non-numeric) data into a **numerical** form, because most ML
algorithms operate only on numbers. The trick is to add the numbers **without inventing meaning** that isn't
there.

The methods
-----------

**One-hot encoding** turns a category into several **binary** columns (exactly one 1) — right for **nominal**
categories with **low cardinality**, since it implies no order. **Ordinal / label** encoding assigns
integers, valid only when categories are genuinely **ordered**. For **high-cardinality** features,
**frequency**, **target**, or learned **embedding** encodings avoid the column explosion of one-hot.

Getting it wrong
----------------

Label-encoding a **nominal** variable (red = 1, blue = 2, green = 3) falsely tells the model green > blue — a
fake ordering that distance- and gradient-based models will believe. Match the encoding to whether the
category is **ordered**, and to its **cardinality**.
"""

CONTENT["Normalize (in Feature Engineering)"] = r"""
What it is
----------

**Normalizing** (feature scaling) rescales numeric features to a **comparable range** so that features with
large magnitudes don't **dominate** the ones with small magnitudes. It changes a feature's **range**, not its
data type.

The two workhorses
------------------

**Min-max scaling** maps values to **[0, 1]**; **standardization (Z-score)** centers to **mean 0, standard
deviation 1**:

.. math::

   x' = \frac{x - \min(x)}{\max(x) - \min(x)} \quad\text{(min-max)}, \qquad x' = \frac{x - \mu}{\sigma} \quad\text{(Z-score)}.

Min-max suits bounded data; Z-score suits Gaussian-ish data and methods like **PCA**. Min-max is
**outlier-sensitive**, so **robust scaling** (median and **IQR**) is used when outliers are present.

When and when not
-----------------

Normalize for **scale-sensitive** models (KNN, SVM, neural nets); it **speeds convergence** and prevents
large-value bias. Don't normalize **one-hot** or categorical columns (they're already 0/1 and it destroys
their meaning), and fit the scaler on the **training set only** to avoid leakage.
"""

MINDMAP.update({
    "Sensitivity in Feature Engineering": [
        "Encode (in Feature Engineering)", "Normalize (in Feature Engineering)", "Feature Values",
        "Outlier", "Support Vector Machines (SVMs)", "Decision Trees",
    ],
    "Encode (in Feature Engineering)": [
        "Normalize (in Feature Engineering)", "Sensitivity in Feature Engineering", "Embedding",
        "Feature Values", "Outlier", "Neural Networks",
    ],
    "Normalize (in Feature Engineering)": [
        "Encode (in Feature Engineering)", "Sensitivity in Feature Engineering", "Outlier",
        "Z-Score", "Normal Distribution", "Neural Networks",
    ],
})


# ----------------------------------------------------------------------
# Theme: hypothesis testing — statistical tests, chi-square, power analysis  (inference)
# ----------------------------------------------------------------------

CONTENT["Statistical Tests"] = r"""
What it is
----------

A **statistical test** is a formal procedure for deciding whether data provide enough evidence to **reject**
a default assumption. Every test follows the same **five steps**: state a **null (H₀)** and **alternative
(Hₐ)** hypothesis, pick a **significance level α**, compute a **test statistic**, find its **p-value**, and
**interpret**.

The decision rule
-----------------

**Reject H₀ when p < α** (the data would be surprising if H₀ were true), otherwise **fail to reject** it.
Crucially, failing to reject is **not** proof that H₀ is true — absence of evidence is not evidence of
absence. Two errors are possible: **Type I** (rejecting a true H₀, rate α) and **Type II** (missing a real
effect, rate β).

The families
------------

Tests split into **parametric** (assuming a distribution — t-test, ANOVA) and **non-parametric**
(assumption-free — **KS**, chi-square), and into one- vs two-sided. The right test depends on the **data
type**, the **question**, and the assumptions you can defend.
"""

CONTENT["Chi-square (χ²) Test"] = r"""
What it is
----------

The **chi-square test** works on **categorical** data, comparing **observed** counts to the **expected**
counts under a null hypothesis:

.. math::

   \chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}.

A large :math:`\chi^2` means observations stray far from expectation.

Its two forms
-------------

**Goodness-of-fit** asks whether one categorical variable follows a **specified distribution** (do dice
rolls look fair?); **independence** asks whether two categorical variables in a **contingency table** are
**associated** (is purchase related to region?). The statistic is compared to the **χ² distribution** with
the appropriate **degrees of freedom**.

Reading it and its limits
-------------------------

A small **p-value** rejects the null (a real fit failure or association); the strength of an association is
then summarized by **Cramér's V**. The test needs **adequate expected counts** per cell and has **low
power** on small samples — a non-significant result is weak evidence, not confirmation.
"""

CONTENT["Power Analysis"] = r"""
What it is
----------

**Power analysis** plans a study around its **statistical power** — the probability of **detecting a real
effect** (correctly rejecting H₀ when it's false), which equals :math:`1 - \beta`. Most often it answers
*how many samples do I need?*

The four levers
---------------

Power, **effect size**, **significance level α**, and **sample size** are locked in a relationship — fix any
**three** and the fourth is determined. Bigger effects, larger samples, or a looser α all **raise** power;
the usual target is **≥ 0.80**.

Why do it first
---------------

Run **before** collecting data, power analysis prevents **underpowered** experiments that waste resources
and are likely to **miss** true effects (a high Type II risk). Run **after**, it tells you how much you could
realistically have detected — and warns against over-reading a **non-significant** result.
"""

MINDMAP.update({
    "Statistical Tests": [
        "Chi-square (χ²) Test", "Kolmogorov–Smirnov (KS) Test", "Power Analysis",
        "Statistical Power", "Confidence Intervals (CIs)", "A/B Testing",
    ],
    "Chi-square (χ²) Test": [
        "Statistical Tests", "Cramér's V", "Kolmogorov–Smirnov (KS) Test", "Power Analysis",
        "Kullback–Leibler (KL) Divergence", "Data Drift",
    ],
    "Power Analysis": [
        "Statistical Tests", "Statistical Power", "Chi-square (χ²) Test", "A/B Testing",
        "Confidence Intervals (CIs)", "Kolmogorov–Smirnov (KS) Test",
    ],
})


# ----------------------------------------------------------------------
# Theme: confidence intervals — CIs, Clopper-Pearson, Wilson score  (inference)
# ----------------------------------------------------------------------

CONTENT["Confidence Intervals (CIs)"] = r"""
What it is
----------

A **confidence interval** is a **range** of plausible values for an unknown parameter — a mean, a
proportion — computed from a sample together with a **confidence level** (typically **95%**). It expresses
the **uncertainty** in a point estimate: a wider interval means less precision.

What the level means
--------------------

The confidence level is a statement about the **procedure**, not any one interval. If you repeated the study
many times, about **95% of the intervals** you built would contain the true value — it is **not** a 95%
probability that the parameter lies in *this* interval (in the frequentist view the parameter is fixed).
Intervals **narrow** as the sample size **grows**.

How they're built
-----------------

A CI is typically an estimate **± a margin of error** (a critical value times a **standard error**), but for
tricky quantities like a **binomial proportion** there are several methods — **Wald**, **Wilson**,
**Clopper–Pearson**, **bootstrap** — that trade **coverage** against **width**.
"""

CONTENT["Clopper–Pearson Interval"] = r"""
What it is
----------

The **Clopper–Pearson interval** is the **"exact"** confidence interval for a **binomial proportion** — built
directly from the **binomial distribution** (via **Beta-distribution** quantiles) rather than a normal
approximation. It **inverts** the binomial CDF to find the proportions consistent with the data.

Its guarantee
-------------

It **never has less than** the nominal coverage — a 95% Clopper–Pearson interval covers the true proportion
**at least** 95% of the time for **every** p and n. That safety is its selling point when you **must not**
under-cover.

The cost
--------

Guaranteeing coverage makes it **conservative** — the actual coverage is often **~99%**, so the interval is
**wider than necessary** and demands larger samples for a given precision. It is the **widest** of the common
methods, best reserved for **very small samples** or when guaranteed coverage is essential.
"""

CONTENT["Wilson Score Interval"] = r"""
What it is
----------

The **Wilson score interval** is a well-calibrated confidence interval for a **binomial proportion**, derived
by improving the crude **normal-approximation (Wald)** interval. Introduced by E. B. Wilson in 1927, it is
**asymmetric** and always stays **within [0, 1]**.

Why it's better
---------------

Unlike the **Wald** interval, it doesn't **overshoot** past 0 or 1 and doesn't collapse to **zero width**
when the observed proportion is 0 or 1; and unlike **Clopper–Pearson**, it isn't overly **conservative** —
its coverage sits **close to nominal**, so its intervals are **narrower**. That balance makes it the
**recommended default** in most applications.

The caveats
-----------

Its coverage can dip **slightly below** nominal for a few awkward proportions, and for **extremely small**
samples the guaranteed **Clopper–Pearson** may still be safer. A continuity-corrected variant exists for
tighter coverage.
"""

MINDMAP.update({
    "Confidence Intervals (CIs)": [
        "Clopper–Pearson Interval", "Wilson Score Interval", "Bootstrap Confidence Intervals (CIs)",
        "Standard Error (SE)", "Statistical Tests", "Normal Distribution",
    ],
    "Clopper–Pearson Interval": [
        "Confidence Intervals (CIs)", "Wilson Score Interval", "Normal Distribution",
        "Statistical Tests", "Bootstrap Confidence Intervals (CIs)", "Standard Error (SE)",
    ],
    "Wilson Score Interval": [
        "Confidence Intervals (CIs)", "Clopper–Pearson Interval", "Normal Distribution",
        "Standard Error (SE)", "Z-Score", "Statistical Tests",
    ],
})


# ----------------------------------------------------------------------
# Theme: averaging strategies — micro, macro, weighted averaging  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Micro Averaging"] = r"""
What it is
----------

**Micro averaging** aggregates a metric by **pooling the counts** — it sums the **true positives, false
positives and false negatives** across all classes first, then computes precision, recall or F1 from those
totals. Every **instance** counts equally, so **frequent classes dominate**.

Its behavior
------------

Because it is driven by raw counts, micro averaging gives an **overall**, accuracy-flavored number — in
fact, for **single-label multiclass**, micro-F1 **equals accuracy**. Its weakness is that strong performance
on a big class can **mask** poor performance on a small one.

When to use it
--------------

Reach for micro averaging on **balanced** problems, in **multilabel** settings, or whenever you want a
**single global** score reflecting overall correctness. Pair it with **macro** to expose whether minority
classes are being hidden.
"""

CONTENT["Macro Averaging"] = r"""
What it is
----------

**Macro averaging** computes the metric **separately for each class** (one-vs-rest), then takes the
**unweighted arithmetic mean**. Every **class** counts the **same**, no matter how many samples it has:

.. math::

   P_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}\frac{TP_c}{TP_c + FP_c}.

Its behavior
------------

Because each class contributes equally, macro averaging **punishes ignoring minorities** — a model that aces
the majority but fails a rare class gets a **low** macro score. That makes it sensitive to **rare-class**
performance and a natural **fairness**-oriented headline.

When to use it
--------------

Use macro averaging when **all classes matter equally**, especially on **imbalanced** data where you don't
want the majority to drown out the rest. It can, however, look **pessimistic** if some tiny classes are
inherently hard.
"""

CONTENT["Weighted Averaging"] = r"""
What it is
----------

**Weighted averaging** is macro averaging with a twist — it computes each class's metric, then averages them
**weighted by support** (the number of true instances in each class). Larger classes therefore **count
more**.

Its behavior
------------

This makes the score **reflect the actual class distribution** — it is essentially macro averaging
**adjusted for imbalance**, sitting between micro and macro. On a **representative** test set, the weighted
average estimates what you'd see on a **random production** example.

When to use it
--------------

Weighted averaging suits **stakeholder and production** reporting, where you want one number that respects
the **real class mix** without letting a tiny class swing the result. Use **macro** instead when minority
classes must be weighted **equally** regardless of frequency.
"""

MINDMAP.update({
    "Micro Averaging": [
        "Macro Averaging", "Weighted Averaging", "Micro AUC", "One-vs-Rest (OvR)",
        "Precision (a.k.a. Positive Predictive Value, PPV)", "Multiclass Classification",
    ],
    "Macro Averaging": [
        "Micro Averaging", "Weighted Averaging", "Macro AUC", "One-vs-Rest (OvR)",
        "F1-score", "Multiclass Classification",
    ],
    "Weighted Averaging": [
        "Micro Averaging", "Macro Averaging", "F1-score", "Multiclass Classification",
        "Precision (a.k.a. Positive Predictive Value, PPV)", "Macro AUC",
    ],
})


# ----------------------------------------------------------------------
# Theme: precision-recall summaries — harmonic mean, F1-score, average precision  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Harmonic Mean"] = r"""
What it is
----------

The **harmonic mean** is an average that **leans toward the smaller** of the values — the reciprocal of the
average of reciprocals. For two numbers it is:

.. math::

   \text{HM} = \frac{2ab}{a + b}.

It is always **≤ the arithmetic mean**, and equal only when the values match.

Its key property
----------------

It **penalizes imbalance**. Averaging precision 0.95 and recall 0.20, the arithmetic mean gives a rosy
**0.575**, but the harmonic mean gives **~0.33** — correctly flagging that one component is poor. A high
harmonic mean requires **all** inputs to be high.

Where it's used
---------------

That property is exactly why the **F1-score** uses it to combine precision and recall, and why harmonic
means are the right average for **rates and ratios** (speeds, P/E ratios) rather than additive quantities.
"""

CONTENT["F1-score"] = r"""
What it is
----------

The **F1-score** combines **precision** and **recall** into one number by taking their **harmonic mean**:

.. math::

   F_1 = 2\cdot\frac{P \cdot R}{P + R} = \frac{2\,TP}{2\,TP + FP + FN}.

It ranges from **0 to 1**, and is high only when **both** precision and recall are high.

Why harmonic
------------

Using the harmonic mean makes F1 **penalize lopsided** models — a classifier with 0.95 precision but 0.20
recall scores a low F1, unlike accuracy or a plain average. This makes F1 far more informative than
**accuracy** on **imbalanced** data, and it deliberately **ignores true negatives**.

Its variants
------------

For **multiclass** problems, F1 is aggregated with **micro**, **macro** or **weighted** averaging; the
general **Fβ** score tilts the balance toward recall (β > 1) or precision (β < 1) when the two errors carry
different costs.
"""

CONTENT["Average Precision (AP)"] = r"""
What it is
----------

**Average precision** summarizes the entire **precision–recall curve** in one number — the mean of precision
across recall levels, computed as precision weighted by the **gain in recall** at each threshold:

.. math::

   \text{AP} = \sum_{n} (R_n - R_{n-1})\,P_n.

It equals the **area under the PR curve** (PR-AUC / AUPRC).

Why it's useful
---------------

Because it sweeps **all thresholds**, AP needs no single cutoff, and because it is built from **precision and
recall** it **ignores true negatives** — making it far more informative than ROC-AUC on **imbalanced** data
where the positive class is rare.

Where it's used
---------------

AP is the standard score for **ranking** and **detection**; averaging it over classes or queries gives
**mean average precision (mAP)**, the headline metric in information retrieval and object detection.
"""

MINDMAP.update({
    "Harmonic Mean": [
        "F1-score", "Mean", "Average Precision (AP)",
        "Precision (a.k.a. Positive Predictive Value, PPV)", "Weighted Averaging", "Macro Averaging",
    ],
    "F1-score": [
        "Harmonic Mean", "Precision (a.k.a. Positive Predictive Value, PPV)", "Average Precision (AP)",
        "Accuracy", "Macro Averaging", "Micro Averaging",
    ],
    "Average Precision (AP)": [
        "Precision–Recall AUC (PR-AUC)", "F1-score", "Precision (a.k.a. Positive Predictive Value, PPV)",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Harmonic Mean", "Macro Averaging",
    ],
})


# ----------------------------------------------------------------------
# Theme: precision variants — per-class, multiclass, multilabel precision  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["Per-class Precision (sometimes called class-wise precision)"] = r"""
What it is
----------

**Per-class precision** is **precision computed separately for each class**, treating that class as the
**positive** one and everything else as negative (**one-vs-rest**). For class :math:`c` it is

.. math::

   \text{precision}_c = \frac{TP_c}{TP_c + FP_c},

answering *of everything predicted as class c, how much really was c?*

Why report it
-------------

A single averaged number can **hide** a class the model handles badly; per-class precision exposes exactly
**which** classes suffer false positives. In scikit-learn, ``precision_score(average=None)`` returns the
whole **array** of per-class values.

Its role
--------

Per-class precision is the **building block** that **micro**, **macro** and **weighted** averaging then
collapse into one score. Best practice is to report the **per-class** values **alongside** any aggregate.
"""

CONTENT["Multiclass Precision"] = r"""
What it is
----------

**Multiclass precision** is precision for a problem with **more than two mutually exclusive** classes. Since
precision is defined on a **binary** positive/negative split, it is computed by treating the task as **K
one-vs-rest** binary problems — the **per-class** precisions — then reduced to a single number.

How it's reduced
----------------

The per-class values are combined by an **averaging** scheme — **micro** (pool counts,
majority-dominated), **macro** (equal weight per class), or **weighted** (by support). The choice
determines whether rare classes are surfaced or hidden, so it must be **stated** with the score.

The caveat
----------

Because each class is scored against "the rest," each binary split is **imbalanced**; reporting the
**per-class** precisions guards against an average that looks good only because the majority class does.
"""

CONTENT["Multilabel Precision"] = r"""
What it is
----------

**Multilabel precision** is precision when each sample can carry **several labels at once** — the labels are
**not** mutually exclusive. Targets are an **indicator matrix** (cell [i, j] = 1 if sample i has label j),
and precision is computed **per label**, each label a binary problem.

How it's aggregated
-------------------

As with multiclass, per-label precisions are combined by **micro**, **macro** or **weighted** averaging —
but multilabel adds a distinctive **'samples'** average, which computes precision **per instance** (across
that sample's labels) and averages over samples.

Why it differs
--------------

Unlike multiclass, where exactly one class is correct, multilabel labels **co-occur**, so a prediction can
be **partly** right (some labels correct, others missed). The averaging choice — especially **samples** vs
**micro** — decides whether you're scoring per-label or per-example accuracy.
"""

MINDMAP.update({
    "Per-class Precision (sometimes called class-wise precision)": [
        "Multiclass Precision", "Multilabel Precision", "Precision (a.k.a. Positive Predictive Value, PPV)",
        "One-vs-Rest (OvR)", "Macro Averaging", "Micro Averaging",
    ],
    "Multiclass Precision": [
        "Per-class Precision (sometimes called class-wise precision)", "Multilabel Precision",
        "Multiclass Classification", "Macro Averaging", "Weighted Averaging",
        "Precision (a.k.a. Positive Predictive Value, PPV)",
    ],
    "Multilabel Precision": [
        "Multiclass Precision", "Per-class Precision (sometimes called class-wise precision)",
        "One-vs-Rest (OvR)", "Micro Averaging", "Precision (a.k.a. Positive Predictive Value, PPV)",
        "F1-score",
    ],
})


# ----------------------------------------------------------------------
# Theme: resampling — bootstrap, upsampling, downsampling  (imbalance / inference)
# ----------------------------------------------------------------------

CONTENT["Bootstrap"] = r"""
What it is
----------

The **bootstrap** is a resampling method that draws new samples **with replacement** from the observed data
— each resample the same size as the original, with some points repeated and others omitted. From many such
resamples it estimates a statistic's **sampling distribution**.

What it's for
-------------

By recomputing a statistic (a mean, an AUC) across hundreds or thousands of bootstrap resamples, you get its
**standard error** and **confidence intervals** **without** assuming a formula or a distribution. That makes
it a flexible, **non-parametric** way to quantify **uncertainty**.

Where it appears
----------------

The same idea powers **bagging** (bootstrap aggregating) and **random forests**, which train each model on a
different bootstrap sample to reduce variance. Its main cost is **compute** — many refits — and it can
struggle with very small samples or extreme statistics.
"""

CONTENT["Upsampling"] = r"""
What it is
----------

**Upsampling** (random **oversampling**) rebalances an **imbalanced** dataset by **inflating the minority
class** — duplicating its examples until the classes are closer to even, so the classifier isn't overwhelmed
by the majority.

The risk
--------

Because it **repeats** existing points, upsampling can cause **overfitting** — the model learns patterns that
only exist in the **duplicated** samples rather than the true minority distribution. The fix is **SMOTE**,
which **interpolates** new synthetic points between a minority point and its nearest neighbors instead of
copying, so no example is an exact duplicate.

When to use it
--------------

Prefer upsampling when the dataset is **small** and discarding data would hurt. Critically, apply it to the
**training set only** — resampling the validation or test set causes **data leakage** and inflates your
metrics.
"""

CONTENT["Downsampling"] = r"""
What it is
----------

**Downsampling** (random **undersampling**) rebalances an **imbalanced** dataset the opposite way — by
**removing majority-class** examples until the classes are closer to even. It keeps all the minority data and
thins out the majority.

The risk
--------

Discarding majority examples can cause **underfitting** — the model loses **informative** cases and may miss
the majority class's general pattern. In extreme imbalance you may throw away the vast bulk of the data
(99%+), damaging its representation.

When to use it
--------------

Prefer downsampling when data is **plentiful**, since it is **computationally efficient** (less data to train
on) and **avoids the overfitting** of duplication. As with upsampling, apply it to the **training set only**
to avoid **data leakage**.
"""

MINDMAP.update({
    "Bootstrap": [
        "Bootstrap Confidence Intervals (CIs)", "Standard Error (SE)", "Upsampling",
        "Downsampling", "Decision Trees", "Confidence Intervals (CIs)",
    ],
    "Upsampling": [
        "Downsampling", "Bootstrap", "SMOTE (Synthetic Minority Over-sampling Technique)",
        "Recall", "Precision (a.k.a. Positive Predictive Value, PPV)", "Model Stability",
    ],
    "Downsampling": [
        "Upsampling", "Bootstrap", "SMOTE (Synthetic Minority Over-sampling Technique)",
        "Recall", "Precision (a.k.a. Positive Predictive Value, PPV)", "Model Stability",
    ],
})


# ----------------------------------------------------------------------
# Theme: online experimentation — A/B testing, sequential testing, interleaving  (abtest)
# ----------------------------------------------------------------------

CONTENT["A/B Testing"] = r"""
What it is
----------

An **A/B test** is a **controlled experiment** that **randomly assigns** users to two variants — **A**
(control) and **B** (treatment) — and measures which performs better on a chosen **metric** (conversion
rate, time on page, retention). Randomization is what lets you read the difference as **causal**.

How it's run
------------

You fix the **metric**, use a **power analysis** to set the **sample size**, pick a **statistical test**
(t-test, chi-square), and choose a **significance level α**. When the data are in, the test decides whether
B's effect is **real** or noise.

Its discipline
--------------

The classic A/B test is **fixed-horizon** — you must wait for the pre-planned sample before deciding.
**Peeking** early and stopping when it looks significant **inflates false positives**, which is exactly the
failure that **sequential** methods are designed to fix.
"""

CONTENT["Sequential Testing (also called sequential analysis)"] = r"""
What it is
----------

**Sequential testing** (sequential analysis) **monitors an experiment continuously** and lets you **stop as
soon as** the evidence is conclusive — rather than waiting for a fixed sample size. It is built for the
streaming data of modern experimentation platforms.

The problem it solves
---------------------

Repeatedly checking a **fixed-horizon** test and stopping when it looks good — **peeking** — badly
**inflates the Type I error**. Sequential methods like the **sequential probability ratio test (SPRT)** and
group-sequential designs keep the false-positive rate controlled **at any time**, so early stopping is
**valid**.

The payoff
----------

Because it can end as soon as a winner (or a dead end) is clear, sequential testing **cuts the average
sample size** and **deployment time**, exposing **fewer users** to an inferior variant — at the price of
slightly more conservative thresholds to preserve error control.
"""

CONTENT["Interleaving Tests"] = r"""
What it is
----------

**Interleaving** compares **two rankers** — say two search algorithms — by **blending their result lists
into one** list shown to a **single user**, then crediting each **click** to whichever ranker supplied that
item. The same user effectively judges both at once.

Why it's powerful
-----------------

Because each user sees **both** rankers' results (a within-user comparison), interleaving is **far more
sensitive** than an A/B test — it reaches a reliable conclusion from **far fewer interactions**, so **fewer
users** are exposed to a possibly worse ranker. **Multileaving** extends the idea to compare **many** rankers
simultaneously.

Where it's used
---------------

Interleaving is a staple of **search and recommendation** evaluation and **learning-to-rank**, where ranking
quality differences are subtle and A/B tests would need huge traffic to detect them.
"""

MINDMAP.update({
    "A/B Testing": [
        "Sequential Testing (also called sequential analysis)", "Interleaving Tests",
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Power Analysis", "Statistical Tests",
        "Conversion Rate (CR)",
    ],
    "Sequential Testing (also called sequential analysis)": [
        "A/B Testing", "Interleaving Tests", "Statistical Tests", "Power Analysis",
        "Bootstrap", "Confidence Intervals (CIs)",
    ],
    "Interleaving Tests": [
        "A/B Testing", "Sequential Testing (also called sequential analysis)", "Statistical Tests",
        "Traditional A/B Test (Fixed-Horizon A/B Test)", "Power Analysis", "A/B/n Test",
    ],
})


# ----------------------------------------------------------------------
# Theme: validation splits — evaluation set, time-based splits, stratified k-fold CV  (validation)
# ----------------------------------------------------------------------

CONTENT["Evaluation Set"] = r"""
What it is
----------

An **evaluation set** is data **held out** from training so a model can be scored on examples it has **never
seen** — the only honest way to estimate how it will **generalize**. In practice it splits into two roles.

Validation vs test
------------------

The **validation set** is used **repeatedly** during development — tuning hyperparameters, early stopping,
choosing between models; the **test set** is touched **once**, at the very end, for a final **unbiased**
estimate. Any peek at the test set during development **contaminates** it and inflates the reported score.

The cardinal rule
-----------------

**Split first**, then fit every preprocessing step (scaling, encoding) on the **training data only** and
apply it to the held-out sets. Fitting on all the data before splitting leaks information from the evaluation
set into training — the classic **data leakage** that makes scores look better than reality.
"""

CONTENT["Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)"] = r"""
What it is
----------

A **time-based split** orders data by **time** and trains on the **past** while validating and testing on the
**future** — the earliest records for training, the most recent held out. It reproduces the reality of
deployment, where **future data doesn't exist** at training time.

Why it's needed
---------------

**Time-series** data violates the **i.i.d.** assumption behind ordinary splitting — observations depend on
**prior** ones. Shuffling or random k-fold would let the model **train on the future** to predict the past, a
**temporal leakage** that badly **overstates** accuracy.

How it's done
-------------

Schemes like a **rolling forecasting origin** (walk-forward) or an **expanding / sliding window** repeatedly
move the training window forward in time, so every evaluation always predicts **later** data than it trained
on. Look-ahead **features** must be avoided too.
"""

CONTENT["k-fold Stratified Cross-Validation (Stratified CV)"] = r"""
What it is
----------

**Stratified k-fold cross-validation** splits the data into **k folds** while **preserving each class's
proportion** in every fold — so a fold of a 5%-positive dataset stays about **5% positive**. It combines the
stability of k-fold CV with balanced folds.

Why stratify
------------

Plain **k-fold** can, by chance, build folds with **too few or missing** minority-class examples, giving
**biased** or unstable metrics — especially on **imbalanced** data. Stratification makes each fold **mirror**
the overall distribution, so the k scores are **reliable** and comparable.

How it's used
-------------

It is the **default** for classification (often **repeated stratified 10-fold**); scikit-learn provides
``StratifiedKFold``. Two cautions carry over from any CV: fit preprocessing on the **training folds only** to
avoid **leakage**, and don't use it on **time-series** data, where **time-based** splits are required instead.
"""

MINDMAP.update({
    "Evaluation Set": [
        "k-fold Stratified Cross-Validation (Stratified CV)",
        "Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)",
        "Cross-Validation (CV)", "Model Score", "Model Stability", "Data Drift",
    ],
    "Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)": [
        "Sliding Window (Rolling Window) Cross-Validation", "Expanding Window Cross-Validation",
        "Time Series Forecasting", "IID (Independent and Identically Distributed)",
        "Evaluation Set", "Cross-Validation (CV)",
    ],
    "k-fold Stratified Cross-Validation (Stratified CV)": [
        "Cross-Validation (CV)", "Stratified Group K-Fold", "Evaluation Set",
        "Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)",
        "Model Stability", "Data Drift",
    ],
})


# ----------------------------------------------------------------------
# Theme: distribution shift — covariate drift, label drift, PSI  (drift / mlops)
# ----------------------------------------------------------------------

CONTENT["Covariate Drift (a.k.a. Covariate Shift)"] = r"""
What it is
----------

**Covariate drift** (covariate shift) is a change in the distribution of the **input features** a model sees
— the production inputs no longer look like the training inputs — while the feature-to-label rule stays the
same:

.. math::

   p_{\text{train}}(x) \neq p_{\text{prod}}(x), \qquad p(y \mid x)\ \text{unchanged}.

The model is being asked about a **different population** than it learned on.

How it differs
--------------

It is one of three **dataset shifts**. **Covariate drift** moves **p(x)** (the inputs), **label drift** moves
**p(y)** (the target mix), and **concept drift** moves **p(y | x)** (the relationship itself). Only concept
drift changes the *rule*; covariate drift changes *who* you're scoring.

Detecting and fixing it
-----------------------

It is caught by comparing feature distributions per column with **PSI** or the **KS** test. Remedies include
**importance weighting** — reweighting training points by the density ratio
:math:`w(x) = p_{\text{prod}}(x) / p_{\text{train}}(x)` — retraining on **recent** data, and building
**robust** features (winsorized, log-scaled, sensible bins).
"""

CONTENT["Label Drift (a.k.a. Target Drift)"] = r"""
What it is
----------

**Label drift** (target drift) is a change in the distribution of the **target** itself — the **class
balance** or outcome mix shifts between training and production, :math:`p_{\text{train}}(y) \neq
p_{\text{prod}}(y)`, even when the feature-to-label relationship may be unchanged. A fraud rate that creeps
from 1% to 3% is label drift.

How it differs
--------------

Like covariate drift it is a **dataset shift**, but it moves **p(y)** rather than **p(x)** or **p(y | x)**.
Because most classifiers implicitly assume the **base rate** they trained on, a shifted target distribution
can throw off **calibrated probabilities** and **thresholds** even if each input still maps to the right
answer.

Detecting and fixing it
-----------------------

Monitor the **label** or **prediction** distribution over time (PSI on predicted classes, tracked class
proportions). Fixes include **recalibrating** decision thresholds to the new base rate, **reweighting** or
resampling to the current mix, and **retraining** on recent labels.
"""

CONTENT["PSI (Population Stability Index)"] = r"""
What it is
----------

The **Population Stability Index** measures how much a variable's distribution has **shifted** between a
**reference** ("expected") sample and a **current** ("actual") one — usually training vs production. It bins
the variable and sums the per-bin discrepancy:

.. math::

   \text{PSI} = \sum_{b} (A_b - E_b)\,\ln\!\frac{A_b}{E_b},

over bins :math:`b`. It is **0** when the distributions match and grows without bound as they diverge.

How it's computed
-----------------

Choose **bins** (often 10, by quantile or equal width) using the **same edges** for both samples, take each
bin's **proportion** in the expected and actual data, and sum the term above across bins. It is closely
related to a **symmetrized KL divergence**.

Reading it
----------

The rules of thumb are **< 0.1** stable, **0.1–0.25** moderate drift (**watch**), and **> 0.25** significant
drift (**retrain**). Two cautions: an **empty bin** makes PSI undefined or unbounded (so proportions are
clipped), and PSI tends to **rise with sample size**, so thresholds may need tuning.
"""

MINDMAP.update({
    "Covariate Drift (a.k.a. Covariate Shift)": [
        "Label Drift (a.k.a. Target Drift)", "Concept Drift", "Dataset Shift", "Data Drift",
        "Drift Detection", "PSI (Population Stability Index)",
    ],
    "Label Drift (a.k.a. Target Drift)": [
        "Covariate Drift (a.k.a. Covariate Shift)", "Concept Drift", "Dataset Shift",
        "PSI (Population Stability Index)", "Data Drift", "Drift Detection",
    ],
    "PSI (Population Stability Index)": [
        "Covariate Drift (a.k.a. Covariate Shift)", "Label Drift (a.k.a. Target Drift)",
        "Kolmogorov–Smirnov (KS) Test", "Kullback–Leibler (KL) Divergence", "Data Drift",
        "Drift Detection",
    ],
})


# ----------------------------------------------------------------------
# Theme: operational cost — compute budgets, inference cost, manual review minutes  (ops / platforms)
# ----------------------------------------------------------------------

CONTENT["Compute budgets"] = r"""
What it is
----------

A **compute budget** is the pool of **computational resources** — GPU / TPU hours, **FLOPs**, and the dollars
behind them — allocated to an ML system's **training** and **serving**. It caps how big a model you can
train and how much traffic you can serve.

The two halves
--------------

**Training** is a **one-time** cost that grows with model and data size (its FLOPs approximated by the
**6ND** rule — roughly 6 × parameters × tokens), while **inference** is an **ongoing** cost (about **2N**
FLOPs per forward pass) that scales with usage:

.. math::

   C_{\text{train}} \approx 6ND, \qquad C_{\text{inf}} \approx 2N \ \text{FLOPs per pass}.

In production, inference usually claims the **majority** of the budget.

Managing it
-----------

Teams set **budgets and alerts**, model **optimistic / expected / pessimistic** scenarios, and track unit
economics like **cost per prediction** and **GPU utilization**. Hidden drains — idle instances, failed runs,
oversized experiments — routinely add **20–40%** over the planned figure.
"""

CONTENT["Inference Cost (Inference $)"] = r"""
What it is
----------

**Inference cost** is the **ongoing** expense of **serving predictions** — the compute (and money) spent
every time the deployed model answers a request. It is usually tracked as the **cost per prediction** (or
per token), the fundamental **unit economic** of an ML product.

Why it dominates
----------------

Unlike **training**, which is paid **once**, inference runs **continuously** and **scales with adoption** —
every user request consumes compute, so cost grows with traffic. At scale it is the **larger** line item,
often **65–80%** of an AI budget, and it is where revenue meets the bill.

Bringing it down
----------------

It is driven by **model size**, **hardware**, and **utilization**, so it falls with **quantization**
(smaller, faster weights), **caching**, **batching**, and **right-sizing** capacity to demand — techniques
that can cut cost per prediction substantially. The target is a **declining** cost-per-prediction over time.
"""

CONTENT["Manual review minutes"] = r"""
What it is
----------

**Manual review minutes** measure the **human time** spent checking model outputs in a **human-in-the-loop**
pipeline — analysts working a **review queue** of **flagged** or **low-confidence** predictions, confirming
or correcting each. It is the **labor** cost of keeping a model's decisions trustworthy.

Why it matters
--------------

Human review is **expensive** — expert annotation runs to tens of dollars per hour, and for some systems
this **labeling / review** cost dwarfs the **compute** cost. It is a real budget line, not a rounding error,
and it scales with **how many** cases the model sends to a person.

The lever
---------

Fewer needless escalations means fewer review minutes, so **precision** and a well-tuned **selection rate**
(the share of cases flagged) directly control the cost. The design trade-off is **automation vs assurance**
— routing more to humans is safer but slower and pricier; routing less is cheaper but riskier.
"""

MINDMAP.update({
    "Compute budgets": [
        "Inference Cost (Inference $)", "TPU Clusters", "Quantization", "Caching",
        "Cloud Inference", "Latency Guardrails",
    ],
    "Inference Cost (Inference $)": [
        "Compute budgets", "Manual review minutes", "Quantization", "Caching",
        "TPU Clusters", "Cloud Inference",
    ],
    "Manual review minutes": [
        "Inference Cost (Inference $)", "Compute budgets",
        "Precision (a.k.a. Positive Predictive Value, PPV)", "Selection Rate",
        "Latency Guardrails", "Cloud Inference",
    ],
})


# ----------------------------------------------------------------------
# Theme: serving infrastructure — caching, quantization, TPU clusters  (platforms / ops)
# ----------------------------------------------------------------------

CONTENT["Caching"] = r"""
What it is
----------

**Caching** stores the results of expensive computation and **reuses** them instead of recomputing —
trading **memory** for **speed** and **cost**. If the same work would produce the same answer, a cache
returns it instantly.

In model serving
----------------

The signature example is the **KV cache** in transformers, which keeps the **key/value** tensors of earlier
tokens so generating each new token doesn't re-process the whole sequence. It grows **linearly with sequence
length** and can exceed the **model weights** in memory — which is why it is often **quantized**. Beyond
that, **prediction caching** reuses answers to repeated requests and **feature caching** precomputes costly
features.

The trade-offs
--------------

A cache must handle **staleness** — cached values can go **out of date** when inputs or the model change — so
it needs **invalidation** and **eviction** policies, and it consumes **memory** that must be budgeted against
the speed it buys.
"""

CONTENT["Quantization"] = r"""
What it is
----------

**Quantization** lowers the **numerical precision** of a model's **weights** (and often activations) — from
32-bit floating point (**FP32**) down to **INT8**, **FP8**, or even **INT4**. Fewer bits per number means a
**smaller, faster, cheaper** model.

The payoff
----------

An **INT8** model uses about **75% less memory** than FP32, and because decoding is often
**memory-bandwidth-bound**, 4-bit weights can be read up to **4× faster** than 16-bit — directly cutting
**latency** and **inference cost** on hardware that supports low precision.

Managing the trade-off
----------------------

Naive quantization **degrades accuracy**. **Post-training quantization (PTQ)** converts a trained model
quickly (with a small calibration set), while **quantization-aware training (QAT)** bakes precision loss into
training to preserve accuracy; advanced schemes like **AWQ** and **GPTQ** protect the most **sensitive**
weights to reach near-FP16 quality at INT4 speeds.
"""

CONTENT["TPU Clusters"] = r"""
What it is
----------

A **TPU** (Tensor Processing Unit) is Google's custom **ASIC** built for machine learning — hardware
specialized for the massive **matrix multiplications** inside neural networks. A **TPU cluster** (or **pod**)
links many of these chips with **high-speed interconnects** to train and serve **very large** models.

Why it exists
-------------

General-purpose **CPUs** are too slow for deep learning and even **GPUs** aren't purpose-built for it; TPUs
pack dense **matrix-multiply** units and high **memory bandwidth** to push far more throughput per watt on
those specific operations. They are typically consumed via the **cloud**, on demand.

How clusters help
-----------------

A single chip can't hold the largest models, so a cluster **splits the work** — across **data**, **model**,
and **pipeline** parallelism — running in parallel over many TPUs. That is what makes training
billion-parameter models, and serving them at scale, feasible.
"""

MINDMAP.update({
    "Caching": [
        "Quantization", "TPU Clusters", "Inference Cost (Inference $)", "Cloud Inference",
        "Latency Guardrails", "Compute budgets",
    ],
    "Quantization": [
        "Caching", "TPU Clusters", "Inference Cost (Inference $)", "Neural Networks",
        "Compute budgets", "ONNX (Open Neural Network Exchange)",
    ],
    "TPU Clusters": [
        "Quantization", "Caching", "Compute budgets", "Inference Cost (Inference $)",
        "Cloud Inference", "Neural Networks",
    ],
})


# ----------------------------------------------------------------------
# Theme: production guardrails — drift, latency, fairness guardrails  (mlops / ops)
# ----------------------------------------------------------------------

CONTENT["Drift Guardrails"] = r"""
What it is
----------

**Drift guardrails** are automated monitoring rules that watch a deployed model's **inputs and predictions**
for **distribution shift** and **trigger action** — an alert, an investigation, or a **retrain** — when the
drift crosses a threshold. They turn passive monitoring into a **response**.

How they're set
---------------

They compare live data to a **rolling baseline** with drift metrics like **PSI** and the **KS** test — for
example, **PSI above 0.2–0.25** on a key feature raises an alert, and **prediction drift that stays over
threshold for several consecutive days** kicks off **automatic retraining** on fresh labels.

Why they matter
---------------

A model silently **degrades** as the world moves away from its training data, and no aggregate dashboard
catches that on its own. Guardrails make the degradation **actionable** — often gating a retrained model
through a **registry** that re-evaluates it against production before it sees traffic.
"""

CONTENT["Latency Guardrails"] = r"""
What it is
----------

**Latency guardrails** are **budgets** on how long the model may take to respond — service-level objectives
(**SLOs**) that trigger an alert when serving gets too slow. They protect the **user experience** and any
latency **SLAs**.

Tail, not average
-----------------

They track **tail percentiles** — **p50, p95, p99** — not just the mean, because a few very slow requests
ruin the experience even when the average looks fine. A typical rule fires when the current **p99** exceeds,
say, **1.5×** a rolling baseline.

Setting the budget
------------------

Acceptable latency is set by the **use case** — on the order of tens of milliseconds for ad serving or fraud
scoring, more for heavier recommendations — and the system is **designed and sized** to stay under it, then
**measured continuously** with alerts on breach.
"""

CONTENT["Fairness Guardrails"] = r"""
What it is
----------

**Fairness guardrails** are automated checks on a model's **disparity across groups** that **block a
deployment** or **trigger retraining** when a fairness metric exceeds an agreed limit. They bake **equity**
and **compliance** into the release process.

How they're enforced
--------------------

A pre-deployment **fairness audit** requires disparity measures — such as the **demographic-parity
difference** or **equalized-odds** gap — to stay **below a threshold** (for example **≤ 0.05**); if **DPD or
EO exceeds** it, the release is **halted** and the model is **retrained**. Post-deployment, real-time
monitoring watches for **bias spikes**.

Why they matter
---------------

Fairness can **degrade** as populations drift, and regulated domains (lending, hiring, healthcare) demand
**auditable** guarantees. Guardrails provide a **hard gate** and an **audit trail**, rather than relying on a
one-time fairness check that goes stale.
"""

MINDMAP.update({
    "Drift Guardrails": [
        "Latency Guardrails", "Fairness Guardrails", "Drift Detection",
        "PSI (Population Stability Index)", "Data Drift", "Kolmogorov–Smirnov (KS) Test",
    ],
    "Latency Guardrails": [
        "Drift Guardrails", "Fairness Guardrails", "Inference Cost (Inference $)",
        "Cloud Inference", "Compute budgets", "Caching",
    ],
    "Fairness Guardrails": [
        "Drift Guardrails", "Latency Guardrails", "Selection Rate", "Drift Detection",
        "PSI (Population Stability Index)", "Statistical Tests",
    ],
})


# ----------------------------------------------------------------------
# Theme: model families — classification, regression, linear models  (concepts / training)
# ----------------------------------------------------------------------

CONTENT["Classification Models"] = r"""
What it is
----------

**Classification models** predict a **discrete category** — spam or not, which digit, which disease. The
output is a **class label** (often via a probability over classes), and the model learns a **decision
boundary** that separates the classes in feature space.

The landscape
-------------

They range from **linear** ones (**logistic regression**, linear SVM) to **non-linear** ones (**decision
trees**, random forests, **neural networks**, kernel SVMs). Tasks split into **binary** (two classes),
**multiclass** (one of many), and **multilabel** (several at once).

How they're judged
------------------

Because the target is categorical, classification uses metrics like **accuracy**, **precision / recall**,
**F1**, and **AUC** — not squared error — and its **loss functions** are typically **cross-entropy** rather
than a distance. The right metric depends on **class balance** and error costs.
"""

CONTENT["Regression Models"] = r"""
What it is
----------

**Regression models** predict a **continuous number** — a price, a temperature, a demand — rather than a
class. They learn a function mapping features to a **real-valued** output, fitting a curve or surface through
the data.

The landscape
-------------

The simplest is **linear regression** (a weighted sum of features), extending to **polynomial**,
**regularized** (ridge, lasso), tree-based (**random forests**, gradient boosting), and **neural**
regressors. The same algorithm family often has both a classification and a regression form.

How they're judged
------------------

Regression is scored by how far predictions land from the truth — **MSE / RMSE**, **MAE**, and **R²** — and
trained to minimize a distance-based **loss**. Because those errors use magnitudes, regression is
**sensitive to outliers**, which is why robust losses and metrics exist.
"""

CONTENT["Linear Models"] = r"""
What it is
----------

A **linear model** predicts from a **weighted sum** of the input features, optionally passed through a link
function:

.. math::

   \hat{y} = \mathbf{w}^\top \mathbf{x} + b.

Its defining trait is that it is **linear in the parameters**, which makes it simple, fast, and highly
**interpretable**.

Both tasks
----------

The family spans **regression** (**linear regression**, ridge, lasso) and **classification** (**logistic
regression**, linear SVM), where the linear combination is squashed by a **sigmoid** or **softmax** into
probabilities. In every case the learned **weights** show each feature's direction and strength.

Strengths and limits
--------------------

Linear models are **data-efficient**, **cheap** to train and serve, and **transparent** — but they can only
capture **linear** relationships unless you add **interactions** or feature transforms. They are the natural
**baseline** against which more complex models must justify themselves.
"""

MINDMAP.update({
    "Classification Models": [
        "Regression Models", "Linear Models", "Logistic Regression", "Decision Trees",
        "Binary Classification", "Multiclass Classification",
    ],
    "Regression Models": [
        "Classification Models", "Linear Models", "Mean Squared Error (MSE)", "Loss Functions",
        "Outlier", "Neural Networks",
    ],
    "Linear Models": [
        "Logistic Regression", "Classification Models", "Regression Models", "Neural Networks",
        "Loss Functions", "Support Vector Machines (SVMs)",
    ],
})


# ----------------------------------------------------------------------
# Theme: perception & data — computer vision, NLP, full annotation  (repr / features)
# ----------------------------------------------------------------------

CONTENT["Computer Vision (CV)"] = r"""
What it is
----------

**Computer vision** teaches machines to **interpret visual data** — images and video — extracting meaning the
way human sight does. It turns pixels into **structured** understanding: what is present, where, and how it
moves.

Its tasks
---------

The core problems are **image classification** (what's in the picture), **object detection** (locating
objects with **bounding boxes**), **segmentation** (labeling every **pixel**), plus recognition, pose, and
tracking. Modern CV is dominated by **deep learning** — **convolutional neural networks** and **vision
transformers**.

What it needs
-------------

CV has historically relied on **large annotated** datasets (ImageNet-scale), because supervised models learn
from labeled examples — which makes **annotation** a major cost and drives interest in self- and
semi-supervised alternatives. It powers medical imaging, autonomous driving, and quality inspection.
"""

CONTENT["Natural Language Processing (NLP)"] = r"""
What it is
----------

**Natural language processing** teaches machines to **understand and generate human language** — text and
speech. It bridges unstructured language and computation, from parsing meaning to producing fluent output.

Its tasks
---------

NLP spans **classification** (sentiment, topic), **named-entity recognition**, **translation**,
**summarization**, **question answering**, and **generation**. Its architectures moved from RNNs and
**LSTMs** to the **transformer**, behind **BERT** and **GPT**-style models.

Its turning point
-----------------

NLP was the first great success of **self-supervised** pretraining — models learn from **masked-token**
prediction on **huge unlabeled** text, then fine-tune on a smaller labeled task. That pretrain-then-finetune
recipe reset the field and now underlies search, chatbots, and translation.
"""

CONTENT["Full Annotation"] = r"""
What it is
----------

**Full annotation** means **labeling every example** in a dataset with its **ground-truth** target — the
complete, high-quality supervision that classic **supervised learning** assumes. Each image gets its boxes,
each sentence its tags.

The cost
--------

It is **manual, slow, and expensive** — often the most **tedious** part of an ML project — and requires
annotators following **guidelines**, whose **disagreements** become a data-quality issue (measured with
inter-annotator agreement). At scale, labeling everything is simply **infeasible**.

Why it persists
---------------

Despite the cost, fully annotated data gives the **strongest** signal and remains the **go-to** for
production and the **gold-standard benchmark**. Its expense is exactly what motivates **weak**, **semi-**,
and **self-supervised** learning, which trade some label quality for far less human effort.
"""

MINDMAP.update({
    "Computer Vision (CV)": [
        "Natural Language Processing (NLP)", "Neural Networks", "Full Annotation",
        "Embedding", "Autoencoder", "Deep Ensembles",
    ],
    "Natural Language Processing (NLP)": [
        "Computer Vision (CV)", "Neural Networks", "Embedding", "Full Annotation",
        "LSTM — Long Short-Term Memory Networks", "Weak Supervision",
    ],
    "Full Annotation": [
        "Weak Supervision", "Label Noise", "Computer Vision (CV)",
        "Natural Language Processing (NLP)", "Neural Networks", "Manual review minutes",
    ],
})


# ----------------------------------------------------------------------
# Theme: post-hoc calibration — temperature scaling, Platt scaling, isotonic regression  (calibration)
# ----------------------------------------------------------------------

CONTENT["Temperature Scaling"] = r"""
What it is
----------

**Temperature scaling** is the simplest **post-hoc calibration** method for neural nets — it divides the
**logits** by a single learned scalar **T** before the **softmax**, softening or sharpening the
probabilities:

.. math::

   \hat{Q} = \mathrm{softmax}(\mathbf{z} / T), \quad T > 0.

It is a one-parameter fix applied **after** training.

What T does
-----------

**T = 1** leaves the model unchanged; **T > 1** makes predictions **less confident** (softer), which corrects
the **overconfidence** typical of modern networks; **T < 1** makes them sharper. T is fit on a **held-out**
validation set by minimizing **negative log-likelihood**.

Why it's popular
----------------

It is **simple**, **effective**, and crucially **accuracy-preserving** — dividing every logit by the same T
never changes the **argmax**, so the decision boundary and accuracy are untouched. Its limit is
**expressiveness**: it can only rescale confidence uniformly, not fix region-specific miscalibration.
"""

CONTENT["Platt Scaling"] = r"""
What it is
----------

**Platt scaling** calibrates a classifier by fitting a **logistic (sigmoid)** function on its raw scores,
mapping them to probabilities:

.. math::

   \hat{Q} = \sigma(a\,z + b).

The two parameters **a** and **b** are fit by **negative log-likelihood** on a validation set. It was invented
by John Platt for **SVMs**.

How it relates
--------------

It is a **parametric** method that assumes a **sigmoid**-shaped miscalibration. **Temperature scaling** is
essentially its **one-parameter, multi-class** special case (fixing the slope and dropping the offset), so
the two are close cousins.

When to use it
--------------

Platt scaling is a solid default for **binary** classifiers with **monotonic** score miscalibration and
limited calibration data, since two parameters rarely overfit — but if the true miscalibration isn't
sigmoid-shaped, a more flexible method like **isotonic regression** fits better.
"""

CONTENT["Isotonic Regression"] = r"""
What it is
----------

**Isotonic regression** is a **non-parametric** calibration method that fits a **monotonic** (non-decreasing)
**step function** mapping raw scores to calibrated probabilities — it assumes only that a higher score should
mean a higher probability, nothing about the shape.

Its strength
------------

Because it is **model-free**, it can correct **arbitrary** monotonic miscalibration that parametric methods
(Platt, temperature) miss — and with **enough** calibration data it typically **outperforms** them. It is
also a general tool for **monotonic regression**, not only calibration.

Its weakness
------------

That flexibility makes it **prone to overfitting** when calibration data is **scarce**, and the
piecewise-constant fit is less smooth. Unlike temperature scaling it does **not** guarantee the model's
accuracy is preserved.
"""

MINDMAP.update({
    "Temperature Scaling": [
        "Platt Scaling", "Isotonic Regression", "Overconfident", "Confidence Level",
        "Softmax Function", "Underconfident",
    ],
    "Platt Scaling": [
        "Temperature Scaling", "Isotonic Regression", "Sigmoid Function", "Logistic Regression",
        "Support Vector Machines (SVMs)", "Confidence Level",
    ],
    "Isotonic Regression": [
        "Platt Scaling", "Temperature Scaling", "Confidence Level", "Overconfident",
        "Regression Models", "Underconfident",
    ],
})


# ----------------------------------------------------------------------
# Theme: calibration error — adaptive ECE, MCE, Murphy's decomposition  (calibration)
# ----------------------------------------------------------------------

CONTENT["Adaptive ECE (Expected Calibration Error with Adaptive Binning)"] = r"""
What it is
----------

**Adaptive ECE** measures a classifier's **miscalibration** — the gap between its **confidence** and its
actual **accuracy** — using **equal-count** bins. Like standard **Expected Calibration Error**, it is a
weighted average of |accuracy − confidence| across bins:

.. math::

   \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

The "adaptive" part changes only **how the bins are drawn**.

The problem it fixes
--------------------

Standard ECE uses **fixed equal-width** bins ([0.0–0.1], …), so when predictions **cluster** (modern nets
pile probabilities near 1.0), some bins hold **few or zero** samples and give **noisy** estimates. Adaptive
binning instead makes each bin hold **the same number** of predictions, so bin **widths vary** with the data.

Why it helps
------------

Equal-count bins yield a **more stable, fairer** calibration estimate on **skewed** predictions, where
equal-width ECE is unreliable. It shares ECE's caveat, though: the result still depends on the **number of
bins**, and neither is a **proper scoring rule**.
"""

CONTENT["Maximum Calibration Error (MCE)"] = r"""
What it is
----------

**Maximum Calibration Error** reports the **worst** calibration gap rather than the average — the **largest**
difference between accuracy and confidence over all bins:

.. math::

   \text{MCE} = \max_{m}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

Where ECE asks *how miscalibrated on average?*, MCE asks *how bad does it get?*

When it matters
---------------

MCE is the right lens for **safety-critical** systems — medical, autonomous, financial — where a single
**badly** miscalibrated confidence region can cause harm, even if the **average** looks fine. Lower is
better, as with ECE.

Its limits
----------

Like ECE it is **binning-dependent** (the answer shifts with bin count and scheme), and it is **not a proper
scoring rule** — a model can achieve low calibration error with **trivial** predictions, so MCE must be read
**alongside** discrimination metrics, not alone.
"""

CONTENT["Murphy's Decomposition"] = r"""
What it is
----------

**Murphy's decomposition** (1973) splits a **proper scoring rule** — classically the **Brier score** — into
three interpretable pieces:

.. math::

   \text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}.

It reveals *why* a probabilistic forecast scores as it does.

The three terms
---------------

**Reliability** is the **calibration** error (how far forecast probabilities sit from observed frequencies —
**lower** is better); **resolution** is how much the forecasts **vary** from the base rate to **separate**
outcomes (**higher** is better); **uncertainty** is the **irreducible** variance of the event itself,
independent of the model.

Why it matters
--------------

It shows a good score needs **both** good calibration **and** good resolution — a perfectly calibrated model
that always predicts the base rate has **zero** reliability error but **zero** resolution, and is useless. It
is the theoretical reason calibration metrics like **ECE** tell only **half** the story.
"""

MINDMAP.update({
    "Adaptive ECE (Expected Calibration Error with Adaptive Binning)": [
        "Maximum Calibration Error (MCE)", "Murphy's Decomposition", "Confidence Level",
        "Temperature Scaling", "Overconfident", "Underconfident",
    ],
    "Maximum Calibration Error (MCE)": [
        "Adaptive ECE (Expected Calibration Error with Adaptive Binning)", "Murphy's Decomposition",
        "Confidence Level", "Temperature Scaling", "Risk-Based Decisions", "Overconfident",
    ],
    "Murphy's Decomposition": [
        "Adaptive ECE (Expected Calibration Error with Adaptive Binning)",
        "Maximum Calibration Error (MCE)", "Strictly Proper Scoring Rules", "Confidence Level",
        "Probabilistic Forecasts", "Temperature Scaling",
    ],
})


# ----------------------------------------------------------------------
# Theme: ranking & IR benchmarks — DCG, Kaggle, TREC  (ranking / evaluation)
# ----------------------------------------------------------------------

CONTENT["DCG (Discounted Cumulative Gain)"] = r"""
What it is
----------

**Discounted Cumulative Gain** scores a **ranked list** by summing each item's **graded relevance**,
discounted by how **far down** it sits — so a relevant result near the top counts far more than the same
result buried lower:

.. math::

   \text{DCG}_p = \sum_{i=1}^{p} \frac{\mathrm{rel}_i}{\log_2(i+1)}.

The **logarithmic** discount encodes that users examine top results most.

Why it beats precision
----------------------

Unlike binary **precision / recall**, DCG uses **multi-level** relevance (say 0–3) **and** position, capturing
both *how relevant* each item is and *where* it was ranked — exactly what matters for **search** and
**recommendation**.

Normalizing it
--------------

Raw DCG isn't comparable across queries with different numbers of relevant items, so
:math:`\text{NDCG} = \text{DCG} / \text{IDCG}` divides by the **ideal** DCG (the best possible ordering),
giving a **0-to-1** score where **1** is a perfect ranking. It is the standard offline ranking metric.
"""

CONTENT["Kaggle"] = r"""
What it is
----------

**Kaggle** is an online **data-science competition** platform (owned by Google) where organizations post a
dataset and a problem, and competitors submit predictions scored on a **held-out test set**, ranked on a
**leaderboard**. It also hosts public **datasets**, **notebooks**, and courses.

How competitions work
---------------------

Entrants train models and submit predictions evaluated by a **fixed metric**; a **public** leaderboard shows
partial-data scores during the contest, while the final **private** leaderboard — on unseen test data —
decides winners, guarding against **overfitting** the public split.

Its influence and caveats
-------------------------

Kaggle **popularized** competitive, benchmark-driven ML and battle-tested techniques like **gradient-boosted
trees** and **ensembling**. But winning solutions often **over-optimize** a single metric and stack many
models, so they don't always translate to **production**, where latency and maintainability matter.
"""

CONTENT["TREC (Text REtrieval Conference)"] = r"""
What it is
----------

The **Text REtrieval Conference** is an annual **information-retrieval benchmark** run by **NIST** since
1992. It provides standard **test collections** — documents, query **topics**, and human **relevance
judgments** — and organizes **tracks** so retrieval systems can be compared on common ground.

Why it matters
--------------

TREC formalized the **shared-task** evaluation paradigm in IR — the **pooling** method for gathering
relevance judgments at scale, reusable **test collections**, and the ``trec_eval`` scoring tool. Ranking
metrics like **DCG / NDCG** and MAP were validated on TREC data.

Its legacy
----------

Its **tracks** (ad-hoc, web, question answering, and more) drove decades of progress in **search** and now
**retrieval-augmented** systems, and its methodology underpins modern IR **leaderboards**. It is to
information retrieval what shared benchmarks are to the rest of ML.
"""

MINDMAP.update({
    "DCG (Discounted Cumulative Gain)": [
        "Kaggle", "TREC (Text REtrieval Conference)", "Average Precision (AP)",
        "Relevance in Recommender Systems", "Intra-List Diversity (ILD)",
        "Cosine Similarity of Item Features",
    ],
    "Kaggle": [
        "TREC (Text REtrieval Conference)", "DCG (Discounted Cumulative Gain)",
        "Forecasting Competitions", "Computer Vision (CV)", "Average Precision (AP)",
        "Natural Language Processing (NLP)",
    ],
    "TREC (Text REtrieval Conference)": [
        "DCG (Discounted Cumulative Gain)", "Kaggle", "Average Precision (AP)",
        "Relevance in Recommender Systems", "Natural Language Processing (NLP)",
        "Forecasting Competitions",
    ],
})


# ----------------------------------------------------------------------
# Theme: score & ROC — model score, ROC curve, AUC  (metrics / evaluation)
# ----------------------------------------------------------------------

CONTENT["ROC Curve (Receiver Operating Characteristic)"] = r"""
What it is
----------

A **ROC curve** (Receiver Operating Characteristic) plots a binary classifier's **true positive rate**
(sensitivity / recall) against its **false positive rate** (1 − specificity) as the **decision threshold**
sweeps from strict to lenient. Each point is one threshold's (FPR, TPR) trade-off.

Reading it
----------

Lowering the threshold labels **more** examples positive, so **both** TPR and FPR rise — the curve runs from
(0, 0) to (1, 1). A curve hugging the **upper-left** corner (high TPR, low FPR) is excellent; the
**diagonal** line is **random guessing**; the closer to the top-left, the better the separation.

Why it's useful
---------------

Because it shows performance at **every** threshold, the ROC curve reveals the full **trade-off** between
catching positives and raising false alarms — letting you pick an operating point for your costs, rather than
being locked to one cutoff. It dates to **radar** signal detection in the 1940s.
"""

CONTENT["Model Score"] = r"""
What it is
----------

A **model score** is the **continuous output** a classifier assigns each example — a **probability** or
real-valued score of belonging to the **positive** class — *before* it becomes a hard label. Logistic
regression, random forests, and neural nets all emit scores.

Score vs label
--------------

Turning a score into a **decision** requires a **threshold** — above it, positive; below, negative. The score
carries **more** information than the label: its **ranking** (are positives scored above negatives?) is what
threshold-free metrics like **AUC** measure, and its **magnitude** matters for ranking and prioritization.

Score vs probability
--------------------

A score need not be a **calibrated** probability — a score of 0.9 doesn't guarantee a 90% chance of being
positive unless the model is calibrated (e.g., via **temperature** or **Platt scaling**). Use the raw score
for **ranking**, the calibrated one for **decisions** that need real probabilities.
"""

CONTENT["AUC (Area Under the Curve)"] = r"""
What it is
----------

**AUC** — the **area under the curve** — condenses an entire **ROC curve** into one number by measuring the
area beneath it. It ranges from **0 to 1**: **1** is perfect, **0.5** is random guessing, and below 0.5 means
the scores are **inverted**.

Its meaning
-----------

AUC has a clean interpretation — it is the **probability that a random positive is scored higher than a
random negative** (the Wilcoxon–Mann–Whitney statistic). So it measures how well the model **ranks**
positives above negatives, independent of any single threshold.

Why it beats accuracy
---------------------

AUC is **threshold-invariant** and always calibrated so **0.5 = useless**, unlike **accuracy**, which is
misleading under **imbalance** (90% accuracy is trivial when 90% of data is negative). But on **heavily
imbalanced** data the **PR-AUC** often tells a more honest story, since ROC-AUC can look optimistic.
"""

MINDMAP.update({
    "ROC Curve (Receiver Operating Characteristic)": [
        "AUC (Area Under the Curve)", "Model Score",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Precision–Recall AUC (PR-AUC)", "Accuracy", "Partial AUC (pAUC)",
    ],
    "Model Score": [
        "ROC Curve (Receiver Operating Characteristic)", "AUC (Area Under the Curve)",
        "Confidence Level", "Log-Odds", "Sigmoid Function", "Temperature Scaling",
    ],
    "AUC (Area Under the Curve)": [
        "ROC Curve (Receiver Operating Characteristic)",
        "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)",
        "Precision–Recall AUC (PR-AUC)", "Model Score", "Partial AUC (pAUC)", "Accuracy",
    ],
})


# ----------------------------------------------------------------------
# Theme: drift core — concept drift, data drift, KS shift  (drift)
# ----------------------------------------------------------------------

CONTENT["Concept Drift"] = r"""
What it is
----------

**Concept drift** is a change in the **relationship** between inputs and the outcome — formally a shift in
:math:`P(Y \mid X)`. The inputs can look **identical**, but what they *mean* for the target has changed: the
rules the model learned no longer hold.

Why it's dangerous
------------------

Because the input distribution may look **normal**, concept drift is **hard to detect** — the model keeps
predicting **confidently** while being **wrong**. It shows up as a **decline** in accuracy, F1, or business
KPIs, which is why performance is monitored on **labeled** or delayed data, aided by detectors like
**ADWIN**, **DDM**, or **Page-Hinkley**.

Its forms and fix
-----------------

Drift can be **sudden** (a regime change), **gradual**, **incremental**, or **recurring** (seasonal patterns
that revert). The remedy is **retraining** on fresh labeled data that reflects the new relationship — the
reason production models need continuous **monitoring** and update loops.
"""

CONTENT["Data Drift"] = r"""
What it is
----------

**Data drift** — also called **covariate shift** — is a change in the **input** distribution :math:`P(X)`
while the model itself stays **fixed**. Its weights and logic are unchanged, but the data arriving at
inference no longer **resembles** the training data, so predictions grow **less reliable**.

Drift vs noise
--------------

Data drift is **systematic** — a sustained, directional shift — not the random fluctuation that's normal and
expected. Examples: a fraud model meeting **new devices and geographies**, or a credit model trained on
salaried workers now scoring **gig workers**.

Detecting and relating it
-------------------------

It's caught by comparing a **production** window to a **reference** window with statistical tests (**KS**,
**Chi-square**), **PSI**, or divergence metrics — the input side, so it's detectable **before** labels
arrive. Data drift can **evolve into** concept drift, which is why teams monitor :math:`P(X)` first, then
investigate the input–output relationship if performance drops.
"""

CONTENT["KS shift (Kolmogorov–Smirnov shift)"] = r"""
What it is
----------

**KS shift** detects **data drift** in a **continuous** feature with the **two-sample Kolmogorov–Smirnov
test** — it compares the feature's **cumulative distribution** in a reference window against the current
production window and measures their **largest** gap:

.. math::

   D = \sup_{x}\,\big| F_{\text{ref}}(x) - F_{\text{prod}}(x) \big|.

A large **D** means the two samples likely come from **different** distributions.

Why it's used
-------------

The KS test is **non-parametric** — it assumes **no** particular distribution shape — so it flags
**arbitrary** changes in a numeric feature's distribution. When **D** exceeds a **critical value** (or its
p-value falls below a threshold), the shift is **statistically significant**.

Using it in practice
--------------------

At production scale, tiny shifts become "significant" on **huge** samples, so the threshold is **calibrated**
to batch size to avoid **alert fatigue**. KS shift complements **PSI** (which grades severity) and
**Chi-square** (for categorical features) as part of a drift-monitoring suite.
"""

MINDMAP.update({
    "Concept Drift": [
        "Data Drift", "Label Drift (a.k.a. Target Drift)", "Covariate Drift (a.k.a. Covariate Shift)",
        "Drift Detection", "Dataset Shift", "Model Stability",
    ],
    "Data Drift": [
        "Concept Drift", "Covariate Drift (a.k.a. Covariate Shift)", "PSI (Population Stability Index)",
        "KS shift (Kolmogorov–Smirnov shift)", "Drift Detection", "Categorical Drift",
    ],
    "KS shift (Kolmogorov–Smirnov shift)": [
        "Kolmogorov–Smirnov (KS) Test", "Cumulative Distribution Function (CDF)", "Data Drift",
        "PSI (Population Stability Index)", "Covariate Drift (a.k.a. Covariate Shift)", "Concept Drift",
    ],
})


# ----------------------------------------------------------------------
# Theme: scale-free forecast error — MASE, WMAPE, sMAPE  (evaluation / metrics)
# ----------------------------------------------------------------------

CONTENT["MASE (Mean Absolute Scaled Error)"] = r"""
What it is
----------

**Mean Absolute Scaled Error** divides a forecast's **mean absolute error** by the MAE of an **in-sample
naive** benchmark — the seasonal-naive or last-value forecast — giving a pure ratio:

.. math::

   \text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}.

It answers a single question: *did the model beat the trivial baseline?*

Reading it
----------

**MASE < 1** means the forecast **outperforms** naive; **= 1** ties it; **> 1** means the naive forecast
**wins** and the model should be reconsidered. Because numerator and denominator share **units**, MASE is
**scale-free** and comparable across series of wildly different magnitudes.

Why it's the gold standard
--------------------------

Unlike percentage errors, MASE is **symmetric** (over- and under-forecasts penalized equally), **robust** to
**zeros** and outliers (the naive step is bounded away from zero unless the series is constant), and
**interpretable**. Proposed by Hyndman & Koehler (2006), it is a default for forecasting **competitions** and
multi-SKU demand.
"""

CONTENT["WMAPE (Weighted Mean Absolute Percentage Error)"] = r"""
What it is
----------

**Weighted Mean Absolute Percentage Error** divides the **total** absolute error by the **total** actual
demand — the sum of errors over the sum of actuals:

.. math::

   \text{WMAPE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}.

Rather than averaging per-item percentages, it weights each error by its **volume**.

Why weighting matters
---------------------

Plain MAPE treats a 50% miss on a **tiny** item the same as on a **huge** one and blows up when actuals are
near **zero**. WMAPE lets **high-volume** items dominate — reflecting real **business impact** — and stays
defined as long as total demand isn't zero, making it a **retail** and demand-planning staple.

Its trade-off
-------------

Because big items dominate, WMAPE can **hide** poor accuracy on the **long tail** of small items — a model can
score well while badly missing many low-volume SKUs. It is closely related to **WAPE**, and best read
**alongside** a per-item metric to catch tail errors.
"""

CONTENT["sMAPE (Symmetric Mean Absolute Percentage Error)"] = r"""
What it is
----------

**Symmetric Mean Absolute Percentage Error** fixes MAPE's asymmetry by putting the **average** of actual and
forecast in the denominator:

.. math::

   \text{sMAPE} = \frac{1}{n}\sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}.

In its common form it is **bounded** between 0% and 200%.

What it fixes (and doesn't)
---------------------------

Plain MAPE penalizes **over-forecasts** more than under-forecasts and explodes as actuals approach zero;
sMAPE is more **balanced** and **bounded**, which is why it served as the official metric of the
**M-competitions**. But it is **not** perfectly symmetric, and it still misbehaves when both actual and
forecast are near **zero** (the error jumps toward 100–200%).

When to use it
--------------

Reach for sMAPE when you want a **bounded**, roughly symmetric percentage error for comparing across series —
but avoid it on **intermittent** or zero-heavy demand, where **MASE** is the safer scale-free choice.
"""

MINDMAP.update({
    "MASE (Mean Absolute Scaled Error)": [
        "WMAPE (Weighted Mean Absolute Percentage Error)",
        "sMAPE (Symmetric Mean Absolute Percentage Error)", "Mean Absolute Percentage Error (MAPE)",
        "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Forecasting Competitions",
    ],
    "WMAPE (Weighted Mean Absolute Percentage Error)": [
        "MASE (Mean Absolute Scaled Error)", "sMAPE (Symmetric Mean Absolute Percentage Error)",
        "Mean Absolute Percentage Error (MAPE)", "WAPE (Weighted Absolute Percentage Error)",
        "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)",
    ],
    "sMAPE (Symmetric Mean Absolute Percentage Error)": [
        "MASE (Mean Absolute Scaled Error)", "WMAPE (Weighted Mean Absolute Percentage Error)",
        "Mean Absolute Percentage Error (MAPE)", "Root Mean Squared Error (RMSE)",
        "Forecasting Competitions", "Mean Absolute Error (MAE)",
    ],
})


# ----------------------------------------------------------------------
# Theme: point-to-distribution error — MAE, pinball loss, CRPS  (evaluation / metrics)
# ----------------------------------------------------------------------

CONTENT["Mean Absolute Error (MAE)"] = r"""
What it is
----------

**Mean Absolute Error** is the average **absolute** gap between prediction and truth — the **L1** error:

.. math::

   \text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|.

It reports the typical error in the **same units** as the target, with no squaring.

How it behaves
--------------

Because it takes **absolute** values rather than squares, MAE weights **all** errors **linearly** and is far
more **robust to outliers** than MSE / RMSE — one huge miss doesn't dominate. The forecast that minimizes MAE
is the **median** of the target (for RMSE it is the mean).

When to use it
--------------

MAE is the right choice when you want an **interpretable**, outlier-**resistant** measure of typical error and
don't need to punish large mistakes extra hard. Its main limits: it is **scale-dependent** (not comparable
across series — use **MASE** for that) and, being **point-only**, it can't score probabilistic forecasts.
"""

CONTENT["Pinball Loss (a.k.a. Quantile Loss)"] = r"""
What it is
----------

**Pinball loss** (a.k.a. **quantile loss**) scores a **quantile** forecast by penalizing errors
**asymmetrically** — under- and over-prediction get different weights set by the target quantile
:math:`\tau`:

.. math::

   L_\tau(y, \hat{y}) = \max\big(\tau(y - \hat{y}),\ (\tau - 1)(y - \hat{y})\big).

Minimizing it makes :math:`\hat{y}` approach the true :math:`\tau`-quantile.

Why asymmetry
-------------

For a high quantile (say :math:`\tau = 0.9`), **under**-predicting is penalized far more than over-predicting,
pushing the forecast **up** to cover the upper tail — exactly what you want for a 90% **prediction interval**.
At :math:`\tau = 0.5` the two weights match and pinball loss reduces to (half) the **MAE**.

Where it's used
---------------

It trains and evaluates **quantile regressors** and probabilistic models that output **intervals** rather than
points, without assuming any distribution. A caveat: fitting several quantiles independently can cause
**quantile crossing**, where a lower quantile's forecast exceeds a higher one's.
"""

CONTENT["Continuous Ranked Probability Score (CRPS)"] = r"""
What it is
----------

The **Continuous Ranked Probability Score** grades a **full probabilistic** forecast by comparing its
predicted **CDF** to the observed outcome — the integrated squared gap between the forecast distribution and a
step at the truth:

.. math::

   \text{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(z) - \mathbb{1}\{y \le z\}\big)^2\, dz.

**Lower** is better, and it rewards mass placed **near** the observation.

Its key properties
------------------

CRPS is a **strictly proper scoring rule** — it is minimized only by **honest**, well-calibrated
distributions, penalizing **overconfidence** — and it reports in the target's **units** (like MAE). It equals
the integral of **pinball loss** over **all** quantiles, tying the whole family together.

How it relates to MAE
---------------------

For a **point** (degenerate) forecast, the predicted CDF becomes a step function and CRPS **collapses to the
MAE**. So CRPS is literally MAE **generalized** to distributions — the natural score for **weather**,
**energy**, and **demand** probabilistic forecasting, though it is **unbounded**.
"""

MINDMAP.update({
    "Mean Absolute Error (MAE)": [
        "Pinball Loss (a.k.a. Quantile Loss)", "Continuous Ranked Probability Score (CRPS)",
        "Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)",
        "MASE (Mean Absolute Scaled Error)", "Mean Absolute Percentage Error (MAPE)",
    ],
    "Pinball Loss (a.k.a. Quantile Loss)": [
        "Continuous Ranked Probability Score (CRPS)", "Mean Absolute Error (MAE)",
        "Probabilistic Forecasts", "Strictly Proper Scoring Rules", "Root Mean Squared Error (RMSE)",
        "MASE (Mean Absolute Scaled Error)",
    ],
    "Continuous Ranked Probability Score (CRPS)": [
        "Pinball Loss (a.k.a. Quantile Loss)", "Mean Absolute Error (MAE)",
        "Strictly Proper Scoring Rules", "Probabilistic Forecasts", "Brier Score",
        "Cumulative Distribution Function (CDF)",
    ],
})


# ----------------------------------------------------------------------
# Theme: top-N recsys accuracy — hit rate, NDCG, MAP  (ranking / recsys)
# ----------------------------------------------------------------------

CONTENT["Hit Rate (HR)"] = r"""
What it is
----------

**Hit Rate** is the simplest top-N recommendation metric — it asks whether **at least one** relevant item
appears in a user's top-**K** list. Each user scores **1** if there's any hit and **0** otherwise, and the
metric is the **average** across users:

.. math::

   \text{HR@}K = \frac{\#\{\text{users with} \ge 1 \text{ relevant item in top } K\}}{|U|}.

What it captures
----------------

HR measures **coverage of intent** at the coarsest level — did we surface *something* the user wanted? — which
is exactly right for feeds, "you might also like" rows, and any setting where a **single** good hit is a win.
It is intuitive and easy to explain to stakeholders.

Its limits
----------

HR is **binary** and **position-blind** — it doesn't care **where** in the list the hit landed or **how
many** relevant items were found, so a hit at rank 1 and a hit at rank 10 score the same. It also **rises**
mechanically with **K**, so always report the cutoff (Hit@5 vs Hit@10) and pair it with a **ranking** metric.
"""

CONTENT["NDCG (Normalized Discounted Cumulative Gain)"] = r"""
What it is
----------

**Normalized Discounted Cumulative Gain** measures **ranking quality** using both **graded relevance** and
**position**. It sums each item's relevance with a **logarithmic** discount for lower ranks (DCG), then
divides by the **ideal** ordering's score (IDCG):

.. math::

   \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K} \in [0, 1].

A perfect ranking scores **1**.

Why it's powerful
-----------------

Unlike Hit Rate, NDCG rewards putting **highly** relevant items **near the top** and handles **multi-level**
relevance (a 5-star match beats a 3-star one). Normalizing by IDCG makes it **comparable** across users with
different numbers of relevant items — the reason it is the **default** offline ranking metric for
recommenders and search.

In recsys
---------

Computed **per user** then **averaged**, NDCG@K captures the personalized-ordering quality that drives
engagement. It is the recommender-system application of the same **DCG** used in information retrieval, so IR
and recsys share this yardstick.
"""

CONTENT["Mean Average Precision (MAP)"] = r"""
What it is
----------

**Mean Average Precision** is the **mean**, across all users, of each user's **Average Precision (AP)**. AP
averages the **precision** measured at **every** rank where a relevant item appears — the area under that
user's **precision–recall** curve:

.. math::

   \text{MAP} = \frac{1}{|U|}\sum_{u \in U} \text{AP}_u.

It rolls per-user ranking quality into one number.

What AP rewards
---------------

Because AP recomputes precision at each **relevant** position, it **emphasizes** getting relevant items
**early** — a relevant item at rank 1 lifts every later precision term, while one at rank 10 lifts few. So MAP
is strongly **order-sensitive**, rewarding front-loaded relevance.

How it compares
---------------

MAP works with **binary** relevance (relevant or not), where **NDCG** handles **graded** relevance; MAP
summarizes the **whole** precision–recall trade-off, where **Hit Rate** only checks for any hit. Reported at a
cutoff (MAP@K), it is a standard top-N metric for search and recommendation.
"""

MINDMAP.update({
    "Hit Rate (HR)": [
        "NDCG (Normalized Discounted Cumulative Gain)", "Mean Average Precision (MAP)",
        "DCG (Discounted Cumulative Gain)", "Average Precision (AP)",
        "Relevance in Recommender Systems", "Recall",
    ],
    "NDCG (Normalized Discounted Cumulative Gain)": [
        "DCG (Discounted Cumulative Gain)", "Mean Average Precision (MAP)", "Hit Rate (HR)",
        "Average Precision (AP)", "Relevance in Recommender Systems", "Intra-List Diversity (ILD)",
    ],
    "Mean Average Precision (MAP)": [
        "Average Precision (AP)", "NDCG (Normalized Discounted Cumulative Gain)", "Hit Rate (HR)",
        "DCG (Discounted Cumulative Gain)", "Precision (a.k.a. Positive Predictive Value, PPV)",
        "Relevance in Recommender Systems",
    ],
})


# ----------------------------------------------------------------------
# Theme: beyond-accuracy recsys — novelty, diversity, coverage  (recsys)
# ----------------------------------------------------------------------

CONTENT["Novelty (in Recommender Systems)"] = r"""
What it is
----------

**Novelty** measures how **new** or **unfamiliar** a recommender's suggestions are to a user — surfacing items
they're **unlikely to have already seen** rather than obvious mainstream hits. It is a **beyond-accuracy**
objective: a technically accurate list of things the user already knows adds little value.

How it's measured
-----------------

Novelty is usually tied to **popularity** — the less popular an item, the more novel — so metrics use
**average recommendation popularity** (lower means more novel) or the **self-information** (the negative log
of an item's popularity). Recommending from the **long tail** raises novelty.

Why it matters
--------------

Novelty drives **discovery** and helps monetize the **long tail**, but pushed too far it sacrifices
**relevance** — nobody wants *random* items. The art is balancing the **accuracy-novelty** trade-off;
combined with relevance and surprise, novelty becomes **serendipity**.
"""

CONTENT["Diversity (in Recommender Systems)"] = r"""
What it is
----------

**Diversity** measures how **varied** the items **within** a single recommendation list are — the opposite of
ten near-identical suggestions. A diverse list spans a user's **multiple** interests rather than hammering
one.

How it's measured
-----------------

The standard gauge is **intra-list dissimilarity** — the average **pairwise** distance between recommended
items (often 1 − **cosine similarity** of their features), captured by **Intra-List Diversity**. At the
catalog level, **Gini** or **entropy** across all recommendations measures aggregate diversity.

Why it matters
--------------

Diversity improves the **experience** — it hedges against a wrong guess about intent and keeps lists
interesting — but there's an **accuracy-diversity** trade-off, since the most "accurate" items are often
**similar**. Good systems tune diversity **without** dumping relevance.
"""

CONTENT["Coverage"] = r"""
What it is
----------

**Coverage** measures how much of the **catalog** a recommender actually uses — the share of available items
it is able to, or chooses to, recommend. A system can be accurate yet only ever surface a **handful** of
popular items, ignoring the rest.

Two flavors
-----------

**Prediction coverage** is the fraction of items for which the model **can** make a prediction at all;
**catalog coverage** is the fraction of items that actually **appear** in the recommendation lists users see.
The latter is the usual beyond-accuracy target.

Why it matters
--------------

High coverage means the **long tail** gets exposure and the catalog isn't wasted — countering **popularity
bias**. A limitation of plain coverage: it counts an item shown **once** the same as one shown **thousands**
of times, which is why **Gini** and **entropy** refine it to capture how **evenly** exposure is spread.
"""

MINDMAP.update({
    "Novelty (in Recommender Systems)": [
        "Diversity (in Recommender Systems)", "Coverage", "Intra-List Diversity (ILD)",
        "Relevance in Recommender Systems", "Hit Rate (HR)", "Cosine Similarity of Item Features",
    ],
    "Diversity (in Recommender Systems)": [
        "Novelty (in Recommender Systems)", "Coverage", "Intra-List Diversity (ILD)",
        "Cosine Similarity of Item Features", "Relevance in Recommender Systems",
        "NDCG (Normalized Discounted Cumulative Gain)",
    ],
    "Coverage": [
        "Novelty (in Recommender Systems)", "Diversity (in Recommender Systems)",
        "Intra-List Diversity (ILD)", "Relevance in Recommender Systems", "Hit Rate (HR)",
        "Mean Average Precision (MAP)",
    ],
})


# ----------------------------------------------------------------------
# Theme: calibration diagnostics — ECE, reliability curves, Brier score  (calibration)
# ----------------------------------------------------------------------

CONTENT["Expected Calibration Error (ECE)"] = r"""
What it is
----------

**Expected Calibration Error** summarizes miscalibration in **one number** — the **weighted average** gap
between a model's **confidence** and its **accuracy**, taken over bins of predictions:

.. math::

   \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

Geometrically, it is the average distance of the **calibration curve** from the diagonal.

How it's computed
-----------------

Predictions are grouped into **bins** by confidence; in each bin you compare the **fraction correct**
(accuracy) to the **average confidence**, and weight each bin's gap by its **size**. The result is
**bounded** in [0, 1] and easy to report — the standard scalar for comparing calibration.

Its caveats
-----------

ECE is **bin-dependent** (the number and placement of bins move the value) and, being an **average**, it can
**hide** a badly miscalibrated region behind well-behaved bins. It is also **not a proper scoring rule** — a
trivial model can score low — so it is read with **reliability curves** and **Brier score**. Its variants are
**MCE** and **Adaptive ECE**.
"""

CONTENT["Reliability Curves (also called Calibration Curves)"] = r"""
What it is
----------

A **reliability curve** (or calibration curve / reliability diagram) is the **visual** check for calibration —
it plots **predicted probability** on the x-axis against the **observed frequency** of the outcome on the
y-axis. A perfectly calibrated model traces the **diagonal** :math:`y = x`.

Reading it
----------

Points **below** the diagonal mean the model is **overconfident** (accuracy falls short of its confidence);
points **above** mean it is **underconfident**. A companion **histogram** of confidences shows whether
predictions pile up at the **extremes** — a hallmark of overconfident networks.

Why it complements ECE
----------------------

A single ECE number can't say **where** miscalibration happens, and two models with the **same** ECE can have
very different curves. The reliability curve **localizes** the problem across the confidence range — and the
weighted gap between it and the diagonal **is** the ECE.
"""

CONTENT["Brier Score"] = r"""
What it is
----------

The **Brier score** is the **mean squared error** of probabilistic predictions — the average squared gap
between the predicted probability and the **actual** (0/1) outcome:

.. math::

   \text{BS} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2.

**Lower** is better, with **0** perfect. It is a single sample-level number for binary or multiclass problems.

Why it's special
----------------

Unlike ECE, the Brier score is a **strictly proper scoring rule** — it is minimized only by **honest**
probabilities, and by **Murphy's decomposition** it splits into **calibration** plus **refinement** terms. So
a low Brier score means the model is **both** well-calibrated **and** discriminating.

Its limitation
--------------

Because it **blends** calibration and discrimination, the Brier score can't tell you **which** is lacking — a
sharp-but-miscalibrated model and a calibrated-but-vague one can score similarly. That is why it is reported
**alongside** ECE and reliability curves, which isolate the calibration piece.
"""

MINDMAP.update({
    "Expected Calibration Error (ECE)": [
        "Adaptive ECE (Expected Calibration Error with Adaptive Binning)",
        "Maximum Calibration Error (MCE)", "Reliability Curves (also called Calibration Curves)",
        "Confidence Level", "Temperature Scaling", "Brier Score",
    ],
    "Reliability Curves (also called Calibration Curves)": [
        "Expected Calibration Error (ECE)", "Overconfident", "Underconfident", "Confidence Level",
        "Brier Score", "Temperature Scaling",
    ],
    "Brier Score": [
        "Expected Calibration Error (ECE)", "Murphy's Decomposition", "Strictly Proper Scoring Rules",
        "Continuous Ranked Probability Score (CRPS)",
        "Reliability Curves (also called Calibration Curves)",
        "Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)",
    ],
})


# ----------------------------------------------------------------------
# Theme: calibration & scores — log loss, calibration quality, logits  (calibration)
# ----------------------------------------------------------------------

CONTENT["Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)"] = r"""
What it is
----------

**Log loss** (logarithmic loss / **cross-entropy** loss) scores a probabilistic classifier by the **negative
log-likelihood** of the true labels — how surprised the model is by reality. For binary labels:

.. math::

   \text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\big[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\big].

It runs from **0** (perfect) to **∞**.

Its defining trait
------------------

The **logarithm** makes log loss punish **confident** mistakes **brutally** — predicting 0.01 for a true
positive costs far more than predicting 0.4. This **exponential** penalty for overconfidence is exactly why
it is the standard **training objective** for logistic regression and neural nets, which optimize it directly.

When to use it (and a caveat)
-----------------------------

Reach for log loss whenever **calibrated** probabilities matter — fraud, diagnosis, risk. It **requires**
genuine probabilities, so raw **logits** must be squashed (softmax / sigmoid) first, and it is harsher on
overconfidence than the **Brier score**. Compare a model's log loss to the **base-rate** log loss to confirm
it adds value.
"""

CONTENT["Calibration quality (Model Calibration)"] = r"""
What it is
----------

**Calibration quality** describes how well a model's predicted **probabilities** match **reality** — a
well-calibrated model that says "80% confident" is right about **80%** of the time. It is a property of the
**probabilities**, separate from whether the model is **accurate**.

Calibration vs accuracy
-----------------------

A model can be **accurate** yet badly calibrated (right often, but its confidence numbers are meaningless) or
**calibrated** yet weakly **discriminating**. The two are distinct axes, which is why **proper scores** like
log loss and Brier — which reward **both** — are read together with pure calibration measures.

Why it matters
--------------

Whenever probabilities feed **decisions** — thresholds, expected-value calculations, downstream systems —
calibration is essential, and modern deep networks are typically **overconfident**. It is measured with
**ECE**, **reliability curves**, and **Brier score**, and repaired **post-hoc** with **temperature**,
**Platt**, or **isotonic** scaling.
"""

CONTENT["Logits"] = r"""
What it is
----------

**Logits** are the **raw, unnormalized** scores a classifier's final layer produces **before** they're turned
into probabilities. They range over **all** real numbers — positive, negative, unbounded — and live in
**log-odds** space, not probability space.

From logits to probabilities
-----------------------------

A **softmax** turns a vector of logits into a probability **distribution** that sums to 1 (for multiclass),
while a **sigmoid** maps a single logit to one probability (for binary). Because softmax **normalizes**,
raising one logit **lowers** the others' probabilities — the competition that sharpens a prediction.

Why keep them raw
-----------------

Exposing logits enables **numerically stable** training (log-softmax beats probabilities-then-log, which is
why frameworks feed **cross-entropy** raw logits) and **post-hoc calibration** — **temperature scaling**
divides logits by T *before* softmax, which is only possible when the logits are available.
"""

MINDMAP.update({
    "Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)": [
        "Brier Score", "Expected Calibration Error (ECE)", "Binary Cross-Entropy (BCE)",
        "Softmax Function", "Logistic Regression", "Strictly Proper Scoring Rules",
    ],
    "Calibration quality (Model Calibration)": [
        "Expected Calibration Error (ECE)", "Reliability Curves (also called Calibration Curves)",
        "Temperature Scaling", "Overconfident", "Confidence Level", "Brier Score",
    ],
    "Logits": [
        "Softmax Function", "Sigmoid Function", "Logit Space", "Log-Odds", "Temperature Scaling",
        "Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)",
    ],
})


# ----------------------------------------------------------------------
# Theme: digital advertising metrics — conversion rate, CPC, CTR  (growth / abtest)
# ----------------------------------------------------------------------

CONTENT["Conversion Rate (CR)"] = r"""
What it is
----------

**Conversion rate** is the share of visitors who complete a **desired action** — a purchase, sign-up, or form
submission — out of all visitors:

.. math::

   \text{CR} = \frac{\text{conversions}}{\text{total visitors}} \times 100.

It measures how effectively traffic turns into **outcomes**.

Where it sits
-------------

CR is a **mid-to-bottom-of-funnel** metric — it captures what people do **after** they've arrived, so it
reflects the quality of the **landing page**, offer, and checkout, not just the ad. It is the KPI most
**directly tied to revenue**, which is why marketers watch it closely.

How to move it
--------------

Conversion rate is improved by sharpening the **offer** and **copy**, streamlining the **funnel**, and running
**A/B tests** on page variants — not by chasing more clicks. A campaign with modest traffic but high CR often
beats a high-traffic, low-CR one on profit.
"""

CONTENT["Cost-Per-Click (CPC) Models"] = r"""
What it is
----------

**Cost-per-click** is the **pay-per-click (PPC)** pricing model where an advertiser pays each time someone
**clicks** their ad — total spend divided by clicks:

.. math::

   \text{CPC} = \frac{\text{ad spend}}{\text{clicks}}.

You pay for **engagement**, not mere exposure.

How it's set
------------

CPC isn't fixed — it comes out of an **auction** shaped by your **bid**, the ad's **quality score /
relevance**, and **competition** for the audience. A higher **CTR** signals relevance and typically
**lowers** your CPC, so better creative pays for itself.

CPC vs CPM
----------

Under CPC you pay only when users **act**, making it ideal for **traffic and conversion** goals; under
**CPM** (cost per thousand impressions) you pay for **visibility** regardless of clicks, better for
**awareness**. The two connect: CPC is roughly CPM divided by (1000 × CTR).
"""

CONTENT["CTR (Click-Through Rate)"] = r"""
What it is
----------

**Click-through rate** is the share of people who **click** an ad or link after **seeing** it — clicks divided
by **impressions**:

.. math::

   \text{CTR} = \frac{\text{clicks}}{\text{impressions}} \times 100.

It is the earliest online-advertising metric, dating to 1994 banner ads.

What it signals
---------------

CTR is a **top-of-funnel** engagement gauge — it measures whether the **creative, targeting, and message**
capture attention. On ad platforms a high CTR raises the **quality score**, which in turn **lowers CPC** and
wins better **placement**, so relevance compounds.

Its trap
--------

A **high** CTR is **not** success by itself — if those clicks don't **convert**, you're paying for traffic
that doesn't pay back. CTR must be read **with conversion rate**: high CTR + low CR usually means the ad
**over-promises** relative to the landing page or offer.
"""

MINDMAP.update({
    "Conversion Rate (CR)": [
        "CTR (Click-Through Rate)", "Cost-Per-Click (CPC) Models", "True Conversion Rate",
        "Conversion Rate Uplift", "Incremental Conversions", "CAC (Customer Acquisition Cost)",
    ],
    "Cost-Per-Click (CPC) Models": [
        "CTR (Click-Through Rate)", "Conversion Rate (CR)", "CAC (Customer Acquisition Cost)",
        "Incremental Conversions", "Conversion Rate Uplift", "Paid CAC (Customer Acquisition Cost)",
    ],
    "CTR (Click-Through Rate)": [
        "Conversion Rate (CR)", "Cost-Per-Click (CPC) Models", "Conversion Rate Uplift",
        "CAC (Customer Acquisition Cost)", "True Conversion Rate", "A/B Testing",
    ],
})


# ----------------------------------------------------------------------
# Theme: retail demand forecasting — overstock, stockouts, WAPE  (supply chain / metrics)
# ----------------------------------------------------------------------

CONTENT["Overstock %"] = r"""
What it is
----------

**Overstock %** is the share of inventory that sits **in excess** of what demand needs — the fraction of
stock held **beyond** target or forecast requirements. It is the "**too much**" side of inventory planning.

What it costs
-------------

Excess stock isn't free — it ties up **capital**, consumes **warehouse** space, and risks **obsolescence**,
spoilage, and eventual **markdowns**; holding costs often run to a fifth or more of the inventory's value.
High overstock usually traces to **over-forecasting** demand or over-ordering to avoid the opposite problem.

The trade-off
-------------

Overstock is one horn of the classic inventory dilemma — cutting it too aggressively invites **stockouts**.
The balance point is a chosen **service level**, and better **demand forecasts** (lower error) are what let a
business hold **less** safety stock without more shortages.
"""

CONTENT["Stockouts"] = r"""
What it is
----------

A **stockout** is when an item is **unavailable** — demand arrives but inventory is **gone**. It is the "**too
little**" side of inventory planning, and each stockout is a **missed** sale at the moment of intent.

What it costs
-------------

Stockouts cause **lost revenue**, frustrated customers, and **substitution** to competitors — and the damage
can outlast the event if shoppers **don't return**. They stem from **under-forecasting**, demand spikes, or
**supply** disruptions, and most are preventable failures of replenishment rather than supplier problems.

The trade-off
-------------

Stockouts are the mirror image of **overstock** — you can eliminate them by holding **more** inventory, but
that raises **holding costs**. Firms manage the tension with **safety stock** sized to a target **service
level**, and sharper forecasts shrink both failure modes at once.
"""

CONTENT["WAPE (Weighted Absolute Percentage Error)"] = r"""
What it is
----------

**Weighted Absolute Percentage Error** divides the **total** absolute forecast error by the **total** actual
demand:

.. math::

   \text{WAPE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}.

It expresses aggregate error as a single percentage, weighting each item by its **volume**.

Why retailers use it
--------------------

Unlike **MAPE**, WAPE doesn't **blow up** when individual actuals are near **zero** (common for slow-moving
SKUs), and it lets **high-volume** items dominate — reflecting real demand. It stays defined as long as total
demand isn't zero, which makes it the **default** accuracy metric in demand planning.

How it connects
---------------

WAPE is effectively the same quantity as **WMAPE**, and it is the number that **drives inventory** — high WAPE
means forecasts are far off, feeding both **overstock** and **stockouts**. Cutting WAPE is how planners
**shrink** those costly failure modes.
"""

MINDMAP.update({
    "Overstock %": [
        "Stockouts", "WAPE (Weighted Absolute Percentage Error)",
        "WMAPE (Weighted Mean Absolute Percentage Error)", "Mean Absolute Percentage Error (MAPE)",
        "Time Series Forecasting", "Seasonality",
    ],
    "Stockouts": [
        "Overstock %", "WAPE (Weighted Absolute Percentage Error)",
        "WMAPE (Weighted Mean Absolute Percentage Error)", "Time Series Forecasting",
        "MASE (Mean Absolute Scaled Error)", "Seasonality",
    ],
    "WAPE (Weighted Absolute Percentage Error)": [
        "WMAPE (Weighted Mean Absolute Percentage Error)", "MASE (Mean Absolute Scaled Error)",
        "Mean Absolute Percentage Error (MAPE)", "Overstock %", "Stockouts",
        "Mean Absolute Error (MAE)",
    ],
})


# ----------------------------------------------------------------------
# Theme: regulation & high-stakes — fair lending, Basel III, high-stakes domains  (fairness / risk)
# ----------------------------------------------------------------------

CONTENT["Fair Lending laws"] = r"""
What they are
-------------

**Fair lending laws** are the US statutes that **prohibit discrimination** in credit — chiefly the **Equal
Credit Opportunity Act (ECOA)**, covering all credit transactions, and the **Fair Housing Act**, covering
mortgages and housing. They bar decisions based on **protected characteristics** like race, sex, age,
religion, or national origin.

Two theories of harm
--------------------

Enforcement rests on **disparate treatment** (treating applicants differently **on purpose**) and **disparate
impact** (a **facially neutral** rule that disproportionately **burdens** a protected group, even with **no**
intent). Crucially, **statistical disparities alone** can trigger scrutiny before any intent is shown.

Why they matter for ML
----------------------

A credit model can discriminate through **proxies** — zip code or surname standing in for race — so compliance
demands **fairness testing**, sensitivity analysis, and ongoing **monitoring**. It also requires
**adverse-action** explanations specific enough to tell a rejected applicant **why**, which is hard for
**opaque** models — a driver of explainable AI in finance.
"""

CONTENT["Basel III"] = r"""
What it is
----------

**Basel III** is the international **banking regulation** framework from the **Basel Committee**, written after
the **2007–09 financial crisis** to make banks more **resilient**. It tightens the **capital**, **liquidity**,
and **leverage** a bank must hold against its risks.

Its core requirements
---------------------

It raises both the **quantity** and **quality** of capital (more **common equity**), adds **liquidity** rules
(holding enough liquid assets to survive stress), a **leverage** cap, and **buffers** that build up in good
times to absorb losses in bad ones — all aimed at reducing **systemic** risk.

Why it matters for ML
---------------------

Banks estimate **credit risk** — probability of default, loss given default — with models whose outputs feed
**capital** calculations and **stress tests**. That puts those models under strict **model-risk management**
and validation, making Basel III a major reason financial ML must be **auditable** and **robust**.
"""

CONTENT["High-Stakes Domains"] = r"""
What they are
-------------

**High-stakes domains** are application areas where a model's **mistakes carry severe consequences** — harm to
**health**, **liberty**, **livelihood**, or **safety**. Think medical diagnosis, criminal-justice risk
scoring, credit and hiring decisions, and autonomous driving.

Why they're different
----------------------

In these settings **accuracy alone is not enough**. A model must also be **interpretable**, **fair**,
**calibrated**, and **robust**, with rigorous **validation**, **uncertainty** estimates, and **human
oversight** — because a confident wrong answer can ruin a life, not just a recommendation.

What they demand
----------------

High-stakes use is where **trustworthy-AI** requirements concentrate — **explainability** (SHAP, LIME),
**fairness** auditing, calibration, guardrails, and regulatory compliance (**fair-lending** law, **Basel
III**, medical-device rules). It is the contrast case to **low-stakes** tasks like movie recommendations,
where an error is trivial.
"""

MINDMAP.update({
    "Fair Lending laws": [
        "High-Stakes Domains", "Basel III", "Fairness Guardrails", "Risk-Based Decisions",
        "SHAP (SHapley Additive exPlanations)",
        "LIME (Local Interpretable Model-agnostic Explanations)",
    ],
    "Basel III": [
        "High-Stakes Domains", "Fair Lending laws", "Risk-Based Decisions", "Model Stability",
        "Fairness Guardrails", "LIME (Local Interpretable Model-agnostic Explanations)",
    ],
    "High-Stakes Domains": [
        "Fair Lending laws", "Basel III", "Fairness Guardrails", "Risk-Based Decisions",
        "SHAP (SHapley Additive exPlanations)", "Model Stability",
    ],
})


# ----------------------------------------------------------------------
# Theme: fairness metrics — fairness parity, selection rate, recall  (fairness / metrics)
# ----------------------------------------------------------------------

CONTENT["Fairness parity"] = r"""
What it is
----------

**Fairness parity** is the family of **group-fairness** criteria that demand some metric be **equal across**
demographic groups — that the model treat protected groups **comparably**. Which metric you equalize defines
the flavor of parity.

The main flavors
----------------

**Demographic (statistical) parity** equalizes the **positive-outcome rate** across groups; **equal
opportunity** equalizes the **true-positive rate** (recall) among those who **qualify**; **equalized odds**
equalizes **both** TPR and FPR; **predictive parity** equalizes **precision**. Each is measured as a
**difference** or a **ratio** between groups.

Why it's hard
-------------

The different parities **conflict** — impossibility results show you generally **can't** satisfy all at once
(the **COMPAS** debate turned on predictive parity holding while equalized odds failed). Enforcing any parity
also usually **costs accuracy**, so teams choose the criterion that fits the **harm** they most need to
prevent. The **four-fifths (80%) rule** is a common legal threshold.
"""

CONTENT["Selection Rate"] = r"""
What it is
----------

The **selection rate** is the fraction of a group that receives a **favorable** decision — the share predicted
**positive**, :math:`P(\hat{Y} = 1)` for that group. If a lender approves 60% of one group and 40% of
another, those are the two selection rates.

What it drives
--------------

Selection rate is the quantity behind **demographic parity** — parity holds exactly when selection rates are
**equal** across groups. The **disparate-impact ratio** divides the **lowest** group's selection rate by the
**highest**; a ratio **below 0.8** (the four-fifths rule) flags potential **adverse impact**.

Where it's used
---------------

It is the core number in auditing **hiring**, **lending**, and admissions for bias, because it captures **who
gets the good outcome** without needing labels. But equal selection rates say **nothing** about whether the
**right** people were chosen — that is why it is paired with error-based metrics like **recall**.
"""

CONTENT["Recall"] = r"""
What it is
----------

**Recall** — also the **true positive rate** or **sensitivity** — is the share of **actual positives** the
model correctly **catches**:

.. math::

   \text{Recall} = \frac{TP}{TP + FN}.

It answers "**of everything that was truly positive, how much did we find?**" and falls when **false
negatives** pile up.

The trade-off
-------------

Recall trades off against **precision** — loosening the threshold catches more positives (higher recall) but
admits more false alarms (lower precision), which is why they're read together via the **F1-score**. High
recall matters most when a **miss** is costly: disease screening, fraud, safety.

Its fairness role
-----------------

Comparing recall **across groups** is exactly the **equal-opportunity** fairness test — it asks whether
**qualified** members of every group have the **same** chance of being correctly selected. Unequal recall
means the model **misses** true positives more often for one group, a common and consequential bias.
"""

MINDMAP.update({
    "Fairness parity": [
        "Selection Rate", "Recall", "Fairness Guardrails", "Fair Lending laws",
        "High-Stakes Domains", "Precision (a.k.a. Positive Predictive Value, PPV)",
    ],
    "Selection Rate": [
        "Fairness parity", "Recall", "Fair Lending laws", "Fairness Guardrails",
        "High-Stakes Domains", "Conversion Rate (CR)",
    ],
    "Recall": [
        "Fairness parity", "Selection Rate", "Precision (a.k.a. Positive Predictive Value, PPV)",
        "F1-score", "Macro Recall", "Micro Recall",
    ],
})


# ----------------------------------------------------------------------
# Theme: unit economics — LTV, CAC, incremental sales  (growth)
# ----------------------------------------------------------------------

CONTENT["LTV (Customer Lifetime Value)"] = r"""
What it is
----------

**Customer lifetime value (LTV)** is the total **revenue** — or, more precisely, **profit** — an average
customer generates over their **entire relationship** with the business, from first purchase to churn. It
answers *how much is a customer actually worth?*

How it's estimated
------------------

A common form multiplies **average revenue per account** by **gross margin** and divides by the **churn
rate** (equivalently, revenue per period times expected lifetime). So LTV rises with **higher retention**,
fatter **margins**, and more **revenue per customer** — retention being the biggest lever.

Why it matters
--------------

LTV is the **value** half of **unit economics**. It sets the ceiling on what you can afford to **spend**
acquiring customers, justifies investment in **retention** and **upsell**, and — paired with acquisition cost
— tells investors whether the business is **sustainable**. A small lift in retention can swing LTV
dramatically.
"""

CONTENT["CAC (Customer Acquisition Cost)"] = r"""
What it is
----------

**Customer acquisition cost (CAC)** is the total **sales and marketing** spend required to win one new
customer — all acquisition costs divided by the **number** of customers acquired in the period. It is the
**cost** half of unit economics.

What goes in it
---------------

CAC bundles **ad spend**, marketing tools and agencies, and **sales** salaries and commissions — not just
media cost. It differs from **CPA** (cost per acquisition), which is usually a **single-channel** metric, and
it must be computed with **matched** time periods so costs line up with the customers they actually acquired.

The LTV:CAC ratio
-----------------

CAC only means something **next to LTV**. The **LTV:CAC ratio** — lifetime value divided by acquisition cost —
gauges whether growth is **profitable**: around **3:1** is the healthy benchmark, **below 1:1** loses money on
every customer, and **above ~5:1** can signal you're **under-investing** in growth.
"""

CONTENT["Incremental Sales"] = r"""
What it is
----------

**Incremental sales** are the **extra** sales a specific action — a campaign, promotion, or discount —
actually **caused**, over and above the **baseline** that would have happened **anyway**. It is the sales
figure with the campaign **minus** the counterfactual without it.

Why it's tricky
---------------

Raw campaign sales **overstate** impact, because some buyers would have purchased **regardless**. Isolating
the true lift requires an **incrementality test** — a **holdout** or control group that doesn't see the
campaign — so the **difference** between treated and control reveals the **causal** effect, not mere
correlation.

Why it matters
--------------

Incremental sales are the honest basis for **marketing ROI** and **budget** decisions — you should credit a
campaign only with the revenue it **truly** drove, not sales it merely **coincided** with. The same logic
underlies **incremental conversions** and **uplift** modeling: measure what **wouldn't** have happened
otherwise.
"""

MINDMAP.update({
    "LTV (Customer Lifetime Value)": [
        "CAC (Customer Acquisition Cost)", "Incremental Sales", "Conversion Rate (CR)",
        "Blended CAC (Customer Acquisition Cost)", "Paid CAC (Customer Acquisition Cost)",
        "Incremental Conversions",
    ],
    "CAC (Customer Acquisition Cost)": [
        "LTV (Customer Lifetime Value)", "Incremental Sales", "Conversion Rate (CR)",
        "CTR (Click-Through Rate)", "Cost-Per-Click (CPC) Models",
        "Blended CAC (Customer Acquisition Cost)",
    ],
    "Incremental Sales": [
        "Incremental Conversions", "LTV (Customer Lifetime Value)", "CAC (Customer Acquisition Cost)",
        "Conversion Rate Uplift", "Conversion Rate (CR)", "Incremental Recovery Rate (IRR)",
    ],
})


# ----------------------------------------------------------------------
# Theme: classical statistics — population proportion, correlation, statistical power  (probstats / inference)
# ----------------------------------------------------------------------

CONTENT["Population Proportion"] = r"""
What it is
----------

The **population proportion** is the **true fraction** of an entire population that has some characteristic —
the real conversion rate, defect rate, or approval share — usually written **p**. It is a **parameter**: a
fixed but usually **unknown** number you want to learn.

Estimating it
-------------

Since you rarely measure everyone, you estimate p with the **sample proportion** :math:`\hat{p} = x/n`
(successes over sample size). :math:`\hat{p}` is a **statistic** that varies from sample to sample, with
**standard error** :math:`\sqrt{p(1-p)/n}` shrinking as **n** grows — the basis for **confidence intervals**
and **hypothesis tests** about p.

Where it shows up
-----------------

Proportions are everywhere in ML and analytics — **conversion rates**, click rates, and accuracy are all
population proportions estimated from samples. That is why **A/B tests** and polls rest on proportion
inference, and why bigger samples give **tighter** estimates.
"""

CONTENT["Correlation"] = r"""
What it is
----------

**Correlation** measures the **strength and direction** of the relationship between two variables. The
**Pearson coefficient** — **r** in a sample, **ρ** in the population — runs from **−1 to +1**: −1 a perfect
**negative** line, +1 a perfect **positive** line, and **0** no **linear** relationship.

How to read it
--------------

The **sign** gives direction, the **magnitude** gives strength; squaring it yields **r²**, the share of one
variable's **variance explained** by the other. Rough effect-size guides call 0.1 small, 0.3 medium, 0.5
large — but a statistically significant r can still be **trivially** small in a large sample.

Its limits
----------

Correlation captures only **linear** association, so it can **miss** strong nonlinear patterns; it is **not
robust** to **outliers**, which can inflate or hide it; and, crucially, **correlation is not causation** — two
variables can move together because a **third** drives both. Use rank correlation (Spearman) for monotonic,
non-linear ties.
"""

CONTENT["Statistical Power"] = r"""
What it is
----------

**Statistical power** is the probability that a test **correctly detects a real effect** — that it **rejects**
the null hypothesis when the null is genuinely **false**. Formally it is :math:`1 - \beta`, where
:math:`\beta` is the **Type II error** (false-negative) rate.

What it depends on
------------------

Power rises with **larger effect sizes**, **bigger samples**, a **looser** significance level :math:`\alpha`,
and **lower** variance. Researchers conventionally target **0.80** — an 80% chance of catching a true effect —
and solve for the **sample size** that achieves it via **power analysis**.

Why it matters
--------------

An **underpowered** study is likely to **miss** true effects and produces findings that **don't replicate**;
power is the guard against **false negatives**, the complement of the :math:`\alpha` that guards against false
positives. Too much power on a huge sample flips the risk — flagging **trivial** effects as significant, which
is why **effect size** is reported alongside significance.
"""

MINDMAP.update({
    "Population Proportion": [
        "Confidence Intervals (CIs)", "Statistical Tests", "Standard Error (SE)", "Statistical Power",
        "Conversion Rate (CR)", "Bootstrap Confidence Intervals (CIs)",
    ],
    "Correlation": [
        "R² (R-squared)", "Statistical Power", "Statistical Tests", "Confidence Intervals (CIs)",
        "Population Proportion", "Outlier",
    ],
    "Statistical Power": [
        "Power Analysis", "Confidence Intervals (CIs)", "Statistical Tests", "Population Proportion",
        "Correlation", "Chi-square (χ²) Test",
    ],
})


# ----------------------------------------------------------------------
# Theme: causal & probabilistic inference — causal trees, IRR, Bayesian inference  (causal / bayes)
# ----------------------------------------------------------------------

CONTENT["Causal Trees"] = r"""
What it is
----------

**Causal trees** estimate **heterogeneous treatment effects** — how an intervention's impact **varies** across
subgroups — by **recursively partitioning** the feature space into **leaves** within which the treatment
effect (treated-vs-control outcome difference) is roughly **constant**. Introduced by **Athey and Imbens**,
they adapt decision trees from **prediction** to **causal** estimation.

The honesty trick
-----------------

Unlike a standard tree, a causal tree uses **honest** estimation — one part of the data **chooses** the
splits, a **separate** part **estimates** the effect in each leaf. This prevents the tree from **overfitting**
the same data it split on, giving effect estimates you can build **confidence intervals** around.

What it's for
-------------

Causal trees answer **who benefits** — the core question of **uplift modeling**, personalized medicine, and
policy targeting. Extended to **causal forests** for stability, and to **Bayesian** versions (causal BART)
that return a full **posterior** over each unit's effect for uncertainty-aware decisions.
"""

CONTENT["Incremental Recovery Rate (IRR)"] = r"""
What it is
----------

The **incremental recovery rate** measures the **lift** an intervention produces in a **recovery** outcome —
reactivating lapsed customers, collecting overdue accounts, or winning back churned users — beyond what would
have recovered **anyway**. It is the treated group's recovery rate **minus** a control group's:

.. math::

   \text{IRR} = \text{recovery rate}_{\text{treated}} - \text{recovery rate}_{\text{control}}.

Why the baseline matters
------------------------

A raw recovery rate **overstates** a campaign's value, because some accounts **self-cure** or return without
any nudge. Subtracting a **holdout** control isolates the **causal** lift — the recoveries the intervention
**actually** caused — the same incrementality logic behind incremental sales and conversions.

Where it's used
---------------

IRR drives **targeting** and **budget** in collections, retention, and win-back — you focus effort on segments
with the **highest** incremental recovery, not the highest raw recovery, since some of those would come back
**for free**. It pairs naturally with **uplift** models that predict per-customer lift.
"""

CONTENT["Bayesian Inference."] = r"""
What it is
----------

**Bayesian inference** updates **beliefs** in light of evidence using **Bayes' theorem** — it combines a
**prior** (what you believed before) with the **likelihood** (how probable the data are under each hypothesis)
to produce a **posterior** (what you believe after):

.. math::

   \text{posterior} \propto \text{prior} \times \text{likelihood}.

Parameters are treated as **random variables** with distributions, not fixed points.

What makes it distinctive
-------------------------

Because it yields a **full posterior distribution**, Bayesian inference quantifies **uncertainty** directly — a
**credible interval** says there's a 95% probability the parameter lies inside it — and it naturally
**incorporates prior knowledge** and **updates sequentially** as data arrive. This contrasts with the
**frequentist** view of fixed parameters and p-values.

The catch and the tools
-----------------------

Posteriors are usually **intractable**, so they're approximated with **Markov Chain Monte Carlo** or
variational methods via tools like **Stan**, **PyMC**, or NumPyro. Bayesian inference underlies **Bayesian A/B
testing**, **Bayesian optimization**, and the **causal** tree models above.
"""

MINDMAP.update({
    "Causal Trees": [
        "Uplift Random Forests", "Decision Trees", "Bayesian Inference.",
        "Incremental Recovery Rate (IRR)", "Incremental Sales", "Conversion Rate Uplift",
    ],
    "Incremental Recovery Rate (IRR)": [
        "Incremental Sales", "Incremental Conversions", "Causal Trees", "Conversion Rate Uplift",
        "Uplift Random Forests", "CAC (Customer Acquisition Cost)",
    ],
    "Bayesian Inference.": [
        "Causal Trees", "Likelihood", "Normal Distribution", "Statistical Tests",
        "Confidence Intervals (CIs)", "Uplift Random Forests",
    ],
})


# ----------------------------------------------------------------------
# Theme: pragmatic modeling — weak supervision, RMSLE, baseline heuristics  (features / evaluation)
# ----------------------------------------------------------------------

CONTENT["Weak Supervision"] = r"""
What it is
----------

**Weak supervision** trains models from **noisy, cheap, or imprecise** label sources instead of costly
**hand-labeling** — a direct answer to the training-data **bottleneck**. Rather than perfect ground truth, it
leans on many **imperfect** signals.

How it works
------------

Users write **labeling functions** — small snippets of **heuristics**, keyword rules, external knowledge, or
other models' outputs — that each **label** or **abstain**, often with **unknown** accuracy and **conflicting**
votes. A **label model** then **de-noises** and combines them, estimating each function's reliability to
produce **probabilistic** consensus labels — with **no** ground truth. Those labels train a downstream
classifier. This is the **Snorkel / data-programming** paradigm.

Its trade-off
-------------

Weak supervision makes labeling **dramatically** faster and its rules **interpretable** and easy to update, at
the cost of **noisier** labels than full annotation. Best practice keeps a small **hand-labeled** set to
**validate** quality and compare against fully supervised baselines.
"""

CONTENT["RMSLE (Root Mean Squared Logarithmic Error)"] = r"""
What it is
----------

**Root Mean Squared Logarithmic Error** is RMSE computed on the **logarithms** of the predictions and
actuals — the root-mean-square of the differences in **log space**:

.. math::

   \text{RMSLE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\big(\log(\hat{y}_i + 1) - \log(y_i + 1)\big)^2}.

The **+1** lets it handle zeros.

What the log changes
--------------------

Taking logs turns absolute errors into **relative** ones, so a fixed **percentage** miss costs the same
whether the value is small or huge — making RMSLE **robust to scale** and to large **outliers**. It also
becomes **asymmetric**: it penalizes **under**-prediction more than over-prediction.

When to use it
--------------

RMSLE suits **positive**, **right-skewed** targets that span orders of magnitude — prices, counts, demand —
and situations where **under-forecasting** is the costlier mistake. Its limits: it **can't** take negative
values, and its log scaling makes the raw number **less intuitive** than RMSE.
"""

CONTENT["Baseline Heuristics"] = r"""
What it is
----------

**Baseline heuristics** are the **simple, naive** reference models a real system must **beat** to earn its
complexity — the "**dumb**" benchmark. Predicting the **mean** or **median**, the **majority class**, the
**last value**, or a basic **if-then rule** are all baselines.

Why they're essential
---------------------

A metric is **meaningless** in isolation — 90% accuracy is trivial if the majority class is already 90%. A
baseline sets the **floor**: if a complex model **can't** beat it, the model adds **no value** and may even
hide a **bug**. Baselines are **cheap**, fast, and interpretable, so they cost almost nothing to run.

Where they show up
------------------

Baselines frame every honest **evaluation** and are **built into** metrics — **MASE**, for instance, divides a
forecast's error by a **naive** baseline's, so a score below 1 literally means "**better than the
heuristic**." Always establish the baseline **first**.
"""

MINDMAP.update({
    "Weak Supervision": [
        "Full Annotation", "Label Noise", "Computer Vision (CV)",
        "Natural Language Processing (NLP)", "Embedding", "Neural Networks",
    ],
    "RMSLE (Root Mean Squared Logarithmic Error)": [
        "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)",
        "MASE (Mean Absolute Scaled Error)", "Mean Absolute Percentage Error (MAPE)",
        "WAPE (Weighted Absolute Percentage Error)", "Outlier",
    ],
    "Baseline Heuristics": [
        "MASE (Mean Absolute Scaled Error)", "Root Mean Squared Error (RMSE)", "Accuracy",
        "Mean Absolute Error (MAE)", "Time Series Forecasting", "Model Score",
    ],
})


# ----------------------------------------------------------------------
# Theme: final miscellany — underflow, crew overtime, advanced spreadsheet sorting
# ----------------------------------------------------------------------

CONTENT["Underflow"] = r"""
What it is
----------

**Underflow** happens when a computation produces a number **too small** to represent in the floating-point
format — smaller than the tiniest positive value — so the computer **rounds it to zero**, destroying a real
nonzero result. It is the small-magnitude counterpart of **overflow**.

Why ML hits it
--------------

Machine learning multiplies **many small probabilities** — in Naive Bayes, HMMs, and likelihoods — and the
product of hundreds of values below 1 quickly drops below the representable floor, collapsing to **0** and
corrupting the result. Low-precision (**float16**) training underflows even sooner, showing up as **vanishing
gradients**.

The fix
-------

Compute in **log space**. Because :math:`\log(a \cdot b) = \log(a) + \log(b)`, a fragile **product** of tiny
probabilities becomes a stable **sum** of log-probabilities — the reason libraries use **log-likelihoods** and
the **LogSumExp** trick, and why scikit-learn's Naive Bayes works with logs internally.
"""

CONTENT["Crew Overtime"] = r"""
What it is
----------

**Crew overtime** is the labor a workforce puts in **beyond** its scheduled hours — and the **premium pay**
that comes with it. It is a core **operational cost** in airlines, logistics, transit, and any **shift-based**
operation, and a number planners work hard to keep down.

What drives it
--------------

Overtime spikes from **disruptions** — delays, absences, demand surges — and from **poor scheduling** or
inaccurate **demand forecasts** that leave too few people rostered. It is a **reactive** cost: you pay it when
the plan doesn't survive contact with reality.

Why it's optimized
------------------

Crew overtime is a **penalty** term in scheduling and rostering models, balanced against **understaffing**
(which causes delays and missed service) and **overstaffing** (idle labor). Better **forecasting** and robust
scheduling shrink it — a direct link between prediction quality and operating cost.
"""

CONTENT["Advanced Sorting in Spreadsheets"] = r"""
What it is
----------

**Advanced sorting** in spreadsheets means ordering data by more than a single column A-to-Z — **multi-level**
sorts that break ties across several keys, **custom** orders, and sorts by **color** or **format**. It goes
well beyond the one-click sort button.

How it works
------------

The core tool is a **multi-key** sort — order by one column, then by a second **within** ties, then a third —
plus **custom lists** (sorting Low, Medium, High in **logical** rather than alphabetical order), case-sensitive
sorting, and sorting **left-to-right** by rows. Excel's Sort dialog and Google Sheets both expose these, and
functions like **SORT / SORTBY** do it dynamically.

Why it matters
--------------

Sorting is a foundational step in **exploring** and **preparing** tabular data — grouping records, surfacing
extremes, and readying data for analysis. The key **caution**: always extend the sort to **all related
columns**, or you'll shuffle one field out of alignment with the rest and silently **corrupt** the rows.
"""

MINDMAP.update({
    "Underflow": [
        "Logits", "Softmax Function",
        "Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)", "Log-Odds",
        "Quantization", "Sigmoid Function",
    ],
    "Crew Overtime": [
        "Manual review minutes", "Compute budgets", "Overstock %", "Stockouts",
        "Time Series Forecasting", "Inference Cost (Inference $)",
    ],
    "Advanced Sorting in Spreadsheets": [
        "Encode (in Feature Engineering)", "Normalize (in Feature Engineering)",
        "Sensitivity in Feature Engineering", "Outlier", "Correlation", "Median",
    ],
})
