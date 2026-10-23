# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic inputs and matplotlib set-up shared by the benchmarks."""

from __future__ import annotations

import os

# Before pyplot is imported anywhere: no display is needed or used.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

#: Seed of every generated input; the same inputs on every machine.
SEED = 0


def classification(n_samples, n_classes):
    """
    Return labels and predicted probabilities for ``n_classes`` classes.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_classes : int
        Number of classes, at least 2.

    Returns
    -------
    y_true : numpy.ndarray of shape (n_samples,)
        Every class occurs at least once.
    y_probas : numpy.ndarray of shape (n_samples, n_classes)
        Rows sum to 1.
    """
    rng = np.random.RandomState(SEED)
    y_true = np.arange(n_samples) % n_classes
    rng.shuffle(y_true)
    y_probas = rng.dirichlet(np.ones(n_classes), size=n_samples)
    return y_true, y_probas


def blobs(n_samples, n_features, n_centers):
    """Return ``n_samples`` points around ``n_centers`` centres, and their labels."""
    rng = np.random.RandomState(SEED)
    centers = rng.uniform(-10, 10, size=(n_centers, n_features))
    labels = np.arange(n_samples) % n_centers
    points = centers[labels] + rng.normal(size=(n_samples, n_features))
    return points, labels


def vectors(n_items, n_dims):
    """Return ``n_items`` random vectors of ``n_dims`` float32 values."""
    return np.random.RandomState(SEED).rand(n_items, n_dims).astype(np.float32)


def close_figures():
    """Close every figure, so that one benchmark's figures do not slow the next."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close("all")
