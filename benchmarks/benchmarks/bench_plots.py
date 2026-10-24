# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""The plotting API on generated inputs of growing size."""

from __future__ import annotations

from typing import ClassVar

from .common import blobs, classification, close_figures


class ClassificationPlots:
    """Curves and matrices drawn from labels and predicted probabilities."""

    params = ([1_000, 20_000], [2, 10])
    param_names: ClassVar[list[str]] = ["n_samples", "n_classes"]
    timeout = 300

    def setup(self, n_samples, n_classes):
        """Prepare classification data for plotting benchmarks."""
        import scikitplot.api  # noqa: PLC0415

        self.api = scikitplot.api
        self.y_true, self.y_probas = classification(n_samples, n_classes)
        self.y_pred = self.y_probas.argmax(axis=1)

    def teardown(self, n_samples, n_classes):
        """Close figures created during the benchmark."""
        close_figures()

    def time_plot_roc(self, n_samples, n_classes):
        """Benchmark plotting the ROC curves for the generated data."""
        self.api.plot_roc(self.y_true, self.y_probas)

    def time_plot_precision_recall(self, n_samples, n_classes):
        """Benchmark plotting precision-recall curves for the generated data."""
        self.api.plot_precision_recall(self.y_true, self.y_probas)

    def time_plot_confusion_matrix(self, n_samples, n_classes):
        """Benchmark plotting the confusion matrix for the generated data."""
        self.api.plot_confusion_matrix(self.y_true, self.y_pred)

    def peakmem_plot_roc(self, n_samples, n_classes):
        """Benchmark peak memory while plotting ROC curves for the generated data."""
        self.api.plot_roc(self.y_true, self.y_probas)


class ClusteringPlots:
    """Silhouette analysis of points around known centres."""

    params = ([500, 3_000],)
    param_names: ClassVar[list[str]] = ["n_samples"]
    timeout = 300

    def setup(self, n_samples):
        """Prepare clustering data for plotting benchmarks."""
        import scikitplot.api  # noqa: PLC0415

        self.api = scikitplot.api
        self.points, self.labels = blobs(n_samples, n_features=8, n_centers=4)

    def teardown(self, n_samples):
        """Close figures created during the benchmark."""
        close_figures()

    def time_plot_silhouette(self, n_samples):
        """Benchmark plotting silhouette analysis for the generated data."""
        self.api.plot_silhouette(self.points, self.labels)
