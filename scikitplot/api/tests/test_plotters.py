import unittest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris as load_data
from sklearn.decomposition import PCA

import scikitplot as sp


class TestPlotPCAComponentVariance(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]
        self.clf_not_fitted = PCA()
        self.clf = PCA().fit(self.X)

    def tearDown(self):
        plt.close("all")

    def test_fitted(self):
        self.assertRaises(
            TypeError, sp.api.plotters.plot_pca_component_variance, self.clf_not_fitted
        )

    def test_target_explained_variance(self):
        ax = sp.api.plotters.plot_pca_component_variance(
            self.clf, target_explained_variance=0
        )
        ax = sp.api.plotters.plot_pca_component_variance(
            self.clf, target_explained_variance=0.5
        )
        ax = sp.api.plotters.plot_pca_component_variance(
            self.clf, target_explained_variance=1
        )
        ax = sp.api.plotters.plot_pca_component_variance(
            self.clf, target_explained_variance=1.5
        )

    def test_ax(self):
        fig, ax = plt.subplots(1, 1)
        out_ax = sp.api.plotters.plot_pca_component_variance(self.clf)
        assert ax is not out_ax
        out_ax = sp.api.plotters.plot_pca_component_variance(self.clf, ax=ax)
        assert ax is out_ax


class TestPlotPCA2DProjection(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]
        self.clf_not_fitted = PCA()
        self.clf = PCA().fit(self.X)

    def tearDown(self):
        plt.close("all")

    def test_ax(self):
        fig, ax = plt.subplots(1, 1)
        out_ax = sp.api.plotters.plot_pca_2d_projection(self.clf, self.X, self.y)
        assert ax is not out_ax
        out_ax = sp.api.plotters.plot_pca_2d_projection(self.clf, self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_cmap(self):
        fig, ax = plt.subplots(1, 1)
        ax = sp.api.plotters.plot_pca_2d_projection(
            self.clf, self.X, self.y, cmap="Spectral"
        )
        ax = sp.api.plotters.plot_pca_2d_projection(
            self.clf, self.X, self.y, cmap=plt.cm.Spectral
        )


class TestValidateLabels(unittest.TestCase):
    def test_valid_equal(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "C"]
        arg_name = "true_labels"

        actual = sp.api._utils._helpers.validate_labels(
            known_labels, passed_labels, arg_name
        )
        self.assertEqual(actual, None)

    def test_valid_subset(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B"]
        arg_name = "true_labels"

        actual = sp.api._utils._helpers.validate_labels(
            known_labels, passed_labels, arg_name
        )
        self.assertEqual(actual, None)

    def test_invalid_one_duplicate(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "B"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            sp.api._utils._helpers.validate_labels(
                known_labels, passed_labels, arg_name
            )

        msg = "The following duplicate labels were passed into true_labels: B"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_two_duplicates(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "A", "B"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            sp.api._utils._helpers.validate_labels(
                known_labels, passed_labels, arg_name
            )

        msg = "The following duplicate labels were passed into true_labels: A, B"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_one_missing(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "D"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            sp.api._utils._helpers.validate_labels(
                known_labels, passed_labels, arg_name
            )

        msg = "The following labels were passed into true_labels, but were not found in labels: D"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_two_missing(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "E", "B", "D"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            sp.api._utils._helpers.validate_labels(
                known_labels, passed_labels, arg_name
            )

        msg = "The following labels were passed into true_labels, but were not found in labels: E, D"
        self.assertEqual(msg, str(context.exception))

    def test_numerical_labels(self):
        known_labels = [0, 1, 2]
        passed_labels = [0, 2]
        arg_name = "true_labels"

        actual = sp.api._utils._helpers.validate_labels(
            known_labels, passed_labels, arg_name
        )
        self.assertEqual(actual, None)

    def test_invalid_duplicate_numerical_labels(self):
        known_labels = [0, 1, 2]
        passed_labels = [0, 2, 2]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            sp.api._utils._helpers.validate_labels(
                known_labels, passed_labels, arg_name
            )

        msg = "The following duplicate labels were passed into true_labels: 2"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_missing_numerical_labels(self):
        known_labels = [0, 1, 2]
        passed_labels = [0, 2, 3]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            sp.api._utils._helpers.validate_labels(
                known_labels, passed_labels, arg_name
            )

        msg = "The following labels were passed into true_labels, but were not found in labels: 3"
        self.assertEqual(msg, str(context.exception))


if __name__ == "__main__":
    unittest.main()
