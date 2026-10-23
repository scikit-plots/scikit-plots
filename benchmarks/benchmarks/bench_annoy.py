# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Approximate nearest neighbours: building an index and querying it."""

from __future__ import annotations

from typing import ClassVar

from .common import vectors


class AnnoyIndexBuild:
    """Adding items and building the forest."""

    params: ClassVar[tuple[list[int], ...]] = ([10_000], [32], [10])
    param_names: ClassVar[list[str]] = ["n_items", "n_dims", "n_trees"]
    timeout = 300

    def setup(self, n_items, n_dims, n_trees):
        """Generate vectors used to build the index."""
        self.data = vectors(n_items, n_dims)

    def time_add_and_build(self, n_items, n_dims, n_trees):
        """Add vectors to an index and build its forest."""
        from scikitplot.annoy import AnnoyIndex  # noqa: PLC0415

        index = AnnoyIndex(n_dims, "angular")
        for i, vector in enumerate(self.data):
            index.add_item(i, vector)
        index.build(n_trees)


class AnnoyIndexQuery:
    """Nearest neighbours of stored vectors in a built index."""

    params: ClassVar[tuple[list[int], ...]] = ([10_000], [32], [10, 100])
    param_names: ClassVar[list[str]] = ["n_items", "n_dims", "n_neighbours"]
    timeout = 300

    def setup(self, n_items, n_dims, n_neighbours):
        """Generate vectors and build the index used for queries."""
        from scikitplot.annoy import AnnoyIndex  # noqa: PLC0415

        self.data = vectors(n_items, n_dims)
        self.index = AnnoyIndex(n_dims, "angular")
        for i, vector in enumerate(self.data):
            self.index.add_item(i, vector)
        self.index.build(10)

    def time_query_100(self, n_items, n_dims, n_neighbours):
        """Query the index for neighbours of the first 100 vectors."""
        for vector in self.data[:100]:
            self.index.get_nns_by_vector(vector, n_neighbours)

    def track_index_items(self, n_items, n_dims, n_neighbours):
        """Return the number of items stored in the index."""
        return self.index.get_n_items()
