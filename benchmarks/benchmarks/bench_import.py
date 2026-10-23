# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Import time: what every user pays first, measured in a fresh interpreter."""

from __future__ import annotations


class Import:
    """Import of the package and of its plotting API, each in a new process."""

    timeout = 300

    def timeraw_import_scikitplot(self):
        """Return the statement used to benchmark importing scikitplot."""
        return "import scikitplot"

    def timeraw_import_scikitplot_api(self):
        """Return the statement used to benchmark importing scikitplot.api."""
        return "import scikitplot.api"
