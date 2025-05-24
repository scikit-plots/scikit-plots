"""
Configuration module for the package.

This module consolidates configuration-related components such as:
- `show`: Configuration display utility.
- `_citation`: BibTeX and citation metadata.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from .__config__ import show_config as show_config
from ._citation import __bibtex__ as __bibtex__
from ._citation import __citation__ as __citation__
