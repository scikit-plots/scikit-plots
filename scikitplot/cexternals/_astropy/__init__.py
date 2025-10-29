# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Astropy is a package intended to contain core functionality and some
common tools needed for performing astronomy and astrophysics research with
Python. It also provides an index for other astronomy packages and tools for
managing them.

Documentation is available in the docstrings and
online at https://www.astropy.org/.
"""
# cexternals/_astropy/__init__.py
# https://docs.astropy.org/en/latest/index.html

#######################################################################

from .version import version as __version__

# Define the astropy version
__author__ = "Astropy Developers"
__author_email__ = "astropy.team@gmail.com"
__git_hash__  = "dbc384f3eeff4576b41a68486fcbb0a77789a8d8"

__all__ = [  # noqa: RUF100, RUF022
    "__version__",
    # "__bibtex__",
    # # Subpackages (mostly lazy-loaded)
    # "config",
    # "constants",
    # "convolution",
    # "coordinates",
    # "cosmology",
    # "io",
    # "modeling",
    # "nddata",
    # "samp",
    "stats",
    # "table",
    # "tests",
    # "time",
    # "timeseries",
    # "uncertainty",
    # "units",
    # "utils",
    # "visualization",
    # "wcs",
    # # Functions
    # "test",
    # "log",
    # "find_api_page",
    # "online_help",
    # "online_docs_root",
    # "conf",
    # "physical_constants",
    # "astronomical_constants",
    # "system_info",
]


def __getattr__(attr):
    if attr in __all__:
        from importlib import import_module

        return import_module("scikitplot.cexternals._astropy." + attr)

    raise AttributeError(f"submodule '_astropy' has no attribute {attr!r}")
