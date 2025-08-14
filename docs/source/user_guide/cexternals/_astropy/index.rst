.. _astropy-index:

======================================================================
Astropy (experimental)
======================================================================

This module contains some functions related to :py:mod:`~._astropy` under :py:mod:`~.stats`.

.. seealso::

   * https://github.com/astropy/astropy
   * https://docs.astropy.org/en/stable/visualization/histogram.html
   * https://docs.astropy.org/en/stable/stats/ref_api.html#module-astropy.stats

Documentation is available in the docstrings and
online at https://www.astropy.org/.

.. important::

    Module also available under :py:mod:`~.stats` library.
    User guide for more :ref:`astrostatistics-index`.

.. .. code-block:: python

.. prompt:: bash >>> auto

   # (recommended)
   from astropy import stats

   # (optionally) scikit-plots stats also include astropy stats
   import scikitplot.stats as stats
   from scikitplot import stats

   # (optionally) Only astropy stats
   import scikitplot.cexternals._astropy.stats as stats
   from scikitplot.cexternals._astropy import stats
