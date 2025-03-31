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

   # (optionally) Only astropy stats
   import scikitplot._astropy.stats as stats
   from scikitplot._astropy import stats

   # (recommended) scikit-plots stats also include astropy stats
   from astropy import stats
   from scikitplot import stats
   import scikitplot.stats as stats
