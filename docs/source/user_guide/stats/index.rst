.. _stats-index:

Stats (experimental)
======================================================================

Examples relevant to the :py:mod:`~.stats` module.

.. _astrostatistics-index:

Astrostatistics Tools (experimental)
----------------------------------------------------------------------

Documentation is available in the docstrings and
online at https://www.astropy.org/.


.. _tweedie-dist-index:

Tweedie Distribution (generalized family)
----------------------------------------------------------------------

Examples relevant to the :py:obj:`~.tweedie` variable.

Tweedie distributions are a very general family of distributions
that includes the Gaussian, Poisson, and Gamma (among many others)
as special cases.

Parameter 𝑝: The Tweedie family is defined by a parameter 𝑝,
which controls the distribution's properties. For different values of 𝑝,
the distribution can take different forms.

Example of the Tweedie Family:

* 𝑝=0: Gaussian (Normal) distribution
* 𝑝=1: Poisson distribution
* 𝑝=2: Gamma distribution
* 𝑝=3: Inverse Gaussian distribution

Common Applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generalized Linear Models (GLMs): The Tweedie distributions are often used in GLMs, especially in cases where the data has non-constant variance or is overdispersed (i.e., variance greater than the mean).

The Tweedie distribution is widely used in:

* Insurance industry: Modeling claim amounts and policy exposure.
* Medical and genomic testing: Analyzing datasets with zero-inflated and continuous positive values.
* Environmental science: Rainfall modeling and hydrology studies.

.. seealso::

   * https://github.com/thequackdaddy/tweedie
   * https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.Tweedie.html
   * https://glum.readthedocs.io/en/latest/glm.html#glum.TweedieDistribution
   * https://glum.readthedocs.io/en/latest/glm.html#glum.TweedieDistribution.log_likelihood
