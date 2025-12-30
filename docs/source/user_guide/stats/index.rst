..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # https://documatt.com/restructuredtext-reference/element/admonition.html
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.stats

.. _stats-index:

======================================================================
Stats
======================================================================

Examples relevant to the :py:mod:`~scikitplot.stats` module.

.. TODO: Describe Astrostatistics Tools
.. TODO: Describe Tweedie Distribution

.. _astrostatistics-index:

Astrostatistics Tools (experimental)
----------------------------------------------------------------------

Examples relevant to the :py:mod:`~scikitplot.cexternals._astropy.stats` module.

Documentation is available in the docstrings and
online at https://www.astropy.org/.

.. _tweedie-dist-index:

Tweedie Distribution (generalized family)
----------------------------------------------------------------------

Examples relevant to the :py:mod:`~scikitplot.externals._tweedie` module.

Tweedie distributions are a very general family of distributions
that includes the Gaussian, Poisson, and Gamma (among many others)
as special cases.

Parameter 𝑝: The Tweedie family is defined by a parameter 𝑝,
which controls the distribution's properties. For different values of 𝑝,
the distribution can take different forms.

Example of the Tweedie Family :py:obj:`~.tweedie` or :py:obj:`~.tweedie_gen` :

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
