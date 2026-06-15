"""Probability scale implementation for matplotlib Axes.

This module provides :class:`ProbScale`, a matplotlib ``ScaleBase`` subclass
that transforms an axis to display data on a probability (or percentage)
scale, and :class:`_minimal_norm`, a scipy-free fallback normal distribution
used by default when scipy is not installed.

Notes
-----
**Matplotlib compatibility (≥ 3.11)**

Starting with matplotlib 3.11, :func:`matplotlib.scale.register_scale`
inspects ``scale_class.__init__`` via :func:`inspect.signature`.  If the
string ``"axis"`` appears anywhere in the parameter list — even with a
default value — it emits a ``PendingDeprecationWarning``
(:class:`~matplotlib.MatplotlibDeprecationWarning`) and will become a hard
error in a future release.  The dispatch in ``scale_factory`` is controlled
by the same flag::

    # mpl 3.11 scale_factory pseudocode
    if "axis" in inspect.signature(scale_cls).parameters:
        return scale_cls(axis, **kwargs)   # old path → warned
    else:
        return scale_cls(**kwargs)          # new path → clean

Using ``*args`` removes the name ``"axis"`` from the inspected signature
while still accepting the positional argument that pre-3.11 ``scale_factory``
passes unconditionally.  This is the only strategy that is simultaneously
clean under 3.11+ inspection *and* backward-compatible with ≤ 3.10 call
conventions.

**Strategy comparison**

+-------------------------------------------+-----------------+-----------------+
| Signature                                 | mpl ≤ 3.10 compat | mpl ≥ 3.11 clean |
+===========================================+=================+=================+
| ``__init__(self, axis, **kwargs)``        | ✓               | ✗ warns         |
+-------------------------------------------+-----------------+-----------------+
| ``__init__(self, axis=None, **kwargs)``   | ✓               | ✗ warns         |
+-------------------------------------------+-----------------+-----------------+
| ``__init__(self, *args, **kwargs)``       | ✓               | ✓               |
+-------------------------------------------+-----------------+-----------------+

The ``axis`` positional argument has **never** been used inside
``ProbScale.__init__``; only ``dist``, ``as_pct``, and ``nonpos`` are
consumed from ``kwargs``.  Dropping the name therefore has zero functional
impact.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy
from matplotlib.scale import ScaleBase
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    NullFormatter,
    NullLocator,
)

from .formatters import PctFormatter, ProbFormatter
from .transforms import ProbTransform

__all__ = [
    "_minimal_norm",
    "ProbScale",
]


# ---------------------------------------------------------------------------
# _minimal_norm
# ---------------------------------------------------------------------------

class _minimal_norm:
    """Minimal normal-distribution implementation compatible with
    ``scipy.stats.norm``.

    Provides ``ppf`` and ``cdf`` class methods so that scipy is *not* a
    hard dependency of the probability scale.  The approximation is based
    on Abramowitz & Stegun (1972) and has a maximum absolute error of
    approximately 1.5 × 10⁻⁷.

    Attributes
    ----------
    _A : float
        Rational-approximation constant derived from (Abramowitz & Stegun,
        eq. 7.1.26).  Value: ``−(8·(π−3)) / (3·π·(π−4))``.

    Notes
    -----
    **Developer note** — This class uses *only* class methods; it is never
    instantiated with state.  All methods are ``@classmethod`` so that the
    class itself can be passed as the ``dist`` argument to
    :class:`ProbScale` and :class:`~.transforms.ProbTransform` without
    requiring ``_minimal_norm()`` instantiation.

    References
    ----------
    .. [1] Abramowitz, M. & Stegun, I. A. (1972).
       *Handbook of Mathematical Functions*, 10th ed., §7.1.26.
       Dover Publications.
    .. [2] Wikipedia, "Error function",
       https://en.wikipedia.org/wiki/Error_function
    .. [3] Wikipedia, "Normal distribution",
       https://en.wikipedia.org/wiki/Normal_distribution

    Examples
    --------
    >>> from scikitplot.externals._probscale.probscale import _minimal_norm
    >>> import numpy as np
    >>> q = np.array([0.1, 0.5, 0.9])
    >>> _minimal_norm.ppf(q)   # doctest: +SKIP
    array([-1.282...,  0.   ,  1.282...])
    >>> _minimal_norm.cdf(np.array([-1.0, 0.0, 1.0]))   # doctest: +SKIP
    array([0.158..., 0.5  , 0.841...])
    """

    _A: float = -(8.0 * (numpy.pi - 3.0) / (3.0 * numpy.pi * (numpy.pi - 4.0)))

    @classmethod
    def _approx_erf(cls, x: numpy.ndarray) -> numpy.ndarray:
        """Approximate the error function.

        Uses the Abramowitz & Stegun rational approximation (eq. 7.1.26)
        with a maximum absolute error of 1.5 × 10⁻⁷.

        Parameters
        ----------
        x : array-like
            Input values.

        Returns
        -------
        erf_x : numpy.ndarray
            Approximate ``erf(x)``.

        Notes
        -----
        The ``numpy.sign`` operation on ``nan`` produces a
        ``RuntimeWarning: invalid value encountered in sign``; the warning
        is suppressed locally because a ``nan`` input correctly propagates
        through :func:`numpy.sign` and :func:`numpy.sqrt` to yield ``nan``
        output.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Error_function
        """
        guts = (
            -(x ** 2)
            * (4.0 / numpy.pi + cls._A * x ** 2)
            / (1.0 + cls._A * x ** 2)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in sign"
            )
            return numpy.sign(x) * numpy.sqrt(1.0 - numpy.exp(guts))

    @classmethod
    def _approx_inv_erf(cls, z: numpy.ndarray) -> numpy.ndarray:
        """Approximate the inverse error function.

        Parameters
        ----------
        z : array-like
            Input values in the range (−1, 1).

        Returns
        -------
        inv_erf_z : numpy.ndarray
            Approximate ``erfinv(z)``.

        Notes
        -----
        The same ``numpy.sign`` ``nan`` warning suppression as
        :meth:`_approx_erf` applies here.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Error_function
        """
        _b = (2.0 / numpy.pi / cls._A) + (0.5 * numpy.log(1.0 - z ** 2))
        _c = numpy.log(1.0 - z ** 2) / cls._A
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in sign"
            )
            return numpy.sign(z) * numpy.sqrt(numpy.sqrt(_b ** 2 - _c) - _b)

    @classmethod
    def ppf(cls, q: numpy.ndarray) -> numpy.ndarray:
        """Percent-point function (quantile function / inverse CDF).

        Parameters
        ----------
        q : array-like
            Probabilities in (0, 1).

        Returns
        -------
        z : numpy.ndarray
            Quantiles of the standard normal distribution corresponding
            to the given probabilities.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Normal_distribution
        """
        return numpy.sqrt(2.0) * cls._approx_inv_erf(2.0 * q - 1.0)

    @classmethod
    def cdf(cls, x: numpy.ndarray) -> numpy.ndarray:
        """Cumulative distribution function of the standard normal.

        Parameters
        ----------
        x : array-like
            Quantile values.

        Returns
        -------
        p : numpy.ndarray
            Cumulative probabilities in (0, 1).

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Normal_distribution
        """
        return 0.5 * (1.0 + cls._approx_erf(x / numpy.sqrt(2.0)))


# ---------------------------------------------------------------------------
# ProbScale
# ---------------------------------------------------------------------------

class ProbScale(ScaleBase):
    """A probability scale for matplotlib Axes.

    Transforms an axis so that normally distributed data plot as a straight
    line.  Tick locations correspond to probability (or percentage) values;
    the underlying quantile mapping is supplied by *dist*.

    Parameters
    ----------
    *args : positional arguments (absorbed, not used)
        Prior to matplotlib 3.11 ``scale_factory`` unconditionally passed
        the ``axis`` artist as the first positional argument to every scale
        constructor.  From 3.11 onwards, the positional argument is only
        passed when ``"axis"`` appears in ``inspect.signature(cls)``.  Using
        ``*args`` absorbs the positional argument on older matplotlib
        versions without placing the string ``"axis"`` in the inspected
        signature, which would trigger a ``PendingDeprecationWarning``.
    dist : scipy.stats-compatible distribution, optional
        Object that provides ``ppf(q)`` (quantile function) and ``cdf(x)``
        (CDF) class or instance methods.  Defaults to
        :class:`_minimal_norm` so that scipy is not a hard dependency.
    as_pct : bool, optional
        When ``True`` (default) tick labels are shown as percentages
        (0 – 100).  When ``False`` they are shown as proportions (0 – 1).
    nonpos : {'mask', 'clip'}, optional
        Strategy for data outside the valid probability range; forwarded to
        :class:`~.transforms.ProbTransform`.  Default is ``'mask'``.

    Attributes
    ----------
    name : str
        The string name used to register this scale with matplotlib:
        ``'prob'``.
    dist : distribution object
        The distribution used for the probability mapping.
    as_pct : bool
        Whether tick labels are rendered as percentages.
    nonpos : str
        Out-of-bounds handling strategy (``'mask'`` or ``'clip'``).

    Notes
    -----
    **User note** — Register and use with::

        import scikitplot.externals._probscale  # registers 'prob' scale
        fig, ax = pyplot.subplots()
        ax.set_yscale('prob')
        ax.set_ylim(0.5, 99.5)

    **Developer note** — ``*args`` is intentional; see the module-level
    docstring for the complete compatibility matrix and rationale.  The
    ``axis`` positional argument is *never* read inside this constructor;
    all configuration is driven by keyword arguments.

    See Also
    --------
    matplotlib.scale.ScaleBase : Abstract base class for all scales.
    .transforms.ProbTransform : The underlying matplotlib Transform.
    _minimal_norm : Default distribution (scipy-free).

    References
    ----------
    .. [1] matplotlib scales documentation:
       https://matplotlib.org/stable/gallery/scales/scales.html
    .. [2] mpl-probscale upstream:
       https://github.com/matplotlib/mpl-probscale

    Examples
    --------
    Basic probability scale (percentage labels):

    .. plot::
      :include-source:
      :align: center
      :context: close-figs
      :alt: ProbScale percentage axis

      >>> import scikitplot.externals._probscale  # registers 'prob' scale
      >>> from matplotlib import pyplot
      >>> fig, ax = pyplot.subplots(figsize=(4, 7))
      >>> ax.set_ylim(bottom=0.5, top=99.5)
      >>> ax.set_yscale('prob')

    Proportion labels and a custom distribution:

    .. plot::
      :include-source:
      :align: center
      :context: close-figs
      :alt: ProbScale proportion axis

      >>> from scipy import stats
      >>> fig, ax = pyplot.subplots(figsize=(4, 7))
      >>> ax.set_ylim(bottom=0.01, top=0.99)
      >>> ax.set_yscale('prob', as_pct=False, dist=stats.norm)
    """

    name: str = "prob"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # *args absorbs the positional ``axis`` argument that matplotlib
        # ≤ 3.10 ``scale_factory`` passes unconditionally.  The value is
        # intentionally never read: only dist, as_pct, and nonpos matter.
        self.dist = kwargs.pop("dist", _minimal_norm)
        self.as_pct = kwargs.pop("as_pct", True)
        self.nonpos = kwargs.pop("nonpos", "mask")
        self._transform = ProbTransform(self.dist, as_pct=self.as_pct)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_probs(cls, nobs: float, as_pct: bool) -> numpy.ndarray:
        """Compute probability tick locations for *nobs* observations.

        Parameters
        ----------
        nobs : float
            Number of observations (controls how many tail ticks are
            generated).  A value of ``1e8`` produces a full set of ticks
            covering the entire probability scale.
        as_pct : bool
            When ``True`` returned values are percentages (0 – 100);
            otherwise proportions (0 – 1).

        Returns
        -------
        locs : numpy.ndarray
            1-D array of probability values suitable for
            :class:`~matplotlib.ticker.FixedLocator`.

        Notes
        -----
        The factor is ``1.0`` for percentages (no scaling needed) and
        ``100.0`` for proportions (divide percentages by 100).  This
        avoids the Python-2-era ``A and B or C`` boolean idiom — see
        :meth:`limit_range_for_scale` for the same correction applied
        there.
        """
        factor = 1.0 if as_pct else 100.0

        order = int(numpy.floor(numpy.log10(nobs)))
        base_probs = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)

        axis_probs = base_probs.copy()
        for n in range(order):
            if n <= 2:
                lower_fringe = numpy.array([1, 2, 5], dtype=float)
                upper_fringe = numpy.array([5, 8, 9], dtype=float)
            else:
                lower_fringe = numpy.array([1], dtype=float)
                upper_fringe = numpy.array([9], dtype=float)

            new_lower = lower_fringe / 10 ** n
            new_upper = upper_fringe / 10 ** n + axis_probs.max()
            axis_probs = numpy.hstack([new_lower, axis_probs, new_upper])

        return axis_probs / factor

    # ------------------------------------------------------------------
    # ScaleBase interface
    # ------------------------------------------------------------------

    def set_default_locators_and_formatters(self, axis: Any) -> None:
        """Configure probability-scale locators and formatters.

        Sets a :class:`~matplotlib.ticker.FixedLocator` at standard
        probability tick positions, a :class:`~matplotlib.ticker.FuncFormatter`
        that renders probabilities as percentages or proportions, and
        ``NullLocator`` / ``NullFormatter`` for minor ticks.

        Parameters
        ----------
        axis : matplotlib.axis.Axis
            The axis whose locators and formatters will be configured.
        """
        axis.set_major_locator(
            FixedLocator(self._get_probs(1e8, self.as_pct))
        )
        if self.as_pct:
            axis.set_major_formatter(FuncFormatter(PctFormatter()))
        else:
            axis.set_major_formatter(FuncFormatter(ProbFormatter()))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self) -> ProbTransform:
        """Return the probability transform for this scale.

        Returns
        -------
        transform : .transforms.ProbTransform
            The :class:`~matplotlib.transforms.Transform` instance that
            maps probability / percentage values to quantile space (and
            vice versa via its :meth:`~.transforms.ProbTransform.inverted`
            twin :class:`~.transforms.QuantileTransform`).
        """
        return self._transform

    def limit_range_for_scale(
        self,
        vmin: float,
        vmax: float,
        minpos: float,
    ) -> tuple[float, float]:
        """Clamp axis limits to positive probability values.

        Any limit at or below zero is replaced by *minpos*, the smallest
        strictly-positive value on the axis, so that the probability
        transform (which is undefined at 0 and 1) always receives a
        valid input.

        Parameters
        ----------
        vmin : float
            Current lower axis limit.
        vmax : float
            Current upper axis limit.
        minpos : float
            Smallest strictly-positive data value visible on the axis;
            supplied by matplotlib.

        Returns
        -------
        vmin_new : float
            Clamped lower limit.
        vmax_new : float
            Clamped upper limit.

        Notes
        -----
        The original implementation used the Python-2-era idiom
        ``vmin <= 0.0 and minpos or vmin``.  This is equivalent to
        ``minpos if vmin <= 0.0 else vmin`` **only** when *minpos* is
        truthy.  If *minpos* were ``0`` or ``None`` the idiom would
        silently return *vmin* (the wrong branch).  The explicit ternary
        form used here is always correct regardless of the truthiness of
        *minpos*.
        """
        return (
            minpos if vmin <= 0.0 else vmin,
            minpos if vmax <= 0.0 else vmax,
        )
