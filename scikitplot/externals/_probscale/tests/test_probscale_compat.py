"""Regression tests for matplotlib-version-independent compatibility.

This module covers the five coverage gaps identified during the
scikitplot/externals/_probscale code review:

1. ``register_scale`` emits zero ``PendingDeprecationWarning`` on mpl ≥ 3.11.
2. ``"axis"`` is absent from ``inspect.signature(ProbScale).parameters``.
3. ``ProbScale.__init__`` absorbs a positional axis argument (mpl ≤ 3.10 compat).
4. ``limit_range_for_scale`` correctly handles ``vmin ≤ 0`` (the Python-2
   boolean-idiom bug: ``A and B or C`` silently returns ``C`` when ``B`` is
   falsy, so the explicit ternary ``(B if A else C)`` is required).
5. ``ProbScale._get_probs`` returns values strictly inside the expected range.

Notes
-----
**Why these tests matter:**

``register_scale`` in matplotlib 3.11 gates on
``"axis" in inspect.signature(scale_cls).parameters``.  If the string is
present — even with a default value (``axis=None``) — it emits a
``PendingDeprecationWarning`` that pytest converts to an error under
``filterwarnings = error``.  The fix (``*args``) is the only strategy that
removes the name from the inspected signature while remaining callable from
pre-3.11 ``scale_factory`` which passes ``axis`` as a positional argument
unconditionally.

The boolean-idiom test (``limit_range_for_scale``) guards against a
regression to the Python-2-era ``A and B or C`` pattern, which silently
returns the wrong branch when ``B`` is falsy.
"""

from __future__ import annotations

import inspect
import warnings

import matplotlib
import numpy
import numpy.testing as nptest
import pytest
from matplotlib import scale as mpl_scale
from packaging.version import Version

from ..probscale import ProbScale, _minimal_norm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MPL_VERSION = Version(matplotlib.__version__)
MPL_GE_311 = MPL_VERSION >= Version("3.11")


# ---------------------------------------------------------------------------
# 1. register_scale — no PendingDeprecationWarning (mpl ≥ 3.11)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MPL_GE_311, reason="PendingDeprecationWarning only in mpl ≥ 3.11")
def test_register_scale_no_pending_deprecation_warning():
    """``register_scale(ProbScale)`` must not emit PendingDeprecationWarning.

    On matplotlib ≥ 3.11 the function inspects the scale class's ``__init__``
    signature.  If ``"axis"`` appears in the named parameters it emits a
    ``PendingDeprecationWarning``.  The ``*args`` signature removes the name
    from ``inspect.signature`` while still accepting the positional argument
    that older ``scale_factory`` passes.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PendingDeprecationWarning)
        mpl_scale.register_scale(ProbScale)

    pdw = [w for w in caught if issubclass(w.category, PendingDeprecationWarning)]
    assert len(pdw) == 0, (
        f"register_scale(ProbScale) emitted {len(pdw)} PendingDeprecationWarning(s) "
        f"on matplotlib {matplotlib.__version__}.  Ensure ProbScale.__init__ uses "
        f"*args, not a named 'axis' parameter."
    )


# ---------------------------------------------------------------------------
# 2. axis absent from inspect.signature
# ---------------------------------------------------------------------------


def test_probscale_axis_not_in_signature():
    """The string ``"axis"`` must be absent from ``inspect.signature(ProbScale)``.

    This is the exact condition that matplotlib 3.11 ``register_scale`` gates
    on.  Both ``def __init__(self, axis, ...)`` and
    ``def __init__(self, axis=None, ...)`` cause the warning; only ``*args``
    (or a signature with no ``axis`` name at all) is safe.
    """
    params = inspect.signature(ProbScale).parameters
    assert "axis" not in params, (
        f"'axis' found in ProbScale.__init__ signature: {list(params)}.  "
        f"Use *args to avoid PendingDeprecationWarning in mpl ≥ 3.11."
    )


# ---------------------------------------------------------------------------
# 3. backward compat — positional axis argument is absorbed silently
# ---------------------------------------------------------------------------


class _FakeAxis:
    """Minimal stand-in for ``matplotlib.axis.Axis`` for constructor tests."""


def test_probscale_init_absorbs_positional_axis_arg():
    """``ProbScale(fake_axis)`` must not raise even though ``axis`` is not named.

    On matplotlib ≤ 3.10 ``scale_factory`` calls
    ``scale_cls(axis, **kwargs)`` unconditionally.  The ``*args`` signature
    absorbs the positional argument without placing ``"axis"`` in the named
    parameter list that mpl 3.11 inspects.
    """
    fake_axis = _FakeAxis()
    scale = ProbScale(fake_axis)  # positional — must not raise
    assert isinstance(scale, ProbScale)
    assert scale.dist is _minimal_norm
    assert scale.as_pct is True
    assert scale.nonpos == "mask"


def test_probscale_init_absorbs_positional_axis_with_kwargs():
    """Positional axis + keyword arguments are absorbed and consumed correctly."""
    fake_axis = _FakeAxis()
    scale = ProbScale(fake_axis, as_pct=False, nonpos="clip")
    assert scale.as_pct is False
    assert scale.nonpos == "clip"


def test_probscale_init_no_positional_arg():
    """``ProbScale()`` (no positional arg — mpl 3.11 path) must also work."""
    scale = ProbScale()
    assert isinstance(scale, ProbScale)
    assert scale.dist is _minimal_norm


# ---------------------------------------------------------------------------
# 4. limit_range_for_scale — explicit ternary, not Python-2 boolean idiom
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vmin", "vmax", "minpos", "expected_vmin", "expected_vmax"),
    [
        # vmin < 0 → replaced by minpos; vmax positive → kept
        (-1.0, 99.0, 0.1, 0.1, 99.0),
        # vmin == 0 → replaced by minpos (boundary: 0 is not strictly positive)
        (0.0, 99.0, 0.1, 0.1, 99.0),
        # vmin > 0 → kept unchanged
        (1.0, 99.0, 0.1, 1.0, 99.0),
        # vmax <= 0 → replaced; vmin positive → kept
        (1.0, -5.0, 0.1, 1.0, 0.1),
        # both limits non-positive → both replaced
        (-5.0, -1.0, 0.1, 0.1, 0.1),
        # typical usage on the probability scale
        (0.5, 99.5, 1e-7, 0.5, 99.5),
    ],
    ids=[
        "vmin_negative",
        "vmin_zero",
        "vmin_positive",
        "vmax_nonpositive",
        "both_nonpositive",
        "typical_prob_scale",
    ],
)
def test_limit_range_for_scale(vmin, vmax, minpos, expected_vmin, expected_vmax):
    """``limit_range_for_scale`` must return the correct (vmin, vmax) pair.

    Regression guard against the Python-2-era idiom
    ``vmin <= 0.0 and minpos or vmin`` which silently returns ``vmin`` when
    ``minpos`` is falsy (0 or None).  The explicit ternary
    ``(minpos if vmin <= 0.0 else vmin)`` is always correct.
    """
    scale = ProbScale()
    result_vmin, result_vmax = scale.limit_range_for_scale(vmin, vmax, minpos)
    assert result_vmin == expected_vmin, (
        f"vmin: got {result_vmin!r}, expected {expected_vmin!r}"
    )
    assert result_vmax == expected_vmax, (
        f"vmax: got {result_vmax!r}, expected {expected_vmax!r}"
    )


# ---------------------------------------------------------------------------
# 5. _get_probs — shape and strict-range validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nobs", "as_pct", "lo", "hi"),
    [
        (1e8, True,  0.0,   100.0),
        (1e8, False, 0.0,   1.0),
        (1e4, True,  0.0,   100.0),
        (1e4, False, 0.0,   1.0),
        (10,  True,  0.0,   100.0),
        (10,  False, 0.0,   1.0),
    ],
    ids=[
        "1e8_pct", "1e8_prop",
        "1e4_pct", "1e4_prop",
        "10_pct",  "10_prop",
    ],
)
def test_get_probs_values_strictly_in_range(nobs, as_pct, lo, hi):
    """``_get_probs`` must return values strictly inside the open interval (lo, hi).

    Tick locations must never be ≤ 0 or ≥ the scale maximum (100 % or 1.0)
    because the probability transform is undefined at those boundaries.
    """
    probs = ProbScale._get_probs(nobs, as_pct)
    assert isinstance(probs, numpy.ndarray)
    assert probs.ndim == 1
    assert len(probs) > 0, "_get_probs returned an empty array"
    assert (probs > lo).all(), (
        f"_get_probs({nobs}, as_pct={as_pct}): found values ≤ {lo}: "
        f"{probs[probs <= lo]}"
    )
    assert (probs < hi).all(), (
        f"_get_probs({nobs}, as_pct={as_pct}): found values ≥ {hi}: "
        f"{probs[probs >= hi]}"
    )


def test_get_probs_as_pct_and_prop_are_consistent():
    """``as_pct=True`` and ``as_pct=False`` must produce the same locations scaled.

    The percentage version should equal the proportion version multiplied by 100
    (to within floating-point rounding).
    """
    pct = ProbScale._get_probs(1e8, True)
    prop = ProbScale._get_probs(1e8, False)
    nptest.assert_allclose(pct, prop * 100.0, rtol=1e-12)


def test_get_probs_sorted():
    """``_get_probs`` output must be monotonically increasing."""
    probs = ProbScale._get_probs(1e8, True)
    assert numpy.all(numpy.diff(probs) > 0), (
        "_get_probs returned non-monotonic tick locations"
    )
