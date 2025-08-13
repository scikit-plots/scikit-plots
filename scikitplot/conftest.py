# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Pytest customization conftest.py

import atexit as _atexit
import gc as _gc
import json as _json
import os as _os
import tempfile as _tempfile
import warnings as _warnings

import hypothesis as _hypothesis
import pytest as _pytest

# Set the backend of matplotlib to prevent build errors.
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.testing as _np_testing
import pandas as _pd

from ._lib import _pep440
from ._lib._array_api import SKPLT_ARRAY_API, SKPLT_DEVICE
from ._reset import reset

_mpl.use("Agg")  # Use non-interactive backend before pyplot is imported

# Auto-clean with atexit
# _atexit.register(reset)

# @pytest.fixture(autouse=True)
# def clean_modules():
#     yield
#     reset()


######################################################################
## pytest_configure
######################################################################

# Following the approach of Scipy's conftest.py...
try:
    import pytest_run_parallel as _pytest_run_parallel  # noqa:F401

    PARALLEL_RUN_AVAILABLE = True
except Exception:
    PARALLEL_RUN_AVAILABLE = False


def pytest_configure(config):
    try:
        import pytest_timeout  # noqa:F401
    except Exception:
        config.addinivalue_line(
            "markers", "timeout: mark a test for a non-default timeout"
        )
    try:
        # This is a more reliable test of whether pytest_fail_slow is installed
        # When I uninstalled it, `import pytest_fail_slow` didn't fail!
        from pytest_fail_slow import (
            parse_duration,  # type: ignore[import-not-found] # noqa: F401
        )
    except Exception:
        config.addinivalue_line(
            "markers", "fail_slow: mark a test for a non-default timeout failure"
        )

    if not PARALLEL_RUN_AVAILABLE:
        config.addinivalue_line(
            "markers",
            "parallel_threads(n): run the given test function in parallel "
            "using `n` threads.",
        )
        config.addinivalue_line(
            "markers", "thread_unsafe: mark the test function as single-threaded"
        )
        config.addinivalue_line(
            "markers",
            "iterations(n): run the given test function `n` times in each thread",
        )


######################################################################
## globally for all tests
######################################################################


def pytest_runtest_setup(item):
    # Before each test
    # Trigger garbage collection
    _gc.collect()

    mark = item.get_closest_marker("xslow")
    if mark is not None:
        try:
            v = int(_os.environ.get("SKPLT_XSLOW", "0"))
        except ValueError:
            v = False
        if not v:
            _pytest.skip(
                "very slow test; set environment variable SKPLT_XSLOW=1 to run it"
            )
    mark = item.get_closest_marker("xfail_on_32bit")
    if mark is not None and _np.intp(0).itemsize < 8:
        _pytest.xfail(f"Fails on our 32-bit test platform(s): {mark.args[0]}")

    # Older versions of threadpoolctl have an issue that may lead to this
    # warning being emitted, see gh-14441
    with _np_testing.suppress_warnings() as sup:
        sup.filter(_pytest.PytestUnraisableExceptionWarning)

        try:
            from threadpoolctl import threadpool_limits

            HAS_THREADPOOLCTL = True
        except Exception:  # observed in gh-14441: (ImportError, AttributeError)
            # Optional dependency only. All exceptions are caught, for robustness
            HAS_THREADPOOLCTL = False

        if HAS_THREADPOOLCTL:
            # Set the number of openmp threads based on the number of workers
            # xdist is using to prevent oversubscription. Simplified version of what
            # sklearn does (it can rely on threadpoolctl and its builtin OpenMP helper
            # functions)
            try:
                xdist_worker_count = int(_os.environ["PYTEST_XDIST_WORKER_COUNT"])
            except KeyError:
                # raises when pytest-xdist is not installed
                return

            if not _os.getenv("OMP_NUM_THREADS"):
                max_openmp_threads = _os.cpu_count() // 2  # use nr of physical cores
                threads_per_worker = max(max_openmp_threads // xdist_worker_count, 1)
                try:
                    threadpool_limits(threads_per_worker, user_api="blas")
                except Exception:
                    # May raise AttributeError for older versions of OpenBLAS.
                    # Catch any error for robustness.
                    return


def pytest_runtest_teardown(item, nextitem):
    # After each test
    # Trigger garbage collection
    _gc.collect()


######################################################################
## pytest: run_gc
######################################################################

# @_pytest.fixture(autouse=True)
# def run_gc():
#     # Run garbage collection before each test
#     _gc.collect()
#     yield
#     # Run garbage collection after each test
#     _gc.collect()

######################################################################
## pytest: num_parallel_threads
######################################################################

if not PARALLEL_RUN_AVAILABLE:

    @_pytest.fixture
    def num_parallel_threads():
        return 1


######################################################################
## _pytest fixture: plotting
######################################################################


# Following the approach of Seaborn's conftest.py...
@_pytest.fixture(autouse=True)
def close_figs():
    yield
    _plt.close("all")


@_pytest.fixture(autouse=True)
def random_seed():
    # seed = sum(map(ord, "seaborn random global"))
    _np.random.seed(0)


@_pytest.fixture
def rng():
    # seed = sum(map(ord, "seaborn random object"))
    return _np.random.RandomState(0)


######################################################################
## _pytest fixture: xarray
######################################################################


@_pytest.fixture
def xr():
    """
    Fixture to import xarray so that the test is skipped when xarray is not installed.
    Use this fixture instead of importing xrray in test files.

    Examples
    --------
    Request the xarray fixture by passing in ``xr`` as an argument to the test ::

        def test_imshow_xarray(xr):
            ds = xr.DataArray(_np.random.randn(2, 3))
            im = plt.figure().subplots().imshow(ds)
            _np.testing.assert_array_equal(im.get_array(), ds)

    """
    return _pytest.importorskip("xarray")


######################################################################
## if not import skip pandas
######################################################################

# @_pytest.fixture
# def pd():
#     """
#     Fixture to import and configure pandas. Using this fixture, the test is skipped when
#     pandas is not installed. Use this fixture instead of importing pandas in test files.

#     Examples
#     --------
#     Request the pandas fixture by passing in ``pd`` as an argument to the test ::

#         def test_matshow_pandas(pd):

#             df = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6]})
#             im = plt.figure().subplots().matshow(df)
#             _np.testing.assert_array_equal(im.get_array(), df)
#     """
#     _pd = _pytest.importorskip('pandas')
#     try:
#         from pandas.plotting import (
#             deregister_matplotlib_converters as deregister
#         )
#         deregister()
#     except ImportError:
#         pass
#     return _pd

######################################################################
## _pytest fixture: numpy, pandas dataset
######################################################################


@_pytest.fixture
def wide_df(rng):
    columns = list("abc")
    index = _pd.RangeIndex(10, 50, 2, name="wide_index")
    values = rng.normal(size=(len(index), len(columns)))
    return _pd.DataFrame(values, index=index, columns=columns)


@_pytest.fixture
def wide_array(wide_df):
    return wide_df.to_numpy()


# TODO s/flat/thin?
@_pytest.fixture
def flat_series(rng):
    index = _pd.RangeIndex(10, 30, name="t")
    return _pd.Series(rng.normal(size=20), index, name="s")


@_pytest.fixture
def flat_array(flat_series):
    return flat_series.to_numpy()


@_pytest.fixture
def flat_list(flat_series):
    return flat_series.to_list()


@_pytest.fixture(params=["series", "array", "list"])
def flat_data(rng, request):
    index = _pd.RangeIndex(10, 30, name="t")
    series = _pd.Series(rng.normal(size=20), index, name="s")

    if request.param == "series":
        data = series
    elif request.param == "array":
        data = series.to_numpy()
    elif request.param == "list":
        data = series.to_list()
    return data


@_pytest.fixture
def wide_list_of_series(rng):
    return [
        _pd.Series(rng.normal(size=20), _np.arange(20), name="a"),
        _pd.Series(rng.normal(size=10), _np.arange(5, 15), name="b"),
    ]


@_pytest.fixture
def wide_list_of_arrays(wide_list_of_series):
    return [s.to_numpy() for s in wide_list_of_series]


@_pytest.fixture
def wide_list_of_lists(wide_list_of_series):
    return [s.to_list() for s in wide_list_of_series]


@_pytest.fixture
def wide_dict_of_series(wide_list_of_series):
    return {s.name: s for s in wide_list_of_series}


@_pytest.fixture
def wide_dict_of_arrays(wide_list_of_series):
    return {s.name: s.to_numpy() for s in wide_list_of_series}


@_pytest.fixture
def wide_dict_of_lists(wide_list_of_series):
    return {s.name: s.to_list() for s in wide_list_of_series}


@_pytest.fixture
def long_df(rng):
    n = 100
    df = _pd.DataFrame(
        dict(
            x=rng.uniform(0, 20, n).round().astype("int"),
            y=rng.normal(size=n),
            z=rng.lognormal(size=n),
            a=rng.choice(list("abc"), n),
            b=rng.choice(list("mnop"), n),
            c=rng.choice([0, 1], n, [0.3, 0.7]),
            d=rng.choice(
                _np.arange("2004-07-30", "2007-07-30", dtype="datetime64[Y]"), n
            ),
            t=rng.choice(
                _np.arange("2004-07-30", "2004-07-31", dtype="datetime64[m]"), n
            ),
            s=rng.choice([2, 4, 8], n),
            f=rng.choice([0.2, 0.3], n),
        )
    )
    a_cat = df["a"].astype("category")
    new_categories = _np.roll(a_cat.cat.categories, 1)
    df["a_cat"] = a_cat.cat.reorder_categories(new_categories)

    df["s_cat"] = df["s"].astype("category")
    df["s_str"] = df["s"].astype(str)

    return df


@_pytest.fixture
def long_dict(long_df):
    return long_df.to_dict()


@_pytest.fixture
def repeated_df(rng):
    n = 100
    return _pd.DataFrame(
        dict(
            x=_np.tile(_np.arange(n // 2), 2),
            y=rng.normal(size=n),
            a=rng.choice(list("abc"), n),
            u=_np.repeat(_np.arange(2), n // 2),
        )
    )


@_pytest.fixture
def null_df(rng, long_df):
    df = long_df.copy()
    for col in df:
        if _pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        idx = rng.permutation(df.index)[:10]
        df.loc[idx, col] = _np.nan
    return df


@_pytest.fixture
def object_df(rng, long_df):
    df = long_df.copy()
    # objectify numeric columns
    for col in ["c", "s", "f"]:
        df[col] = df[col].astype(object)
    return df


@_pytest.fixture
def null_series(flat_series):
    return _pd.Series(index=flat_series.index, dtype="float64")


class MockInterchangeableDataFrame:
    # Mock object that is not a pandas.DataFrame but that can
    # be converted to one via the DataFrame exchange protocol
    def __init__(self, data):
        self._data = data

    def __dataframe__(self, *args, **kwargs):
        return self._data.__dataframe__(*args, **kwargs)


@_pytest.fixture
def mock_long_df(long_df):
    return MockInterchangeableDataFrame(long_df)


######################################################################
## Array API xp backend from scipy
######################################################################

# Array API backend handling
xp_available_backends = {"numpy": _np}

if SKPLT_ARRAY_API and isinstance(SKPLT_ARRAY_API, str):
    # fill the dict of backends with available libraries
    try:
        from ._lib import array_api_strict

        xp_available_backends.update({"array_api_strict": array_api_strict})
        if _pep440.parse(array_api_strict.__version__) < _pep440.Version("2.0"):
            raise ImportError("array-api-strict must be >= version 2.0")
        array_api_strict.set_array_api_strict_flags(api_version="2023.12")
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]

        xp_available_backends.update({"torch": torch})
        # can use `mps` or `cpu`
        torch.set_default_device(SKPLT_DEVICE)

        # default to float64 unless explicitly requested
        default = _os.getenv("SKPLT_DEFAULT_DTYPE", default="float64")
        if default == "float64":
            torch.set_default_dtype(torch.float64)
        elif default != "float32":
            raise ValueError(
                "SKPLT_DEFAULT_DTYPE env var, if set, can only be either 'float64' "
                f"or 'float32'. Got '{default}' instead."
            )
    except ImportError:
        pass

    try:
        import cupy  # type: ignore[import-not-found]

        xp_available_backends.update({"cupy": cupy})
    except ImportError:
        pass

    try:
        import jax.numpy  # type: ignore[import-not-found]

        xp_available_backends.update({"jax.numpy": jax.numpy})
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_device", jax.devices(SKPLT_DEVICE)[0])
    except ImportError:
        pass

    # by default, use all available backends
    if SKPLT_ARRAY_API.lower() not in ("1", "true"):
        SKPLT_ARRAY_API_ = _json.loads(SKPLT_ARRAY_API)

        if "all" in SKPLT_ARRAY_API_:
            pass  # same as True
        else:
            # only select a subset of backend by filtering out the dict
            try:
                xp_available_backends = {
                    backend: xp_available_backends[backend]
                    for backend in SKPLT_ARRAY_API_
                }
            except KeyError:
                msg = f"'--array-api-backend' must be in {xp_available_backends.keys()}"
                raise ValueError(msg)


if "cupy" in xp_available_backends:
    SKPLT_DEVICE = "cuda"

    # this is annoying in CuPy 13.x
    _warnings.filterwarnings(
        "ignore", "cupyx.jit.rawkernel is experimental", category=FutureWarning
    )
    from cupyx.scipy import signal

    del signal


@_pytest.fixture(
    params=[
        _pytest.param(v, id=k, marks=_pytest.mark.array_api_backends)
        for k, v in xp_available_backends.items()
    ]
)
def xp(request):
    """
    Run the test that uses this fixture on each available array API library.

    You can select all and only the tests that use the `xp` fixture by
    passing `-m array_api_backends` to pytest.

    Please read: https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
    """
    if SKPLT_ARRAY_API:
        from ._lib._array_api import default_xp

        # Throughout all calls to assert_almost_equal, assert_array_almost_equal, and
        # xp_assert_* functions, test that the array namespace is xp in both the
        # expected and actual arrays. This is to detect the case where both arrays are
        # erroneously just plain numpy while xp is something else.
        with default_xp(request.param):
            yield request.param
    else:
        yield request.param


# array_api_compatible = (
#   _pytest.mark.parametrize("xp", xp_available_backends.values())
# )
array_api_compatible = _pytest.mark.array_api_compatible

skip_xp_invalid_arg = _pytest.mark.skipif(
    SKPLT_ARRAY_API,
    reason=(
        "Test involves masked arrays, object arrays, or other types "
        "that are not valid input when `SKPLT_ARRAY_API` is used."
    ),
)

# pytestmark = pytest.mark.skipif(
#     any(
#         pytest.importorskip(pkg, reason=f"{pkg} not installed") is None
#         for pkg in ["cupy", "dask", "torch"]
#     ),
#     reason="Required module missing: cupy, dask, or torch"
# )

# # @requires_modules("cupy", "torch")
# def requires_modules(*modules):
#     """Decorator to skip test if any given module is missing."""
#     def decorator(func):
#         for m in modules:
#             if pytest.importorskip(m, reason=f"{m} not installed") is None:
#                 return pytest.mark.skip(reason=f"{m} not installed")(func)
#         return func
#     return decorator

######################################################################
## array API xp backends
######################################################################


def _backends_kwargs_from_request(request, skip_or_xfail):
    """A helper for {skip,xfail}_xp_backends"""
    # do not allow multiple backends
    args_ = request.keywords[f"{skip_or_xfail}_xp_backends"].args
    if len(args_) > 1:
        # np_only / cpu_only has args=(), otherwise it's ('numpy',)
        # and we do not allow ('numpy', 'cupy')
        raise ValueError(f"multiple backends: {args_}")

    markers = list(request.node.iter_markers(f"{skip_or_xfail}_xp_backends"))
    backends = []
    kwargs = {}
    for marker in markers:
        if marker.kwargs.get("np_only"):
            kwargs["np_only"] = True
            kwargs["exceptions"] = marker.kwargs.get("exceptions", [])
            kwargs["reason"] = marker.kwargs.get("reason", None)
        elif marker.kwargs.get("cpu_only"):
            if not kwargs.get("np_only"):
                # if np_only is given, it is certainly cpu only
                kwargs["cpu_only"] = True
                kwargs["exceptions"] = marker.kwargs.get("exceptions", [])
                kwargs["reason"] = marker.kwargs.get("reason", None)

        # add backends, if any
        if len(marker.args) > 0:
            backend = marker.args[0]  # was a tuple, ('numpy',) etc
            backends.append(backend)
            kwargs.update(**{backend: marker.kwargs})

    return backends, kwargs


@_pytest.fixture
def skip_xp_backends(xp, request):
    """
    skip_xp_backends(backend=None, reason=None, np_only=False, cpu_only=False, exceptions=None)

    Skip a decorated test for the provided backend, or skip a category of backends.

    See ``skip_or_xfail_backends`` docstring for details. Note that, contrary to
    ``skip_or_xfail_backends``, the ``backend`` and ``reason`` arguments are optional
    single strings: this function only skips a single backend at a time.
    To skip multiple backends, provide multiple decorators.
    """
    if "skip_xp_backends" not in request.keywords:
        return

    backends, kwargs = _backends_kwargs_from_request(request, skip_or_xfail="skip")
    skip_or_xfail_xp_backends(xp, backends, kwargs, skip_or_xfail="skip")


@_pytest.fixture
def xfail_xp_backends(xp, request):
    """
    xfail_xp_backends(backend=None, reason=None, np_only=False, cpu_only=False, exceptions=None)

    xfail a decorated test for the provided backend, or xfail a category of backends.

    See ``skip_or_xfail_backends`` docstring for details. Note that, contrary to
    ``skip_or_xfail_backends``, the ``backend`` and ``reason`` arguments are optional
    single strings: this function only xfails a single backend at a time.
    To xfail multiple backends, provide multiple decorators.
    """
    if "xfail_xp_backends" not in request.keywords:
        return
    backends, kwargs = _backends_kwargs_from_request(request, skip_or_xfail="xfail")
    skip_or_xfail_xp_backends(xp, backends, kwargs, skip_or_xfail="xfail")


def skip_or_xfail_xp_backends(xp, backends, kwargs, skip_or_xfail="skip"):
    """
    Skip based on the ``skip_xp_backends`` or ``xfail_xp_backends`` marker.

    See the "Support for the array API standard" docs page for usage examples.

    Parameters
    ----------
    backends : tuple
        Backends to skip/xfail, e.g. ``("array_api_strict", "torch")``.
        These are overridden when ``np_only`` is ``True``, and are not
        necessary to provide for non-CPU backends when ``cpu_only`` is ``True``.
        For a custom reason to apply, you should pass
        ``kwargs={<backend name>: {'reason': '...'}, ...}``.
    np_only : bool, optional
        When ``True``, the test is skipped/xfailed for all backends other
        than the default NumPy backend. There is no need to provide
        any ``backends`` in this case. Default: ``False``.
    cpu_only : bool, optional
        When ``True``, the test is skipped/xfailed on non-CPU devices.
        There is no need to provide any ``backends`` in this case,
        but any ``backends`` will also be skipped on the CPU.
        Default: ``False``.
    reason : str, optional
        A reason for the skip/xfail in the case of ``np_only=True`` or
        ``cpu_only=True``. If omitted, a default reason is used.
    exceptions : list, optional
        A list of exceptions for use with ``cpu_only`` or ``np_only``.
        This should be provided when delegation is implemented for some,
        but not all, non-CPU/non-NumPy backends.
    skip_or_xfail : str
        ``'skip'`` to skip, ``'xfail'`` to xfail.

    """
    skip_or_xfail = getattr(_pytest, skip_or_xfail)
    np_only = kwargs.get("np_only", False)
    cpu_only = kwargs.get("cpu_only", False)
    exceptions = kwargs.get("exceptions", [])

    if reasons := kwargs.get("reasons"):
        raise ValueError(f"provide a single `reason=` kwarg; got {reasons=} instead")

    # input validation
    if np_only and cpu_only:
        # np_only is a stricter subset of cpu_only
        cpu_only = False
    if exceptions and not (cpu_only or np_only):
        raise ValueError("`exceptions` is only valid alongside `cpu_only` or `np_only`")

    # Test explicit backends first so that their reason can override
    # those from np_only/cpu_only
    if backends is not None:
        for i, backend in enumerate(backends):
            if xp.__name__ == backend:
                reason = kwargs[backend].get("reason")
                if not reason:
                    reason = f"do not run with array API backend: {backend}"

                skip_or_xfail(reason=reason)

    if np_only:
        reason = kwargs.get("reason")
        if not reason:
            reason = "do not run with non-NumPy backends"

        if xp.__name__ != "numpy" and xp.__name__ not in exceptions:
            skip_or_xfail(reason=reason)
        return

    if cpu_only:
        reason = kwargs.get("reason")
        if not reason:
            reason = (
                "no array-agnostic implementation or delegation available "
                "for this backend and device"
            )

        exceptions = [] if exceptions is None else exceptions
        if SKPLT_ARRAY_API and SKPLT_DEVICE != "cpu":
            if xp.__name__ == "cupy" and "cupy" not in exceptions:
                skip_or_xfail(reason=reason)
            elif xp.__name__ == "torch" and "torch" not in exceptions:
                if "cpu" not in xp.empty(0).device.type:
                    skip_or_xfail(reason=reason)
            elif xp.__name__ == "jax.numpy" and "jax.numpy" not in exceptions:
                for d in xp.empty(0).devices():
                    if "cpu" not in d.device_kind:
                        skip_or_xfail(reason=reason)


######################################################################
## hypothesis profiles
######################################################################

# Following the approach of NumPy's conftest.py...
# Use a known and persistent tmpdir for hypothesis' caches, which
# can be automatically cleared by the OS or user.
_hypothesis.configuration.set_hypothesis_home_dir(
    _os.path.join(_tempfile.gettempdir(), ".hypothesis")
)
# We register two custom profiles for SciPy - for details see
# https://hypothesis.readthedocs.io/en/latest/settings.html
# The first is designed for our own CI runs; the latter also
# forces determinism and is designed for use via scipy.test()
_hypothesis.settings.register_profile(
    name="nondeterministic", deadline=None, print_blob=True
)
_hypothesis.settings.register_profile(
    name="deterministic",
    deadline=None,
    print_blob=True,
    database=None,
    derandomize=True,
    suppress_health_check=list(_hypothesis.HealthCheck),
)
# Profile is currently set by environment variable `SKPLT_HYPOTHESIS_PROFILE`
# In the future, it would be good to work the choice into dev.py.
SKPLT_HYPOTHESIS_PROFILE = _os.environ.get("SKPLT_HYPOTHESIS_PROFILE", "deterministic")
_hypothesis.settings.load_profile(SKPLT_HYPOTHESIS_PROFILE)

######################################################################
## doctesting stuff
######################################################################

# try:
#     from scipy_doctest.conftest import dt_config
#     HAVE_SCPDT = True
# except ModuleNotFoundError:
#     HAVE_SCPDT = False

# if HAVE_SCPDT:

#     # FIXME: populate the dict once
#     @contextmanager
#     def warnings_errors_and_rng(test=None):
#         """Temporarily turn (almost) all warnings to errors.

#         Filter out known warnings which we allow.
#         """
#         known_warnings = dict()

#         # these functions are known to emit "divide by zero" RuntimeWarnings
#         divide_by_zero = [
#             'scipy.linalg.norm', 'scipy.ndimage.center_of_mass',
#         ]
#         for name in divide_by_zero:
#             known_warnings[name] = dict(category=RuntimeWarning,
#                                         message='divide by zero')

#         # Deprecated stuff in scipy.signal and elsewhere
#         deprecated = [
#             'scipy.signal.cwt', 'scipy.signal.morlet', 'scipy.signal.morlet2',
#             'scipy.signal.ricker',
#             'scipy.integrate.simpson',
#             'scipy.interpolate.interp2d',
#             'scipy.linalg.kron',
#         ]
#         for name in deprecated:
#             known_warnings[name] = dict(category=DeprecationWarning)

#         from scipy import integrate
#         # the functions are known to emit IntegrationWarnings
#         integration_w = ['scipy.special.ellip_normal',
#                          'scipy.special.ellip_harm_2',
#         ]
#         for name in integration_w:
#             known_warnings[name] = dict(category=integrate.IntegrationWarning,
#                                         message='The occurrence of roundoff')

#         # scipy.stats deliberately emits UserWarnings sometimes
#         user_w = ['scipy.stats.anderson_ksamp', 'scipy.stats.kurtosistest',
#                   'scipy.stats.normaltest', 'scipy.sparse.linalg.norm']
#         for name in user_w:
#             known_warnings[name] = dict(category=UserWarning)

#         # additional one-off warnings to filter
#         dct = {
#             'scipy.sparse.linalg.norm':
#                 dict(category=UserWarning, message="Exited at iteration"),
#             # tutorials
#             'linalg.rst':
#                 dict(message='the matrix subclass is not',
#                      category=PendingDeprecationWarning),
#             'stats.rst':
#                 dict(message='The maximum number of subdivisions',
#                      category=integrate.IntegrationWarning),
#         }
#         known_warnings.update(dct)

#         # these legitimately emit warnings in examples
#         legit = set('scipy.signal.normalize')

#         # Now, the meat of the matter: filter warnings,
#         # also control the random seed for each doctest.

#         # XXX: this matches the refguide-check behavior, but is a tad strange:
#         # makes sure that the seed the old-fashioned _np.random* methods is
#         # *NOT* reproducible but the new-style `default_rng()` *IS* reproducible.
#         # Should these two be either both repro or both not repro?

#         from scipy._lib._util import _fixed_default_rng
#         import numpy as _np
#         with _fixed_default_rng():
#             _np.random.seed(None)
#             with _warnings.catch_warnings():
#                 if test and test.name in known_warnings:
#                     _warnings.filterwarnings('ignore',
#                                             **known_warnings[test.name])
#                     yield
#                 elif test and test.name in legit:
#                     yield
#                 else:
#                     _warnings.simplefilter('error', Warning)
#                     yield

#     dt_config.user_context_mgr = warnings_errors_and_rng
#     dt_config.skiplist = set([
#         'scipy.linalg.LinAlgError',     # comes from numpy
#         'scipy.fftpack.fftshift',       # fftpack stuff is also from numpy
#         'scipy.fftpack.ifftshift',
#         'scipy.fftpack.fftfreq',
#         'scipy.special.sinc',           # sinc is from numpy
#         'scipy.optimize.show_options',  # does not have much to doctest
#         'scipy.signal.normalize',       # manipulates warnings (XXX temp skip)
#         'scipy.sparse.linalg.norm',     # XXX temp skip
#         # these below test things which inherit from _np.ndarray
#         # cross-ref https://github.com/numpy/numpy/issues/28019
#         'scipy.io.matlab.MatlabObject.strides',
#         'scipy.io.matlab.MatlabObject.dtype',
#         'scipy.io.matlab.MatlabOpaque.dtype',
#         'scipy.io.matlab.MatlabOpaque.strides',
#         'scipy.io.matlab.MatlabFunction.strides',
#         'scipy.io.matlab.MatlabFunction.dtype'
#     ])

#     # these are affected by NumPy 2.0 scalar repr: rely on string comparison
#     if _np.__version__ < "2":
#         dt_config.skiplist.update(set([
#             'scipy.io.hb_read',
#             'scipy.io.hb_write',
#             'scipy.sparse.csgraph.connected_components',
#             'scipy.sparse.csgraph.depth_first_order',
#             'scipy.sparse.csgraph.shortest_path',
#             'scipy.sparse.csgraph.floyd_warshall',
#             'scipy.sparse.csgraph.dijkstra',
#             'scipy.sparse.csgraph.bellman_ford',
#             'scipy.sparse.csgraph.johnson',
#             'scipy.sparse.csgraph.yen',
#             'scipy.sparse.csgraph.breadth_first_order',
#             'scipy.sparse.csgraph.reverse_cuthill_mckee',
#             'scipy.sparse.csgraph.structural_rank',
#             'scipy.sparse.csgraph.construct_dist_matrix',
#             'scipy.sparse.csgraph.reconstruct_path',
#             'scipy.ndimage.value_indices',
#             'scipy.stats.mstats.describe',
#     ]))

#     # help pytest collection a bit: these names are either private
#     # (distributions), or just do not need doctesting.
#     dt_config.pytest_extra_ignore = [
#         "scipy.stats.distributions",
#         "scipy.optimize.cython_optimize",
#         "scipy.test",
#         "scipy.show_config",
#         # equivalent to "pytest --ignore=path/to/file"
#         "scipy/special/_precompute",
#         "scipy/interpolate/_interpnd_info.py",
#         "scipy/_lib/array_api_compat",
#         "scipy/_lib/highs",
#         "scipy/_lib/unuran",
#         "scipy/_lib/_gcutils.py",
#         "scipy/_lib/doccer.py",
#         "scipy/_lib/_uarray",
#     ]

#     dt_config.pytest_extra_xfail = {
#         # name: reason
#         "ND_regular_grid.rst": "ReST parser limitation",
#         "extrapolation_examples.rst": "ReST parser limitation",
#         "sampling_pinv.rst": "__cinit__ unexpected argument",
#         "sampling_srou.rst": "nan in scalar_power",
#         "probability_distributions.rst": "integration warning",
#     }

#     # tutorials
#     dt_config.pseudocode = set(['integrate.nquad(func,'])
#     dt_config.local_resources = {
#         'io.rst': [
#             "octave_a.mat",
#             "octave_cells.mat",
#             "octave_struct.mat"
#         ]
#     }

#     dt_config.strict_check = True
