# scikitplot/_testing/_pytesttester.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Pytest test running.

This module implements the ``test()`` function for library sub-packages.
The typical boiler-plate for a sub-package ``__init__.py``::

    from ._testing._pytesttester import PytestTester

    test = PytestTester(__name__)
    del PytestTester

Warnings filtering and other runtime settings should be configured in the
``pytest.ini`` file at the repository root:

* ``pytest.ini`` present (develop mode)
    All unfiltered warnings are raised as errors.
* ``pytest.ini`` absent (release mode)
    ``DeprecationWarning`` and ``PendingDeprecationWarning`` are ignored;
    other warnings pass through.

Notes
-----
This module is imported by every sub-package that exposes a ``test()``
function.  It therefore contains **no library imports at module scope** to
avoid slowing test collection and to prevent circular imports.
"""

from __future__ import annotations

import os
import sys

__all__ = ["PytestTester"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _show_numpy_info() -> None:
    """Print NumPy version and relaxed-strides status to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Developer note: Called before ``pytest.main`` as a diagnostic aid.  If
    NumPy is not installed, a single :class:`ImportWarning` is emitted and the
    function returns silently rather than raising :class:`ImportError`.

    Examples
    --------
    >>> _show_numpy_info()  # doctest: +SKIP
    NumPy version 2.x.y
    NumPy relaxed strides checking option: True
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        import warnings as _warnings

        _warnings.warn(
            "NumPy is not installed; skipping NumPy diagnostic output.",
            ImportWarning,
            stacklevel=2,
        )
        return
    print(f"NumPy version {np.__version__}")
    relaxed_strides = np.ones((10, 1), order="C").flags.f_contiguous
    print(f"NumPy relaxed strides checking option: {relaxed_strides}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PytestTester:
    """Pytest test runner entry point for a library sub-package.

    A ``test`` attribute is typically wired into a sub-package
    ``__init__.py``::

        from ._testing._pytesttester import PytestTester

        test = PytestTester(__name__)
        del PytestTester

    Calling the instance discovers and runs all tests associated with that
    sub-package and all of its nested sub-packages.

    Parameters
    ----------
    module_name : str
        Fully-qualified module name whose tests should be run, e.g.
        ``"scikitplot.decomposition"``.

    Attributes
    ----------
    module_name : str
        The module name provided at construction time.

    Notes
    -----
    User note: Use the instance as a callable — the public interface is the
    :meth:`__call__` signature.  Do not access named methods directly.

    Developer note: All library imports are intentionally deferred to
    :meth:`__call__` time so that importing this module during test collection
    does not trigger heavy dependency loading or circular imports.

    Examples
    --------
    >>> tester = PytestTester("scikitplot.decomposition")  # doctest: +SKIP
    >>> tester()                                           # doctest: +SKIP
    True
    """

    def __init__(self, module_name: str) -> None:
        if not isinstance(module_name, str):
            raise TypeError(
                f"'module_name' must be a str, got {type(module_name).__name__!r}."
            )
        self.module_name = module_name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(module_name={self.module_name!r})"

    def __call__(
        self,
        label: str = "fast",
        verbose: int = 1,
        extra_argv: list | None = None,
        doctests: bool = False,
        coverage: bool = False,
        durations: int = -1,
        tests: list | None = None,
    ) -> bool:
        """Run tests for the module using pytest.

        Parameters
        ----------
        label : str, default='fast'
            Which tests to run.  ``'fast'`` skips tests decorated with
            ``pytest.mark.slow``; ``'full'`` includes them; any other value
            is forwarded verbatim as a pytest ``-m`` marker expression.
        verbose : int, default=1
            Verbosity level in the range ``[1, inf)``.  ``1`` maps to quiet
            mode only; each increment adds one ``-v`` flag.
        extra_argv : list of str or None, default=None
            Extra arguments forwarded verbatim to ``pytest.main``.
        doctests : bool, default=False
            .. note::
               Doctest mode is **not** supported.  Passing ``True`` raises
               :class:`ValueError`.
        coverage : bool, default=False
            When ``True``, append ``--cov=<module_path>`` to enable
            pytest-cov coverage reporting.
        durations : int, default=-1
            Timing report threshold.  ``-1`` disables; ``0`` reports all;
            positive *n* reports the *n* slowest tests.
        tests : list of str or None, default=None
            Explicit test targets forwarded via ``--pyargs``.  When ``None``,
            defaults to ``[self.module_name]``.

        Returns
        -------
        success : bool
            ``True`` when all collected tests pass; ``False`` otherwise.

        Raises
        ------
        ValueError
            If *doctests* is ``True`` (not supported).
        KeyError
            If :attr:`module_name` is not found in ``sys.modules``.

        See Also
        --------
        _show_numpy_info : Diagnostic helper called before running tests.

        Notes
        -----
        Developer note: ``pytest.main`` may raise ``SystemExit`` in some
        environments (e.g. Jupyter, IPython).  The exit code is normalised to
        ``int`` so that ``pytest.ExitCode`` enum values (which inherit from
        ``int``), raw integers, ``None`` (POSIX success convention), and
        strings (any non-empty string = failure) are all handled correctly.

        Examples
        --------
        >>> tester = PytestTester("scikitplot")  # doctest: +SKIP
        >>> tester(label="fast", verbose=1)      # doctest: +SKIP
        True
        """
        import pytest

        if doctests:
            raise ValueError(
                "Doctests are not supported by PytestTester. "
                "Use pytest's --doctest-modules flag directly instead."
            )

        module = sys.modules[self.module_name]

        # __path__ exists on packages; single-file modules only have __file__.
        path_list = getattr(module, "__path__", None)
        if path_list:
            module_path = os.path.abspath(path_list[0])
        else:
            file_attr = getattr(module, "__file__", None) or "."
            module_path = os.path.abspath(file_attr)

        # Base arguments: show locals on failure (-l) and quiet output (-q).
        pytest_args = ["-l", "-q"]

        # Suppress known false-positive / environment-specific warnings that
        # are irrelevant to user code and pollute test output.
        pytest_args += [
            "-W ignore:Not importing directory",
            "-W ignore:numpy.dtype size changed",
            "-W ignore:numpy.ufunc size changed",
            "-W ignore::UserWarning:cpuinfo",
            "-W ignore:the matrix subclass is not",
        ]

        if extra_argv:
            pytest_args += list(extra_argv)

        # Each verbosity level above 1 adds one -v flag.
        if verbose > 1:
            pytest_args += ["-" + "v" * (verbose - 1)]

        if coverage:
            pytest_args += [f"--cov={module_path}"]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if durations >= 0:
            pytest_args += [f"--durations={durations}"]

        if tests is None:
            tests = [self.module_name]

        pytest_args += ["--pyargs"] + list(tests)

        _show_numpy_info()

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            # Normalise:
            #   int / ExitCode enum  -> use directly
            #   None                 -> 0  (POSIX: exit() with no arg = success)
            #   non-empty str        -> 1  (any message = failure)
            raw = exc.code
            if isinstance(raw, int):
                code = raw
            else:
                code = 0 if not raw else 1

        return int(code) == 0
