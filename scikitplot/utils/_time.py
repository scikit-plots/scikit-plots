# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Lightweight timing context manager with :py:obj:`~.logger` support.
"""

from __future__ import annotations

import time
from typing import Self

from .. import logger


class Timer:
    """
    Lightweight ⏱ timing context manager with :py:obj:`~.logger` support.

    Examples
    --------
    >>> from scikitplot.utils._time import Timer
    >>> with Timer("Building Annoy index...", verbose=True):
    ...     build_index()

    >>> import scikitplot.utils as sp

    >>> with sp.Timer(verbose=True, logging_level="debug"):
    ...     sp.PathNamer()

    >>> with sp.Timer(logging_level="debug"):
    ...     sp.PathNamer().make_filename()

    >>> with sp.Timer(logging_level="info"):
    ...     sp.PathNamer().make_path()

    >>> with sp.Timer(logging_level="warn"):
    ...     sp.make_path()

    >>> with sp.Timer(logging_level="warning"):
    ...     sp.make_path()

    >>> with sp.Timer(logging_level="exception"):
    ...     sp.make_path()

    >>> with sp.Timer(logging_level="error"):
    ...     sp.make_path()

    >>> with sp.Timer(logging_level="fatal"):
    ...     sp.make_path()

    >>> with sp.Timer(logging_level="critical"):
    ...     sp.make_path()
    """

    def __init__(
        self,
        message: str = "",
        *,
        precision: int = 3,
        logging_level: str = "info",
        verbose: bool = False,
    ):
        self.message = message
        self.precision = precision
        self.logging_level = logging_level
        self.verbose = verbose

        self._start: float = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()

        # log at selected level
        log_fn = getattr(logger, self.logging_level, logger.info)
        log_fn(self.message)

        if self.verbose and self.message:
            print(f"[⏱] {self.message}")  # noqa: T201

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: any | None,  # noqa: PYI036
    ) -> None:
        # elapsed_time = time() - self.start_time
        elapsed = time.perf_counter() - self._start
        formatted = f"{elapsed:.{self.precision}f}s"

        log_fn = getattr(logger, self.logging_level, logger.info)
        log_fn(f"⏱ → Completed in {formatted}")

        if self.verbose:
            print(f"⏱ → Completed in {formatted}")  # noqa: T201

        # Let exceptions propagate normally
        return False
