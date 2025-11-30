# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Lightweight timing context manager with :py:obj:`~.logger` support.
"""

from __future__ import annotations

import time

from .. import logger


class Timer:
    """
    Lightweight timing context manager with :py:obj:`~.logger` support.

    Examples
    --------
    >>> with Timer("Building Annoy index...", verbose=True):
    >>>     build_index()
    """

    def __init__(
        self,
        message: str,
        *,
        verbose: bool = False,
        log_level: str = "info",
        precision: int = 3,
    ):
        self.message = message
        self.verbose = verbose
        self.log_level = log_level
        self.precision = precision

        self._start: float = 0.0

    def __enter__(self) -> Timer:  # noqa: PYI034
        self._start = time.perf_counter()

        # log at selected level
        log_fn = getattr(logger, self.log_level, logger.info)
        log_fn(self.message)

        if self.verbose:
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

        log_fn = getattr(logger, self.log_level, logger.info)
        log_fn(f"Completed in {formatted}")

        if self.verbose:
            print(f"   → Completed in {formatted}")  # noqa: T201

        # Let exceptions propagate normally
        return False
