# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

from typing import Any, Optional, Type, TypeVar  # noqa: F401

from typing_extensions import Self

class Timer:
    message: str
    verbose: bool
    log_level: str
    precision: int
    _start: float

    def __init__(
        self,
        message: str,
        *,
        verbose: bool = ...,
        log_level: str = ...,
        precision: int = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,  # noqa: PYI036
    ) -> bool | None: ...
