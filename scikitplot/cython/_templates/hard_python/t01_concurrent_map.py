"""Hard Python template: concurrent execution.

Demonstrates a safe ThreadPoolExecutor pattern with bounded workers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable


def thread_map(
    fn: Callable[[int], int], items: Iterable[int], *, max_workers: int = 4
) -> list[int]:
    """Map ``fn`` over items using a thread pool."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))
