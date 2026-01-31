"""LRU-cached fibonacci (stdlib caching)."""

from functools import lru_cache


@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
