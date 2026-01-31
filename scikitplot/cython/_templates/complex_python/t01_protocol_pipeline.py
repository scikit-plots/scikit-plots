"""Complex Python template: Protocol-based pipeline typing."""

from __future__ import annotations

from typing import Iterable, Protocol, TypeVar

T = TypeVar("T")


class Transformer(Protocol[T]):
    def transform(self, xs: Iterable[T]) -> list[T]: ...


def apply(t: Transformer[T], xs: Iterable[T]) -> list[T]:
    return t.transform(xs)
