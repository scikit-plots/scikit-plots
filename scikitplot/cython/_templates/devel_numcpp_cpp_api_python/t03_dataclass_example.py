"""Dataclass example."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def norm2(p: Point) -> float:
    return (p.x * p.x + p.y * p.y) ** 0.5
