"""Module Python template: simple registry pattern."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> ModelSpec:
    return _REGISTRY[name]
