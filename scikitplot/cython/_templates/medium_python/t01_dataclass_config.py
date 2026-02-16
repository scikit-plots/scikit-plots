"""Medium Python template: dataclass config pattern."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    lr: float = 0.1
    max_iter: int = 100


def describe(cfg: Config) -> str:
    return f"lr={cfg.lr} max_iter={cfg.max_iter}"
