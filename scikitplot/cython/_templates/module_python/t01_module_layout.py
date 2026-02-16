"""Module Python template: recommended module layout."""

from __future__ import annotations

__all__ = ["Greeter"]


class Greeter:
    def __init__(self, name: str) -> None:
        self.name = name

    def hello(self) -> str:
        return f"Hello, {self.name}!"
