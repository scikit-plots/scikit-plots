"""Complex Python template: Protocol-based API."""

from __future__ import annotations

from typing import Protocol


class SupportsPredictProba(Protocol):
    def predict_proba(self, X): ...


def positive_class_probability(model: SupportsPredictProba, X) -> list[float]:
    proba = model.predict_proba(X)
    return [float(row[1]) for row in proba]
