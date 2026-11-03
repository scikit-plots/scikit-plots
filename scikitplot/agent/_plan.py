# scikitplot/agent/_plan.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A run's plan: items the planner commits to, and the rules for changing them.

Notes
-----
**User.** A planner may attach the *whole* plan to any decision
(``Decision.act(..., plan=[...])`` or ``Decision.finish(..., plan=[...])``).
The runtime checks the change against the current plan and records it with
the decision, so the plan is saved, audited and resumed with everything else;
:attr:`RunState.plan` is the plan as of the last decision that carried one.
A plan is the planner's statement of intent: it is recorded and checked, and
never gates an action -- the five gates do that.

**Developer.** The rules exist so a plan cannot misreport progress:

* every item of the current plan appears in the next one -- an item is
  removed only by marking it ``dropped``, so nothing disappears unrecorded;
* ``done`` and ``dropped`` are final: such an item keeps its status and its
  text;
* at most one item is ``in_progress``;
* ids are unique; sizes are bounded (:data:`MAX_PLAN_ITEMS`, text length).

The plan is not stored apart from the decisions that carry it; it is derived
from them, so the saved plan and the audited plan cannot disagree, and
:meth:`RunState.check` replays every change when a state is loaded.
"""

from __future__ import annotations

import dataclasses
import re
from enum import unique
from typing import Any, Iterable, Mapping, Sequence

from ._errors import _StrEnum

__all__ = [
    "MAX_PLAN_ITEMS",
    "PlanItem",
    "PlanStatus",
    "check_plan_change",
]

#: Most items one plan may hold.
MAX_PLAN_ITEMS = 64
_MAX_TEXT = 500
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@unique
class PlanStatus(_StrEnum):
    """Where one plan item stands."""

    PENDING = "pending"
    """Not started."""

    IN_PROGRESS = "in_progress"
    """Being worked on; at most one item at a time."""

    DONE = "done"
    """Finished. Final."""

    DROPPED = "dropped"
    """Abandoned, kept on record. Final."""


_FINAL = frozenset({PlanStatus.DONE, PlanStatus.DROPPED})


@dataclasses.dataclass(frozen=True)
class PlanItem:
    """
    One item of a plan.

    Parameters
    ----------
    id : str
        Stable identity: 1-64 characters of letters, digits, ``_``, ``.``,
        ``-``, starting with a letter or digit.
    text : str
        What the item is, non-empty, at most 500 characters.
    status : PlanStatus or str, default="pending"

    Examples
    --------
    >>> PlanItem("search", "find the chunker docs", "in_progress").status.value
    'in_progress'
    """

    id: str
    text: str
    status: PlanStatus = PlanStatus.PENDING

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID.fullmatch(self.id):
            raise ValueError(f"plan item id {self.id!r} is not a valid id.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"plan item {self.id!r} has no text.")
        if len(self.text) > _MAX_TEXT:
            raise ValueError(
                f"plan item {self.id!r} text is {len(self.text)} characters; "
                f"at most {_MAX_TEXT}."
            )
        object.__setattr__(self, "status", PlanStatus(self.status))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {"id": self.id, "text": self.text, "status": self.status.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanItem:
        """Rebuild an item written by :meth:`to_dict`."""
        if not isinstance(data, Mapping) or set(data) != {"id", "text", "status"}:
            raise ValueError(f"a plan item must have id, text and status: {data!r}")
        return cls(data["id"], data["text"], data["status"])


def normalize_plan(items: Iterable[Any]) -> tuple[PlanItem, ...]:
    """
    Return ``items`` as a validated tuple of :class:`PlanItem`.

    Raises
    ------
    TypeError
        If an item is neither a ``PlanItem`` nor a mapping, or ``items`` is a
        string.
    ValueError
        For repeated ids, more than :data:`MAX_PLAN_ITEMS` items, or more than
        one ``in_progress`` item.
    """
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise TypeError("a plan must be a sequence of PlanItem")
    out = tuple(
        item if isinstance(item, PlanItem) else PlanItem.from_dict(item)
        for item in items
    )
    if len(out) > MAX_PLAN_ITEMS:
        raise ValueError(
            f"a plan holds at most {MAX_PLAN_ITEMS} items, got {len(out)}.",
        )
    ids = [item.id for item in out]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"plan item ids repeat: {repeated}.")
    active = [item.id for item in out if item.status is PlanStatus.IN_PROGRESS]
    if len(active) > 1:
        raise ValueError(f"at most one plan item is in_progress, got {active}.")
    return out


def check_plan_change(
    current: Sequence[PlanItem], proposed: Sequence[PlanItem]
) -> None:
    """
    Refuse a plan change that would hide or rewrite progress.

    Parameters
    ----------
    current : sequence of PlanItem
        The plan in force.
    proposed : sequence of PlanItem
        The replacement, already validated by :func:`normalize_plan`.

    Raises
    ------
    ValueError
        If an item of ``current`` is missing from ``proposed``, or a ``done``
        or ``dropped`` item changes status or text.

    Examples
    --------
    >>> old = [PlanItem("a", "read", "done")]
    >>> check_plan_change(old, [PlanItem("a", "read", "done"), PlanItem("b", "write")])
    >>> check_plan_change(old, [PlanItem("b", "write")])
    Traceback (most recent call last):
    ...
    ValueError: plan items ['a'] are missing; mark an item dropped instead of removing it.
    """
    after = {item.id: item for item in proposed}
    missing = [item.id for item in current if item.id not in after]
    if missing:
        raise ValueError(
            f"plan items {missing} are missing; mark an item dropped instead of "
            "removing it."
        )
    for item in current:
        if item.status in _FINAL and after[item.id] != item:
            raise ValueError(
                f"plan item {item.id!r} is {item.status.value}, which is final; "
                "its status and text cannot change."
            )
