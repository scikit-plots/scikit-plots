# scikitplot/agent/_evaluate.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate agent runs from their audit trail.

Notes
-----
**User.** :func:`evaluate_audit` reads the records an agent wrote (a sink's
``records`` or :func:`~scikitplot.agent.read_audit` of a JSON Lines file) and
returns, per run and over all runs: how runs ended, decisions, tool calls,
denials by gate code, tool failures, approvals asked and granted, sub-runs,
plan changes, tokens, cost and wall time. Nothing is instrumented: the trail
already holds every one of these, and it is the record that cannot be edited
unnoticed.

**Developer.** The trail is verified first (:func:`verify_audit`) and an
evaluation of a broken trail is refused: numbers computed from a trail that
was changed would describe a run that did not happen. A run that has not
ended (paused, or stopped by a crash) is reported with status ``None`` and is
left out of the rates that need an end. Sub-runs are linked to their parent by
its ``subrun`` records and aggregated separately, so a delegation is not
counted twice.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Iterable, Mapping

from ._audit import verify_audit
from ._errors import StateError

__all__ = ["AuditEvaluation", "RunSummary", "evaluate_audit"]


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """
    What one run did, read from its audit records.

    Parameters
    ----------
    run_id : str
    goal : str
    parent : str or None
        The parent run's id, for a sub-run.
    status, stop_reason : str or None
        From the run's last ``run_ended`` record; ``None`` while it has not
        ended.
    decisions : int
    tool_calls : int
        Calls started, retries included.
    outcomes : dict of str to int
        Steps by outcome (``executed``, ``failed``, ``denied``, ...).
    denials : dict of str to int
        Denied and failed steps by error code.
    approvals_requested, approvals_granted, approvals_refused : int
    subruns : int
    plan_changes : int
        Decisions that carried a plan.
    tokens : int or None
    cost : float or None
        From the ``run_ended`` record.
    wall_seconds : float
        Last record time minus first.
    """

    run_id: str
    goal: str
    parent: str | None
    status: str | None
    stop_reason: str | None
    decisions: int
    tool_calls: int
    outcomes: dict[str, int]
    denials: dict[str, int]
    approvals_requested: int
    approvals_granted: int
    approvals_refused: int
    subruns: int
    plan_changes: int
    tokens: int | None
    cost: float | None
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AuditEvaluation:
    """
    Per-run summaries and aggregate rates.

    Parameters
    ----------
    runs : tuple of RunSummary
        Every run in the trail, in order of first appearance.
    summary : dict
        Over top-level runs (sub-runs excluded): ``n_runs``, ``n_ended``,
        ``completion_rate`` (completed / ended), ``stop_reasons``,
        ``mean_decisions``, ``mean_tool_calls``, ``denial_rate`` (denied /
        steps), ``tool_failure_rate`` (failed / (executed + failed)),
        ``approval_grant_rate`` (granted / answered), ``total_cost``,
        ``total_tokens``; a rate with no denominator is ``None``.
    """

    runs: tuple[RunSummary, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {"runs": [r.to_dict() for r in self.runs], "summary": dict(self.summary)}


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_audit(records: Iterable[Mapping[str, Any]]) -> AuditEvaluation:
    """
    Summarise every run in an audit trail.

    Parameters
    ----------
    records : iterable of mapping
        Audit records as written; several runs may be interleaved.

    Returns
    -------
    AuditEvaluation

    Raises
    ------
    StateError
        If the trail does not verify (:func:`verify_audit` reports a
        problem), naming the first problem.
    """
    records = list(records)
    problems = verify_audit(records)
    if problems:
        raise StateError(f"the audit trail does not verify: {problems[0]}")
    order: list[str] = []
    by_run: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        run = record["run_id"]
        if run not in by_run:
            order.append(run)
            by_run[run] = []
        by_run[run].append(record)
    parents = {
        record["data"]["run_id"]: record["run_id"]
        for record in records
        if record["kind"] == "subrun"
    }
    runs = tuple(_summarise(run, by_run[run], parents.get(run)) for run in order)
    return AuditEvaluation(runs, _aggregate([r for r in runs if r.parent is None]))


def _summarise(run: str, records: list, parent: str | None) -> RunSummary:
    kinds = Counter(r["kind"] for r in records)
    outcomes: Counter = Counter()
    denials: Counter = Counter()
    for r in records:
        if r["kind"] == "step":
            outcomes[r["data"]["outcome"]] += 1
            if r["data"]["error"] is not None:
                denials[r["data"]["error"]["code"]] += 1
    granted = sum(
        1 for r in records if r["kind"] == "approval" and r["data"]["granted"]
    )
    refused = sum(
        1 for r in records if r["kind"] == "approval" and not r["data"]["granted"]
    )
    plans = sum(
        1
        for r in records
        if r["kind"] == "decision" and "plan" in r["data"]["decision"]
    )
    started = next((r for r in records if r["kind"] == "run_started"), None)
    ended = [r for r in records if r["kind"] == "run_ended"]
    last = ended[-1]["data"] if ended else None
    # A resumed run that ends again writes a new run_ended; the last one counts.
    if last is not None and any(
        r["kind"] == "run_resumed" and r["seq"] > ended[-1]["seq"] for r in records
    ):
        last = None
    return RunSummary(
        run_id=run,
        goal="" if started is None else started["data"]["goal"],
        parent=parent,
        status=None if last is None else last["status"],
        stop_reason=None if last is None else last["stop_reason"],
        decisions=kinds["decision"],
        tool_calls=kinds["tool_call"],
        outcomes=dict(sorted(outcomes.items())),
        denials=dict(sorted(denials.items())),
        approvals_requested=kinds["approval_requested"],
        approvals_granted=granted,
        approvals_refused=refused,
        subruns=kinds["subrun"],
        plan_changes=plans,
        tokens=None if last is None else int(last["spent"]["tokens"]),
        cost=None if last is None else float(last["spent"]["cost"]),
        wall_seconds=max(0.0, float(records[-1]["time"]) - float(records[0]["time"])),
    )


def _aggregate(runs: list[RunSummary]) -> dict[str, Any]:
    ended = [r for r in runs if r.status is not None]
    steps = sum(sum(r.outcomes.values()) for r in runs)
    executed = sum(r.outcomes.get("executed", 0) for r in runs)
    failed = sum(r.outcomes.get("failed", 0) for r in runs)
    denied = sum(r.outcomes.get("denied", 0) for r in runs)
    answered = sum(r.approvals_granted + r.approvals_refused for r in runs)
    return {
        "n_runs": len(runs),
        "n_ended": len(ended),
        "completion_rate": _ratio(
            sum(r.status == "completed" for r in ended),
            len(ended),
        ),
        "stop_reasons": dict(sorted(Counter(r.stop_reason for r in ended).items())),
        "mean_decisions": _ratio(sum(r.decisions for r in runs), len(runs)),
        "mean_tool_calls": _ratio(sum(r.tool_calls for r in runs), len(runs)),
        "denial_rate": _ratio(denied, steps),
        "tool_failure_rate": _ratio(failed, executed + failed),
        "approval_grant_rate": _ratio(sum(r.approvals_granted for r in runs), answered),
        "total_cost": sum(r.cost for r in ended if r.cost is not None),
        "total_tokens": sum(r.tokens for r in ended if r.tokens is not None),
    }
