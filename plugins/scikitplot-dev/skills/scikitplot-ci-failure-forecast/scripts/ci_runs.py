#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Read a scheduled workflow's baseline run, and score a forecast against a run.

Usage (``gh`` must be logged in)::

    python3 ci_runs.py --repo OWNER/NAME baseline --workflow nightly.yml --out baseline.json
    python3 ci_runs.py --repo OWNER/NAME score forecast.json --run 123456789

``baseline`` writes the newest completed run of the workflow for an event
(``schedule`` by default; ``push`` when the history holds no scheduled run):
its id, commit, date, and every job of attempt 1 with its conclusion.
Attempt 1 is the record, because a manual rerun can turn a run green and
hide what failed. A job name that repeats (a reusable workflow called
twice) keeps its worst conclusion.

``score`` reads a forecast, ``{"workflow": ..., "baseline_run": ..., "jobs":
{"<job name>": "carries over" | "recovers" | "breaks" | "passes"}}``, and
compares it with attempt 1 of the forecast run: a forecast failure is
"carries over" or "breaks"; an actual failure is any conclusion other than
success, skipped or neutral. A cancelled job and a job missing from the run
are "unknown", not scored. Scoring a run that has not finished is refused.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

OUTCOMES = ("carries over", "recovers", "breaks", "passes")
FAILING = frozenset({"carries over", "breaks"})
NOT_FAILED = frozenset({"success", "skipped", "neutral"})
#: Conclusions that say nothing about the code: not scored.
UNDECIDED = frozenset({"cancelled", "pending"})


def _gh(args: List[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh failed: {result.stderr.strip()}")
    return result.stdout


def attempt_jobs(repo: str, run: int) -> Dict[str, str]:
    """Return job name -> conclusion for attempt 1 of *run*."""
    text = _gh(["api", "--paginate", f"repos/{repo}/actions/runs/{run}/attempts/1/jobs",
                "--jq", ".jobs[] | {name, conclusion}"])
    return merge_jobs(json.loads(line) for line in text.splitlines() if line.strip())


def merge_jobs(jobs) -> Dict[str, str]:
    """
    Return job name -> conclusion, a repeated name keeping its worst one.

    A job still running has no conclusion and is ``"pending"``.
    """
    out: Dict[str, str] = {}
    for job in jobs:
        name, conclusion = job["name"], job.get("conclusion") or "pending"
        prior = out.get(name)
        if prior is None or _rank(conclusion) > _rank(prior):
            out[name] = conclusion
    return out


def _rank(conclusion: str) -> int:
    if conclusion in NOT_FAILED:
        return 0
    return 1 if conclusion in UNDECIDED else 2


def baseline(repo: str, workflow: str, event: str = "schedule", branch: Optional[str] = None) -> dict:
    """Return the newest completed *event* run of *workflow* with its attempt-1 jobs."""
    text = _gh(["api", f"repos/{repo}/actions/workflows/{workflow}/runs?event={event}"
                "&status=completed&per_page=1" + (f"&branch={branch}" if branch else ""),
                "--jq", ".workflow_runs[] | {id, head_sha, created_at, html_url}"])
    runs = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not runs:
        raise RuntimeError(f"{workflow} has no completed {event} run in the Actions history;"
                           " try --event push --branch main")
    run = runs[0]
    jobs = attempt_jobs(repo, run["id"])
    return {"workflow": workflow, "event": event, "run": run["id"], "sha": run["head_sha"],
            "created": run["created_at"], "url": run["html_url"], "jobs": jobs,
            "failed": sorted(n for n, c in jobs.items() if c not in NOT_FAILED)}


def score(forecast: dict, actual: Dict[str, str]) -> dict:
    """
    Compare a forecast with the jobs of a run.

    Parameters
    ----------
    forecast : dict
        The forecast; its ``jobs`` map job names to one of :data:`OUTCOMES`.
    actual : dict
        Job name -> attempt-1 conclusion of the run that was forecast.

    Returns
    -------
    dict
        ``hits`` (failure forecast and seen), ``misses`` (failure seen, not
        forecast), ``false_alarms`` (failure forecast, not seen), ``unknown``
        (forecast jobs absent from the run) and ``correct`` (jobs whose
        failure or pass was forecast right).

    Raises
    ------
    ValueError
        When a forecast outcome is not one of :data:`OUTCOMES`.
    """
    jobs = forecast.get("jobs") or {}
    bad = {name: value for name, value in jobs.items() if value not in OUTCOMES}
    if bad:
        raise ValueError(f"unknown outcomes {bad}; use one of {OUTCOMES}")
    if "pending" in actual.values():
        raise ValueError("the run has not finished; score it once every job has a conclusion")
    decided = {name: c for name, c in actual.items() if c not in UNDECIDED}
    failed = {name for name, c in decided.items() if c not in NOT_FAILED}
    said = {name for name, value in jobs.items() if value in FAILING}
    known = [name for name in jobs if name in decided]
    return {
        "hits": sorted(said & failed),
        "misses": sorted(failed - said),
        "false_alarms": sorted((said & set(decided)) - failed),
        "unknown": sorted(set(jobs) - set(decided)),
        "correct": sum(1 for name in known if (name in said) == (name in failed)),
        "total": len(known),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--workflow", required=True, help="workflow file name, such as ci_wheels_conda.yml")
    b.add_argument("--event", default="schedule", help="the trigger to read (default: schedule)")
    b.add_argument("--branch", help="only runs on this branch, such as main (use with --event push)")
    b.add_argument("--out", type=pathlib.Path)
    s = sub.add_parser("score")
    s.add_argument("forecast", type=pathlib.Path)
    s.add_argument("--run", required=True, type=int, help="the run the forecast was for")
    args = parser.parse_args(argv)
    if args.command == "baseline":
        try:
            data = baseline(args.repo, args.workflow, args.event, args.branch)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        text = json.dumps(data, indent=1)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        print(text)
        return 0
    try:
        result = score(json.loads(args.forecast.read_text(encoding="utf-8")),
                       attempt_jobs(args.repo, args.run))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
