"""Review all maintained Sphinx-extension subsystems independently, then reconcile."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from review_subsystem import review_subsystem, _markdown as subsystem_markdown

FAMILY = HERE.parent.parent


def review_all(jobs: int = 1) -> dict:
    manifests = sorted(p for p in FAMILY.glob("_*/MAINTENANCE.json") if p.parent.name != "_maintenance_core")
    workers = max(1, min(jobs, len(manifests) or 1))
    if workers == 1:
        reports = [review_subsystem(path, jobs=1) for path in manifests]
    else:
        # Parallelize at subsystem boundary.  Lenses inside each subsystem run serially
        # here to avoid nested unbounded pools; direct review_subsystem supports lens-level
        # parallelism when a maintainer wants to review one subsystem independently.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sphinx-subsystem-review") as pool:
            reports = list(pool.map(lambda p: review_subsystem(p, jobs=1), manifests))
    counts = {severity: 0 for severity in ("ERROR", "WARNING", "UNAVAILABLE", "INFO")}
    for report in reports:
        for severity, value in report.get("counts", {}).items():
            counts[severity] = counts.get(severity, 0) + value
    blocked = [r.get("runtime_dir") for r in reports if r.get("status") == "BLOCKED"]
    release_blocked = [r.get("runtime_dir") for r in reports if r.get("release_promotion") == "BLOCKED"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "independent_then_reconcile",
        "subsystems": reports,
        "summary": {
            "status": "PR_READY" if not blocked else "BLOCKED",
            "release_promotion": "ELIGIBLE" if not release_blocked else "BLOCKED",
            "blocked_subsystems": blocked,
            "release_blocked_subsystems": release_blocked,
            "counts": counts,
        },
    }


def _markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# `_sphinx_ext` independent review reconciliation",
        "",
        f"Family PR status: **{summary['status']}**  ",
        f"Family release promotion: **{summary['release_promotion']}**  ",
        f"Findings: {summary['counts']}",
        "",
        "Each subsystem was reviewed independently and only then reconciled. "
        "UNAVAILABLE evidence does not automatically block PR review, but it blocks release promotion when the subsystem policy requires it.",
        "",
    ]
    for subsystem in report["subsystems"]:
        rendered = subsystem_markdown(subsystem).strip().splitlines()
        lines += ["---", ""] + rendered + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--require-release", action="store_true", help="fail if release promotion is blocked, including unavailable required release evidence")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = review_all(args.jobs)
    if args.format == "json":
        output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    elif args.format == "markdown":
        output = _markdown(report)
    else:
        s = report["summary"]
        output = f"_sphinx_ext review: {s['status']} / release {s['release_promotion']} / findings {s['counts']}\n"
        for item in report["subsystems"]:
            output += f" - {item.get('runtime_dir')}: {item['status']} / release {item.get('release_promotion')}\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 1 if report["summary"]["status"] == "BLOCKED" or (args.require_release and report["summary"].get("release_promotion") != "ELIGIBLE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
