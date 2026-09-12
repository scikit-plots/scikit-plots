"""Run deterministic review lenses for one maintained Sphinx-extension subsystem."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from review import build_context, review_package, run_lens, validate_profile


def review_subsystem(manifest_path: Path, jobs: int = 1) -> dict:
    try:
        ctx = build_context(manifest_path)
    except Exception as exc:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        errors = [f"cannot load safe review context: {exc}"]
        return {
            "schema_version": 1,
            "subsystem": manifest.get("subsystem"),
            "runtime_dir": manifest.get("runtime_dir"),
            "status": "BLOCKED",
            "release_promotion": "BLOCKED",
            "counts": {"ERROR": 1, "WARNING": 0, "UNAVAILABLE": 0, "INFO": 0},
            "profile_errors": errors,
            "package_reviews": [],
            "lenses": [],
        }
    errors = validate_profile(ctx.profile, ctx.manifest)
    if errors:
        return {
            "schema_version": 1,
            "subsystem": ctx.manifest.get("subsystem"),
            "runtime_dir": ctx.manifest.get("runtime_dir"),
            "status": "BLOCKED",
            "release_promotion": "BLOCKED",
            "counts": {"ERROR": len(errors), "WARNING": 0, "UNAVAILABLE": 0, "INFO": 0},
            "profile_errors": errors,
            "package_reviews": [],
            "lenses": [],
        }
    lenses = list(ctx.profile["lenses"])
    targets = list(ctx.profile["package_targets"])
    workers = max(1, min(jobs, max(len(lenses), len(targets))))
    if workers == 1:
        results = [run_lens(ctx, lens) for lens in lenses]
        package_results = [review_package(ctx, package) for package in targets]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sphinx-review") as pool:
            results = list(pool.map(lambda lens: run_lens(ctx, lens), lenses))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sphinx-package-review") as pool:
            package_results = list(pool.map(lambda package: review_package(ctx, package), targets))
    required_blocked = any(lens["required"] and lens["status"] == "BLOCKED" for lens in results)
    package_blocked = any(package["status"] == "BLOCKED" for package in package_results)
    policy = ctx.profile["pr_policy"]
    unavailable = sum(
        1 for lens in results for finding in lens["findings"] if finding["severity"] == "UNAVAILABLE"
    )
    counts = {severity: 0 for severity in ("ERROR", "WARNING", "UNAVAILABLE", "INFO")}
    for collection in (results, package_results):
        for item in collection:
            for finding in item["findings"]:
                counts[finding["severity"]] += 1
    pr_ready = not required_blocked and not package_blocked
    release_promotable = pr_ready and not (
        policy.get("unavailable_release_gate_blocks_promotion", True) and unavailable
    )
    return {
        "schema_version": 1,
        "subsystem": ctx.manifest.get("subsystem"),
        "runtime_dir": ctx.manifest.get("runtime_dir"),
        "status": "PR_READY" if pr_ready else "BLOCKED",
        "release_promotion": "ELIGIBLE" if release_promotable else "BLOCKED",
        "counts": counts,
        "package_reviews": package_results,
        "lenses": results,
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# Review — `{report.get('runtime_dir', report.get('subsystem'))}`",
        "",
        f"PR status: **{report.get('status')}**  ",
        f"Release promotion: **{report.get('release_promotion', 'BLOCKED')}**",
        "",
    ]
    if report.get("profile_errors"):
        lines.append("## Profile errors")
        lines.extend(f"- {e}" for e in report["profile_errors"])
        return "\n".join(lines) + "\n"
    if report.get("package_reviews"):
        lines += ["## Runtime package reviews", ""]
        for package in report["package_reviews"]:
            deps = ", ".join(package.get("observed_sibling_imports", [])) or "none"
            lines.append(f"- `{package['package']}` — **{package['status']}**; Python files: {package.get('python_files', 0)}; observed sibling imports: {deps}.")
            for finding in package.get("findings", []):
                lines.append(f"  - **{finding['severity']} `{finding['code']}`** — {finding['message']} Evidence: `{finding['evidence']}`. Remediation: {finding['remediation']}")
        lines.append("")
    for lens in report.get("lenses", []):
        lines += [f"## {lens['title']} — {lens['status']}", "", str(lens.get("focus", "")), ""]
        if not lens["findings"]:
            lines.append("No findings.")
            lines.append("")
            continue
        for finding in lens["findings"]:
            lines.append(
                f"- **{finding['severity']} `{finding['code']}`** — {finding['message']} "
                f"Evidence: `{finding['evidence']}`. Remediation: {finding['remediation']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--require-release", action="store_true", help="fail if release promotion is blocked, including unavailable required release evidence")
    args = parser.parse_args()
    report = review_subsystem(args.manifest, args.jobs)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.format == "markdown":
        print(_markdown(report), end="")
    else:
        print(
            f"{report.get('runtime_dir', report.get('subsystem'))}: "
            f"{report['status']} / release {report.get('release_promotion', 'BLOCKED')}"
        )
        for package in report.get("package_reviews", []):
            print(
                f"  {package['status']:7} package:{package['package']} "
                f"({len(package['findings'])} findings)"
            )
        for lens in report.get("lenses", []):
            print(f"  {lens['status']:7} {lens['id']} ({len(lens['findings'])} findings)")
            for finding in lens["findings"]:
                print(f"    {finding['severity']:11} {finding['code']}: {finding['message']}")
        for error in report.get("profile_errors", []):
            print(f"  ERROR profile: {error}")
    return 1 if report["status"] == "BLOCKED" or (args.require_release and report.get("release_promotion") != "ELIGIBLE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
