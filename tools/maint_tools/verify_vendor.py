#!/usr/bin/env python3
# verify_vendor.py
"""
verify_vendor.py
================

Integrity verifier for vendored repositories managed by `vendor_repo.sh`.

Features:
  • Scans recursively for `vendor.lock.json` files.
  • Recomputes the SHA256 tree hash of each vendored directory.
  • Verifies that all recorded metadata (commit_hash, tree_hash) is correct.
  • Outputs a colorized human-readable report
  • JSON report mode integrity report for CI/CD dashboards.
  • Exits with nonzero status if any mismatches are detected.
  • Deterministic & reproducible

Usage:
  python verify_vendor.py [root_dir] [--json] [--pretty]

Human-readable report
python ./tools/maint_tools/verify_vendor.py "/work/scikitplot/cexternals/NumCpp"  # --json --pretty

Compact JSON (for CI logs)
python verify_vendor.py --json > integrity.json

Pretty JSON (for local inspection)
python verify_vendor.py --pretty

Sample JSON output
{
  "timestamp": "2025-10-16T01:30:45.987Z",
  "total": 2,
  "passed": 1,
  "failed": 1,
  "vendors": [
    {
      "path": "scikitplot/externals/array_api_compat",
      "repository": "https://github.com/data-apis/array-api-compat.git",
      "version": "1.12",
      "commit_ok": true,
      "hash_ok": true,
      "expected_commit": "a7b93e4",
      "actual_commit": "a7b93e4",
      "expected_hash": "9af21b...",
      "actual_hash": "9af21b..."
    },
    {
      "path": "scikitplot/externals/array_api_extra",
      "repository": "https://github.com/data-apis/array-api-extra.git",
      "version": "v0.7.1",
      "commit_ok": false,
      "hash_ok": false,
      "expected_commit": "9b732e0",
      "actual_commit": "N/A",
      "expected_hash": "01f3d...",
      "actual_hash": "11c4b..."
    }
  ]
}
"""

from __future__ import annotations
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# Use datetime.UTC if available (Python 3.11+), else fallback to datetime.timezone.utc
UTC = getattr(datetime, "UTC", datetime.timezone.utc)  # datetime.datetime.now(tz=UTC)

# ANSI colors (fallback to plain if not a tty)
# GREEN = "\033[92m" if sys.stdout.isatty() else ""
# RED = "\033[91m" if sys.stdout.isatty() else ""
# YELLOW = "\033[93m" if sys.stdout.isatty() else ""
# BOLD = "\033[1m" if sys.stdout.isatty() else ""
# RESET = "\033[0m" if sys.stdout.isatty() else ""
def _color(s, color):
    if not sys.stdout.isatty():
        return s
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "bold": "\033[1m", "reset": "\033[0m"}
    return f"{colors.get(color, '')}{s}{colors['reset']}"


def compute_tree_hash(directory: Path) -> Tuple[str, str]:
    """
    Compute deterministic SHA256 hash of all files in a directory.

    Returns:
        (mode, hash)
        - mode: "bash-sha256sum" or "python-hashlib"
        - hash: the computed tree hash
    """
    excludes = {"vendor.lock.json", "README.md", ".gitignore"}

    # Try Bash + sha256sum mode if available
    try:
        subprocess.run(["sha256sum", "--version"], check=True, capture_output=True)
        subprocess.run(["find", "--version"], check=True, capture_output=True)
        mode = "bash-sha256sum"

        # Build exclusion expression for bash
        # exclude_args = " ".join(f"\\( -name {f!r} -prune \\) -o" for f in excludes)
        cmd = (
            f"find {directory} -type f "
            f"{' '.join(f'-not -name ' + repr(f) for f in excludes)} "
            r"-print0 | sort -z | xargs -0 sha256sum | sort | sha256sum | awk '{print $1}'"
        )

        hash_value = (
            subprocess.check_output(["bash", "-c", cmd], text=True)
            .strip()
            .split()[-1]
        )
        print(hash_value)
        return mode, hash_value

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to pure Python hashlib implementation
        mode = "python-hashlib"
        hasher = hashlib.sha256()

        # for file in sorted(directory.rglob("*")):
        for path, _, files in os.walk(directory):
            for fname in sorted(files):
                if fname in excludes:
                    continue
                full = Path(path) / fname
                rel = full.relative_to(directory)
                hasher.update(str(rel).encode())
                with open(full, "rb") as fp:
                    while chunk := fp.read(8192):
                        hasher.update(chunk)

        return mode, hasher.hexdigest()


def verify_commit(directory: Path, expected_commit: str) -> Tuple[bool, str]:
    """Verify Git commit hash if .git present (if directory is a git repo)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        actual_commit = result.stdout.strip()
        return actual_commit == expected_commit, actual_commit
    except subprocess.CalledProcessError:
        # Not a Git repo or missing metadata
        return False, "N/A"


def verify_vendor_lock(lock_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate one vendor.lock.json file and return its verification data."""
    with open(lock_path, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    repo_dir = lock_path.parent
    # expected_commit = lock_data.get("commit_hash")
    # ok_commit, actual_commit = verify_commit(repo_dir, expected_commit)

    expected_hash = lock_data.get("tree_hash")
    actual_mode, actual_hash = compute_tree_hash(repo_dir)
    ok_hash = expected_hash == actual_hash

    return (
        ok_hash,
        {
            "path": str(repo_dir),
            "repository": lock_data.get("repository"),
            "version": lock_data.get("version"),
            # "commit_ok": ok_commit,
            # "expected_commit": expected_commit,
            # "actual_commit": actual_commit,
            "hash_ok": ok_hash,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
        },
    )


def print_human_report(results: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print human-readable colored verification results."""
    # print(f"{BOLD}Vendor Integrity Report — {datetime.utcnow().isoformat()} UTC{RESET}")
    print(_color(f"Vendor Integrity Report — {datetime.datetime.now(tz=UTC).isoformat()} UTC", "bold"))
    print("=" * 80)

    for path, info in results.items():
        status_ok = info["hash_ok"]
        color = "green" if status_ok else "red"
        status = "OK" if status_ok else "FAIL"
        print(f"{_color(status, color):<5} {path}")
        print(f"   repo:     {info['repository']}")
        print(f"   version:  {info['version']}")
        # print(f"   commit:   {info['expected_commit']} → {info['actual_commit']}")
        print(f"   hash:     {info['expected_hash']} → {info['actual_hash']}")
        print()

    total = len(results)
    failed = sum(1 for i in results.values() if not i["hash_ok"])
    passed = total - failed

    print(f"{_color('✔ Passed:', 'green')} {passed} / {_color('✘ Failed:', 'red')} {failed} / Total: {total}")
    print("=" * 80)

    # if failed:
    #     sys.exit(2)


def print_json_report(results: Dict[str, Dict[str, Any]], pretty: bool = False) -> None:
    """Print JSON-formatted report (for CI dashboards)."""
    summary = {
        "timestamp": datetime.datetime.now(tz=UTC).isoformat(),
        "total": len(results),
        "passed": sum(1 for i in results.values() if i["hash_ok"]),
        "failed": sum(1 for i in results.values() if not i["hash_ok"]),
        "vendors": list(results.values()),
    }
    if pretty:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, separators=(",", ":")))


def find_lock_files(root: Path) -> list[Path]:
    """Recursively find all vendor.lock.json files."""
    return list(root.rglob("vendor.lock.json"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify integrity of vendored repositories.")
    parser.add_argument("root", nargs="?", default=".", help="Root directory to scan (default: current).")
    parser.add_argument("--json", action="store_true", help="Output results as JSON (for CI dashboards).")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output (implies --json).")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    lock_files = find_lock_files(root)
    if not lock_files:
        print(_color(f"⚠ No vendor.lock.json files found under {root}", "yellow"))
        sys.exit(0)

    results = {}
    for lock_path in lock_files:
        ok, data = verify_vendor_lock(lock_path)
        results[str(lock_path.parent)] = data

    if args.json or args.pretty:
        print_json_report(results, pretty=args.pretty)
    else:
        print_human_report(results)

    failed = any(not i["hash_ok"] for i in results.values())
    sys.exit(2 if failed else 0)


if __name__ == "__main__":
    main()
