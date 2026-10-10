"""Report current Levenshtein maintenance state and open findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
MAINT = ROOT / "maintenances" / "levenshtein"


def review() -> dict[str, object]:
    state = json.loads((MAINT / "_maintenance" / "STATE.json").read_text(encoding="utf-8"))
    review_data = json.loads((MAINT / "REVIEW.json").read_text(encoding="utf-8"))
    open_findings = [f for f in review_data["findings"] if f.get("status") == "open"]
    return {
        "subsystem": "scikitplot.levenshtein",
        "recorded_state": state,
        "open_findings": open_findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = review()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        state = result["recorded_state"]
        print(
            f"runtime={state['runtime_status']} "
            f"maintenance={state['maintenance_status']} "
            f"release={state['release_status']}"
        )
        for finding in result["open_findings"]:
            print(f"{finding['id']}: {finding['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
