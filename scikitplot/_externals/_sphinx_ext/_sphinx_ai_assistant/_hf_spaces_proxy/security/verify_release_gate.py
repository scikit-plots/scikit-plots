#!/usr/bin/env python3
"""One fail-closed command for source policy plus production evidence binding."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import verify_release_evidence  # noqa: E402
import verify_supply_chain  # noqa: E402

logger = logging.getLogger(__name__)


def verify(  # ruff: ignore[undocumented-public-function]
    evidence: Path,
) -> dict[str, object]:
    source = verify_supply_chain.verify()
    release = verify_release_evidence.verify(evidence)
    return {
        "ok": True,
        "source_policy": source,
        "release_evidence": release,
    }


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.evidence)
    except Exception as exc:  # ruff: ignore[blind-except]
        code = getattr(exc, "args", ["RELEASE_GATE_FAILED"])[0] or "RELEASE_GATE_FAILED"
        sys.stderr.write(
            json.dumps({"ok": False, "code": str(code)}, sort_keys=True) + "\n"
        )
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
