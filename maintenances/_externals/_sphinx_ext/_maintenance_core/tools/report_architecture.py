"""Print the auto-discovered Sphinx-extension architecture graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from architecture import architecture_snapshot
from paths import discover_runtime_sphinx_ext

FAMILY = HERE.parent.parent


def _manifests() -> list[Path]:
    return [
        p for p in sorted(FAMILY.glob("_*/MAINTENANCE.json"))
        if p.parent.name != "_maintenance_core"
    ]


def _text(snapshot: dict) -> str:
    lines = ["Sphinx extension architecture", "", "Capabilities:"]
    for item in snapshot["capabilities"]:
        lines.append(f" - {item['id']} -> {item['owner']} (steward: {item['steward']})")
    lines.extend(["", "Typed dependency edges:"])
    for edge in snapshot["declared_edges"]:
        caps = ", ".join(edge["capabilities"])
        lines.append(
            f" - {edge['source']} -> {edge['target']} [{edge['kind']}; {edge['evidence']}] ({caps})"
        )
    lines.extend(["", "Effective runtime graph:"])
    for source, targets in snapshot["effective_runtime_graph"].items():
        lines.append(f" - {source}: {', '.join(targets) if targets else '-'}")
    lines.append("")
    lines.append("Cycle: " + (" -> ".join(snapshot["cycle"]) if snapshot["cycle"] else "none"))
    return "\n".join(lines)


def _mermaid(snapshot: dict) -> str:
    lines = ["graph TD"]
    for source, targets in snapshot["effective_runtime_graph"].items():
        if not targets:
            lines.append(f"    {source}")
        for target in targets:
            lines.append(f"    {source} --> {target}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("text", "json", "mermaid"), default="text")
    args = parser.parse_args()
    runtime = discover_runtime_sphinx_ext(FAMILY)
    snapshot = architecture_snapshot(_manifests(), runtime)
    if args.format == "json":
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    elif args.format == "mermaid":
        print(_mermaid(snapshot))
    else:
        print(_text(snapshot))
    return 1 if snapshot["cycle"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
