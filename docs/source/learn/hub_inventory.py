#!/usr/bin/env python3
"""
hub_inventory.py — single-source-of-truth tracker for the whole ``learn/`` hub.

Generalises the per-course ``_discovered.json`` idea to every course at once: it
reads the hand-authored ``roadmap`` block of ``hub_inventory.json``, scans each
course's *actual* state from **both** its ``*_inventory.tsv`` and its built
``.rst`` tree, flags where the two disagree, computes **gaps** (roadmap sections
that exist in the plan but have no content yet — e.g. Deep Learning 2–5),
**diffs** against the previous run, prints a dashboard, and (optionally) scaffolds
placeholder skeletons for the gaps.

WHY
    Per-course tracking already exists (each course has ``*_content.py`` +
    ``*_inventory.tsv`` + ``build_*.py`` + ``_buildcheck.py``). What was missing is
    a *hub-level* layer that unifies them, encodes the expected roadmap (including
    not-yet-built sections), detects gaps automatically, and shows what changed
    between runs — so "what do I implement next" is answered by a scan, not by eye.

DATA MODEL  (``hub_inventory.json``)
    roadmap   HAND-AUTHORED. Per course: ``kind`` in {flat, sectioned, mirror} and,
              for sectioned courses, a list of ``sections``. A section maps to a
              tsv ``section``-column key and/or a ``dir`` folder; ``whole_course``
              means "this section == the whole flat course" (used for Deep
              Learning section 1); a section with none of these is roadmap-only and
              becomes a gap. NEVER overwritten by this script.
    actual    GENERATED. Per course/section: tsv count, rst count, mismatch flag.
    gaps      GENERATED. Roadmap sections whose actual content is 0.
    diff      GENERATED. Per-course/section deltas vs the previous run.
    generated GENERATED. ISO timestamp of the scan.

COUNTING RULES (deterministic)
    flat course     tsv = data rows in ``*_inventory.tsv``;
                    rst = ``*.rst`` at the course root excluding ``index.rst``.
    sectioned       per section: tsv = rows whose ``section`` column == the key;
                    rst = ``*.rst`` in the section ``dir`` excluding ``index.rst``.
                    ``whole_course`` section uses the flat-course counts.
    mirror          rst = ``pages`` from ``<course>/_sources_cache/_discovered.json``
                    (tsv not applicable).

SAFETY
    * Read-only by default. ``--scaffold`` only *creates* placeholder ``.rst`` under
      ``<course>/_roadmap/`` for gap sections, never overwriting anything and never
      touching a course's generated lesson tree.
    * Deterministic and idempotent: same tree -> same JSON (bar the timestamp).
    * No third-party dependencies.

USAGE
    python3 hub_inventory.py                 # scan, update JSON, print dashboard
    python3 hub_inventory.py --check         # non-zero exit if any mismatch/gap (CI)
    python3 hub_inventory.py --scaffold      # also stub gap sections under _roadmap/
    python3 hub_inventory.py --json-only     # update JSON, no dashboard
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent
INVENTORY = HUB / "hub_inventory.json"
PREV = HUB / "hub_inventory.prev.json"
LOG = HUB / "hub_inventory_log.jsonl"


# --- small helpers --------------------------------------------------------
def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


def _course_dir(key: str) -> Path:
    """Course key may be nested (e.g. ``hands-on/edtech``)."""
    return HUB / key


def _count_rst(directory: Path) -> int:
    """``.rst`` files directly in ``directory`` excluding ``index.rst``."""
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.glob("*.rst") if p.name != "index.rst")


def _read_tsv(course: Path):
    """
    Return ``(rows, has_section)``.

    For the course's ``*_inventory.tsv`` (or
    ``([], False)`` if none). Rows are dicts keyed by the header.
    """
    tsvs = sorted(course.glob("*_inventory.tsv"))
    if not tsvs:
        return [], False
    with tsvs[0].open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = [r for r in reader]
        has_section = reader.fieldnames is not None and "section" in reader.fieldnames
    return rows, has_section


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --- scanning -------------------------------------------------------------
def scan_course(key: str, spec: dict) -> dict:
    """Scan one course into an ``actual`` record per the counting rules."""
    course = _course_dir(key)
    kind = spec.get("kind", "flat")

    if kind == "mirror":
        disc = _load(course / "_sources_cache" / "_discovered.json", {})
        pages = len(disc.get("pages", []))
        return {"kind": "mirror", "rst_total": pages, "tsv_total": None,
                "mismatch": False,
                "extra": {"fragments": len(disc.get("fragments", [])),
                          "images": len(disc.get("images", []))}}

    rows, has_section = _read_tsv(course)
    tsv_total = len(rows)
    root_rst = _count_rst(course)

    if kind == "flat":
        return {"kind": "flat", "tsv_total": tsv_total, "rst_total": root_rst,
                "mismatch": tsv_total != root_rst}

    # sectioned
    sections = {}
    sum_tsv = sum_rst = 0
    for sec in spec.get("sections", []):
        name = sec["name"]
        if sec.get("whole_course"):
            s_tsv, s_rst = tsv_total, root_rst
        else:
            s_tsv = (sum(1 for r in rows if r.get("section") == sec["tsv"])
                     if has_section and "tsv" in sec else 0)
            s_rst = _count_rst(course / sec["dir"]) if "dir" in sec else 0
        # mismatch only meaningful when at least one source is expected to be > 0
        expected_both = sec.get("whole_course") or ("tsv" in sec and "dir" in sec)
        sections[name] = {"tsv": s_tsv, "rst": s_rst,
                          "mismatch": bool(expected_both and s_tsv != s_rst)}
        sum_tsv += s_tsv
        sum_rst += s_rst
    return {"kind": "sectioned", "tsv_total": sum_tsv, "rst_total": sum_rst,
            "tsv_file_total": tsv_total,
            "mismatch": sum_tsv != sum_rst or (has_section and sum_tsv != tsv_total),
            "sections": sections}


def compute_gaps(roadmap: dict, actual: dict) -> list:
    """Roadmap sections whose actual content is 0 (expected but not built)."""
    gaps = []
    for key, spec in roadmap.items():
        rec = actual.get(key, {})
        if spec.get("kind") == "sectioned":
            for name, s in rec.get("sections", {}).items():
                if s["tsv"] == 0 and s["rst"] == 0:
                    gaps.append({"course": key, "section": name})
        else:
            if (rec.get("rst_total") or 0) == 0 and (rec.get("tsv_total") or 0) == 0:
                gaps.append({"course": key, "section": None})
    return gaps


def compute_diff(prev: dict, cur: dict) -> dict:
    """Per-course delta in total lessons/pages vs the previous scan."""
    diff = {}
    keys = set(prev) | set(cur)
    for k in sorted(keys):
        p = (prev.get(k) or {}).get("rst_total") or 0
        c = (cur.get(k) or {}).get("rst_total") or 0
        if p != c or k not in prev or k not in cur:
            diff[k] = {"was": p, "now": c, "delta": c - p}
    return diff


# --- reporting ------------------------------------------------------------
def dashboard(roadmap: dict, actual: dict, gaps: list, diff: dict) -> str:
    out = ["", "learn/ hub inventory", "=" * 60]
    grand = 0
    for key, spec in roadmap.items():
        rec = actual.get(key, {})
        title = spec.get("title", key)
        if spec.get("kind") == "mirror":
            n = rec.get("rst_total", 0)
            grand += n
            ex = rec.get("extra", {})
            out.append(f"{title} ({n} pages, {ex.get('fragments', 0)} frags, "
                       f"{ex.get('images', 0)} imgs)")
            continue
        total = rec.get("rst_total", 0)
        grand += total
        flag = "  ⚠ tsv≠rst" if rec.get("mismatch") else ""
        out.append(f"{title} ({total}){flag}")
        for name, s in rec.get("sections", {}).items():
            mark = " ⚠" if s["mismatch"] else (" ·gap" if s["tsv"] == 0 and s["rst"] == 0 else "")
            shown = s["rst"] if s["rst"] or not s["tsv"] else s["tsv"]
            out.append(f"    {name} ({shown}){mark}")
    out.append("-" * 60)
    out.append(f"total tracked lessons/pages: {grand}")
    if gaps:
        out.append("")
        out.append(f"GAPS — not yet built ({len(gaps)}):")
        for g in gaps:
            where = f"{g['course']} / {g['section']}" if g["section"] else g["course"]
            out.append(f"    - {where}")
    mism = [k for k, r in actual.items() if r.get("mismatch")]
    if mism:
        out.append("")
        out.append(f"MISMATCHES — tsv vs built .rst ({len(mism)}): {', '.join(mism)}")
    if diff:
        out.append("")
        out.append("CHANGED since last run:")
        for k, d in diff.items():
            out.append(f"    {k}: {d['was']} -> {d['now']} ({d['delta']:+d})")
    out.append("")
    return "\n".join(out)


# --- scaffold -------------------------------------------------------------
def scaffold(gaps: list) -> list:
    """
    Create a placeholder ``.rst`` under ``<course>/_roadmap/`` for each gap section.

    Idempotent; never overwrites; staging only (outside the lesson tree).
    """
    made = []
    for g in gaps:
        if not g["section"]:
            continue
        dest = _course_dir(g["course"]) / "_roadmap" / f"{_slug(g['section'])}.rst"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        title = g["section"]
        bar = "*" * (len(title) + 4)
        dest.write_text(
            f"{bar}\n  {title}\n{bar}\n\n"
            ".. admonition:: Roadmap placeholder\n\n"
            "   Content pending. This section is in the hub roadmap but not yet\n"
            "   built. Populate it through this course's normal generator workflow,\n"
            "   then delete this placeholder. (Created by ``hub_inventory.py "
            "--scaffold``.)\n",
            encoding="utf-8")
        made.append(str(dest.relative_to(HUB)))
    return made


# --- main -----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Scan and track the learn/ hub.")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any mismatch or gap (for CI)")
    ap.add_argument("--scaffold", action="store_true",
                    help="create placeholder stubs for gap sections")
    ap.add_argument("--json-only", action="store_true", help="no dashboard output")
    args = ap.parse_args()

    doc = _load(INVENTORY, None)
    if doc is None or "roadmap" not in doc:
        print(f"error: {INVENTORY.name} with a 'roadmap' block is required.",
              file=sys.stderr)
        return 2
    roadmap = doc["roadmap"]
    prev_actual = doc.get("actual", {}) or {}

    actual = {key: scan_course(key, spec) for key, spec in roadmap.items()}
    gaps = compute_gaps(roadmap, actual)
    diff = compute_diff(prev_actual, actual)

    # snapshot the previous inventory, then write the fresh one (roadmap preserved)
    if doc.get("generated"):
        PREV.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc.update({"actual": actual, "gaps": gaps, "diff": diff, "generated": stamp})
    INVENTORY.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    grand = sum((r.get("rst_total") or 0) for r in actual.values())
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": stamp, "total": grand, "gaps": len(gaps),
                             "totals": {k: (r.get("rst_total") or 0)
                                        for k, r in actual.items()}}) + "\n")

    made = scaffold(gaps) if args.scaffold else []

    if not args.json_only:
        print(dashboard(roadmap, actual, gaps, diff))
        if made:
            print(f"scaffolded {len(made)} placeholder(s):")
            for m in made:
                print(f"    + {m}")
        print(f"wrote {INVENTORY.name} @ {stamp}")

    if args.check and (gaps or any(r.get("mismatch") for r in actual.values())):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
