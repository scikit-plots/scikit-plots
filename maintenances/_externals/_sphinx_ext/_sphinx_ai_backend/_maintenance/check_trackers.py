"""
Maintenance drift checker for ``_sphinx_ai_backend``.

No third-party dependencies are required.

This submodule is **proposed**: its maintenance control plane exists, but the
deployable service code it will own still lives inside ``_sphinx_ai_assistant``.
The checker is written around that fact. While ``proposed`` is true it verifies
that the recorded service paths are *absent here and present there*, and it
**recomputes** every physical total rather than trusting the recorded number.

User note
---------
Run it from anywhere; paths are resolved from this file's location::

    python scikitplot/_externals/_sphinx_ext/_sphinx_ai_backend/_maintenance/check_trackers.py

Exit status is ``0`` when the submodule is coherent and ``1`` otherwise. Every
failure line names the artifact and the exact disagreement.

Developer note
--------------
This file was previously a byte-identical copy of the ``_sphinx_ai_assistant``
checker. That copy announced the wrong subsystem and enforced the assistant's
required-file list and ``STATE.json`` key set against this directory, so it
could never pass — it reported 23 errors by construction. A gate that cannot go
green teaches maintainers to ignore it, so it was replaced rather than tuned.

The counting rule for physical totals is stated once, here, and applied
everywhere: ``source_loc_all_files`` counts newlines in every regular file under
a recorded path, and ``source_loc_code_only`` counts newlines in files whose
suffix is in :data:`CODE_SUFFIXES`. Both are recorded and both are gated,
because a single ambiguous figure is what allowed the earlier ``9 896`` total to
be assembled from two different rules and drift unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMODULE_ROOT = HERE.parent
FAMILY_ROOT = SUBMODULE_ROOT.parent

MODULE = "scikitplot._externals._sphinx_ext._sphinx_ai_backend"
CLIENT = "_sphinx_ai_assistant"

REQUIRED = [
    "DEPENDENCY_MAP.md",
    "MAINTENANCE_MODEL.md",
    "README.md",
    "SUBMODULE_STRUCTURE.md",
    "TRACKER_LOGICAL.md",
    "TRACKER_PHYSICAL.md",
    "VERIFICATION.md",
    "HISTORY.md",
    "TRACKER.json",
    "STATE.json",
]
REQUIRED_SIBLING = ["MAINTAINING.md"]

STATE_KEYS = [
    "schema_version",
    "module",
    "role",
    "proposed",
    "family",
    "source",
    "maintenance_set",
    "physical_totals",
    "open_findings",
    "unverified_claims",
    "fresh_chat_read_order",
    "next_exact_action",
]
TRACKER_KEYS = [
    "schema_version",
    "module",
    "role",
    "proposed",
    "family",
    "source",
    "physical",
    "antecessors",
    "successors",
]
FINDING_KEYS = ["id", "severity", "title", "evidence", "fix"]

#: Suffixes counted as executable service code.
CODE_SUFFIXES = {".py", ".js"}

#: Directory names that never contribute to physical totals.
EPHEMERAL_PARTS = {"__pycache__", ".pytest_cache", ".git"}

#: File suffixes that never contribute to physical totals.
EPHEMERAL_SUFFIXES = {".pyc", ".pyo"}


def fail(msg: str, errors: list[str]) -> None:
    """Record a drift failure."""
    errors.append(msg)


def load_json(path: Path, errors: list[str]) -> dict | None:
    """Parse JSON, reporting syntax errors as drift rather than raising."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"unreadable JSON: {path.name}: {exc}", errors)
        return None


def is_countable(path: Path) -> bool:
    """Return whether ``path`` contributes to a physical total."""
    if not path.is_file():
        return False
    if path.suffix in EPHEMERAL_SUFFIXES:
        return False
    return not EPHEMERAL_PARTS.intersection(path.parts)


def line_count(path: Path) -> int:
    """Count newline-delimited lines, tolerating undecodable bytes."""
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def collect(paths: list[str], root: Path) -> list[Path]:
    """Return every countable file under the recorded service ``paths``."""
    files: list[Path] = []
    for rel in paths:
        target = root / rel
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(target.rglob("*"))
    return [p for p in files if is_countable(p)]


def measure(paths: list[str], root: Path) -> dict:
    """
    Recompute physical totals for ``paths`` relative to ``root``.

    Parameters
    ----------
    paths : list of str
        Recorded service paths, each a file or directory name.
    root : pathlib.Path
        Directory the paths are resolved against.

    Returns
    -------
    dict
        Keys ``source_files``, ``source_loc_all_files``, ``code_files``,
        ``source_loc_code_only`` and ``by_extension``.
    """
    files = collect(paths, root)
    code = [p for p in files if p.suffix in CODE_SUFFIXES]
    by_extension: dict[str, int] = {}
    for p in files:
        by_extension[p.suffix] = by_extension.get(p.suffix, 0) + 1
    return {
        "source_files": len(files),
        "source_loc_all_files": sum(line_count(p) for p in files),
        "code_files": len(code),
        "source_loc_code_only": sum(line_count(p) for p in code),
        "by_extension": by_extension,
    }


def check_presence(errors: list[str]) -> None:
    """Verify the maintenance control plane exists."""
    for name in REQUIRED:
        if not (HERE / name).is_file():
            fail(f"missing maintenance file: {name}", errors)
    for name in REQUIRED_SIBLING:
        if not (SUBMODULE_ROOT / name).is_file():
            fail(f"missing submodule entry point: {name}", errors)


def check_identity(state: dict | None, tracker: dict | None, errors: list[str]) -> None:
    """Verify STATE/TRACKER shape, module identity and anchor agreement."""
    for doc, keys, label in (
        (state, STATE_KEYS, "STATE.json"),
        (tracker, TRACKER_KEYS, "TRACKER.json"),
    ):
        if doc is None:
            continue
        for key in keys:
            if key not in doc:
                fail(f"{label} missing key: {key}", errors)
        if doc.get("module") != MODULE:
            fail(f"{label} module identity drift: {doc.get('module')!r}", errors)
    if state and tracker:
        if state.get("proposed") != tracker.get("proposed"):
            fail("STATE.json and TRACKER.json disagree on `proposed`", errors)
        if state.get("source", {}).get("sha256") != tracker.get("source", {}).get(
            "sha256"
        ):
            fail("STATE.json and TRACKER.json record different source anchors", errors)


def check_proposed_invariant(tracker: dict | None, errors: list[str]) -> list[str]:
    """
    Verify the service paths sit on the side the tracker claims.

    While ``proposed`` is true the code must be absent here and present in the
    client submodule. A path found on both sides is a half-completed move, which
    is the failure this check exists to make impossible to miss.
    """
    if not tracker:
        return []
    paths = tracker.get("physical", {}).get("paths", [])
    if not paths:
        fail("TRACKER.json records no service paths", errors)
        return []
    proposed = bool(tracker.get("proposed"))
    client_root = FAMILY_ROOT / CLIENT
    if not client_root.is_dir():
        fail(f"client submodule not found: {CLIENT}", errors)
        return paths
    for rel in paths:
        here_side = (SUBMODULE_ROOT / rel).exists()
        there_side = (client_root / rel).exists()
        if here_side and there_side:
            fail(
                f"service path exists on BOTH sides, the move is half-done: {rel}",
                errors,
            )
        elif proposed and not there_side:
            fail(f"proposed service path missing from {CLIENT}: {rel}", errors)
        elif proposed and here_side:
            fail(
                f"service path already moved here while `proposed` is still true: {rel}",
                errors,
            )
        elif not proposed and not here_side:
            fail(
                f"service path missing after the move was recorded complete: {rel}",
                errors,
            )
    return paths


def check_physical(
    tracker: dict | None, state: dict | None, paths: list[str], errors: list[str]
) -> None:
    """Recompute physical totals and gate the recorded values against them."""
    if not tracker or not paths:
        return
    root = FAMILY_ROOT / CLIENT if tracker.get("proposed") else SUBMODULE_ROOT
    actual = measure(paths, root)
    recorded = tracker.get("physical", {}).get("totals", {})
    for key in (
        "source_files",
        "source_loc_all_files",
        "code_files",
        "source_loc_code_only",
    ):
        if key not in recorded:
            fail(f"TRACKER.json physical totals missing key: {key}", errors)
        elif recorded[key] != actual[key]:
            fail(
                f"physical total drift: {key} recorded {recorded[key]}, "
                f"measured {actual[key]}",
                errors,
            )
    recorded_ext = tracker.get("physical", {}).get("by_extension", {})
    if recorded_ext != actual["by_extension"]:
        fail(
            f"by_extension drift: recorded {recorded_ext}, measured {actual['by_extension']}",
            errors,
        )

    # The zero-test claim is this submodule's headline finding, so recompute it
    # rather than trusting the recorded figure.
    tests = [
        p
        for p in collect(paths, root)
        if p.name.startswith("test_") or p.name.endswith("_test.py")
    ]
    if recorded.get("test_files") != len(tests):
        fail(
            f"test_files drift: recorded {recorded.get('test_files')}, measured {len(tests)}",
            errors,
        )
    if state is not None:
        state_files = state.get("physical_totals", {}).get("source_files")
        if state_files != actual["source_files"]:
            fail(
                f"STATE.json physical_totals.source_files is {state_files}, "
                f"measured {actual['source_files']}",
                errors,
            )


def check_findings(state: dict | None, errors: list[str]) -> None:
    """Verify the open findings and unverified claims stay well-formed."""
    if not state:
        return
    findings = state.get("open_findings", [])
    if not isinstance(findings, list):
        fail("STATE.json open_findings must be a list", errors)
        return
    seen: set[str] = set()
    for entry in findings:
        if not isinstance(entry, dict):
            fail("STATE.json open_findings entry is not an object", errors)
            continue
        for key in FINDING_KEYS:
            if not entry.get(key):
                fail(f"finding {entry.get('id', '?')} missing field: {key}", errors)
        fid = entry.get("id")
        if fid in seen:
            fail(f"duplicate finding id: {fid}", errors)
        seen.add(fid)
    if state.get("proposed") and not state.get("unverified_claims"):
        fail(
            "unverified_claims may not be empty while the submodule is proposed", errors
        )


def check_isolation(errors: list[str]) -> None:
    """
    Verify the backend stays a service: imported by nothing, importing nothing.

    The family contract is that the assistant reaches the backend over HTTP. A
    Python import in either direction would silently convert a deployment
    boundary into a packaging one.
    """
    this_file = Path(__file__).resolve()
    unexpected = sorted(
        p.name
        for p in SUBMODULE_ROOT.rglob("*.py")
        if is_countable(p) and p.resolve() != this_file
    )
    if unexpected:
        fail(f"unexpected Python in the maintenance-only shell: {unexpected}", errors)
    for sibling in sorted(FAMILY_ROOT.iterdir()):
        if not sibling.is_dir() or sibling.name == SUBMODULE_ROOT.name:
            continue
        for path in sibling.rglob("*.py"):
            if not is_countable(path):
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if (
                    stripped.startswith(("import ", "from "))
                    and "_sphinx_ai_backend" in stripped
                ):
                    rel = path.relative_to(FAMILY_ROOT)
                    fail(
                        f"family member imports the backend, which must stay HTTP-only: {rel}",
                        errors,
                    )
                    break


def main() -> int:
    """
    Run every check and report drift.

    Returns
    -------
    int
        ``0`` when no drift was found, ``1`` otherwise.
    """
    errors: list[str] = []
    check_presence(errors)
    state = (
        load_json(HERE / "STATE.json", errors)
        if (HERE / "STATE.json").is_file()
        else None
    )
    tracker = (
        load_json(HERE / "TRACKER.json", errors)
        if (HERE / "TRACKER.json").is_file()
        else None
    )
    check_identity(state, tracker, errors)
    paths = check_proposed_invariant(tracker, errors)
    check_physical(tracker, state, paths, errors)
    check_findings(state, errors)
    check_isolation(errors)

    if errors:
        print("_sphinx_ai_backend maintenance drift: FAIL")  # ruff: ignore[print]
        for message in errors:
            print(f" - {message}")  # ruff: ignore[print]
        return 1
    print("_sphinx_ai_backend maintenance drift: GREEN")  # ruff: ignore[print]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
