"""
Maintenance drift checker for ``_sphinx_llm``.

No third-party dependencies are required. The checker validates durable
maintenance truth, vendored-source integrity, and architectural boundaries.
Behavioral/upstream parity tests remain in ``VERIFICATION.md`` and numbered
checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIST = HERE / "history"


def mpath(rel: str | Path) -> Path:
    """
    Resolve a maintenance-relative path, falling back to ``history/``.

    The A02 working set was deliberately filed under ``_maintenance/history/``.
    Resolution is two-step so the recorded *path strings* in STATE.json,
    A02_MATRIX_PLAN.json and UPSTREAM_COMPATIBILITY_BASELINE.json stay
    byte-identical: only lookup gains a fallback, never a recorded digest.
    """
    direct = HERE / rel
    return direct if direct.exists() else HIST / rel


def rpath(rel: str | Path) -> Path:
    """Resolve a repository-root-relative path with the same history fallback."""
    direct = ROOT / rel
    if direct.exists():
        return direct
    parts = Path(rel).parts
    if parts and parts[0] == "_maintenance":
        return HIST.joinpath(*parts[1:])
    return direct


REQUIRED = [
    "RECONCILIATION.md",
    "MAINTENANCE_MODEL.md",
    "RULESET.md",
    "UPSTREAM.md",
    "DEPENDENCY_MAP.md",
    "STANDARDS_BASELINE.md",
    "BUILD_FLOW.md",
    "TRACKER_LOGICAL.md",
    "TRACKER_PHYSICAL.md",
    "TRACKER.json",
    "STATE.json",
    "VENDOR_BASELINE.json",
    "SUBMODULE_STRUCTURE.md",
    "DIRECTIVE_COMPATIBILITY.md",
    "ARTIFACT_CONTRACT.md",
    "SECURITY_MODEL.md",
    "REGISTRY.md",
    "VERIFICATION.md",
    "CHECKPOINT_TEMPLATE.md",
    "HISTORY.md",
    "UPSTREAM_CI_CORROBORATION.md",
    "UPSTREAM_CI_BASELINE.json",
    "UPSTREAM_TEST_HARNESS.md",
    "UPSTREAM_TEST_ENVIRONMENT.json",
    "UPSTREAM_TEST_LOCKSET.json",
    "UPSTREAM_COMPATIBILITY.md",
    "UPSTREAM_COMPATIBILITY_BASELINE.json",
    "run_upstream_tests.py",
    "verify_upstream_compatibility.py",
    "prepare_upstream_test_environment.py",
    "run_a02_config_parity.py",
    "A02_MATRIX_EXECUTION.md",
    "A02_MATRIX_PLAN.json",
    "run_a02_matrix.py",
    "ci/run_a02_cell.sh",
    # DORMANT under A02 DEFERRED_PERMANENTLY: the CircleCI transport is retired,
    # but its baseline stays required so the integrated-and-verified workflow
    # digest remains recorded rather than forgotten.
    "CIRCLECI_INTEGRATION.md",
    "CIRCLECI_INTEGRATION_BASELINE.json",
    "verify_a02_circleci_integration.py",
    "render_a02_circleci_rebase.py",
    "A02_CLOSURE_EVIDENCE.md",
    "verify_a02_closure_evidence.py",
    "A02_RECONCILIATION_READINESS.md",
    "prepare_a02_reconciliation.py",
]
REQUIRED_SCHEMAS = [
    "state.schema.json",
    "tracker.schema.json",
    "checkpoint.schema.json",
    "manifest.schema.json",
    "compatibility.schema.json",
    "provenance.schema.json",
    "llms-config.schema.json",
    "vendor-baseline.schema.json",
    "upstream-test-environment.schema.json",
    "upstream-test-lockset.schema.json",
    "upstream-ci-baseline.schema.json",
    "upstream-compatibility-baseline.schema.json",
    "a02-matrix-plan.schema.json",
    "a02-parity-cell-evidence.schema.json",
    "a02-parity-matrix-evidence.schema.json",
    "circleci-integration-baseline.schema.json",
    "a02-closure-decision.schema.json",
    "a02-reconciliation-readiness.schema.json",
]
ALLOWED_CHECKPOINT_STATUS = {
    "NOT_STARTED",
    "IN_PROGRESS",
    "BLOCKED",
    "COMPLETE",
    "DEFERRED",
    "DEFERRED_PERMANENTLY",
    "SUPERSEDED",
}
# Checkpoints whose closure proof was permanently abandoned. Every later
# checkpoint that closes while one of these is deferred must say so explicitly.
DEFERRED_STATUSES = {"DEFERRED_PERMANENTLY"}
EXPECTED_VENDOR_FILES = {
    "LICENSE",
    "LICENSE_HEADER",
    "__init__.py",
    "docref.py",
    "markdown_builder.py",
    "summary.py",
    "txt.py",
    "version.py",
    "vendor.lock.json",
}
EPHEMERAL_SUFFIXES = {".pyc", ".pyo"}
EPHEMERAL_PARTS = {"__pycache__", ".pytest_cache"}


def fail(  # ruff: ignore[undocumented-public-function]
    msg: str,
    errors: list[str],
) -> None:
    errors.append(msg)


def load_json(  # ruff: ignore[undocumented-public-function]
    path: Path,
    errors: list[str],
):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # ruff: ignore[blind-except]
        fail(f"invalid JSON {path}: {exc}", errors)
        return None


def sha256_file(path: Path) -> str:  # ruff: ignore[undocumented-public-function]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_ephemeral(rel: Path) -> bool:  # ruff: ignore[undocumented-public-function]
    return (
        any(part in EPHEMERAL_PARTS for part in rel.parts)
        or rel.suffix in EPHEMERAL_SUFFIXES
    )


def portable_vendor_hash(records: list[tuple[str, str]]) -> str:
    r"""Hash canonical ``<file-sha>  <relative-posix-path>\n`` records."""
    h = hashlib.sha256()
    canonical = sorted(f"{digest}  {rel}\n" for rel, digest in records)
    for record in canonical:
        h.update(record.encode("utf-8"))
    return h.hexdigest()


def a02_implementation_hash(plan: dict, errors: list[str]) -> str | None:
    """Recompute the implementation fingerprint declared by the A02 plan."""
    spec = plan.get("implementation_fingerprint", {})
    if spec.get("algorithm") != "sha256-sorted-relative-file-digests-v1":
        fail("A02 matrix plan implementation fingerprint algorithm drift", errors)
        return None
    rels = spec.get("files", [])
    if not isinstance(rels, list) or not rels:
        fail("A02 matrix plan implementation fingerprint has no files", errors)
        return None
    records: list[str] = []
    for raw in rels:
        rel = Path(str(raw))
        if rel.is_absolute() or ".." in rel.parts:
            fail(f"A02 matrix plan fingerprint path is unsafe: {raw!r}", errors)
            return None
        path = rpath(rel)
        if not path.is_file():
            fail(f"A02 matrix plan fingerprint file missing: {raw}", errors)
            return None
        # The recorded path string, not the resolved location, is hashed, so the
        # history/ fallback above cannot alter a recorded fingerprint.
        records.append(f"{sha256_file(path)}  {rel.as_posix()}\n")
    return hashlib.sha256("".join(sorted(records)).encode("utf-8")).hexdigest()


def main() -> int:  # ruff: ignore[too-many-branches, undocumented-public-function]
    errors: list[str] = []

    for name in REQUIRED:
        if not mpath(name).is_file():
            fail(f"missing maintenance file: {name}", errors)
    for name in REQUIRED_SCHEMAS:
        p = mpath(Path("schemas") / name)
        if not p.is_file():
            fail(f"missing schema: schemas/{name}", errors)
        else:
            load_json(p, errors)

    # Fail-closed: a missing evidence file is an ERROR, never a skipped check.
    # Previously each of these was `… if (HERE / X).exists() else None`, so
    # relocating an evidence file silently disabled every check that consumed
    # it. Only STATE.json/TRACKER.json keep the tolerant form, because their
    # absence is already reported by the REQUIRED loop above and the remaining
    # checks degrade explicitly rather than silently.
    def evidence(name: str):
        p = mpath(name)
        if not p.is_file():
            fail(
                f"required A0x evidence file is missing, checks depending on it cannot run: {name}",
                errors,
            )
            return None
        return load_json(p, errors)

    state = (
        load_json(mpath("STATE.json"), errors) if mpath("STATE.json").exists() else None
    )
    if state:
        checkpoints = state.get("checkpoints", {})
        unproved = sorted(
            cid
            for cid, rec in checkpoints.items()
            if isinstance(rec, dict) and rec.get("status") in DEFERRED_STATUSES
        )
        if unproved:
            for cid, rec in sorted(checkpoints.items()):
                if not isinstance(rec, dict) or rec.get("status") != "COMPLETE":
                    continue
                if cid <= max(unproved):
                    continue
                declared = rec.get("depends_on_unproved")
                if not isinstance(declared, list) or not set(unproved) <= set(declared):
                    fail(
                        f"checkpoint {cid} is COMPLETE while {', '.join(unproved)} "
                        f"is permanently deferred; it must record "
                        f"depends_on_unproved containing {unproved}",
                        errors,
                    )
    tracker = (
        load_json(mpath("TRACKER.json"), errors)
        if mpath("TRACKER.json").exists()
        else None
    )
    baseline = evidence("VENDOR_BASELINE.json")
    test_environment = evidence("UPSTREAM_TEST_ENVIRONMENT.json")
    test_lockset = evidence("UPSTREAM_TEST_LOCKSET.json")
    ci_baseline = evidence("UPSTREAM_CI_BASELINE.json")
    compatibility = evidence("UPSTREAM_COMPATIBILITY_BASELINE.json")
    a02_plan = evidence("A02_MATRIX_PLAN.json")
    circleci = evidence("CIRCLECI_INTEGRATION_BASELINE.json")

    if state:
        required_state = [
            "schema_version",
            "subsystem",
            "source_anchor",
            "governing_rule",
            "phase",
            "active_checkpoint",
            "production_code_modified",
            "checkpoints",
            "verification_snapshot",
            "next_actions",
        ]
        for key in required_state:
            if key not in state:
                fail(f"STATE.json missing key: {key}", errors)
        sha = state.get("source_anchor", {}).get("sha256", "")
        if sha and not re.fullmatch(r"[0-9a-f]{64}", sha):
            fail("STATE.json source sha256 is not 64 lowercase hex chars", errors)

        checkpoints = state.get("checkpoints", {})
        in_progress = []
        for key, value in checkpoints.items():
            status = value.get("status")
            if status not in ALLOWED_CHECKPOINT_STATUS:
                fail(
                    f"STATE.json checkpoint {key} has invalid status: {status}", errors
                )
            if status == "IN_PROGRESS":
                in_progress.append(key)
        if len(in_progress) > 1:
            fail(f"multiple checkpoints IN_PROGRESS: {in_progress}", errors)
        active = state.get("active_checkpoint")
        if active is not None:
            if active not in checkpoints:
                fail(f"active_checkpoint not present in checkpoints: {active}", errors)
            elif checkpoints[active].get("status") != "IN_PROGRESS":
                fail(f"active_checkpoint {active} is not IN_PROGRESS", errors)
        elif in_progress:
            fail(
                f"checkpoint is IN_PROGRESS but active_checkpoint is null: {in_progress[0]}",
                errors,
            )

    if state and tracker:
        state_anchor = state.get("source_anchor", {})
        tracker_anchor = tracker.get("source_anchor", {})
        for anchor_key in ("archive", "sha256", "lineage_archive", "lineage_sha256"):
            if state_anchor.get(anchor_key) != tracker_anchor.get(anchor_key):
                fail(
                    f"STATE/TRACKER source_anchor mismatch for {anchor_key}: "
                    f"{state_anchor.get(anchor_key)!r} != {tracker_anchor.get(anchor_key)!r}",
                    errors,
                )

    if tracker:
        for key in [
            "schema_version",
            "subsystem",
            "source_anchor",
            "upstreams",
            "physical",
            "logical_contracts",
        ]:
            if key not in tracker:
                fail(f"TRACKER.json missing key: {key}", errors)
        ids = [c.get("id") for c in tracker.get("logical_contracts", [])]
        if len(ids) != len(set(ids)):
            fail("TRACKER.json has duplicate logical contract IDs", errors)

    vendor = ROOT / "sphinx_llm"
    lock_path = vendor / "vendor.lock.json"
    if lock_path.is_file():
        lock = load_json(lock_path, errors)
        missing = sorted(
            name for name in EXPECTED_VENDOR_FILES if not (vendor / name).is_file()
        )
        if missing:
            fail(f"vendored baseline missing expected files: {missing}", errors)
        if not (vendor / "tests").is_dir():
            fail("vendored baseline missing tests/", errors)
        if lock:
            for key in [
                "repository",
                "commit_hash",
                "tree_mode",
                "tree_hash",
                "generated_utc",
            ]:
                if key not in lock:
                    fail(f"vendor.lock.json missing key: {key}", errors)
            tree_hash = lock.get("tree_hash", "")
            if tree_hash and not re.fullmatch(r"[0-9a-f]{64}", tree_hash):
                fail("vendor.lock.json tree_hash is not 64 lowercase hex chars", errors)

        if state:
            up_status = str(state.get("upstream_anchor", {}).get("status", ""))
            tests_status = str(
                state.get("verification_snapshot", {}).get("upstream_tests", "")
            )
            if "NOT_VENDORED" in up_status or "NOT_VENDORED" in tests_status:
                fail(
                    "STATE.json contradicts existing vendor tree with NOT_VENDORED status",
                    errors,
                )
        if tracker:
            impl = str(tracker.get("physical", {}).get("implementation_status", ""))
            if "NOT_YET_VENDORED" in impl:
                fail(
                    "TRACKER.json says NOT_YET_VENDORED although vendor.lock.json exists",
                    errors,
                )
            nvidia = next(
                (
                    u
                    for u in tracker.get("upstreams", [])
                    if u.get("name") == "NVIDIA/sphinx-llm"
                ),
                None,
            )
            if nvidia and "NOT_VENDORED" in str(nvidia.get("status", "")):
                fail(
                    "TRACKER.json NVIDIA status says NOT_VENDORED although vendor tree exists",
                    errors,
                )
            c01 = next(
                (
                    c
                    for c in tracker.get("logical_contracts", [])
                    if c.get("id") == "SLLM-C01"
                ),
                None,
            )
            if c01 and c01.get("owner") != "sphinx_llm/":
                fail(
                    f"SLLM-C01 owner must be sphinx_llm/ for current tree, got {c01.get('owner')!r}",
                    errors,
                )

    # A01 portable vendored-source evidence. The legacy bash tree hash is kept
    # as historical evidence because its sha256sum input embeds target paths.
    if baseline and vendor.is_dir():
        allowed = {"UPSTREAM_PRESERVED", "UPSTREAM_PATCHED"}
        represented: set[str] = set()
        records: list[tuple[str, str]] = []
        for item in baseline.get("files", []):
            rel = item.get("path", "")
            classification = item.get("classification")
            if classification not in allowed:
                fail(
                    f"VENDOR_BASELINE invalid classification for {rel}: {classification}",
                    errors,
                )
                continue
            represented.add(rel)
            path = vendor / rel
            if not path.is_file():
                fail(f"VENDOR_BASELINE file missing from vendor tree: {rel}", errors)
                continue
            actual = sha256_file(path)
            expected = item.get("sha256", "")
            if actual != expected:
                fail(
                    f"vendor file digest drift: {rel}: expected {expected}, got {actual}",
                    errors,
                )
            records.append((Path(rel).as_posix(), actual))
            if (
                classification == "UPSTREAM_PRESERVED"
                and item.get("byte_identical_to_pinned") is not True
            ):
                fail(
                    f"UPSTREAM_PRESERVED file lacks byte-identical proof: {rel}", errors
                )

        for item in baseline.get("local_metadata", []):
            rel = item.get("path", "")
            represented.add(rel)
            path = vendor / rel
            if not path.is_file():
                fail(f"vendor metadata file missing: {rel}", errors)
                continue
            expected = item.get("sha256", "")
            actual = sha256_file(path)
            if expected and actual != expected:
                fail(
                    f"vendor metadata digest drift: {rel}: expected {expected}, got {actual}",
                    errors,
                )

        actual_files = {
            p.relative_to(vendor).as_posix()
            for p in vendor.rglob("*")
            if p.is_file() and not is_ephemeral(p.relative_to(vendor))
        }
        untracked = sorted(actual_files - represented)
        missing_from_tree = sorted(represented - actual_files)
        if untracked:
            fail(f"vendor tree has unclassified files: {untracked}", errors)
        if missing_from_tree:
            fail(
                f"VENDOR_BASELINE represents files absent from vendor tree: {missing_from_tree}",
                errors,
            )

        portable = baseline.get("portable_manifest", {})
        aggregate = portable_vendor_hash(records)
        expected_aggregate = portable.get("aggregate_sha256", "")
        if aggregate != expected_aggregate:
            fail(
                f"portable vendor aggregate drift: expected {expected_aggregate}, got {aggregate}",
                errors,
            )
        if len(records) != portable.get("file_count"):
            fail(
                f"portable vendor file_count drift: expected {portable.get('file_count')}, got {len(records)}",
                errors,
            )

        # The preserved tests rely on NVIDIA's repo-relative docs/source fixture.
        # Keep an exact maintenance-only copy and verify it independently of the
        # production vendor boundary.
        fixture = baseline.get("upstream_test_fixture", {})
        fixture_root_text = fixture.get("root", "")
        fixture_root = rpath(fixture_root_text) if fixture_root_text else None
        fixture_items = fixture.get("files", [])
        if not fixture_root_text or not fixture_items:
            fail("VENDOR_BASELINE missing upstream_test_fixture evidence", errors)
        elif fixture_root is None or not fixture_root.is_dir():
            fail(f"upstream test fixture root missing: {fixture_root_text}", errors)
        else:
            represented_fixture: set[str] = set()
            for item in fixture_items:
                rel_from_maintenance = item.get("path", "")
                classification = item.get("classification")
                if classification != "UPSTREAM_TEST_FIXTURE_PRESERVED":
                    fail(
                        f"invalid upstream test fixture classification for {rel_from_maintenance}: "
                        f"{classification!r}",
                        errors,
                    )
                    continue
                fixture_path = mpath(rel_from_maintenance)
                try:
                    rel_under_root = fixture_path.resolve().relative_to(
                        fixture_root.resolve()
                    )
                except ValueError:
                    fail(
                        f"upstream test fixture path escapes fixture root: {rel_from_maintenance}",
                        errors,
                    )
                    continue
                represented_fixture.add(rel_under_root.as_posix())
                if not fixture_path.is_file():
                    fail(
                        f"upstream test fixture file missing: {rel_from_maintenance}",
                        errors,
                    )
                    continue
                actual = sha256_file(fixture_path)
                expected = item.get("sha256", "")
                if actual != expected:
                    fail(
                        f"upstream test fixture digest drift: {rel_from_maintenance}: "
                        f"expected {expected}, got {actual}",
                        errors,
                    )
                if item.get("byte_identical_to_pinned") is not True:
                    fail(
                        f"upstream test fixture lacks pinned byte-parity proof: {rel_from_maintenance}",
                        errors,
                    )
            actual_fixture = {
                q.relative_to(fixture_root).as_posix()
                for q in fixture_root.rglob("*")
                if q.is_file() and not is_ephemeral(q.relative_to(fixture_root))
            }
            untracked_fixture = sorted(actual_fixture - represented_fixture)
            missing_fixture = sorted(represented_fixture - actual_fixture)
            if untracked_fixture:
                fail(
                    f"upstream test fixture has unclassified files: {untracked_fixture}",
                    errors,
                )
            if missing_fixture:
                fail(
                    f"upstream test fixture evidence references missing files: {missing_fixture}",
                    errors,
                )
            if len(represented_fixture) != fixture.get("file_count"):
                fail(
                    f"upstream test fixture file_count drift: expected {fixture.get('file_count')}, "
                    f"got {len(represented_fixture)}",
                    errors,
                )

        # Baseline and lock must describe the same pinned revision. Do not make
        # the path-dependent legacy tree hash a portable hard gate.
        if lock_path.is_file():
            lock = load_json(lock_path, errors)
            if lock:
                if lock.get("commit_hash") != baseline.get("pinned_upstream", {}).get(
                    "commit"
                ):
                    fail(
                        "vendor.lock.json commit differs from VENDOR_BASELINE pinned commit",
                        errors,
                    )
                legacy = baseline.get("legacy_lock", {})
                if lock.get("tree_hash") != legacy.get("expected_tree_hash"):
                    fail(
                        "vendor.lock.json tree_hash differs from recorded legacy lock evidence",
                        errors,
                    )

        # The exact A01 test environment is pinned independently of A02's future
        # compatibility matrix.  Keep the original upstream project metadata
        # byte-identical and fail closed if the fixture/evidence drifts.
        env_status = ""
        if test_environment:
            expected_commit = test_environment.get("upstream", {}).get("commit")
            pinned_commit = baseline.get("pinned_upstream", {}).get("commit")
            if expected_commit != pinned_commit:
                fail(
                    "UPSTREAM_TEST_ENVIRONMENT pinned commit differs from VENDOR_BASELINE",
                    errors,
                )
            target_python = str(
                test_environment.get("target", {}).get("python_major_minor", "")
            )
            if target_python != "3.13":
                fail(
                    f"A01 exact environment target must remain Python 3.13, got {target_python!r}",
                    errors,
                )

            env_fixture_dir = mpath("upstream_test_environment")
            for key, filename in (
                ("pyproject", "pyproject.toml"),
                ("uv_lock", "uv.lock"),
            ):
                item = test_environment.get("fixtures", {}).get(key, {})
                rel = item.get("path", "")
                path = mpath(rel) if rel else None
                if path is None or not path.is_file():
                    fail(
                        f"A01 upstream environment fixture missing: {rel or filename}",
                        errors,
                    )
                    continue
                try:
                    path.resolve().relative_to(env_fixture_dir.resolve())
                except ValueError:
                    fail(
                        f"A01 upstream environment fixture escapes expected directory: {rel}",
                        errors,
                    )
                    continue
                actual = sha256_file(path)
                expected = item.get("sha256", "")
                if actual != expected:
                    fail(
                        f"A01 upstream environment fixture digest drift: {rel}: expected {expected}, got {actual}",
                        errors,
                    )
                if item.get("classification") != "UPSTREAM_PRESERVED_TEST_ENVIRONMENT":
                    fail(
                        f"A01 environment fixture classification invalid: {rel}", errors
                    )

            # Byte parity is the primary proof.  Parse the lock with stdlib TOML
            # as an additional semantic guard that the recorded anchor versions
            # are actually represented in the preserved lock.
            try:
                import tomllib  # ruff: ignore[import-outside-top-level]

                lock_doc = tomllib.loads(
                    (env_fixture_dir / "uv.lock").read_text(encoding="utf-8")
                )
                pairs = {
                    (pkg.get("name"), pkg.get("version"))
                    for pkg in lock_doc.get("package", [])
                }
                for name, version in test_environment.get(
                    "anchor_packages", {}
                ).items():
                    if (name, version) not in pairs:
                        fail(
                            f"A01 anchor package absent from preserved uv.lock: {name}=={version}",
                            errors,
                        )
            except Exception as exc:  # ruff: ignore[blind-except]
                fail(f"unable to parse preserved A01 uv.lock: {exc}", errors)
            env_status = str(test_environment.get("status", ""))

        # Full selected lockset proof: anchors alone are insufficient to claim
        # GREEN_EXACT_LOCK. The derived selection is frozen to the preserved
        # lock and checked independently of runtime package availability.
        if test_lockset:
            selection = test_lockset.get("selection", {})
            packages = selection.get("packages", [])
            names = [item.get("name") for item in packages]
            if len(names) != len(set(names)):
                fail("UPSTREAM_TEST_LOCKSET has duplicate distribution names", errors)
            if len(packages) != selection.get("package_count"):
                fail(
                    f"UPSTREAM_TEST_LOCKSET package_count drift: expected "
                    f"{selection.get('package_count')}, got {len(packages)}",
                    errors,
                )
            records = [
                f"{item.get('name')}=={item.get('version')}\n"
                for item in sorted(packages, key=lambda item: str(item.get("name", "")))
            ]
            aggregate = hashlib.sha256("".join(records).encode("utf-8")).hexdigest()
            if aggregate != selection.get("aggregate_sha256"):
                fail(
                    f"UPSTREAM_TEST_LOCKSET aggregate drift: expected "
                    f"{selection.get('aggregate_sha256')}, got {aggregate}",
                    errors,
                )
            actual_lock_sha = sha256_file(
                mpath("upstream_test_environment") / "uv.lock"
            )
            if test_lockset.get("source", {}).get("uv_lock_sha256") != actual_lock_sha:
                fail(
                    "UPSTREAM_TEST_LOCKSET source uv.lock digest differs from preserved fixture",
                    errors,
                )
            try:
                import tomllib  # ruff: ignore[import-outside-top-level]

                lock_doc_for_set = tomllib.loads(
                    (mpath("upstream_test_environment") / "uv.lock").read_text(
                        encoding="utf-8"
                    )
                )
                lock_pairs = {
                    (pkg.get("name"), pkg.get("version"))
                    for pkg in lock_doc_for_set.get("package", [])
                    if pkg.get("version") is not None
                }
                for item in packages:
                    pair = (item.get("name"), item.get("version"))
                    if pair not in lock_pairs:
                        fail(
                            f"UPSTREAM_TEST_LOCKSET package/version absent from preserved uv.lock: "
                            f"{pair[0]}=={pair[1]}",
                            errors,
                        )
            except Exception as exc:  # ruff: ignore[blind-except]
                fail(
                    f"unable to cross-check selected lockset against uv.lock: {exc}",
                    errors,
                )
            if test_environment:
                fixture = test_environment.get("fixtures", {}).get("lockset", {})
                if fixture.get("path") != "UPSTREAM_TEST_LOCKSET.json":
                    fail(
                        "UPSTREAM_TEST_ENVIRONMENT lockset fixture path is invalid",
                        errors,
                    )
                actual_lockset_sha = sha256_file(mpath("UPSTREAM_TEST_LOCKSET.json"))
                if fixture.get("sha256") != actual_lockset_sha:
                    fail(
                        "UPSTREAM_TEST_ENVIRONMENT lockset fixture digest drift", errors
                    )
                recorded_lockset = test_environment.get("lockset", {})
                if recorded_lockset.get("selected_external_distributions") != len(
                    packages
                ):
                    fail(
                        "UPSTREAM_TEST_ENVIRONMENT selected lockset count disagrees with lockset evidence",
                        errors,
                    )
                if recorded_lockset.get("aggregate_sha256") != selection.get(
                    "aggregate_sha256"
                ):
                    fail(
                        "UPSTREAM_TEST_ENVIRONMENT lockset aggregate disagrees with lockset evidence",
                        errors,
                    )
                expected_anchors = test_environment.get("anchor_packages", {})
                selected_map = {
                    item.get("name"): item.get("version") for item in packages
                }
                for name, version in expected_anchors.items():
                    if selected_map.get(name) != version:
                        fail(
                            f"A01 anchor disagrees with full selected lockset: "
                            f"{name}=={version}, selected={selected_map.get(name)!r}",
                            errors,
                        )

        # A01 has two explicit behavior-proof modes:
        #   1. a local GREEN_EXACT_LOCK staged run; or
        #   2. official pinned NVIDIA CI for the exact same commit, accepted only
        #      when the local vendor/tests/docs/project metadata/lock/workflow are
        #      byte-verified against that commit.  Local ENVIRONMENT_BLOCKED must
        #      remain visible; it is never relabeled GREEN.
        tests_status = str(baseline.get("tests", {}).get("status", ""))
        ci_proof_green = False
        if ci_baseline:
            pinned_commit = baseline.get("pinned_upstream", {}).get("commit")
            ci_commit = ci_baseline.get("upstream", {}).get("commit")
            run_commit = ci_baseline.get("github_run", {}).get("commit")
            if ci_commit != pinned_commit or run_commit != pinned_commit:
                fail(
                    "UPSTREAM_CI_BASELINE commit differs from VENDOR_BASELINE pinned commit",
                    errors,
                )

            fixtures = ci_baseline.get("fixtures", {})
            ci_fixture_root = mpath("upstream_ci_fixture")
            workflow_item = fixtures.get("test_workflow", {})
            workflow_rel = workflow_item.get("path", "")
            workflow_path = mpath(workflow_rel) if workflow_rel else None
            if workflow_path is None or not workflow_path.is_file():
                fail(
                    f"A01 pinned CI workflow fixture missing: {workflow_rel!r}", errors
                )
            else:
                try:
                    workflow_path.resolve().relative_to(ci_fixture_root.resolve())
                except ValueError:
                    fail(
                        f"A01 pinned CI workflow fixture escapes expected directory: {workflow_rel}",
                        errors,
                    )
                actual = sha256_file(workflow_path)
                expected = workflow_item.get("sha256", "")
                if actual != expected:
                    fail(
                        f"A01 pinned CI workflow digest drift: expected {expected}, got {actual}",
                        errors,
                    )
                workflow_text = workflow_path.read_text(encoding="utf-8")
                required_fragments = [
                    'python-version: "3.13"',
                    'sphinx-version: ">=9,<10"',
                    'uv run --dev --extra gen --with "sphinx${{ matrix.sphinx-version }}" pytest src/sphinx_llm/tests/',
                ]
                for fragment in required_fragments:
                    if fragment not in workflow_text:
                        fail(
                            f"A01 pinned CI workflow missing required behavior fragment: {fragment}",
                            errors,
                        )

            for key, expected_rel in (
                ("pyproject", "upstream_test_environment/pyproject.toml"),
                ("uv_lock", "upstream_test_environment/uv.lock"),
            ):
                item = fixtures.get(key, {})
                rel = item.get("path", "")
                if rel != expected_rel:
                    fail(
                        f"UPSTREAM_CI_BASELINE {key} fixture path drift: {rel!r}",
                        errors,
                    )
                    continue
                path = mpath(rel)
                if not path.is_file():
                    fail(f"UPSTREAM_CI_BASELINE {key} fixture missing: {rel}", errors)
                elif sha256_file(path) != item.get("sha256", ""):
                    fail(f"UPSTREAM_CI_BASELINE {key} fixture digest drift", errors)

            run = ci_baseline.get("github_run", {})
            job = ci_baseline.get("job", {})
            closure = ci_baseline.get("closure", {})
            workflow = ci_baseline.get("workflow", {})
            if run.get("conclusion") != "SUCCESS":
                fail("UPSTREAM_CI_BASELINE official run is not SUCCESS", errors)
            if job.get("conclusion") != "SUCCESS":
                fail(
                    "UPSTREAM_CI_BASELINE Python-3.13/Sphinx-9 job is not SUCCESS",
                    errors,
                )
            if str(job.get("python")) != "3.13" or job.get("sphinx_range") != ">=9,<10":
                fail(
                    "UPSTREAM_CI_BASELINE does not identify the required Python-3.13/Sphinx-9 job",
                    errors,
                )
            if (
                workflow.get("matrix_python") != "3.13"
                or workflow.get("matrix_sphinx") != ">=9,<10"
            ):
                fail("UPSTREAM_CI_BASELINE workflow matrix metadata drift", errors)
            if ci_baseline.get("proof_mode") != "GREEN_PINNED_UPSTREAM_CI_EQUIVALENT":
                fail(
                    "UPSTREAM_CI_BASELINE proof_mode is not the approved A01 mode",
                    errors,
                )
            if (
                closure.get("accepted_for_A01") is not True
                or closure.get("accepted_proof_mode") != "PINNED_UPSTREAM_CI_EQUIVALENT"
            ):
                fail(
                    "UPSTREAM_CI_BASELINE closure metadata does not authorize A01 proof mode",
                    errors,
                )
            local = ci_baseline.get("local_equivalence", {})
            required_local = {
                "vendor_baseline": "GREEN_13_UPSTREAM_PRESERVED_BYTE_IDENTICAL",
                "upstream_docs_fixture": "GREEN_9_FILES_BYTE_IDENTICAL",
                "upstream_project_metadata": (
                    "GREEN_PYPROJECT_AND_UV_LOCK_BYTE_IDENTICAL"
                ),
                "workflow_fixture": "GREEN_TEST_YML_BYTE_IDENTICAL",
                "staged_upstream_layout": "GREEN",
                "selected_python_3_13_lockset": (
                    "GREEN_50_DISTRIBUTIONS_DERIVED_FROM_PRESERVED_UV_LOCK"
                ),
            }
            for key, expected in required_local.items():
                if local.get(key) != expected:
                    fail(
                        f"UPSTREAM_CI_BASELINE local equivalence drift for {key}: {local.get(key)!r}",
                        errors,
                    )
            if local.get("vendor_runtime_modified") is not False:
                fail(
                    "UPSTREAM_CI_BASELINE requires unmodified vendor/runtime source",
                    errors,
                )
            ci_proof_green = not any(
                msg.startswith(("UPSTREAM_CI_BASELINE", "A01 pinned CI"))
                for msg in errors
            )

        if state:
            a01_status = state.get("checkpoints", {}).get("A01", {}).get("status")
            local_proof_green = (
                tests_status == "GREEN" and env_status == "GREEN_EXACT_LOCK"
            )
            if a01_status == "COMPLETE" and not (local_proof_green or ci_proof_green):
                fail(
                    "A01 cannot be COMPLETE without either local GREEN_EXACT_LOCK + GREEN tests "
                    "or the fully validated GREEN_PINNED_UPSTREAM_CI_EQUIVALENT proof",
                    errors,
                )
            if a01_status == "COMPLETE":
                state_mode = str(
                    state.get("verification_snapshot", {}).get(
                        "a01_behavior_proof_mode", ""
                    )
                )
                allowed_modes = {
                    "LOCAL_GREEN_EXACT_LOCK",
                    "PINNED_UPSTREAM_CI_EQUIVALENT",
                }
                if state_mode not in allowed_modes:
                    fail(
                        f"A01 COMPLETE has invalid behavior proof mode: {state_mode!r}",
                        errors,
                    )
                if state_mode == "PINNED_UPSTREAM_CI_EQUIVALENT" and not ci_proof_green:
                    fail(
                        "A01 state selects pinned upstream CI proof but CI baseline is not fully green",
                        errors,
                    )

        if tracker:
            tracked_origin = {
                item.get("path"): item.get("origin_classification")
                for item in tracker.get("physical", {}).get("files", [])
            }
            for item in baseline.get("files", []):
                tracker_path = f"sphinx_llm/{item.get('path')}"
                if tracker_path in tracked_origin and tracked_origin[
                    tracker_path
                ] != item.get("classification"):
                    fail(
                        f"TRACKER origin disagrees with VENDOR_BASELINE for {tracker_path}: "
                        f"{tracked_origin[tracker_path]!r} != {item.get('classification')!r}",
                        errors,
                    )

    # A02 pinned upstream compatibility evidence. This is allowed to be
    # internally GREEN while the checkpoint itself is BLOCKED_PRODUCT: the
    # verifier proves that the recorded blocker is truthful, not that it vanished.
    if compatibility:
        if compatibility.get("checkpoint") != "A02":
            fail("UPSTREAM_COMPATIBILITY_BASELINE checkpoint is not A02", errors)
        if (
            compatibility.get("matrix_proof_mode")
            != "PINNED_UPSTREAM_MATRIX_CI_EQUIVALENT"
        ):
            fail("A02 compatibility proof mode drift", errors)
        if baseline and compatibility.get("upstream", {}).get("commit") != baseline.get(
            "pinned_upstream", {}
        ).get("commit"):
            fail(
                "A02 compatibility commit differs from VENDOR_BASELINE pinned commit",
                errors,
            )
        if state:
            ca = compatibility.get("source_anchor", {})
            sa = state.get("source_anchor", {})
            if ca.get("archive") != sa.get("archive") or ca.get("sha256") != sa.get(
                "sha256"
            ):
                fail(
                    "A02 compatibility source anchor disagrees with STATE.json", errors
                )
        matrix = compatibility.get("matrix", [])
        cells = {(str(x.get("python")), x.get("sphinx")) for x in matrix}
        expected_cells = {
            ("3.12", ">=5.1,<6"),
            ("3.12", ">=6,<7"),
            ("3.12", ">=7,<8"),
            ("3.12", ">=8,<9"),
            ("3.12", ">=9,<10"),
            ("3.9", ">=7,<8"),
            ("3.10", ">=7,<8"),
            ("3.11", ">=8,<9"),
            ("3.13", ">=9,<10"),
            ("3.14", ">=9,<10"),
        }
        if cells != expected_cells:
            fail(f"A02 compatibility matrix drift: {sorted(cells)!r}", errors)
        if len(matrix) != 10 or any(  # ruff: ignore[magic-value-comparison]
            x.get("conclusion") != "SUCCESS" for x in matrix
        ):
            fail(
                "A02 compatibility matrix must retain ten recorded SUCCESS cells",
                errors,
            )
        probes = compatibility.get("source_probes", {})
        if (
            probes.get("tags_forwarded") is not True
            or probes.get("confdir_forwarded") is not True
        ):
            fail("A02 source probes lost known tag/confdir forwarding", errors)
        if probes.get("config_overrides_forwarded") is not False:
            fail(
                "A02 config-override blocker changed; re-run executable compatibility proof before updating state",
                errors,
            )
        fix = compatibility.get("downstream_fix", {})
        if fix.get("strategy") != "DOWNSTREAM_COMPAT_INTEGRITY_CHECKED_CONFIG_SNAPSHOT":
            fail("A02 downstream compatibility strategy drift", errors)
        if fix.get("vendor_files_modified") is not False:
            fail(
                "A02 downstream compatibility fix must preserve NVIDIA vendor files",
                errors,
            )
        required_fix_paths = [
            rpath(str(fix.get("outer_extension", ""))),
            rpath(str(fix.get("generator", ""))),
            rpath(str(fix.get("context_transfer", ""))),
            rpath(str(fix.get("child_bootstrap", ""))),
            rpath(str(fix.get("unit_test", ""))),
            rpath(str(fix.get("matrix_harness_test", ""))),
            rpath(str(fix.get("integration_test", ""))),
            rpath(str(fix.get("integration_fixture", ""))) / "conf.py",
            rpath(str(fix.get("integration_fixture", ""))) / "index.rst",
            rpath(str(fix.get("matrix_plan", ""))),
            rpath(str(fix.get("matrix_orchestrator", ""))),
            rpath(str(fix.get("ci_cell_runner", ""))),
            rpath(str(fix.get("closure_harness_test", ""))),
            rpath(str(fix.get("closure_evidence_verifier", ""))),
            rpath(str(fix.get("closure_evidence_doc", ""))),
            rpath(str(fix.get("closure_decision_schema", ""))),
            rpath(str(fix.get("reconciliation_readiness_test", ""))),
            rpath(str(fix.get("reconciliation_readiness_preparer", ""))),
            rpath(str(fix.get("reconciliation_readiness_doc", ""))),
            rpath(str(fix.get("reconciliation_readiness_schema", ""))),
            rpath(str(fix.get("circleci_rebase_renderer", ""))),
            rpath(str(fix.get("circleci_rebase_test", ""))),
            mpath(Path(str(fix.get("integration_runner", ""))).name),
        ]
        for path in required_fix_paths:
            if not path.is_file():
                fail(f"A02 downstream compatibility path missing: {path}", errors)
        if fix.get("unit_test_result") != "GREEN_10_PASSED":
            fail(
                "A02 dependency-light helper test evidence is not GREEN_10_PASSED",
                errors,
            )
        if fix.get("matrix_harness_test_result") != "GREEN_14_PASSED":
            fail("A02 matrix harness test evidence is not GREEN_14_PASSED", errors)
        if fix.get("closure_harness_test_result") != "GREEN_11_PASSED":
            fail("A02 closure harness test evidence is not GREEN_11_PASSED", errors)
        if fix.get("reconciliation_readiness_test_result") != "GREEN_6_PASSED":
            fail(
                "A02 reconciliation-readiness test evidence is not GREEN_6_PASSED",
                errors,
            )
        if fix.get("circleci_rebase_test_result") != "GREEN_7_PASSED":
            fail(
                "A02 CircleCI semantic-rebase test evidence is not GREEN_7_PASSED",
                errors,
            )
        if (
            fix.get("circleci_rebase_renderer")
            != "_maintenance/render_a02_circleci_rebase.py"
        ):
            fail("A02 CircleCI semantic-rebase renderer path drift", errors)
        if fix.get("circleci_rebase_test") != "tests/test_a02_circleci_rebase.py":
            fail("A02 CircleCI semantic-rebase test path drift", errors)
        if (
            fix.get("closure_evidence_verifier")
            != "_maintenance/verify_a02_closure_evidence.py"
        ):
            fail("A02 closure-evidence verifier path drift", errors)
        if (
            fix.get("closure_decision_schema")
            != "_maintenance/schemas/a02-closure-decision.schema.json"
        ):
            fail("A02 closure-decision schema path drift", errors)
        if (
            fix.get("reconciliation_readiness_preparer")
            != "_maintenance/prepare_a02_reconciliation.py"
        ):
            fail("A02 reconciliation-readiness preparer path drift", errors)
        if (
            fix.get("reconciliation_readiness_schema")
            != "_maintenance/schemas/a02-reconciliation-readiness.schema.json"
        ):
            fail("A02 reconciliation-readiness schema path drift", errors)
        props = fix.get("properties", {})
        if (
            props.get("cell_execution_provenance")
            != "CIRCLECI_BUILTIN_PIPELINE_WORKFLOW_JOB_PROJECT_REVISION_CAPTURED_LOCAL_EXPLICIT"
        ):
            fail("A02 cell execution-provenance contract drift", errors)
        if (
            props.get("post_ci_evidence_gate")
            != "READ_ONLY_RECOMPUTE_MATCH_AGGREGATE_REQUIRE_COHERENT_EXPECTED_CIRCLECI_PROJECT_NO_AUTO_STATE_MUTATION"
        ):
            fail("A02 post-CI provenance gate drift", errors)
        if (
            props.get("reconciliation_boundary")
            != "READ_ONLY_RECEIPT_PINS_EVIDENCE_AND_TARGET_DIGESTS_NO_AUTO_STATE_MUTATION"
        ):
            fail("A02 reconciliation-readiness boundary drift", errors)
        if (
            props.get("circleci_rebase_boundary")
            != "READ_CURRENT_CONFIG_RENDER_SEPARATE_CANDIDATE_NO_IN_PLACE_MUTATION_SEMANTIC_VERIFY"
        ):
            fail("A02 CircleCI semantic-rebase boundary drift", errors)

        parity = fix.get("required_parity_matrix", {})
        parity_cells = parity.get("cells", [])
        parity_pairs = [(str(x.get("python")), x.get("sphinx")) for x in parity_cells]
        upstream_pairs = [(str(x.get("python")), x.get("sphinx")) for x in matrix]
        if not a02_plan:
            fail("A02 canonical matrix plan is unavailable", errors)
        else:
            plan_pairs = [
                (str(x.get("python")), x.get("sphinx"))
                for x in a02_plan.get("cells", [])
            ]
            if (
                a02_plan.get("checkpoint") != "A02"
                or a02_plan.get("plan_id")
                != "A02_CONFIG_PARITY_PINNED_UPSTREAM_MATRIX_V1"
            ):
                fail("A02 canonical matrix plan identity drift", errors)
            if a02_plan.get("upstream_commit") != compatibility.get("upstream", {}).get(
                "commit"
            ):
                fail("A02 matrix plan upstream commit drift", errors)
            if a02_plan.get("sphinx_markdown_builder") != "==0.6.10":
                fail("A02 matrix plan sphinx-markdown-builder pin drift", errors)
            expected_project = a02_plan.get("circleci_expected_project", {})
            if expected_project != {
                "username": "scikit-plots",
                "reponame": "scikit-plots",
            }:
                fail("A02 matrix plan CircleCI project identity drift", errors)
            actual_impl = a02_implementation_hash(a02_plan, errors)
            expected_impl = a02_plan.get("implementation_fingerprint", {}).get("sha256")
            if actual_impl is not None and actual_impl != expected_impl:
                fail(
                    f"A02 matrix plan implementation fingerprint drift: {actual_impl} != {expected_impl}",
                    errors,
                )
            if state:
                pa = a02_plan.get("source_anchor", {})
                sa = state.get("source_anchor", {})
                if pa.get("archive") != sa.get("archive") or pa.get("sha256") != sa.get(
                    "sha256"
                ):
                    fail(
                        "A02 matrix plan source anchor disagrees with STATE.json",
                        errors,
                    )
            if plan_pairs != upstream_pairs:
                fail(
                    "A02 canonical matrix plan must mirror pinned upstream matrix order",
                    errors,
                )
            plan_ids = [x.get("id") for x in a02_plan.get("cells", [])]
            plan_files = [x.get("evidence_file") for x in a02_plan.get("cells", [])]
            plan_indexes = [x.get("index") for x in a02_plan.get("cells", [])]
            if (
                len(plan_ids) != 10  # ruff: ignore[magic-value-comparison]
                or len(set(plan_ids)) != 10  # ruff: ignore[magic-value-comparison]
                or len(set(plan_files)) != 10  # ruff: ignore[magic-value-comparison]
            ):
                fail(
                    "A02 matrix plan requires ten unique cell IDs/evidence files",
                    errors,
                )
            if plan_indexes != list(range(1, 11)):
                fail(
                    "A02 matrix plan indexes must be exactly 1..10 in matrix order",
                    errors,
                )
            for cell in a02_plan.get("cells", []):
                if cell.get("evidence_file") != f"{cell.get('id')}.json":
                    fail(
                        f"A02 matrix evidence filename does not match cell ID: {cell!r}",
                        errors,
                    )
            if fix.get("matrix_plan") != "_maintenance/A02_MATRIX_PLAN.json":
                fail("A02 downstream fix matrix-plan path drift", errors)
            if fix.get("matrix_orchestrator") != "_maintenance/run_a02_matrix.py":
                fail("A02 downstream fix matrix-orchestrator path drift", errors)
            if fix.get("ci_cell_runner") != "_maintenance/ci/run_a02_cell.sh":
                fail("A02 downstream fix CI cell-runner path drift", errors)
        if parity.get("proof_mode") != "MATCH_PINNED_UPSTREAM_10_CELL_MATRIX":
            fail("A02 downstream parity proof mode drift", errors)
        if parity_pairs != upstream_pairs:
            fail(
                "A02 downstream parity matrix must mirror the pinned upstream matrix in the same order",
                errors,
            )
        if (
            len(parity_cells) != 10  # ruff: ignore[magic-value-comparison]
            or parity.get("total_cells") != 10  # ruff: ignore[magic-value-comparison]
        ):
            fail("A02 downstream parity matrix must retain exactly ten cells", errors)
        allowed_parity = {
            "NOT_RUN_DOWNSTREAM_SHIM",
            "GREEN",
            "RED",
            "ENVIRONMENT_BLOCKED",
        }
        for cell in parity_cells:
            cell_status = cell.get("status")
            if cell_status not in allowed_parity:
                fail(
                    f"A02 downstream parity cell has invalid status: {cell_status!r}",
                    errors,
                )
            if cell_status == "GREEN" and not isinstance(cell.get("evidence"), str):
                fail("A02 GREEN downstream parity cell requires evidence", errors)
        green_cells = sum(1 for cell in parity_cells if cell.get("status") == "GREEN")
        if parity.get("green_cells") != green_cells:
            fail(
                "A02 downstream parity green-cell count disagrees with recorded cells",
                errors,
            )
        expected_matrix_state = (
            "GREEN_10_OF_10"
            if green_cells == 10  # ruff: ignore[magic-value-comparison]
            else (
                "NOT_RUN_0_OF_10_GREEN"
                if green_cells == 0
                and all(
                    c.get("status") == "NOT_RUN_DOWNSTREAM_SHIM" for c in parity_cells
                )
                else "PARTIAL"
            )
        )
        if (
            compatibility.get("closure", {}).get("downstream_parity_matrix")
            != expected_matrix_state
        ):
            fail(
                "A02 closure downstream-parity state disagrees with recorded cells",
                errors,
            )

        gaps = {g.get("id"): g for g in compatibility.get("gaps", [])}
        closure = compatibility.get("closure", {})
        if green_cells < 10:  # ruff: ignore[magic-value-comparison]
            # RULE 34 RATIONALE (weakened gate, recorded deliberately):
            # A02's ten-cell downstream matrix was permanently abandoned by
            # maintainer decision, so BLOCKED is no longer the only honest
            # status. DEFERRED_PERMANENTLY is admitted as a second terminal
            # state, and it is held to the *same* strictness: the gap stays
            # open, semantic parity stays un-GREEN, and checkpoint_complete
            # stays false. COMPLETE remains unreachable below 10/10 — that
            # assertion is deliberately NOT weakened.
            a02_status = (
                (state or {}).get("checkpoints", {}).get("A02", {}).get("status")
            )
            deferred = a02_status in DEFERRED_STATUSES
            expected_gap = (
                "DEFERRED_PERMANENTLY"
                if deferred
                else "IMPLEMENTED_AWAITING_EXECUTABLE_REGRESSION"
            )
            if gaps.get("A02-G01", {}).get("status") != expected_gap:
                fail(
                    f"A02-G01 status must be {expected_gap} while downstream parity is below 10/10 GREEN",
                    errors,
                )
            if closure.get("config_override_semantic_parity") != expected_gap:
                fail(
                    "A02 semantic parity cannot be GREEN before 10/10 downstream cells are GREEN",
                    errors,
                )
            if closure.get("checkpoint_complete") is not False:
                fail(
                    "A02 compatibility baseline cannot close before 10/10 downstream cells are GREEN",
                    errors,
                )
            if state and a02_status not in {"BLOCKED"} | DEFERRED_STATUSES:
                fail(
                    f"A02 state must be BLOCKED or DEFERRED_PERMANENTLY until downstream parity is 10/10 GREEN, got {a02_status!r}",
                    errors,
                )
            if deferred:
                a02_record = state.get("checkpoints", {}).get("A02", {})
                for field in ("deferral_rationale", "residual_risk"):
                    if (
                        not isinstance(a02_record.get(field), str)
                        or not a02_record[field].strip()
                    ):
                        fail(
                            f"A02 {field} must be recorded when the checkpoint is DEFERRED_PERMANENTLY",
                            errors,
                        )
        else:
            if gaps.get("A02-G01", {}).get("status") != "CLOSED":
                fail(
                    "A02-G01 must be CLOSED when downstream parity is 10/10 GREEN",
                    errors,
                )
            if (
                closure.get("config_override_semantic_parity") != "GREEN"
                or closure.get("checkpoint_complete") is not True
            ):
                fail(
                    "A02 closure fields must be GREEN/complete when downstream parity is 10/10 GREEN",
                    errors,
                )
            if (
                state
                and state.get("checkpoints", {}).get("A02", {}).get("status")
                != "COMPLETE"
            ):
                fail(
                    "STATE.json A02 must be COMPLETE when downstream parity is 10/10 GREEN",
                    errors,
                )

        expected_integration_result = (
            "MATRIX_PERMANENTLY_DEFERRED_0_OF_10_GREEN_NO_RUN_PLANNED"
            if (state or {}).get("checkpoints", {}).get("A02", {}).get("status")
            in DEFERRED_STATUSES
            else "MATRIX_NOT_RUN_0_OF_10_GREEN_CIRCLECI_INTEGRATED_LOCAL_ENVIRONMENT_BLOCKED"
        )
        if (
            green_cells == 0
            and fix.get("integration_result") != expected_integration_result
        ):
            fail(
                f"A02 integration-result evidence must be {expected_integration_result}",
                errors,
            )
        if state:
            if (
                state.get("checkpoints", {})
                .get("A02", {})
                .get("production_code_modified")
                is not True
            ):
                fail(
                    "A02 checkpoint must record production_code_modified=true after downstream compat implementation",
                    errors,
                )
            if state.get("production_code_modified") is not True:
                fail(
                    "STATE.json must record production_code_modified=true for the current A02 implementation increment",
                    errors,
                )

    # A02 repository-CI integration is exact while this one-shot compatibility
    # campaign is active. The uploaded source archive omitted hidden .circleci
    # files, so the baseline records the public-main source used for the bounded
    # integration and the exact reviewed integrated digest.
    if circleci:
        # RULE 34 RATIONALE (weakened gate, recorded deliberately):
        # under A02 DEFERRED_PERMANENTLY the CircleCI transport is retired, so a
        # missing repository config is no longer a defect. The recorded
        # integrated digest is still verified whenever the file IS present, so
        # a drifting workflow is still caught -- only its absence is tolerated.
        a02_deferred = (state or {}).get("checkpoints", {}).get("A02", {}).get(
            "status"
        ) in DEFERRED_STATUSES
        repo = HERE.parents[4]
        config = repo / ".circleci" / "config.yml"
        if not config.is_file():
            if not a02_deferred:
                fail(
                    "A02 CircleCI integration missing repository .circleci/config.yml",
                    errors,
                )
        else:
            actual_ci = sha256_file(config)
            expected_ci = circleci.get("integration", {}).get("integrated_sha256")
            if actual_ci != expected_ci:
                fail(
                    f"A02 CircleCI integrated digest drift: {actual_ci} != {expected_ci}",
                    errors,
                )
            text = config.read_text(encoding="utf-8", errors="ignore")
            for token in (
                "run_sphinx_llm_a02:",
                "sphinx_llm_a02_parity:",
                "sphinx_llm_a02_aggregate:",
                "sphinx_llm_a02_compatibility:",
                "equal: [true, << pipeline.parameters.run_sphinx_llm_a02 >>]",
                "--require-green",
            ):
                if token not in text:
                    fail(f"A02 CircleCI integration token missing: {token}", errors)
            if "equal: [true, << pipeline.parameters.run_docs_build >>]" not in text:
                fail(
                    "A02 CircleCI integration damaged the existing run_docs_build gate",
                    errors,
                )
        if (
            circleci.get(
                "integration",
                {},
            ).get(
                "required_cells",
            )
            != 10  # ruff: ignore[magic-value-comparison]
        ):
            fail("A02 CircleCI integration must retain exactly ten cells", errors)
        if (
            circleci.get("integration", {}).get("pipeline_parameter_default")
            is not False
        ):
            fail("A02 CircleCI integration parameter must default false", errors)
        if (
            circleci.get("integration", {}).get("aggregate_artifact_retention")
            != "ALWAYS_GREEN_RED_OR_ENVIRONMENT_BLOCKED"
        ):
            fail(
                "A02 CircleCI aggregate artifact retention must remain always-on",
                errors,
            )
        if (
            circleci.get("integration", {}).get("rebase_renderer")
            != "_maintenance/render_a02_circleci_rebase.py"
        ):
            fail("A02 CircleCI rebase renderer baseline drift", errors)
        if (
            circleci.get("integration", {}).get("candidate_verifier_mode")
            != "STRUCTURAL_YAML_CANDIDATE_NO_HISTORICAL_DIGEST_PIN"
        ):
            fail("A02 CircleCI candidate-verifier mode drift", errors)
        observation = circleci.get("source", {}).get(
            "current_public_main_observation", {}
        )
        if (
            observation.get("rebase_policy")
            != "PARSED_YAML_CLASSIFICATION_TOP_LEVEL_TEXT_INSERTION_SEPARATE_OUTPUT_NO_IN_PLACE_MUTATION"
        ):
            fail("A02 current-main semantic rebase policy drift", errors)

    # Retired placeholder must never become a second implementation tree.
    legacy_upstream = ROOT / "upstream"
    if legacy_upstream.is_dir():
        extras = [
            p
            for p in legacy_upstream.rglob("*")
            if p.is_file() and p.name != "README.md"
        ]
        if extras:
            fail("retired upstream/ placeholder contains production-like files", errors)

    # Producer must never depend on assistant runtime/private code.
    for p in ROOT.rglob("*.py"):
        if HERE in p.parents or "tests" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "_sphinx_ai_assistant" in text:
            fail(
                f"reverse dependency detected: {p.relative_to(ROOT)} references _sphinx_ai_assistant",
                errors,
            )

    if errors:
        print("_sphinx_llm maintenance drift: FAIL")  # ruff: ignore[print]
        for e in errors:
            print(f" - {e}")  # ruff: ignore[print]
        return 1
    print("_sphinx_llm maintenance drift: GREEN")  # ruff: ignore[print]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
