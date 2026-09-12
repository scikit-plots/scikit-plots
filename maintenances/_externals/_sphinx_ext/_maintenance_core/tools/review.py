"""Deterministic, agent-compatible review lanes for the Sphinx extension family.

Profiles select only registered checks.  They intentionally cannot embed shell commands,
Python expressions, or arbitrary file reads.  This keeps a future agent orchestration
layer advisory while executable repository evidence stays deterministic and reviewable.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from architecture import normalize_dependency_edges, runtime_imports, validate_manifest_architecture
from check_subsystem import check_subsystem
from paths import discover_repository_root, discover_runtime_sphinx_ext

SEVERITIES = ("ERROR", "WARNING", "UNAVAILABLE", "INFO")
_PROFILE_ID = re.compile(r"[a-z][a-z0-9_.-]{2,63}")
_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ReviewFinding:
    severity: str
    code: str
    message: str
    evidence: str
    remediation: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid review severity: {self.severity}")


@dataclass(frozen=True)
class ReviewContext:
    manifest_path: Path
    manifest: Mapping[str, object]
    profile: Mapping[str, object]
    repo: Path
    runtime_family: Path
    maintenance_root: Path
    skill_file: Path | None

    @property
    def runtime_root(self) -> Path:
        return self.runtime_family / str(self.manifest.get("runtime_dir", ""))

    @property
    def state_path(self) -> Path:
        return self.maintenance_root / str(self.manifest.get("state", "_maintenance/STATE.json"))

    @property
    def tracker_path(self) -> Path:
        return self.maintenance_root / str(self.manifest.get("tracker", "_maintenance/TRACKER.json"))

    @property
    def handoff_path(self) -> Path:
        return self.maintenance_root / str(self.manifest.get("handoff", "_maintenance/FRESH_CHAT_HANDOFF.md"))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finding(severity: str, code: str, message: str, evidence: str, remediation: str) -> ReviewFinding:
    return ReviewFinding(severity, code, message, evidence, remediation)


def check_common_maintenance(ctx: ReviewContext) -> list[ReviewFinding]:
    return [
        _finding(
            "ERROR",
            "review.common-maintenance",
            error,
            str(ctx.manifest_path.relative_to(ctx.repo)),
            "Repair the common maintenance invariant before treating the subsystem as PR-ready.",
        )
        for error in check_subsystem(ctx.manifest_path)
    ]


def check_runtime_syntax(ctx: ReviewContext) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for path in sorted(ctx.runtime_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            findings.append(
                _finding(
                    "ERROR",
                    "review.runtime-syntax",
                    f"runtime source cannot be parsed: {path.name}: {exc}",
                    str(path.relative_to(ctx.repo)),
                    "Fix the source syntax/encoding and rerun the focused owner tests.",
                )
            )
    return findings


def check_state_tracker(ctx: ReviewContext) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    state = _json(ctx.state_path)
    tracker = _json(ctx.tracker_path)
    expected = str(ctx.manifest.get("subsystem", ""))
    for label, data, path in (("STATE", state, ctx.state_path), ("TRACKER", tracker, ctx.tracker_path)):
        if data.get("subsystem") != expected:
            findings.append(
                _finding(
                    "ERROR",
                    "review.subsystem-identity",
                    f"{label}.json subsystem does not match MAINTENANCE.json",
                    str(path.relative_to(ctx.repo)),
                    "Use one canonical fully-qualified subsystem identity in manifest, state, and tracker.",
                )
            )
    active = state.get("active_checkpoint")
    checkpoints = state.get("checkpoints", {})
    if active and isinstance(checkpoints, dict):
        active_text = str(active)
        # Mature subsystems historically used either the logical checkpoint ID (R173T80)
        # or the checkpoint filename stem (R173T80_MENU_ITEM_GLYPHS).  Treat a unique
        # key-prefix match as the same checkpoint instead of manufacturing drift.
        matches = [key for key in checkpoints if active_text == key or active_text.startswith(f"{key}_")]
        if len(matches) != 1:
            findings.append(
                _finding(
                    "ERROR",
                    "review.active-checkpoint-missing",
                    f"active checkpoint {active!r} does not resolve to exactly one STATE.json checkpoint",
                    str(ctx.state_path.relative_to(ctx.repo)),
                    "Use the logical checkpoint ID or its unique filename-stem form.",
                )
            )
    contracts = tracker.get("logical_contracts", [])
    active_contracts = [c for c in contracts if isinstance(c, dict) and c.get("status") == "ACTIVE"]
    if not active_contracts:
        findings.append(
            _finding(
                "WARNING",
                "review.no-active-contracts",
                "tracker has no ACTIVE logical contracts",
                str(ctx.tracker_path.relative_to(ctx.repo)),
                "Record the currently enforced logical contracts or explicitly document why none are active.",
            )
        )
    return findings


def _ordered_positions(text: str, tokens: Iterable[str]) -> tuple[bool, str]:
    cursor = -1
    for token in tokens:
        pos = text.find(token, cursor + 1)
        if pos < 0:
            return False, token
        cursor = pos
    return True, ""


def check_fresh_chat_parity(ctx: ReviewContext) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    required = ("SKILL.md", "MAINTAINING.md", "STATE.json", "TRACKER.json")
    handoff = ctx.handoff_path.read_text(encoding="utf-8", errors="ignore")
    # Inside the handoff itself, "this file" is the canonical self-reference and is
    # intentionally accepted; requiring its basename would reject clearer prose.
    ok, missing = _ordered_positions(handoff, required)
    if not ok:
        findings.append(
            _finding(
                "ERROR",
                "review.handoff-read-order",
                f"fresh-chat handoff does not encode the canonical authority order; first missing/out-of-order token: {missing}",
                str(ctx.handoff_path.relative_to(ctx.repo)),
                "Use SKILL -> MAINTAINING -> handoff/self -> STATE -> TRACKER before todo/lessons and executable evidence.",
            )
        )
    if ctx.skill_file is not None:
        skill = ctx.skill_file.read_text(encoding="utf-8", errors="ignore")
        # The skill is already the entry point.  Unlike the handoff, it must name the
        # handoff explicitly so the next authority is discoverable.
        skill_required = ("MAINTAINING.md", "FRESH_CHAT_HANDOFF.md", "STATE.json", "TRACKER.json")
        ok, missing = _ordered_positions(skill, skill_required)
        if not ok:
            findings.append(
                _finding(
                    "ERROR",
                    "review.skill-read-order",
                    f"SKILL.md does not route through the canonical fresh-chat authorities; first missing/out-of-order token: {missing}",
                    str(ctx.skill_file.relative_to(ctx.repo)),
                    "Route MAINTAINING -> FRESH_CHAT_HANDOFF -> STATE -> TRACKER before domain todo/lessons.",
                )
            )
    return findings


def check_architecture(ctx: ReviewContext) -> list[ReviewFinding]:
    return [
        _finding(
            "ERROR",
            "review.architecture",
            error,
            str(ctx.manifest_path.relative_to(ctx.repo)),
            "Repair the typed dependency/capability contract at its owning package before review.",
        )
        for error in validate_manifest_architecture(ctx.manifest, ctx.runtime_family)
    ]


def check_verification_state(ctx: ReviewContext) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    state = _json(ctx.state_path)
    snapshot = state.get("verification_snapshot", {})
    blockers = state.get("blockers", []) if isinstance(state.get("blockers", []), list) else []
    unavailable = []
    if isinstance(snapshot, dict):
        unavailable = [key for key, value in snapshot.items() if str(value).upper().startswith(_UNAVAILABLE)]
        stale = [key for key, value in snapshot.items() if "PENDING_CURRENT_WORKSPACE_RUN" in str(value)]
        for key in stale:
            findings.append(
                _finding(
                    "ERROR",
                    "review.stale-pending-evidence",
                    f"verification snapshot still contains stale pending placeholder: {key}",
                    str(ctx.state_path.relative_to(ctx.repo)),
                    "Replace the placeholder only with observed GREEN/FAIL/UNAVAILABLE evidence.",
                )
            )
    if unavailable and not blockers:
        findings.append(
            _finding(
                "WARNING",
                "review.unavailable-without-blocker",
                f"{len(unavailable)} verification layer(s) are unavailable but STATE.json has no blockers",
                str(ctx.state_path.relative_to(ctx.repo)),
                "Record what dependency/environment is missing and what exact downstream gate closes it.",
            )
        )
    for key in unavailable:
        findings.append(
            _finding(
                "UNAVAILABLE",
                "review.verification-unavailable",
                f"verification layer unavailable: {key} = {snapshot[key]}",
                str(ctx.state_path.relative_to(ctx.repo)),
                "Run this layer in an environment that provides its optional dependencies before release promotion.",
            )
        )
    return findings


def check_skill_capability_coverage(ctx: ReviewContext) -> list[ReviewFinding]:
    if ctx.skill_file is None:
        return []
    text = ctx.skill_file.read_text(encoding="utf-8", errors="ignore")
    findings: list[ReviewFinding] = []
    for item in ctx.manifest.get("capability_ownership", []) or []:
        if not isinstance(item, Mapping):
            continue
        owner = str(item.get("owner", ""))
        capability = str(item.get("id", ""))
        # Capability IDs are maintenance vocabulary; skills may communicate by owner
        # package instead.  Require at least one of those stable anchors.
        if capability not in text and owner not in text:
            findings.append(
                _finding(
                    "WARNING",
                    "review.skill-capability-gap",
                    f"skill does not mention capability {capability!r} or its owner {owner!r}",
                    str(ctx.skill_file.relative_to(ctx.repo)),
                    "Add routing guidance so a fresh reviewer knows which package owns this capability.",
                )
            )
    return findings


def check_profile_safety(ctx: ReviewContext) -> list[ReviewFinding]:
    # Profiles are declarative by construction; reject common executable escape hatches
    # even if a future schema edit accidentally permits extra properties.
    findings: list[ReviewFinding] = []
    forbidden = {"command", "shell", "python", "exec", "script", "cwd", "env"}

    def walk(value: object, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    findings.append(
                        _finding(
                            "ERROR",
                            "review.profile-executable-metadata",
                            f"review profile contains forbidden executable field: {'.'.join((*trail, str(key)))}",
                            str((ctx.maintenance_root / str(ctx.manifest.get('review_profile', '_maintenance/REVIEW.json'))).relative_to(ctx.repo)),
                            "Use a registered deterministic check ID; never embed executable commands in REVIEW.json.",
                        )
                    )
                walk(child, (*trail, str(key)))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, (*trail, str(i)))

    walk(ctx.profile)
    return findings


CHECKS: dict[str, Callable[[ReviewContext], list[ReviewFinding]]] = {
    "common-maintenance": check_common_maintenance,
    "runtime-syntax": check_runtime_syntax,
    "state-tracker-consistency": check_state_tracker,
    "fresh-chat-parity": check_fresh_chat_parity,
    "architecture-consistency": check_architecture,
    "verification-state": check_verification_state,
    "skill-capability-coverage": check_skill_capability_coverage,
    "profile-safety": check_profile_safety,
}


def validate_profile(profile: Mapping[str, object], manifest: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    allowed_top = {"schema_version", "subsystem", "mode", "package_targets", "lenses", "pr_policy"}
    extra_top = sorted(set(profile) - allowed_top)
    if extra_top:
        errors.append("REVIEW.json contains unsupported top-level fields: " + ", ".join(extra_top))
    forbidden = {"command", "shell", "python", "exec", "script", "cwd", "env"}
    def executable_fields(value: object, trail: tuple[str, ...] = ()) -> list[str]:
        found: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = (*trail, str(key))
                if str(key).lower() in forbidden:
                    found.append(".".join(path))
                found.extend(executable_fields(child, path))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                found.extend(executable_fields(child, (*trail, str(i))))
        return found
    for field in executable_fields(profile):
        errors.append(f"REVIEW.json contains forbidden executable field: {field}")
    if profile.get("schema_version") != 1:
        errors.append("REVIEW.json schema_version must be 1")
    if profile.get("subsystem") != manifest.get("subsystem"):
        errors.append("REVIEW.json subsystem must match MAINTENANCE.json")
    if profile.get("mode") != "independent_then_reconcile":
        errors.append("REVIEW.json mode must be independent_then_reconcile")
    targets = profile.get("package_targets")
    allowed_targets = {str(manifest.get("runtime_dir", ""))}
    allowed_targets.update(edge.target for edge in normalize_dependency_edges(manifest) if edge.target)
    for owner in manifest.get("capability_ownership", []) or []:
        if isinstance(owner, Mapping) and owner.get("owner"):
            allowed_targets.add(str(owner["owner"]))
    if not isinstance(targets, list) or not targets:
        errors.append("REVIEW.json requires at least one package_target")
    else:
        seen_targets: set[str] = set()
        for target in targets:
            if not isinstance(target, str) or not re.fullmatch(r"_[A-Za-z0-9_]+", target):
                errors.append(f"invalid review package target: {target!r}")
                continue
            if target not in allowed_targets:
                errors.append(
                    f"review package target is outside declared subsystem architecture boundary: {target}"
                )
            if target in seen_targets:
                errors.append(f"duplicate review package target: {target}")
            seen_targets.add(target)
    lenses = profile.get("lenses")
    if not isinstance(lenses, list) or not lenses:
        errors.append("REVIEW.json requires at least one review lens")
        return errors
    ids: list[str] = []
    for i, lens in enumerate(lenses):
        if not isinstance(lens, Mapping):
            errors.append(f"review lens {i} must be an object")
            continue
        allowed_lens = {"id", "title", "required", "focus", "checks"}
        extra_lens = sorted(set(lens) - allowed_lens)
        if extra_lens:
            errors.append(f"review lens {i} contains unsupported fields: {', '.join(extra_lens)}")
        lid = str(lens.get("id", ""))
        ids.append(lid)
        if not _PROFILE_ID.fullmatch(lid):
            errors.append(f"invalid review lens id: {lid!r}")
        if not str(lens.get("title", "")).strip():
            errors.append(f"review lens {lid!r} missing title")
        if not isinstance(lens.get("required"), bool):
            errors.append(f"review lens {lid!r} missing boolean required")
        if not str(lens.get("focus", "")).strip():
            errors.append(f"review lens {lid!r} missing focus")
        checks = lens.get("checks")
        if not isinstance(checks, list) or not checks:
            errors.append(f"review lens {lid!r} requires checks")
            continue
        if len(checks) != len(set(str(v) for v in checks)):
            errors.append(f"review lens {lid!r} has duplicate checks")
        for check in checks:
            if check not in CHECKS:
                errors.append(f"review lens {lid!r} references unknown registered check: {check}")
    if len(ids) != len(set(ids)):
        errors.append("REVIEW.json has duplicate review lens IDs")
    policy = profile.get("pr_policy")
    if not isinstance(policy, Mapping):
        errors.append("REVIEW.json requires pr_policy")
    else:
        allowed_policy = {"blocking_severities", "require_required_lenses", "unavailable_release_gate_blocks_promotion"}
        extra_policy = sorted(set(policy) - allowed_policy)
        if extra_policy:
            errors.append("pr_policy contains unsupported fields: " + ", ".join(extra_policy))
        blocking = policy.get("blocking_severities")
        if not isinstance(blocking, list) or any(v not in {"ERROR", "WARNING"} for v in blocking):
            errors.append("pr_policy.blocking_severities must contain ERROR/WARNING only")
        elif len(blocking) != len(set(blocking)):
            errors.append("pr_policy.blocking_severities must be unique")
        for key in ("require_required_lenses", "unavailable_release_gate_blocks_promotion"):
            if not isinstance(policy.get(key), bool):
                errors.append(f"pr_policy.{key} must be boolean")
    return errors


def build_context(manifest_path: Path) -> ReviewContext:
    manifest_path = manifest_path.resolve()
    manifest = _json(manifest_path)
    repo = discover_repository_root(manifest_path)
    if repo is None:
        raise FileNotFoundError(f"cannot discover repository root from {manifest_path}")
    runtime_family = discover_runtime_sphinx_ext(manifest_path)
    maintenance_root = manifest_path.parent
    profile_rel = str(manifest.get("review_profile", ""))
    if not profile_rel:
        raise FileNotFoundError("MAINTENANCE.json does not declare review_profile")
    rel_path = Path(profile_rel)
    if rel_path.is_absolute():
        raise ValueError("review_profile must be a relative path inside the subsystem maintenance root")
    profile_path = (maintenance_root / rel_path).resolve()
    try:
        profile_path.relative_to(maintenance_root.resolve())
    except ValueError as exc:
        raise ValueError("review_profile escapes the subsystem maintenance root") from exc
    profile = _json(profile_path)
    skill_file: Path | None = None
    if manifest.get("skill_root"):
        candidate = repo / str(manifest["skill_root"]) / "SKILL.md"
        if candidate.is_file():
            skill_file = candidate
    return ReviewContext(manifest_path, manifest, profile, repo, runtime_family, maintenance_root, skill_file)


def review_package(ctx: ReviewContext, package: str) -> dict:
    """Review one sibling runtime package independently with safe static evidence."""
    findings: list[ReviewFinding] = []
    root = ctx.runtime_family / package
    if not root.is_dir():
        findings.append(_finding(
            "ERROR", "review.package-missing", f"review target package is missing: {package}",
            str(ctx.runtime_family.relative_to(ctx.repo)),
            "Remove stale review scope or restore the owned runtime package."
        ))
        return {"package": package, "status": "BLOCKED", "findings": [asdict(f) for f in findings]}
    py_files = sorted(root.rglob("*.py"))
    if not (root / "__init__.py").is_file():
        findings.append(_finding(
            "ERROR", "review.package-init-missing", f"runtime package lacks __init__.py: {package}",
            str(root.relative_to(ctx.repo)), "Restore the package boundary or remove the stale review target."
        ))
    if not py_files:
        findings.append(_finding(
            "ERROR", "review.package-empty", f"runtime package has no Python source: {package}",
            str(root.relative_to(ctx.repo)), "Restore the runtime package source or remove the stale review target."
        ))
    for path in py_files:
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            findings.append(_finding(
                "ERROR", "review.package-syntax", f"cannot parse {path.name}: {exc}",
                str(path.relative_to(ctx.repo)), "Fix syntax/encoding before reviewing behavior."
            ))
            continue
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        if any(name == "maintenances" or name.startswith("maintenances.") or name == "skills" or name.startswith("skills.") for name in imported):
            findings.append(_finding(
                "ERROR", "review.package-plane-leak", "runtime package imports maintenance/skill plane",
                str(path.relative_to(ctx.repo)), "Move maintenance/review logic out of runtime code."
            ))
    deps = sorted(runtime_imports(root, ctx.runtime_family))
    blocking = set(ctx.profile.get("pr_policy", {}).get("blocking_severities", ["ERROR"]))
    return {
        "package": package,
        "status": "BLOCKED" if any(f.severity in blocking for f in findings) else "GREEN",
        "python_files": len(py_files),
        "observed_sibling_imports": deps,
        "findings": [asdict(f) for f in findings],
    }


def run_lens(ctx: ReviewContext, lens: Mapping[str, object]) -> dict:
    findings: list[ReviewFinding] = []
    for check_id in lens.get("checks", []):
        findings.extend(CHECKS[str(check_id)](ctx))
    # Stable dedupe because several lenses may intentionally share a foundational check.
    unique: dict[tuple[str, str, str, str], ReviewFinding] = {}
    for finding in findings:
        key = (finding.severity, finding.code, finding.message, finding.evidence)
        unique[key] = finding
    findings = sorted(unique.values(), key=lambda f: (SEVERITIES.index(f.severity), f.code, f.evidence, f.message))
    blocking = set(ctx.profile.get("pr_policy", {}).get("blocking_severities", ["ERROR"]))
    return {
        "id": lens.get("id"),
        "title": lens.get("title"),
        "required": bool(lens.get("required")),
        "focus": lens.get("focus"),
        "status": "BLOCKED" if any(f.severity in blocking for f in findings) else "GREEN",
        "findings": [asdict(f) for f in findings],
    }
