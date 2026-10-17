"""
Static consistency lint for a Docker image-builder GitHub Actions workflow.

Verifies that ``jobs.<job>.strategy.matrix.include[*].python_version`` and the
job-level ``PY_VERSIONS_JSON`` env value describe the same Python-version set,
and that the "default version collapses to a bare tag alias" rule used by the
workflow's tag-assembly script can actually be satisfied.

Requires:

PyYAML (``yaml.safe_load`` only; the file is never executed).

Exit status:

0 : no findings.
1 : at least one finding.
2 : the workflow could not be read or does not have the expected structure.

Examples
--------
>>> # python check_py_versions_consistency.py .github/workflows/ci_docker_image_builder.yml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "Finding",
    "lint_workflow",
    "main",
]

_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+$")
_HEADER_EXPORT_RE = re.compile(r"export\s+PY_VERSIONS_JSON='([^']*)'")
_ROW_KEYS = frozenset({"variant", "interpreter", "python_version"})


class StructureError(ValueError):
    """Raised when the workflow lacks the structure this lint depends on."""


@dataclass(frozen=True)
class Finding:
    """A single lint finding.

    Parameters
    ----------
    code : str
        Stable identifier (``E001`` ...), suitable for grep / suppression.
    message : str
        Human-readable, actionable description.
    """

    code: str
    message: str

    def __str__(self) -> str:
        """Return the finding as a formatted string."""
        return f"{self.code}: {self.message}"


def _version_key(
    version: str,
) -> tuple[int, ...]:
    r"""Return a numeric sort key for an ``X.Y`` version string.

    Parameters
    ----------
    version : str
        Version string matching ``^[0-9]+\.[0-9]+$``.

    Returns
    -------
    tuple of int
        ``("3.10")`` -> ``(3, 10)``; avoids the lexicographic trap
        ``"3.9" > "3.10"``.
    """
    return tuple(int(part) for part in version.split("."))


def _load_job(
    path: Path,
    job_id: str,
) -> tuple[str, dict[str, Any]]:
    """Read the workflow text and return ``(raw_text, job_mapping)``.

    Raises
    ------
    StructureError
        If the file is unreadable, not valid YAML, or lacks ``jobs.<job_id>``.
    """
    try:
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        raise StructureError(f"cannot read/parse {path}: {exc}") from exc
    job = (doc or {}).get("jobs", {}).get(job_id) if isinstance(doc, dict) else None
    if not isinstance(job, dict):
        raise StructureError(f"job {job_id!r} not found under 'jobs'")
    return text, job


def _extract_rows(
    job: dict[str, Any],
) -> list[Any]:
    """Return the static ``strategy.matrix.include`` list of a job.

    Raises
    ------
    StructureError
        If ``include`` is absent or is not a static list (e.g. ``fromJSON``).
    """
    include = ((job.get("strategy") or {}).get("matrix") or {}).get("include")
    if not isinstance(include, list):
        raise StructureError(
            "strategy.matrix.include is missing or not a static list",
        )
    return include


def _check_rows(
    rows: Sequence[Any],
    variants: Iterable[str],
    interpreters: Iterable[str],
) -> tuple[list[Finding], list[tuple[str, str, str]]]:
    """Validate row schema and duplicates; return findings and clean row keys."""
    findings: list[Finding] = []
    clean: list[tuple[str, str, str]] = []
    allowed_v, allowed_i = set(variants), set(interpreters)
    for idx, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            findings.append(
                Finding(
                    "E001",
                    f"row #{idx}: keys must be exactly {sorted(_ROW_KEYS)}, got {row!r}",
                ),
            )
            continue
        variant, interp, pyver = (
            row["variant"],
            row["interpreter"],
            row["python_version"],
        )
        if not isinstance(pyver, str):
            findings.append(
                Finding(
                    "E001",
                    f"row #{idx}: python_version {pyver!r} is {type(pyver).__name__}, not str "
                    "(unquoted YAML float? 3.10 parses as 3.1) -- quote it",
                )
            )
            continue
        if not _VERSION_RE.match(pyver):
            findings.append(
                Finding(
                    "E001", f"row #{idx}: python_version {pyver!r} does not match X.Y"
                )
            )
            continue
        if variant not in allowed_v or interp not in allowed_i:
            findings.append(
                Finding(
                    "E001",
                    f"row #{idx}: variant={variant!r}/interpreter={interp!r} outside "
                    f"{sorted(allowed_v)} x {sorted(allowed_i)} (docker_plan exits 2 on these)",
                )
            )
            continue
        clean.append((variant, interp, pyver))
    for key, count in Counter(clean).items():
        if count > 1:
            findings.append(
                Finding(
                    "E002",
                    f"duplicate matrix row {key} appears {count} times",
                ),
            )
    return findings, clean


def _parse_py_versions(
    raw: Any,
) -> tuple[list[Finding], list[str] | None]:
    """Parse and validate the ``PY_VERSIONS_JSON`` contract.

    Contract: non-empty JSON array of unique ``"X.Y"`` strings, ordered
    oldest -> newest by *numeric* comparison (the last element is the default).
    """
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError as exc:
        return [Finding("E003", f"PY_VERSIONS_JSON is not valid JSON: {exc}")], None
    if not isinstance(parsed, list) or not parsed:
        return [
            Finding("E003", "PY_VERSIONS_JSON must be a non-empty JSON array")
        ], None
    bad = [v for v in parsed if not (isinstance(v, str) and _VERSION_RE.match(v))]
    if bad:
        return [
            Finding(
                "E003",
                f"PY_VERSIONS_JSON elements must be quoted 'X.Y' strings; bad: {bad!r}",
            )
        ], None
    findings: list[Finding] = []
    if len(set(parsed)) != len(parsed):
        findings.append(
            Finding("E003", f"PY_VERSIONS_JSON has duplicates: {parsed}"),
        )
    if parsed != sorted(parsed, key=_version_key):
        findings.append(
            Finding(
                "E003",
                f"PY_VERSIONS_JSON must be ordered oldest -> newest (numeric); got {parsed}. "
                f"The LAST element is the default version.",
            )
        )
    return findings, parsed


def lint_workflow(
    path: Path,
    job_id: str = "docker",
    env_key: str = "PY_VERSIONS_JSON",
    variants: Sequence[str] = ("runtime", "devel"),
    interpreters: Sequence[str] = ("python", "jupyter"),
) -> list[Finding]:
    """Lint one workflow file.

    Parameters
    ----------
    path : pathlib.Path
        Workflow YAML file.
    job_id : str, default="docker"
        Job that owns the matrix and the version env.
    env_key : str, default="PY_VERSIONS_JSON"
        Job-level env key holding the JSON array of supported versions.
    variants, interpreters : sequence of str
        Allowed matrix axis values (mirrors the ``case`` statements in the
        workflow's tag-plan script).

    Returns
    -------
    list of Finding
        Empty when the workflow is consistent.

    Raises
    ------
    StructureError
        If the workflow lacks the expected job / matrix / env structure.
    """
    text, job = _load_job(path, job_id)
    rows = _extract_rows(job)
    env = job.get("env") or {}
    if env_key not in env:
        raise StructureError(f"jobs.{job_id}.env.{env_key} not found")

    findings, clean = _check_rows(rows, variants, interpreters)
    json_findings, versions = _parse_py_versions(env[env_key])
    findings += json_findings
    if versions is None:
        return findings

    matrix_versions = {v for _, _, v in clean}
    for ver in sorted(matrix_versions - set(versions), key=_version_key):
        rows_hit = [f"{a}/{b}" for a, b, c in clean if c == ver]
        findings.append(
            Finding(
                "E004",
                f"python_version {ver!r} used by {rows_hit} is not in {env_key}; those rows exit 2 at the in-workflow guard",
            )
        )
    for ver in sorted(set(versions) - matrix_versions, key=_version_key):
        findings.append(
            Finding(
                "E005",
                f"{env_key} lists {ver!r} but no matrix row builds it (never built; not caught by the in-workflow guard)",
            )
        )

    default = versions[-1]
    for variant, interp in product(variants, interpreters):
        if (variant, interp, default) not in set(clean):
            findings.append(
                Finding(
                    "E006",
                    f"default version {default!r} (last of {env_key}) has no row for {variant}/{interp}: "
                    "its bare tag alias is never published and any existing one goes stale silently",
                )
            )

    for match in _HEADER_EXPORT_RE.finditer(text):
        try:
            copy = json.loads(match.group(1))
        except json.JSONDecodeError:
            copy = None
        if copy != versions:
            findings.append(
                Finding(
                    "E007",
                    f"comment copy of {env_key} ({match.group(1)}) differs from the job env value {versions}",
                )
            )
    return findings


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Command-line entry point; see module docstring for exit codes."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "workflow",
        type=Path,
        help="path to the workflow YAML",
    )
    parser.add_argument(
        "--job",
        default="docker",
        help="job id (default: docker)",
    )
    parser.add_argument(
        "--env-key",
        default="PY_VERSIONS_JSON",
        help="env key (default: PY_VERSIONS_JSON)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        findings = lint_workflow(
            args.workflow,
            job_id=args.job,
            env_key=args.env_key,
        )
    except StructureError as exc:
        logger.error("ERROR: %s", exc)
        return 2
    for finding in findings:
        logger.info("%s", finding)
    logger.info("%d finding(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
